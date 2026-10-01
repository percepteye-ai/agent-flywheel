"""Raw error payloads are the load-bearing signal, not a diagnostic nicety.

ReTool §3.3 attributes emergent self-correction to feeding verbatim tracebacks
back to the model. Everything downstream that learns from failure — recovery-
window preference mining, judge distillation, hint extraction — reads this
text. If it does not survive capture, those channels are starved silently.

The failure mode this pins is specific and was live: the error body got a
QUARTER of the ordinary output budget and was head-truncated, so on any real
traceback the first thing deleted was the exception class and message at the
end — the only part a model can learn to correct from.
"""
from __future__ import annotations

import json
import re

import pytest

from agent_flywheel._body_bound import ELISION as _ELISION
from agent_flywheel._body_bound import MAX_BODY_CHARS as _MAX_ERROR_CHARS
from agent_flywheel._body_bound import bounded as _bounded
from agent_flywheel.event_stream import StreamFormat, parse_tool_calls

_TRACEBACK_TAIL = 'ValueError: city "Paris " has a trailing space'


def _long_traceback(n_frames: int = 400) -> str:
    frames = "".join(
        f'  File "/app/mod{i}.py", line {i}, in handler\n    return step{i}()\n'
        for i in range(n_frames)
    )
    return "Traceback (most recent call last):\n" + frames + _TRACEBACK_TAIL


def _stated_count(out: str) -> int:
    match = re.search(r"…\[(\d+) chars elided by the SDK capture bound\]…", out)
    assert match is not None, out
    return int(match.group(1))


def test_a_body_within_the_bound_is_returned_byte_identical():
    """The common case must not acquire a marker it did not earn."""
    body = "stdout: fine\n"
    assert _bounded(body) == body
    assert _ELISION.format(dropped=1) not in _bounded(body)


def test_the_exception_line_survives_a_long_traceback():
    """Head-truncation deleted exactly this."""
    out = _bounded(_long_traceback(), _MAX_ERROR_CHARS)
    assert out.endswith(_TRACEBACK_TAIL)
    assert out.startswith("Traceback (most recent call last):")


def test_the_bound_is_honoured_including_its_own_marker():
    """A bound that its own annotation can push past is not a bound."""
    out = _bounded(_long_traceback(), _MAX_ERROR_CHARS)
    assert len(out) == _MAX_ERROR_CHARS


def test_the_elision_is_visible_and_counts_what_went():
    """A reader must be able to tell a window from a whole, and a census must
    be able to count it without a new wire field. The count is what was
    actually removed: the marker's own length comes out of the budget, so
    ``len(text) - limit`` would understate it by exactly that length."""
    text = _long_traceback()
    out = _bounded(text, _MAX_ERROR_CHARS)
    marker = _ELISION.format(dropped=_stated_count(out))
    kept = len(out) - len(marker)
    assert _stated_count(out) == len(text) - kept


def test_the_count_is_exact_where_the_marker_gains_a_digit():
    """The marker's length depends on the count's digits, and the count on the
    marker's length. Across the 999 -> 1000 boundary both must still agree."""
    for size in range(1000, 1100):
        text = "x" * size
        out = _bounded(text, 100)
        marker = _ELISION.format(dropped=_stated_count(out))
        assert len(out) == 100
        assert _stated_count(out) == size - (len(out) - len(marker))


def test_a_limit_too_small_for_the_marker_keeps_the_tail():
    """Degenerate, but it must degrade toward the half worth keeping."""
    out = _bounded(_long_traceback(), 20)
    assert len(out) == 20
    assert out.endswith(_TRACEBACK_TAIL[-20:])


_FORMAT = StreamFormat(
    name="test-tool-events",
    start_type="tool.start",
    end_type="tool.end",
    call_id=("id",),
    tool_name=("name",),
    args=("args",),
    result=("result",),
    error_flag=("isError",),
)


def _failed_call(result: object) -> str:
    start = {"type": "tool.start", "id": "c1", "name": "ipython", "args": {}}
    end = {
        "type": "tool.end", "id": "c1", "name": "ipython", "isError": True,
        "result": result,
    }
    return json.dumps(start) + "\n" + json.dumps(end)


def _as_content(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}]}


# ── the parser, one test per place a body is read ──────────────────────────
#
# Each body path bounds the body once and a failed call's error IS that body.
# A test that drives only one path proves that path: the content[] case alone
# stayed green with the details.stdout bound reverted to a head cut.


@pytest.mark.parametrize("result", [
    pytest.param(_as_content(_long_traceback()), id="content[].text"),
    pytest.param({"details": {"stdout": _long_traceback()}}, id="details.stdout"),
    pytest.param(_long_traceback(), id="a result that is a bare string"),
])
def test_every_body_path_reports_the_exception_line(result):
    """Through the parser, not just the helper: the output and the error a
    failed call reports both end in the exception, within the bound, with
    exactly one marker, whichever shape the result had."""
    calls = parse_tool_calls(_failed_call(result), formats=(_FORMAT,))
    assert calls is not None and len(calls) == 1
    call = calls[0]
    assert call.outcome == "failed"
    for body in (call.output, call.error):
        assert isinstance(body, str)
        assert body.startswith("Traceback (most recent call last):")
        assert body.endswith(_TRACEBACK_TAIL)
        assert len(body) == _MAX_ERROR_CHARS
        assert body.count("elided by the SDK capture bound") == 1
    assert call.error == _bounded(_long_traceback())


def test_a_short_failure_is_reported_verbatim():
    calls = parse_tool_calls(_failed_call(_as_content(_TRACEBACK_TAIL)),
                             formats=(_FORMAT,))
    assert calls is not None and calls[0].error == _TRACEBACK_TAIL


# ── the host recorder, driven as the host drives it ────────────────────────

def _record_bytes_cap() -> int:
    import importlib.util

    from agent_flywheel.hosts import RECORDER

    spec = importlib.util.spec_from_file_location("_rec_under_test", RECORDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._MAX_RECORD_BYTES


def _after_tool(tmp_path, response: dict, tool_input: dict | None = None,
                *, importtime: bool = False):
    """One Gemini ``AfterTool`` event through the real recorder: argv, JSON on
    stdin. Returns the one line it appended (bytes) and its stderr."""
    import os
    import subprocess
    import sys

    from agent_flywheel.hosts import RECORDER

    event = {"hook_event_name": "AfterTool", "session_id": "s1",
             "tool_name": "run_shell_command",
             "tool_input": tool_input or {"command": "pytest -x"},
             "tool_response": response}
    env = {k: v for k, v in os.environ.items() if not k.startswith("PERCEPTEYE_")}
    env["PERCEPTEYE_TRAJECTORY_DIR"] = str(tmp_path)
    argv = [sys.executable] + (["-X", "importtime"] if importtime else [])
    r = subprocess.run([*argv, str(RECORDER)], input=json.dumps(event),
                       text=True, capture_output=True, env=env, timeout=60)
    assert r.returncode == 0 and r.stdout == "", (r.returncode, r.stdout)
    lines = (tmp_path / "tool_calls.jsonl").read_bytes().splitlines()
    assert len(lines) == 1, lines
    return lines[0], r.stderr


def test_the_recorder_reports_the_error_the_parser_does(tmp_path):
    """It cut an error to its first 1000 chars after the parser stopped doing
    so: one wire field, two shapes, and the recorder's deleted the exception."""
    line, _ = _after_tool(tmp_path, {"error": _long_traceback()})
    record = json.loads(line)
    assert record["outcome"] == "failed"
    assert record["error"].endswith(_TRACEBACK_TAIL)
    assert record["error"] == _bounded(_long_traceback())


def test_both_bodies_long_still_fit_one_write(tmp_path):
    """The recorder's line cap is what keeps concurrent appends whole, and two
    bodies at the full bound do not fit it together -- less still once JSON
    escapes a traceback's quotes and newlines and a CJK message's every char.
    The bodies give way, each still ending in its tail and counting what it
    lost from the body the tool returned; the arguments, which retry detection
    keys on, survive."""
    error = _long_traceback() + " 文件不存在" * 300
    output = "测试输出 \"quoted\"\n" * 2000 + _TRACEBACK_TAIL
    args = {f"arg{i}": "v" * 900 for i in range(3)}
    line, _ = _after_tool(tmp_path, {"llmContent": output, "error": error}, args)
    assert len(line) <= _record_bytes_cap()
    record = json.loads(line)
    assert set(record["arguments"]) == set(args)
    for raw, kept in ((error.strip(), record["error"]),
                      (output.strip(), record["output"])):
        assert kept.endswith(raw[-20:])
        assert kept.count("elided by the SDK capture bound") == 1
        marker = _ELISION.format(dropped=_stated_count(kept))
        assert _stated_count(kept) == len(raw) - (len(kept) - len(marker))


def test_bounding_an_error_does_not_cost_a_tool_call_the_sdk_import(tmp_path):
    """The recorder loads the bound from its file, never ``import
    agent_flywheel`` -- the per-tool-call cost it exists to avoid. The marker
    in the record is the positive control: the bound did run."""
    line, stderr = _after_tool(tmp_path, {"error": _long_traceback()},
                               importtime=True)
    assert "import time:" in stderr, "positive control: -X importtime printed"
    assert "agent_flywheel" not in stderr, "the per-tool-call path imported the SDK"
    assert "elided by the SDK capture bound" in json.loads(line)["error"]


# ── one traceback, one error, whichever producer saw it ────────────────────


def test_every_producer_reports_one_error_for_one_traceback(tmp_path):
    """The same field from the parser, the host recorder and both adapters'
    returned-error paths. Their raised-error paths are pinned beside each
    adapter's other tests; the Node SDK's, in test_body_bound_parity."""
    pytest.importorskip("langchain_core")
    from langchain_core.messages.tool import ToolMessage

    from agent_flywheel.adapters.adk import _classify as adk_classify
    from agent_flywheel.adapters.langchain import _classify_output

    tb = _long_traceback()
    (parsed,) = parse_tool_calls(_failed_call(tb), formats=(_FORMAT,)) or [None]
    recorded = json.loads(_after_tool(tmp_path, {"error": tb})[0])
    seen = {
        "parser": parsed.error,
        "host recorder": recorded["error"],
        "adk": adk_classify({"error": tb})[1],
        "langchain": _classify_output(
            ToolMessage(content=tb, tool_call_id="c", status="error"),
            ToolMessage)[1],
    }
    assert seen == dict.fromkeys(seen, _bounded(tb))
