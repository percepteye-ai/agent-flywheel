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

from agent_flywheel.event_stream import (
    _ELISION,
    _MAX_ERROR_CHARS,
    _MAX_OUTPUT_CHARS,
    StreamFormat,
    _bounded,
    parse_tool_calls,
)

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


def test_errors_are_not_budgeted_below_ordinary_output():
    """They used to get a quarter. The payload most worth keeping was the one
    kept least."""
    assert _MAX_ERROR_CHARS >= _MAX_OUTPUT_CHARS


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


def _failed_call(text: str) -> str:
    start = {"type": "tool.start", "id": "c1", "name": "ipython", "args": {}}
    end = {
        "type": "tool.end", "id": "c1", "name": "ipython", "isError": True,
        "result": {"content": [{"type": "text", "text": text}]},
    }
    return json.dumps(start) + "\n" + json.dumps(end)


def test_a_failed_tool_call_reports_the_exception_line():
    """Through the parser, not just the helper: the error a failed call
    reports still ends in the exception, within the error budget, and its
    output is bounded the same way."""
    calls = parse_tool_calls(_failed_call(_long_traceback()), formats=(_FORMAT,))
    assert calls is not None and len(calls) == 1
    call = calls[0]
    assert call.outcome == "failed"
    assert call.error is not None
    assert call.error.endswith(_TRACEBACK_TAIL)
    assert call.error.startswith("Traceback (most recent call last):")
    assert len(call.error) <= _MAX_ERROR_CHARS
    assert call.error.count("elided by the SDK capture bound") == 1


def test_a_short_failure_is_reported_verbatim():
    calls = parse_tool_calls(_failed_call(_TRACEBACK_TAIL), formats=(_FORMAT,))
    assert calls is not None and calls[0].error == _TRACEBACK_TAIL
