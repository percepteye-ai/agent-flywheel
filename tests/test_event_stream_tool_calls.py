"""A process agent can report tool calls without being instrumented.

The framework adapters bind to a Python object. A Node CLI or a compiled binary
has none, so its tool calls were unobservable and ``reports_tool_calls=False``
was the honest setting — which costs the per-call verdict the reward reads, the
retry signal, and every bit of traceability a customer would debug a rollout
from. An observability stack that cannot say which tools ran and whether they
worked is not one.

A modern CLI agent already emits that stream for its own UI. These events are
verbatim from prime-agent ``--mode json``.
"""
from __future__ import annotations

import json
import os

from percepteye_agent_flywheel.event_stream import parse_tool_calls

_START = {
    "type": "tool_execution_start",
    "toolCallId": "toolu_016F",
    "toolName": "ipython",
    "args": {"code": "open('hi.txt','w').write('hello')"},
}
_END_OK = {
    "type": "tool_execution_end",
    "toolCallId": "toolu_016F",
    "toolName": "ipython",
    "result": {
        "content": [{"type": "text", "text": "hello\n"}],
        "details": {"durationMs": 7, "status": "ok", "stdout": "hello\n"},
        "isError": False,
    },
    "isError": False,
}


def _stream(*events) -> str:
    return "\n".join(json.dumps(e) for e in events)


# ── the capture ─────────────────────────────────────────────────────────────


def test_a_successful_call_is_recovered_from_the_stream():
    calls = parse_tool_calls(_stream(_START, _END_OK))
    assert calls is not None and len(calls) == 1
    c = calls[0]
    assert c.name == "ipython"
    assert c.outcome == "ok"
    assert c.tool_call_id == "toolu_016F"
    assert c.arguments["code"].startswith("open('hi.txt'")
    assert c.output == "hello\n"
    assert c.latency_ms == 7.0


def test_a_failed_call_carries_its_error_body():
    end = {**_END_OK, "isError": True,
           "result": {**_END_OK["result"], "isError": True,
                      "content": [{"type": "text", "text": "NameError: x"}]}}
    calls = parse_tool_calls(_stream(_START, end))
    assert calls[0].outcome == "failed"
    assert "NameError" in calls[0].error


def test_arguments_come_from_the_start_event_when_the_end_omits_them():
    """The pair is joined on toolCallId — the arguments are what retry
    detection keys on, so losing them loses the signal."""
    end = {k: v for k, v in _END_OK.items() if k != "args"}
    calls = parse_tool_calls(_stream(_START, end))
    assert calls[0].arguments["code"].startswith("open('hi.txt'")


def test_multiple_calls_are_recovered_in_order():
    s2 = {**_START, "toolCallId": "t2", "toolName": "bash", "args": {"cmd": "ls"}}
    e2 = {**_END_OK, "toolCallId": "t2", "toolName": "bash"}
    calls = parse_tool_calls(_stream(_START, _END_OK, s2, e2))
    assert [c.name for c in calls] == ["ipython", "bash"]


# ── absence stays absence ───────────────────────────────────────────────────


def test_a_stream_with_no_tool_events_returns_None_not_empty():
    """``None`` is "we observed nothing"; ``[]`` would positively assert the
    agent made zero calls. A stream we cannot read must never become a claim."""
    assert parse_tool_calls('{"type":"text","text":"hi"}') is None
    assert parse_tool_calls("") is None
    assert parse_tool_calls("not json at all\nplain log line") is None


def test_a_call_that_never_says_whether_it_worked_is_unknown():
    """Tri-state by omission. Rounding this up to "ok" is the exact failure the
    whole outcome contract exists to prevent."""
    end = {k: v for k, v in _END_OK.items() if k != "isError"}
    end["result"] = {"content": [{"type": "text", "text": "done"}]}
    calls = parse_tool_calls(_stream(_START, end))
    assert calls[0].outcome == "unknown"
    # The contract discards status when the outcome is unknown; we never
    # manufacture one for a local tool anyway.
    assert calls[0].status_code is None


def test_an_unnameable_call_is_dropped_rather_than_invented():
    """Retry detection joins on name + arguments. A fabricated name would
    corrupt every such join; losing one outcome is the cheaper error."""
    end = {"type": "tool_execution_end", "toolCallId": "zz", "isError": False}
    assert parse_tool_calls(_stream(end)) == []


def test_interleaved_noise_does_not_discard_the_outcomes():
    """A CLI logs its own chatter alongside the event stream."""
    noisy = "\n".join([
        "starting up…", json.dumps(_START), "{ not valid json",
        json.dumps(_END_OK), "done.",
    ])
    calls = parse_tool_calls(noisy)
    assert len(calls) == 1 and calls[0].outcome == "ok"


# ── the ledger wins ─────────────────────────────────────────────────────────


def test_the_ledger_is_not_overwritten_by_the_stream():
    """An agent that called record_tool_call asserted something deliberate
    about itself; a parse must never overwrite it."""
    from percepteye_agent_flywheel.contract import RolloutOutput, ToolCallOutcome
    from percepteye_agent_flywheel.runner import _attach_stream_tool_calls

    recorded = [ToolCallOutcome(
        name="from_ledger", arguments={}, outcome="ok", status_code=None)]
    out = RolloutOutput(final_text="", tool_calls=list(recorded))
    _attach_stream_tool_calls(out, _stream(_START, _END_OK))
    assert [c.name for c in out.tool_calls] == ["from_ledger"]


def test_the_stream_fills_in_when_the_ledger_recorded_nothing():
    from percepteye_agent_flywheel.contract import RolloutOutput
    from percepteye_agent_flywheel.runner import _attach_stream_tool_calls

    out = RolloutOutput(final_text="")
    assert out.tool_calls is None
    _attach_stream_tool_calls(out, _stream(_START, _END_OK))
    assert out.tool_calls is not None
    assert [c.name for c in out.tool_calls] == ["ipython"]


def test_an_empty_assertion_of_zero_calls_is_preserved():
    """``[]`` from the ledger is a positive claim and must survive."""
    from percepteye_agent_flywheel.contract import RolloutOutput
    from percepteye_agent_flywheel.runner import _attach_stream_tool_calls

    out = RolloutOutput(final_text="", tool_calls=[])
    _attach_stream_tool_calls(out, _stream(_START, _END_OK))
    assert out.tool_calls == []


# ── the vocabulary is data, not a parser ────────────────────────────────────


def test_an_otel_instrumented_agent_is_read_with_no_new_code():
    """The generality claim, made concrete. A process agent that already emits
    GenAI spans as JSON lines gets outcome capture for free — supporting the
    next CLI agent is an entry in FORMATS, not a parser."""
    stream = _stream(
        {"type": "gen_ai.tool.start", "gen_ai.tool.call.id": "c1",
         "gen_ai.tool.name": "search", "gen_ai.tool.input": {"q": "x"}},
        {"type": "gen_ai.tool.end", "gen_ai.tool.call.id": "c1",
         "gen_ai.tool.name": "search", "gen_ai.tool.output": "3 hits"},
    )
    calls = parse_tool_calls(stream)
    assert calls is not None and len(calls) == 1
    assert calls[0].name == "search"
    assert calls[0].arguments == {"q": "x"}
    # No error.type on the span, and OTel omits it on success — but omission
    # is not an assertion of success, so this stays unknown.
    assert calls[0].outcome == "unknown"


def test_the_otel_error_convention_is_read_as_a_failure():
    """``error.type`` carries the exception class and is absent on success —
    a non-empty string IS the failure verdict in that convention."""
    calls = parse_tool_calls(_stream(
        {"type": "gen_ai.tool.end", "gen_ai.tool.call.id": "c1",
         "gen_ai.tool.name": "search", "error.type": "TimeoutError"},
    ))
    assert calls[0].outcome == "failed"


def test_an_empty_error_string_is_absence_not_a_failure():
    calls = parse_tool_calls(_stream(
        {"type": "gen_ai.tool.end", "gen_ai.tool.call.id": "c1",
         "gen_ai.tool.name": "search", "error.type": "  "},
    ))
    assert calls[0].outcome == "unknown"


def test_a_stream_is_matched_by_its_events_not_by_a_declared_name():
    """A name is a claim; an event type is evidence. Nothing here trusts an
    agent's self-description."""
    from percepteye_agent_flywheel.event_stream import FORMATS

    assert {f.name for f in FORMATS} >= {"prime-agent", "otel-genai"}
    # Two vocabularies in one stream: the first seen wins and the other's
    # events are ignored rather than silently merged under one call id.
    mixed = _stream(_START, _END_OK,
                    {"type": "gen_ai.tool.end", "gen_ai.tool.call.id": "z",
                     "gen_ai.tool.name": "other"})
    names = [c.name for c in parse_tool_calls(mixed)]
    assert names == ["ipython"]


# ── every exit path captures, not just the clean one ────────────────────────


def _agent_that_emits_then(exit_code: int, tmp_path):
    """A stub CLI that prints a tool-execution pair and then exits as told."""
    import stat as _stat
    p = tmp_path / "agent.sh"
    p.write_text(
        "#!/bin/sh\n"
        f"cat <<'JSON'\n{json.dumps(_START)}\n{json.dumps(_END_OK)}\nJSON\n"
        f"exit {exit_code}\n"
    )
    p.chmod(p.stat().st_mode | _stat.S_IEXEC)
    return str(p)


def _run_command(tmp_path, script, timeout_s=30.0):
    import time as _t

    from percepteye_agent_flywheel.contract import RolloutRequest
    from percepteye_agent_flywheel.runner import run_rollout_as_command

    req = RolloutRequest(
        rollout_id="rl_x", turn_input="go",
        llm_url="https://gw.invalid/v1", api_key="k",
        artifacts_dir="rollouts/rl_x", deadline_ts=_t.time() + 300,
        lease_token="lt",
    )
    return run_rollout_as_command(
        ["sh", script], req, ledger_dir=str(tmp_path), timeout_s=timeout_s,
        env=dict(os.environ), cwd=str(tmp_path))


def test_a_nonzero_exit_still_captures_its_tool_calls(tmp_path):
    """THE regression. Every rollout on run b60067db exited non-zero, so
    tool_calls came back null even though the parser worked — capture had been
    attached to the clean-exit path only, exactly as the artifact collection
    had been before it."""
    out = _run_command(tmp_path, _agent_that_emits_then(1, tmp_path))
    # The agent RAN and exited non-zero — its own failure, not a
    # failure to start it. Popen raises for the latter.
    assert out.exclusion_reason == "agent_exit_nonzero"
    assert out.tool_calls is not None, "capture skipped on the failure path"
    assert [c.name for c in out.tool_calls] == ["ipython"]


def test_a_clean_exit_captures_its_tool_calls(tmp_path):
    out = _run_command(tmp_path, _agent_that_emits_then(0, tmp_path))
    assert out.exclusion_reason is None
    assert [c.name for c in out.tool_calls] == ["ipython"]


def test_a_timeout_kill_still_captures_what_ran(tmp_path):
    """Killed on the clock, but it ran. Whatever it did before the kill is
    evidence."""
    import stat as _stat
    p = tmp_path / "slow.sh"
    p.write_text(
        "#!/bin/sh\n"
        f"cat <<'JSON'\n{json.dumps(_START)}\n{json.dumps(_END_OK)}\nJSON\n"
        "sleep 30\n"
    )
    p.chmod(p.stat().st_mode | _stat.S_IEXEC)
    out = _run_command(tmp_path, str(p), timeout_s=1.0)
    assert out.exclusion_reason == "deadline_exceeded"
    assert out.tool_calls is not None, "capture skipped on the timeout path"
    assert [c.name for c in out.tool_calls] == ["ipython"]
