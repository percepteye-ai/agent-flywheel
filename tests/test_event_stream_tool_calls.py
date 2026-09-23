"""A process agent can report tool calls without being instrumented.

The framework adapters bind to a Python object. A Node CLI or a compiled binary
has none, so its tool calls were unobservable and ``reports_tool_calls=False``
was the honest setting — which costs the per-call verdict the reward reads, the
retry signal, and every bit of traceability a customer would debug a rollout
from. An observability stack that cannot say which tools ran and whether they
worked is not one.

A modern CLI agent already emits that stream for its own UI. These fixtures are
verbatim from prime-agent ``--mode json`` and are interpreted only through the
explicit optional Prime adapter.
"""
from __future__ import annotations

import json
import os

from agent_flywheel.adapters.prime_agent import (
    PRIME_TOOL_FORMAT,
    PrimeAgentOutputAdapter,
)
from agent_flywheel.event_stream import (
    parse_tool_calls,
    parse_tool_calls_with_omissions,
)

_PRIME = PrimeAgentOutputAdapter()


def _prime_calls(stdout: str):
    return parse_tool_calls(stdout, formats=(PRIME_TOOL_FORMAT,))


def _prime_calls_with_omissions(stdout: str):
    return parse_tool_calls_with_omissions(
        stdout, formats=(PRIME_TOOL_FORMAT,)
    )

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
    calls = _prime_calls(_stream(_START, _END_OK))
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
    calls = _prime_calls(_stream(_START, end))
    assert calls[0].outcome == "failed"
    assert "NameError" in calls[0].error


def test_prime_details_error_overrides_both_false_error_flags():
    """The live Prime shape: execute() returned, so both generic flags are
    false, but the IPython cell itself raised and details carries the truth."""
    end = {
        **_END_OK,
        "isError": False,
        "result": {
            **_END_OK["result"],
            "isError": False,
            "content": [{"type": "text", "text": "NameError: missing"}],
            "details": {
                "durationMs": 7,
                "status": "error",
                "stdout": "NameError: name 'missing' is not defined",
            },
        },
    }
    calls = _prime_calls(_stream(_START, end))
    assert calls[0].outcome == "failed"
    assert "NameError" in (calls[0].error or "")


def test_prime_details_aborted_overrides_both_false_error_flags():
    """Prime's IPython executor uses ``aborted`` for an interrupted cell;
    its own adapter treats that as an error even when generic flags are false.
    """
    end = {
        **_END_OK,
        "isError": False,
        "result": {
            **_END_OK["result"],
            "isError": False,
            "content": [{"type": "text", "text": "Execution aborted"}],
            "details": {
                "durationMs": 7,
                "status": "aborted",
                "stdout": "Execution aborted",
            },
        },
    }
    calls = _prime_calls(_stream(_START, end))
    assert calls[0].outcome == "failed"
    assert "aborted" in (calls[0].error or "").lower()


def test_nested_true_error_flag_also_overrides_an_outer_false_flag():
    end = {
        **_END_OK,
        "isError": False,
        "result": {**_END_OK["result"], "isError": True},
    }
    assert _prime_calls(_stream(_START, end))[0].outcome == "failed"


def test_arguments_come_from_the_start_event_when_the_end_omits_them():
    """The pair is joined on toolCallId — the arguments are what retry
    detection keys on, so losing them loses the signal."""
    end = {k: v for k, v in _END_OK.items() if k != "args"}
    calls = _prime_calls(_stream(_START, end))
    assert calls[0].arguments["code"].startswith("open('hi.txt'")


def test_multiple_calls_are_recovered_in_order():
    s2 = {**_START, "toolCallId": "t2", "toolName": "bash", "args": {"cmd": "ls"}}
    e2 = {**_END_OK, "toolCallId": "t2", "toolName": "bash"}
    calls = _prime_calls(_stream(_START, _END_OK, s2, e2))
    assert [c.name for c in calls] == ["ipython", "bash"]


# ── absence stays absence ───────────────────────────────────────────────────


def test_a_stream_with_no_tool_events_returns_None_not_empty():
    """``None`` is "we observed nothing"; ``[]`` would positively assert the
    agent made zero calls. A stream we cannot read must never become a claim."""
    assert _prime_calls('{"type":"text","text":"hi"}') is None
    assert _prime_calls("") is None
    assert _prime_calls("not json at all\nplain log line") is None


def test_a_call_that_never_says_whether_it_worked_is_unknown():
    """Tri-state by omission. Rounding this up to "ok" is the exact failure the
    whole outcome contract exists to prevent."""
    end = {k: v for k, v in _END_OK.items() if k != "isError"}
    end["result"] = {"content": [{"type": "text", "text": "done"}]}
    calls = _prime_calls(_stream(_START, end))
    assert calls[0].outcome == "unknown"
    # The contract discards status when the outcome is unknown; we never
    # manufacture one for a local tool anyway.
    assert calls[0].status_code is None


def test_an_unnameable_call_is_dropped_rather_than_invented():
    """Retry detection joins on name + arguments. A fabricated name would
    corrupt every such join; losing one outcome is the cheaper error."""
    end = {"type": "tool_execution_end", "toolCallId": "zz", "isError": False}
    assert _prime_calls(_stream(end)) == []


def test_an_unnameable_call_is_reported_as_omitted_to_the_runner():
    end = {"type": "tool_execution_end", "toolCallId": "zz", "isError": False}
    parsed = _prime_calls_with_omissions(_stream(end))
    assert parsed.calls == []
    assert parsed.omitted_count == 1


def test_an_incomplete_start_is_reported_as_omitted_alongside_survivors():
    incomplete = {
        **_START,
        "toolCallId": "never-ended",
        "toolName": "write_file",
    }
    parsed = _prime_calls_with_omissions(
        _stream(_START, _END_OK, incomplete)
    )
    assert [call.name for call in (parsed.calls or [])] == ["ipython"]
    assert parsed.omitted_count == 1


def test_interleaved_noise_does_not_discard_the_outcomes():
    """A CLI logs its own chatter alongside the event stream."""
    noisy = "\n".join([
        "starting up…", json.dumps(_START), "{ not valid json",
        json.dumps(_END_OK), "done.",
    ])
    calls = _prime_calls(noisy)
    assert len(calls) == 1 and calls[0].outcome == "ok"


# ── the trajectory wins ─────────────────────────────────────────────────────────


def test_the_trajectory_is_not_overwritten_by_the_stream():
    """An agent that called record_tool_call asserted something deliberate
    about itself; a parse must never overwrite it."""
    from agent_flywheel.contract import RolloutOutput, ToolCallOutcome
    from agent_flywheel.runner import _attach_command_projection

    recorded = [ToolCallOutcome(
        name="from_trajectory", arguments={}, outcome="ok", status_code=None)]
    out = RolloutOutput(final_text="", tool_calls=list(recorded))
    _attach_command_projection(out, _stream(_START, _END_OK), _PRIME)
    assert [c.name for c in out.tool_calls] == ["from_trajectory"]


def test_the_stream_fills_in_when_the_trajectory_recorded_nothing():
    from agent_flywheel.contract import RolloutOutput
    from agent_flywheel.runner import _attach_command_projection

    out = RolloutOutput(final_text="")
    assert out.tool_calls is None
    _attach_command_projection(out, _stream(_START, _END_OK), _PRIME)
    assert out.tool_calls is not None
    assert [c.name for c in out.tool_calls] == ["ipython"]


def test_an_empty_assertion_of_zero_calls_is_preserved():
    """``[]`` from the trajectory is a positive claim and must survive."""
    from agent_flywheel.contract import RolloutOutput
    from agent_flywheel.runner import _attach_command_projection

    out = RolloutOutput(final_text="", tool_calls=[])
    _attach_command_projection(out, _stream(_START, _END_OK), _PRIME)
    assert out.tool_calls == []


# ── the vocabulary is data, not a parser ────────────────────────────────────


def test_the_generic_vocabulary_is_read_with_no_new_code():
    """The generality claim, made concrete. A process agent that prints the
    generic tool events as JSON lines gets outcome capture for free --
    supporting the next CLI agent is an entry in FORMATS, not a parser."""
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


def test_the_generic_vocabulary_does_not_claim_to_be_opentelemetry():
    """OTel GenAI defines no tool start/end events -- a tool call is one
    ``execute_tool`` span -- so a vocabulary built on them must not be named
    or exported as the OTel format."""
    import agent_flywheel.event_stream as mod

    assert not hasattr(mod, "OTEL_GENAI")
    assert all("otel" not in f.name for f in mod.FORMATS)


def test_the_conventions_real_attribute_names_are_read_first():
    """``gen_ai.tool.call.arguments`` / ``.result`` are the convention's own
    names; the old ``.input``/``.output`` spellings still parse."""
    calls = parse_tool_calls(_stream(
        {"type": "gen_ai.tool.end", "gen_ai.tool.call.id": "c1",
         "gen_ai.tool.name": "search",
         "gen_ai.tool.call.arguments": {"q": "real"},
         "gen_ai.tool.input": {"q": "legacy"},
         "gen_ai.tool.call.result": {"hits": 3}},
    ))
    assert calls[0].arguments == {"q": "real"}
    assert calls[0].output == {"hits": 3}


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
    from agent_flywheel.event_stream import FORMATS

    assert {f.name for f in FORMATS} == {"generic-tool-events"}
    # Two vocabularies in one stream: the first seen wins and the other's
    # events are ignored rather than silently merged under one call id.
    mixed = _stream(_START, _END_OK,
                    {"type": "gen_ai.tool.end", "gen_ai.tool.call.id": "z",
                     "gen_ai.tool.name": "other"})
    names = [c.name for c in _prime_calls(mixed)]
    assert names == ["ipython"]


# ── Prime JSONL is telemetry, not final prose ───────────────────────────────


def _assistant_message(text: str):
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stopReason": "stop",
    }


def test_final_text_comes_from_the_terminal_assistant_event():
    first = _assistant_message("working")
    final = {
        **_assistant_message("Shipped the fix."),
        "content": [
            {"type": "thinking", "thinking": "private"},
            {"type": "text", "text": "Shipped"},
            {"type": "text", "text": "the fix."},
        ],
    }
    stdout = _stream(
        {"type": "session", "version": 3},
        {"type": "agent_start"},
        {"type": "message_end", "message": first},
        {"type": "message_end", "message": final},
        {"type": "agent_end", "messages": [first, final]},
    )
    projected = _PRIME.project(stdout)
    assert projected.final_text == "Shipped\nthe fix."
    assert projected.llm_call_count == 2


def test_agent_end_snapshot_does_not_double_count_model_calls():
    first = _assistant_message("working")
    final = _assistant_message("done")
    stdout = _stream(
        {"type": "agent_start"},
        {"type": "message_end", "message": first},
        {"type": "message_end", "message": final},
        {"type": "agent_end", "messages": [first, final]},
    )
    assert _PRIME.project(stdout).llm_call_count == 2


def test_recognised_incomplete_stream_abstains_instead_of_asserting_zero():
    projected = _PRIME.project(_stream({"type": "agent_start"}))
    assert projected.llm_call_count is None
    assert projected.exclusion_reason == "cancelled"
    assert _PRIME.project('{"final_text":"ordinary JSON"}').llm_call_count is None


def test_command_projection_carries_the_independent_model_call_count():
    from agent_flywheel.contract import RolloutOutput
    from agent_flywheel.runner import _attach_command_projection

    stdout = _stream(
        {"type": "agent_start"},
        {"type": "message_end", "message": _assistant_message("working")},
        {"type": "message_end", "message": _assistant_message("done")},
        {"type": "agent_end", "messages": [_assistant_message("done")]},
    )
    out = RolloutOutput(final_text="raw telemetry")
    _attach_command_projection(out, stdout, _PRIME)
    assert out.final_text == "done"
    assert out.llm_call_count == 2


def test_prime_stream_without_a_terminal_message_is_honestly_empty():
    assert _PRIME.project(_stream(_START, _END_OK)).final_text == ""


def test_agent_end_without_an_assistant_clears_an_earlier_partial_message():
    stdout = _stream(
        {"type": "message_end", "message": _assistant_message("not terminal")},
        {"type": "agent_end", "messages": [{"role": "user", "content": "task"}]},
    )
    assert _PRIME.project(stdout).final_text == ""


def test_ordinary_json_stdout_is_not_mistaken_for_prime_jsonl():
    stdout = '{"final_text":"an ordinary command result"}'
    assert _PRIME.project(stdout).final_text is None


def test_refinement_and_compaction_events_are_reported_as_contamination():
    stdout = _stream(
        {"type": "compaction_start", "reason": "threshold"},
        {"type": "compaction_end", "reason": "threshold"},
        {"type": "refine_complete", "id": "r1"},
        {"type": "refine_failed", "error": "cancelled"},
        {"type": "compaction_start", "reason": "overflow"},
    )
    detail = _PRIME.project(stdout).exclusion_detail or ""
    for kind in (
        "compaction_start", "compaction_end", "refine_complete", "refine_failed",
    ):
        assert kind in detail
    single = _PRIME.project(
        _stream({"type": "compaction_start", "reason": "overflow"})
    )
    assert single.exclusion_reason == "cancelled"
    assert "compaction_start" in (single.exclusion_detail or "")


def test_terminal_assistant_failure_uses_agent_end_as_authority():
    errored = {**_assistant_message("partial"), "stopReason": "error",
               "errorMessage": "secret=/customer/path"}
    recovered = _assistant_message("recovered")
    stdout = _stream(
        {"type": "message_end", "message": recovered},
        {"type": "agent_end", "messages": [recovered, errored]},
    )
    projected = _PRIME.project(stdout)
    assert projected.exclusion_reason == "cancelled"
    assert "stopReason=error" in (projected.exclusion_detail or "")


def test_a_nonterminal_error_does_not_override_a_terminal_success():
    errored = {**_assistant_message("partial"), "stopReason": "aborted"}
    recovered = _assistant_message("recovered")
    stdout = _stream(
        {"type": "agent_start"},
        {"type": "message_end", "message": errored},
        {"type": "agent_end", "messages": [errored, recovered]},
    )
    assert _PRIME.project(stdout).exclusion_reason is None


# ── every exit path captures, not just the clean one ────────────────────────


def _agent_that_emits_then(exit_code: int, tmp_path):
    """A stub CLI that prints a tool-execution pair and then exits as told."""
    import stat as _stat
    p = tmp_path / "agent.sh"
    p.write_text(
        "#!/bin/sh\n"
        f"cat <<'JSON'\n{json.dumps({'type': 'agent_start'})}\n"
        f"{json.dumps(_START)}\n{json.dumps(_END_OK)}\n"
        f"{json.dumps({'type': 'agent_end', 'messages': []})}\nJSON\n"
        f"exit {exit_code}\n"
    )
    p.chmod(p.stat().st_mode | _stat.S_IEXEC)
    return str(p)


def _run_command(tmp_path, script, timeout_s=30.0, *, output_adapter=_PRIME):
    import time as _t

    from agent_flywheel.contract import RolloutRequest
    from agent_flywheel.runner import run_rollout_as_command

    req = RolloutRequest(
        rollout_id="rl_x", turn_input="go",
        llm_url="https://gw.invalid/v1", api_key="k",
        artifacts_dir="rollouts/rl_x", deadline_ts=_t.time() + 300,
        lease_token="lt",
    )
    return run_rollout_as_command(
        ["sh", script], req, trajectory_dir=str(tmp_path), timeout_s=timeout_s,
        env=dict(os.environ), cwd=str(tmp_path), output_adapter=output_adapter)


def test_core_does_not_guess_prime_semantics_without_the_optional_adapter(tmp_path):
    script = _agent_that_emits_then(0, tmp_path)
    out = _run_command(tmp_path, script, output_adapter=None)
    assert out.tool_calls is None
    assert out.llm_call_count is None
    assert out.final_text.startswith('{"type": "agent_start"')


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


def test_command_runner_marks_an_incomplete_stream_as_omitted(tmp_path):
    import stat as _stat

    incomplete = {**_START, "toolCallId": "never-ended"}
    p = tmp_path / "incomplete-agent.sh"
    p.write_text(
        "#!/bin/sh\n"
        f"cat <<'JSON'\n{json.dumps(_START)}\n{json.dumps(_END_OK)}\n"
        f"{json.dumps(incomplete)}\nJSON\n"
    )
    p.chmod(p.stat().st_mode | _stat.S_IEXEC)
    out = _run_command(tmp_path, str(p))
    assert [call.name for call in (out.tool_calls or [])] == ["ipython"]
    assert out.tool_calls_omitted_count == 1


def test_command_runner_reports_assistant_text_not_the_jsonl_stream(tmp_path):
    import stat as _stat

    final = _assistant_message("The actual answer")
    p = tmp_path / "json-agent.sh"
    p.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{json.dumps({'type': 'session', 'version': 3})}'\n"
        f"printf '%s\\n' '{json.dumps({'type': 'agent_start'})}'\n"
        f"printf '%s\\n' '{json.dumps({'type': 'message_end', 'message': final})}'\n"
        f"printf '%s\\n' '{json.dumps({'type': 'agent_end', 'messages': [final]})}'\n"
    )
    p.chmod(p.stat().st_mode | _stat.S_IEXEC)
    out = _run_command(tmp_path, str(p))
    assert out.final_text == "The actual answer"
    assert out.llm_call_count == 1


def test_command_runner_excludes_a_control_contaminated_trace(tmp_path):
    import stat as _stat

    final = _assistant_message("looks successful")
    p = tmp_path / "compacted-agent.sh"
    p.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{json.dumps({'type': 'compaction_start', 'reason': 'threshold'})}'\n"
        f"printf '%s\\n' '{json.dumps({'type': 'message_end', 'message': final})}'\n"
    )
    p.chmod(p.stat().st_mode | _stat.S_IEXEC)
    out = _run_command(tmp_path, str(p))
    assert out.final_text == "looks successful"
    assert out.exclusion_reason == "cancelled"
    assert "compaction_start" in (out.exclusion_detail or "")


def test_json_mode_terminal_error_is_excluded_even_when_process_exits_zero(tmp_path):
    import stat as _stat

    failed = {
        **_assistant_message("a misleading partial answer"),
        "stopReason": "error",
        "errorMessage": "Authorization: Bearer must-not-leak /customer/path",
    }
    p = tmp_path / "failed-json-agent.sh"
    p.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' '{json.dumps({'type': 'message_end', 'message': failed})}'\n"
        f"printf '%s\\n' '{json.dumps({'type': 'agent_end', 'messages': [failed]})}'\n"
    )
    p.chmod(p.stat().st_mode | _stat.S_IEXEC)
    out = _run_command(tmp_path, str(p))
    assert out.exclusion_reason == "cancelled"
    assert "stopReason=error" in (out.exclusion_detail or "")
    assert "must-not-leak" not in (out.exclusion_detail or "")
    assert "/customer/path" not in (out.exclusion_detail or "")


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
