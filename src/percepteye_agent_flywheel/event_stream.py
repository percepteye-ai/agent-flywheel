"""Tool outcomes from a COMMAND agent's own structured event stream.

The framework adapters bind to a Python object. An agent that is a process --
a Node CLI, a compiled binary, anything not importable -- has no object to bind
to, so its tool calls were simply unobservable and the honest setting was
``reports_tool_calls=False``. That is a real hole: without tool outcomes there
is no per-call verdict to grade, no retry to detect, and nothing a customer can
debug a rollout from. An observability stack that cannot say which tools ran
and whether they worked is not an observability stack.

But a modern CLI agent already emits exactly that, because it needs to for its
own UI. Rather than ask the customer to instrument anything, we read the stream
they already produce. This module is the process equivalent of the framework
adapters, and it follows the same rule as the OTel projection in ``tracing``:
absence is preserved, never rounded up.

Precedence: the ledger WINS. An agent that calls ``record_tool_call`` has made
a deliberate assertion about its own behaviour, and a stream parse must never
overwrite it. This is only consulted when the ledger recorded nothing.

The recognised shape (verified against prime-agent ``--mode json``):

    {"type":"tool_execution_start","toolCallId":"t1","toolName":"ipython",
     "args":{...}}
    {"type":"tool_execution_end","toolCallId":"t1","toolName":"ipython",
     "result":{"details":{"durationMs":7,"status":"ok"},"isError":false},
     "isError":false}

``isError`` is the verdict and it is TRI-STATE by omission: true -> failed,
false -> ok, absent -> unknown. An agent that ends a call without saying
whether it worked has not asserted success, and we do not assert it on the
agent's behalf.
"""
from __future__ import annotations

import json
from typing import Any

from .contract import Outcome, ToolCallOutcome

__all__ = ["parse_tool_calls", "TOOL_START", "TOOL_END"]

TOOL_START = "tool_execution_start"
TOOL_END = "tool_execution_end"

#: Bodies can be large (a REPL that printed a file). Bounded so a report stays
#: a report; the ledger, not the stream, is where full fidelity lives.
_MAX_OUTPUT_CHARS = 4000
_MAX_ERROR_CHARS = 1000


def _outcome_of(end_event: dict[str, Any]) -> Outcome:
    """``isError`` as a tri-state.

    Checked on the event and then on its ``result`` — some emitters put the
    flag in one place, some in both. Absent from both is ``unknown``: the call
    ended without a verdict, which is not the same as it having succeeded.
    """
    for holder in (end_event, end_event.get("result")):
        if isinstance(holder, dict) and isinstance(holder.get("isError"), bool):
            return "failed" if holder["isError"] else "ok"
    return "unknown"


def _text_of(result: Any) -> Any:
    """The readable body of a tool result, or the raw value.

    Prefers the ``content: [{type: text, text: ...}]`` shape, falling back to
    stdout and finally to the value itself. A grader needs bytes it can read,
    not a shape it has to guess at.
    """
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if isinstance(content, list):
        parts = [
            c["text"] for c in content
            if isinstance(c, dict) and isinstance(c.get("text"), str)
        ]
        if parts:
            return "".join(parts)[:_MAX_OUTPUT_CHARS]
    details = result.get("details")
    if isinstance(details, dict) and isinstance(details.get("stdout"), str):
        return details["stdout"][:_MAX_OUTPUT_CHARS]
    return result


def parse_tool_calls(stdout: str) -> list[ToolCallOutcome] | None:
    """Tool outcomes from an agent's event stream, or ``None`` if it has none.

    ``None`` and ``[]`` mean different things and the difference is preserved:
    ``None`` is "this stream carries no tool-execution events, so we observed
    nothing", while ``[]`` would be a positive claim that the agent made zero
    calls. A stream we cannot read must never be rendered as an assertion.

    Unparseable lines are skipped rather than fatal — a CLI interleaves its own
    logging with the event stream, and one stray line must not discard every
    outcome around it.
    """
    if not stdout:
        return None

    started: dict[str, dict[str, Any]] = {}
    calls: list[ToolCallOutcome] = []
    saw_any = False

    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except (ValueError, TypeError):
            continue
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        if kind == TOOL_START:
            cid = str(ev.get("toolCallId") or "")
            if cid:
                started[cid] = ev
            saw_any = True
            continue
        if kind != TOOL_END:
            continue
        saw_any = True

        cid = str(ev.get("toolCallId") or "")
        start = started.pop(cid, {})
        name = str(ev.get("toolName") or start.get("toolName") or "")
        if not name:
            # A call we cannot name is a call we cannot attribute. Dropping it
            # loses one outcome; inventing a name would corrupt every join
            # downstream (retry detection keys on name + arguments).
            continue
        args = ev.get("args") or start.get("args") or {}
        if not isinstance(args, dict):
            args = {"_raw": args}
        result = ev.get("result")
        outcome = _outcome_of(ev)

        error: str | None = None
        if outcome == "failed":
            body = _text_of(result)
            if isinstance(body, str) and body.strip():
                error = body[:_MAX_ERROR_CHARS]

        latency = None
        details = result.get("details") if isinstance(result, dict) else None
        if isinstance(details, dict):
            ms = details.get("durationMs")
            if isinstance(ms, (int, float)) and not isinstance(ms, bool):
                latency = float(ms)

        calls.append(ToolCallOutcome(
            name=name,
            arguments=args,
            outcome=outcome,
            # A local tool has no HTTP status. ``None`` is the honest value and
            # the contract discards status entirely when the outcome is
            # unknown, so this never manufactures a verdict.
            status_code=None,
            output=_text_of(result),
            error=error,
            latency_ms=latency,
            tool_call_id=cid or None,
        ))

    if not saw_any:
        return None
    return calls
