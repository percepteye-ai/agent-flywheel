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
from dataclasses import dataclass
from typing import Any

from .contract import Outcome, ToolCallOutcome

__all__ = ["parse_tool_calls", "StreamFormat", "FORMATS", "TOOL_START", "TOOL_END"]


@dataclass(frozen=True)
class StreamFormat:
    """How one agent's event stream names the things we need.

    Declarative on purpose. The framework adapters are a directory of small
    modules because every Python framework hooks tools differently; a process
    agent needs no hook at all, only a vocabulary. So supporting the next CLI
    agent is an entry here — not a parser — and nothing about the extraction,
    the tri-state or the absence rules is re-implemented per agent.

    Field names are CANDIDATE tuples, tried in order, because emitters
    disagree about camelCase vs snake_case far more often than they disagree
    about meaning.
    """

    name: str
    start_type: str
    end_type: str
    call_id: tuple[str, ...] = ("toolCallId", "tool_call_id", "id")
    tool_name: tuple[str, ...] = ("toolName", "tool_name", "name")
    args: tuple[str, ...] = ("args", "arguments", "input")
    result: tuple[str, ...] = ("result", "output", "response")
    error_flag: tuple[str, ...] = ("isError", "is_error", "error")


#: prime-agent (`--mode json`), verified against a live run. The default field
#: candidates already cover it; it is listed explicitly so the vocabulary is
#: visible rather than implied.
PRIME_AGENT = StreamFormat(
    name="prime-agent",
    start_type="tool_execution_start",
    end_type="tool_execution_end",
)

#: The OpenTelemetry GenAI convention, as emitted by an agent that logs its
#: spans as JSON lines. Included because it is the one vocabulary that is not
#: any single vendor's — an agent already instrumented for observability gets
#: outcome capture with no work at all, which is the point.
OTEL_GENAI = StreamFormat(
    name="otel-genai",
    start_type="gen_ai.tool.start",
    end_type="gen_ai.tool.end",
    call_id=("gen_ai.tool.call.id", "toolCallId", "id"),
    tool_name=("gen_ai.tool.name", "toolName", "name"),
    args=("gen_ai.tool.input", "args", "arguments"),
    result=("gen_ai.tool.output", "result", "output"),
    error_flag=("error.type", "isError", "is_error"),
)

#: Order is irrelevant — a stream is matched by the event types it actually
#: carries, never by position or by an agent name we were told.
FORMATS: tuple[StreamFormat, ...] = (PRIME_AGENT, OTEL_GENAI)

# Back-compat aliases for the prime-agent vocabulary.
TOOL_START = PRIME_AGENT.start_type
TOOL_END = PRIME_AGENT.end_type


def _first(d: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return None

#: Bodies can be large (a REPL that printed a file). Bounded so a report stays
#: a report; the ledger, not the stream, is where full fidelity lives.
_MAX_OUTPUT_CHARS = 4000

#: Error bodies get the SAME budget as ordinary output, not a quarter of it.
#: They used to get 1000 chars, which reads as prudence and is the opposite:
#: a raw traceback is the load-bearing signal for learned self-correction, and
#: the error body is the one payload where the interesting part is reliably at
#: the END — the exception class and its message, after however many frames.
#: A 1000-char head-truncation deletes exactly that.
_MAX_ERROR_CHARS = 4000

#: Stamped into a body we had to shorten. Deliberately a fixed, greppable
#: string with the dropped count in it: it is what lets a downstream census
#: distinguish "the tool printed 300 chars" from "the tool printed 300,000 and
#: we kept a window", without adding a field to the wire contract that every
#: producer would then have to learn to set.
_ELISION = "\n…[{dropped} chars elided by the SDK capture bound]…\n"


def _bounded(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    """Shorten ``text`` to ``limit`` while keeping BOTH ends.

    Head-truncation is the wrong shape for the payloads that matter here. A
    traceback opens with a banner and closes with the exception type and
    message; a REPL transcript opens with a command and closes with what it
    produced. Keeping only the head keeps the least informative half of both,
    and does it silently.

    So: half the budget from the front, the rest from the back, with an
    explicit marker between them stating how much went. A reader can see that
    the body is a window rather than a whole, and a census can count it.
    """
    if len(text) <= limit:
        return text
    dropped = len(text) - limit
    marker = _ELISION.format(dropped=dropped)
    # The marker is part of the budget — a bound that its own annotation can
    # push past is not a bound. A limit too small to hold the marker keeps the
    # tail, which is the half worth keeping.
    room = limit - len(marker)
    if room <= 0:
        return text[-limit:]
    head = room // 2
    return text[:head] + marker + text[len(text) - (room - head):]


def _outcome_of(end_event: dict[str, Any], fmt: StreamFormat) -> Outcome:
    """The error flag as a TRI-STATE.

    Checked on the event and then on its result — emitters disagree about
    which carries it, and some set both. Absent from both is ``unknown``: the
    call ended without a verdict, which is not the same as it having
    succeeded, and this is the one place that distinction can still be made.

    A boolean is read as a boolean. A non-empty STRING is read as a failure
    because that is how the OTel convention says it (``error.type`` carries the
    exception class and is simply absent on success) — but an empty string is
    absence, not a claim.
    """
    for holder in (end_event, _first(end_event, fmt.result)):
        if not isinstance(holder, dict):
            continue
        flag = _first(holder, fmt.error_flag)
        if isinstance(flag, bool):
            return "failed" if flag else "ok"
        if isinstance(flag, str) and flag.strip():
            return "failed"
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
            return _bounded("".join(parts))
    details = result.get("details")
    if isinstance(details, dict) and isinstance(details.get("stdout"), str):
        return _bounded(details["stdout"])
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

    matched: StreamFormat | None = None
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

        # A stream is identified by the event types it CARRIES, never by an
        # agent name we were told — a name is a claim, an event type is
        # evidence. The first format whose vocabulary appears wins, and one
        # stream is never read through two vocabularies at once.
        fmt = matched
        if fmt is None:
            for cand in FORMATS:
                if kind in (cand.start_type, cand.end_type):
                    fmt = matched = cand
                    break
        if fmt is None or kind not in (fmt.start_type, fmt.end_type):
            continue

        if kind == fmt.start_type:
            cid = str(_first(ev, fmt.call_id) or "")
            if cid:
                started[cid] = ev
            saw_any = True
            continue
        saw_any = True

        cid = str(_first(ev, fmt.call_id) or "")
        start = started.pop(cid, {})
        name = str(_first(ev, fmt.tool_name) or _first(start, fmt.tool_name) or "")
        if not name:
            # A call we cannot name is a call we cannot attribute. Dropping it
            # loses one outcome; inventing a name would corrupt every join
            # downstream (retry detection keys on name + arguments).
            continue
        args = _first(ev, fmt.args)
        if args is None:
            args = _first(start, fmt.args)
        if not isinstance(args, dict):
            args = {} if args is None else {"_raw": args}
        result = _first(ev, fmt.result)
        outcome = _outcome_of(ev, fmt)

        error: str | None = None
        if outcome == "failed":
            body = _text_of(result)
            if isinstance(body, str) and body.strip():
                error = _bounded(body, _MAX_ERROR_CHARS)

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
