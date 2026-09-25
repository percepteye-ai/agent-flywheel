"""One production turn's model boundary, rendered as a real OTel GenAI trace.

WHAT THIS IS. When an adapter observes its agent's MODEL BOUNDARY during a
production turn -- the exact messages a model call was sent, the system
instructions, the tool definitions it was offered, what it answered, and what
each tool it asked for returned -- that observation is written beside the turn
(:data:`OBSERVATION_FILENAME`). This module turns it into ONE OpenTelemetry
GenAI trace, an OTLP/JSON ``ExportTraceServiceRequest``, which is what the
turn upload carries as ``genai_trace``: an ``invoke_agent`` root with one
``chat`` span per model call and one ``execute_tool`` span per tool call,
content per the GenAI semantic conventions.

PURE. :func:`render_turn_trace` and :func:`observation_from_lines` take values
and return values: no clock, no randomness, no filesystem. That is what lets
the Node SDK's ``genai-otlp.js`` be held to the SAME output for the same
observation, byte for byte, by a golden file both test suites read
(``tests/fixtures/genai_capture_golden.json``) and by
``scripts/cross_language_abi_check.sh``, which runs both.

WHAT IT NEVER DOES, each a rule the control plane's intake enforces and a
reason this module exists rather than the customer's own TracerProvider:

* it never truncates. A prompt cut short is a prompt the model never saw;
* it never invents content. Every message, instruction and definition here was
  observed at a model call. A turn with no observed model call renders to
  ``None`` -- never to a trace rebuilt from the turn's ``input_text`` and
  ``final_text``, which would assert a prompt nobody sent;
* it never marks a decision (the synthetic dataset's decision-point attribute)
  and never stamps ``percepteye.rollout.id``: a production turn is the
  customer's own traffic, and a span carrying that id is dropped downstream as
  our training traffic.

ONE ENCODING FOR CONTENT. Content attributes (``gen_ai.input.messages`` and
the rest) are JSON strings, the form the OTel attribute APIs can carry. They
are compact JSON in the value's own key order, because that order is what the
agent's runtime serialized to the model. The one thing the two SDKs cannot
share is the key order of an object whose keys look like integers: JavaScript
enumerates those first. Each SDK renders its own runtime's order, and the
golden file pins that divergence as a vector of its own rather than leaving it
to be discovered.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Iterable, Mapping
from typing import Any

__all__ = [
    "OBSERVATION_FILENAME",
    "SCOPE_NAME",
    "content_json",
    "end_record",
    "new_span_id",
    "new_trace_id",
    "now_unix_nano",
    "observation_from_lines",
    "operation_record",
    "render_turn_trace",
    "trace_record",
]

#: Where a turn directory keeps its observed model boundary: one JSON record
#: per line (see :func:`observation_from_lines`). Both SDKs write and read
#: this one format.
OBSERVATION_FILENAME = "genai_observation.jsonl"

#: The instrumentation scope every rendered trace names.
SCOPE_NAME = "percepteye.agent_flywheel"

_HEX32 = re.compile(r"[0-9a-f]{32}")
_HEX16 = re.compile(r"[0-9a-f]{16}")
_DECIMAL = re.compile(r"[0-9]+")

#: OTLP/JSON SpanKind integers: the agent span and a tool execution run in
#: process (INTERNAL); a model call is a request to a provider (CLIENT).
_INTERNAL = 1
_CLIENT = 3
#: OTLP Status.StatusCode ERROR. An operation that ended in an error carries
#: it, with ``error.type``; every other span's status is unset (``{}``).
_STATUS_ERROR = 2


def content_json(value: Any) -> str:
    """A content value as the JSON string carried on the span."""
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def now_unix_nano() -> str:
    """Wall-clock nanoseconds as the decimal string OTLP/JSON carries.

    A STRING because the value exceeds 2**53: JavaScript cannot hold it as a
    number, and the observation is one format for both SDKs.
    """
    return str(time.time_ns())


def _random_hex(n_bytes: int) -> str:
    while True:
        value = os.urandom(n_bytes).hex()
        # The all-zero id is INVALID in OTLP; vanishingly rare, never emitted.
        if value.strip("0"):
            return value


def new_trace_id() -> str:
    """A fresh OTLP trace id: 32 lowercase hex, never all zeros."""
    return _random_hex(16)


def new_span_id() -> str:
    """A fresh OTLP span id: 16 lowercase hex, never all zeros."""
    return _random_hex(8)


def _id(value: Any, pattern: re.Pattern[str]) -> str | None:
    if isinstance(value, str) and pattern.fullmatch(value) and value.strip("0"):
        return value
    return None


def _nanos(value: Any) -> str | None:
    if isinstance(value, str) and _DECIMAL.fullmatch(value):
        return value
    return None


def _s(value: str) -> dict[str, Any]:
    return {"stringValue": value}


def _attr(key: str, value: dict[str, Any]) -> dict[str, Any]:
    return {"key": key, "value": value}


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _chat_span(
    op: Mapping[str, Any], *, trace_id: str, parent: str,
    conversation_id: str | None,
) -> dict[str, Any] | None:
    span_id = _id(op.get("span_id"), _HEX16)
    start = _nanos(op.get("start_time_unix_nano"))
    end = _nanos(op.get("end_time_unix_nano"))
    provider = _text(op.get("provider_name"))
    # The conventions REQUIRE the provider and the requested model on an
    # inference span; a call observed without either is skipped rather than
    # rendered as a span the published standard does not recognise.
    request_model = _text(op.get("request_model"))
    inputs = op.get("input_messages")
    if not (span_id and start and end and provider and request_model) \
            or not isinstance(inputs, list):
        return None
    attrs = [
        _attr("gen_ai.operation.name", _s("chat")),
        _attr("gen_ai.provider.name", _s(provider)),
        _attr("gen_ai.request.model", _s(request_model)),
    ]
    response_model = _text(op.get("response_model"))
    if response_model:
        attrs.append(_attr("gen_ai.response.model", _s(response_model)))
    if conversation_id:
        attrs.append(_attr("gen_ai.conversation.id", _s(conversation_id)))
    finish = op.get("finish_reasons")
    if isinstance(finish, list) and finish and all(
            isinstance(f, str) for f in finish):
        attrs.append(_attr("gen_ai.response.finish_reasons", {
            "arrayValue": {"values": [_s(f) for f in finish]}}))
    system = op.get("system_instructions")
    if isinstance(system, list) and system:
        attrs.append(_attr("gen_ai.system_instructions", _s(content_json(system))))
    attrs.append(_attr("gen_ai.input.messages", _s(content_json(inputs))))
    outputs = op.get("output_messages")
    if isinstance(outputs, list) and outputs:
        attrs.append(_attr("gen_ai.output.messages", _s(content_json(outputs))))
    tools = op.get("tool_definitions")
    if isinstance(tools, list) and tools:
        attrs.append(_attr("gen_ai.tool.definitions", _s(content_json(tools))))
    return _finish_span(
        op, attrs,
        trace_id=trace_id, span_id=span_id, parent=parent,
        name=f"chat {request_model}", kind=_CLIENT, start=start, end=end,
    )


def _tool_span(
    op: Mapping[str, Any], *, trace_id: str, parent: str,
    conversation_id: str | None, agent_name: str,
) -> dict[str, Any] | None:
    span_id = _id(op.get("span_id"), _HEX16)
    start = _nanos(op.get("start_time_unix_nano"))
    end = _nanos(op.get("end_time_unix_nano"))
    name = _text(op.get("tool_name"))
    if not (span_id and start and end and name):
        return None
    attrs = [
        _attr("gen_ai.operation.name", _s("execute_tool")),
        _attr("gen_ai.tool.name", _s(name)),
    ]
    tool_type = _text(op.get("tool_type"))
    if tool_type:
        attrs.append(_attr("gen_ai.tool.type", _s(tool_type)))
    call_id = _text(op.get("tool_call_id"))
    if call_id:
        attrs.append(_attr("gen_ai.tool.call.id", _s(call_id)))
    if "arguments" in op:
        attrs.append(_attr("gen_ai.tool.call.arguments",
                           _s(content_json(op["arguments"]))))
    # A result and an error are exclusive: an execution that raised returned
    # nothing, and recording both would say it did.
    if "result" in op and not _text(op.get("error_type")):
        attrs.append(_attr("gen_ai.tool.call.result",
                           _s(content_json(op["result"]))))
    if conversation_id:
        attrs.append(_attr("gen_ai.conversation.id", _s(conversation_id)))
    attrs.append(_attr("gen_ai.agent.name", _s(agent_name)))
    return _finish_span(
        op, attrs,
        trace_id=trace_id, span_id=span_id, parent=parent,
        name=f"execute_tool {name}", kind=_INTERNAL, start=start, end=end,
    )


def _finish_span(
    op: Mapping[str, Any], attrs: list[dict[str, Any]], *,
    trace_id: str, span_id: str, parent: str, name: str, kind: int,
    start: str, end: str,
) -> dict[str, Any]:
    error_type = _text(op.get("error_type"))
    if error_type:
        attrs.append(_attr("error.type", _s(error_type)))
    return {
        "traceId": trace_id,
        "spanId": span_id,
        "parentSpanId": parent,
        "name": name,
        "kind": kind,
        "startTimeUnixNano": start,
        "endTimeUnixNano": end,
        "attributes": attrs,
        "status": {"code": _STATUS_ERROR} if error_type else {},
    }


def render_turn_trace(
    observation: Mapping[str, Any] | None,
    *,
    agent_name: str,
    conversation_id: str | None = None,
) -> dict[str, Any] | None:
    """The turn's OTLP/JSON ``ExportTraceServiceRequest``, or ``None``.

    ``None`` when nothing was observed at a model call: no observation, no
    chat operation that can be rendered, or ids and clocks that are not the
    OTLP encodings. A trace missing its model boundary asserts nothing a
    training row could be built on, and the turn uploads without one.

    ``agent_name`` names the agent (the resource's ``service.name`` and
    ``gen_ai.agent.name``); ``conversation_id`` is the TURN's own, and every
    span carries exactly it -- the control plane refuses a trace naming any
    other, because that id is the only thing that joins this turn's trace to
    the turns before and after it.

    An operation this cannot render (a record cut short by a crash, say) is
    skipped, never guessed at. Each chat span carries its call's WHOLE input,
    so a skipped operation costs that one call and cannot corrupt another's
    prefix.
    """
    if not isinstance(observation, Mapping) or not _text(agent_name):
        return None
    trace_id = _id(observation.get("trace_id"), _HEX32)
    root_id = _id(observation.get("span_id"), _HEX16)
    start = _nanos(observation.get("start_time_unix_nano"))
    end = _nanos(observation.get("end_time_unix_nano"))
    operations = observation.get("operations")
    if not (trace_id and root_id and start and end) or not isinstance(
            operations, list):
        return None
    conversation = _text(conversation_id)
    children: list[dict[str, Any]] = []
    provider: str | None = None
    for op in operations:
        if not isinstance(op, Mapping):
            continue
        if op.get("operation") == "chat":
            span = _chat_span(op, trace_id=trace_id, parent=root_id,
                              conversation_id=conversation)
            if span is not None and provider is None:
                provider = op["provider_name"]
        elif op.get("operation") == "execute_tool":
            span = _tool_span(op, trace_id=trace_id, parent=root_id,
                              conversation_id=conversation,
                              agent_name=agent_name)
        else:
            span = None
        if span is not None:
            children.append(span)
    if provider is None:
        return None
    root_attrs = [
        _attr("gen_ai.operation.name", _s("invoke_agent")),
        _attr("gen_ai.provider.name", _s(provider)),
        _attr("gen_ai.agent.name", _s(agent_name)),
    ]
    if conversation:
        root_attrs.append(_attr("gen_ai.conversation.id", _s(conversation)))
    root = {
        "traceId": trace_id,
        "spanId": root_id,
        "name": f"invoke_agent {agent_name}",
        "kind": _INTERNAL,
        "startTimeUnixNano": start,
        "endTimeUnixNano": end,
        "attributes": root_attrs,
        "status": {},
    }
    return {"resourceSpans": [{
        "resource": {"attributes": [_attr("service.name", _s(agent_name))]},
        "scopeSpans": [{
            "scope": {"name": SCOPE_NAME},
            "spans": [root, *children],
        }],
    }]}


def _record(line: str) -> Mapping[str, Any] | None:
    try:
        value = json.loads(line)
    except ValueError:
        return None
    return value if isinstance(value, Mapping) else None


def observation_from_lines(lines: Iterable[str]) -> dict[str, Any] | None:
    """The observation a turn directory's :data:`OBSERVATION_FILENAME` holds.

    One JSON record per line, three kinds::

        {"record": "trace", "trace_id", "span_id", "start_time_unix_nano"}
        {"record": "operation", "operation": "chat" | "execute_tool", ...}
        {"record": "end", "end_time_unix_nano"}

    The FIRST ``trace`` record names the trace (a second writer racing the
    first cannot re-mint it). Operations are ordered by when they started --
    the order they were APPENDED is when they finished, and parallel tool
    calls finish out of order -- with append order breaking ties, so the same
    file always renders the same trace. The root span covers the turn and
    every operation in it: from the earlier of the turn's start and its first
    operation, to the latest end recorded.

    A line that does not parse (a write cut short by a crash) is skipped. The
    result is deterministic in the file's bytes, which is what makes a retried
    upload of the same turn carry the same trace -- the control plane treats a
    different trace for the same turn as a conflict.
    """
    header: Mapping[str, Any] | None = None
    ops: list[tuple[int, int, Mapping[str, Any]]] = []
    ends: list[int] = []
    for position, line in enumerate(lines):
        if not line.strip():
            continue
        rec = _record(line)
        if rec is None:
            continue
        kind = rec.get("record")
        if kind == "trace" and header is None:
            if (_id(rec.get("trace_id"), _HEX32) and _id(rec.get("span_id"), _HEX16)
                    and _nanos(rec.get("start_time_unix_nano"))):
                header = rec
        elif kind == "operation":
            start = _nanos(rec.get("start_time_unix_nano"))
            end = _nanos(rec.get("end_time_unix_nano"))
            if start and end:
                op = {k: v for k, v in rec.items() if k != "record"}
                ops.append((int(start), position, op))
                ends.append(int(end))
        elif kind == "end":
            end = _nanos(rec.get("end_time_unix_nano"))
            if end:
                ends.append(int(end))
    if header is None or not ops:
        return None
    ops.sort(key=lambda item: (item[0], item[1]))
    start = min(int(header["start_time_unix_nano"]), ops[0][0])
    end = max([start, *ends])
    return {
        "trace_id": header["trace_id"],
        "span_id": header["span_id"],
        "start_time_unix_nano": str(start),
        "end_time_unix_nano": str(end),
        "operations": [op for _, _, op in ops],
    }


def trace_record(*, trace_id: str, span_id: str, start: str) -> str:
    """The ``trace`` line, as written."""
    return content_json({"record": "trace", "trace_id": trace_id,
                         "span_id": span_id, "start_time_unix_nano": start})


def operation_record(op: Mapping[str, Any]) -> str:
    """One ``operation`` line, as written.

    A value JSON cannot encode (a framework object a tool returned) is
    written as its ``str``, which is also what a provider client would have
    had to send in its place; the record is never dropped for it.
    """
    return json.dumps({"record": "operation", **op}, separators=(",", ":"),
                      ensure_ascii=False, default=str)


def end_record(end: str) -> str:
    """The ``end`` line, as written."""
    return content_json({"record": "end", "end_time_unix_nano": end})
