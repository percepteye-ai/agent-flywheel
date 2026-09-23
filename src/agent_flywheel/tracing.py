"""OpenTelemetry egress: OTel GenAI spans, rendered to the published standard.

WHAT "THE STANDARD" MEANS HERE. The GenAI semantic conventions at
``open-telemetry/semantic-conventions-genai@8ffdf568e1``: span names, kinds and
attributes from its ``spans.yaml``/``registry.yaml``, and every content
attribute (``gen_ai.input.messages``, ``gen_ai.output.messages``,
``gen_ai.system_instructions``, ``gen_ai.tool.call.arguments``,
``gen_ai.tool.call.result``) in the shape its JSON schemas define. Content is
written as a JSON STRING, which the convention permits on spans and which is
what its own reference instrumentations write.

PARITY WITH THE NODE SDK IS A TEST, NOT A PROMISE. A customer deploys this
package or the npm one, never both, so the two must emit the same spans for the
same facts. Both repositories carry the same golden fixture
(``tests/fixtures/genai-golden.json`` here, ``test/fixtures/genai-golden.json``
there, byte-identical): a scripted scenario and the exact spans it must
produce, content JSON included. Each SDK renders the scenario and must match it
byte for byte, and this side also validates every content value against the
convention's own schemas (vendored under ``tests/genai_schemas``).

TWO SPANS, both INTERNAL:

``execute_tool {tool}``
    one per recorded tool call, with its real start and end, its arguments and
    (on success) its result as objects, and ``error.type`` plus ERROR on a
    reported failure.
``invoke_agent {agent}``
    one per TURN, carrying the whole turn: the system instructions, the history
    the agent was given and the answer it returned. It replaced two half-turn
    ``chat`` spans, which described neither half as what it was -- no model
    call happened on either.

NOTHING IS CUT. There is no size limit on any attribute: a cut at a fixed
character count turned a JSON argument into invalid JSON. A customer who needs
a bound sets one in their own exporter or collector, where it applies to every
instrumentation alike.

WHY A SPAN CAN BE A CAPTURE SURFACE AT ALL. The conventions define no attribute
for per-token logprobs or token ids, so a span cannot carry what an objective
forming an importance ratio against the sampling policy needs. Every other
objective this SDK's evidence feeds either recomputes what it needs by forward
pass or has no logprob in its loss, and for those the prompt, the completion
and a verdict -- all of which a span carries -- are enough.

**The turn span is OPT-IN; tool spans are not.** A tool name and an outcome are
telemetry; a prompt and a completion are the user's own words, so only text the
customer handed over reaches a turn span. Nothing here reads spans back, so the
customer's sampler, queue overflow and exporter failures cost us nothing.

**We never install a TracerProvider.** ``set_tracer_provider`` is once-only:
the second caller gets a log line, not an exception. So calling it would either
lose all our spans (if the customer configured theirs first) or silently kill
the customer's entire GenAI telemetry (if we won). We take whatever provider is
already registered and emit into it. With no provider configured, the API's
no-op tracer swallows everything, which is the correct outcome for a library.

``percepteye.*`` carries what the convention has no word for -- notably the
tri-state outcome. The convention cannot say "this tool call's result was never
observed", so the projection is documented as lossy rather than pretending
otherwise.
"""
from __future__ import annotations

import json
import math
import time
from typing import Any

__all__ = ["record_tool_call_span", "record_turn_span", "tracing_available"]

#: Resolved once. ``None`` means the API is not installed, which is the normal
#: case: the ``otel`` extra is opt-in and the package ships with one runtime
#: dependency.
_TRACER: Any = None
_RESOLVED = False

#: The Node SDK's ``JSON.stringify``, byte for byte: no spacing, and non-ASCII
#: text as itself rather than ``\\u`` escapes.
_DUMPS: dict[str, Any] = {
    "ensure_ascii": False,
    "separators": (",", ":"),
    "allow_nan": False,
    "default": str,
}

#: The largest integer a JavaScript number holds exactly.
_MAX_SAFE_INTEGER = 2**53 - 1


def _tracer() -> Any:
    """The customer's tracer, or None. Never constructs a provider."""
    global _TRACER, _RESOLVED
    if _RESOLVED:
        return _TRACER
    _RESOLVED = True
    try:
        from opentelemetry import trace  # type: ignore

        # get_tracer ONLY. No SDK import, no provider construction, no
        # set_tracer_provider -- see the module docstring.
        _TRACER = trace.get_tracer("agent-flywheel")
    except Exception:
        _TRACER = None
    return _TRACER


def tracing_available() -> bool:
    """True when the OTel API is importable. Says nothing about whether the
    customer configured a provider -- with none, spans go to a no-op."""
    return _tracer() is not None


def _finite(value: Any) -> Any:
    """``value`` with every NaN/infinity replaced by ``None``, which is what
    ``JSON.stringify`` writes for them: JSON has no spelling for either."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def _genai_json(value: Any) -> str | None:
    """A content attribute's value: the JSON text of ``value``, or ``None``
    when it has none (a cycle, a key JSON cannot spell).

    Always VALID JSON. ``json.dumps`` writes ``NaN`` by default, which no JSON
    parser outside Python accepts; those become ``null`` instead.
    """
    try:
        return json.dumps(value, **_DUMPS)
    except (TypeError, ValueError):
        pass
    try:
        return json.dumps(_finite(value), **_DUMPS)
    except (TypeError, ValueError, RecursionError):
        return None


def _exact_int(text: str) -> int:
    value = int(text)
    if abs(value) > _MAX_SAFE_INTEGER:
        # Node cannot hold this integer exactly and refuses the same input;
        # agreeing costs a deserialization, disagreeing costs parity.
        raise ValueError("integer beyond the exactly representable range")
    return value


def _reject_constant(text: str) -> Any:
    raise ValueError(f"{text} is not JSON")


def _as_object(value: Any) -> dict[str, Any] | None:
    """``value`` as a JSON object, or ``None``.

    The convention expects tool arguments and results to be OBJECTS, and says
    a serialized string SHOULD be deserialized on a best-effort basis. So a
    string is parsed -- with the Node SDK's own grammar: no ``NaN``/
    ``Infinity`` tokens, and no integer beyond +/-(2**53 - 1), which Node could
    not hold exactly.
    """
    if isinstance(value, str):
        try:
            value = json.loads(
                value, parse_int=_exact_int, parse_constant=_reject_constant,
            )
        except ValueError:
            return None
    return value if isinstance(value, dict) else None


def _genai_messages(messages: Any) -> list[Any] | None:
    """Messages in the convention's shape, ready to serialize.

    Callers hand over ``{"role", "parts", ...}`` dicts already. The one thing
    done here is the convention's best-effort rule for a ``tool_call`` part
    whose ``arguments`` arrived as a JSON string (the OpenAI wire form): it
    becomes the object it encodes. Everything else is carried exactly as
    given, ``None`` included -- a tool response's ``null`` is data.
    """
    if not isinstance(messages, (list, tuple)):
        return None
    out: list[Any] = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(
                message.get("parts"), (list, tuple)):
            out.append(message)
            continue
        parts = []
        for part in message["parts"]:
            if (isinstance(part, dict) and part.get("type") == "tool_call"
                    and isinstance(part.get("arguments"), str)):
                obj = _as_object(part["arguments"])
                if obj is not None:
                    part = {**part, "arguments": obj}
            parts.append(part)
        out.append({**message, "parts": parts})
    return out


def _text_message(role: str, text: str) -> dict[str, Any]:
    """One user or assistant text message, the shape a bare string becomes."""
    return {"role": role, "parts": [{"type": "text", "content": str(text)}]}


def _finite_number(value: Any) -> int | float | None:
    """The number, or ``None``. An int STAYS an int: a whole number of
    milliseconds must become an exact number of nanoseconds, and a latency
    Node sends as an integer must not arrive here as a double."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _ns(ms: int | float) -> int:
    """Epoch milliseconds as the nanoseconds the OTel API takes -- exactly,
    for whole milliseconds, where a float product would round."""
    if isinstance(ms, int):
        return ms * 1_000_000
    return round(ms * 1_000_000)


def _span_kind_internal() -> Any:
    from opentelemetry.trace import SpanKind  # type: ignore

    return SpanKind.INTERNAL


def _mark_error(span: Any, message: str) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode  # type: ignore

        span.set_status(Status(StatusCode.ERROR, str(message)))
    except Exception:
        # Older/partial API surfaces: `error.type` still carries the verdict.
        pass


def _set_conversation(span: Any, conversation_id: str | None) -> None:
    """``gen_ai.conversation.id`` AND ``percepteye.conversation.id``, the same
    value. The first is the convention's name and what any consumer reads; the
    second is this SDK's earlier spelling, kept for one release so a consumer
    reading it keeps working while it moves."""
    if conversation_id:
        span.set_attribute("gen_ai.conversation.id", str(conversation_id))
        span.set_attribute("percepteye.conversation.id", str(conversation_id))


def _set_correlation(
    span: Any, *, rollout_id: str | None, turn_id: str | None,
    conversation_id: str | None, task_id: str | None,
    execution_sha256: str | None,
) -> None:
    """The attributes both span kinds carry for correlation.

    ``rollout_id`` is stamped only on work that IS a rollout; a production turn
    carries ``percepteye.turn.id`` and no rollout id, so a consumer counting
    "how much of my traffic was training" is not told every served turn was.
    """
    _set_conversation(span, conversation_id)
    if rollout_id:
        span.set_attribute("percepteye.rollout.id", str(rollout_id))
    if turn_id:
        span.set_attribute("percepteye.turn.id", str(turn_id))
    if task_id:
        # Caller-authored correlation evidence only. This does not assert
        # success, policy identity or training eligibility.
        span.set_attribute("percepteye.task.id", str(task_id))
    if execution_sha256:
        span.set_attribute(
            "percepteye.agent.execution_sha256", str(execution_sha256))


def _error_type(error_class: str | None, status_code: int | None) -> str:
    """``error.type`` for a reported failure: the error's class when one was
    classified, else the status code, else the convention's ``_OTHER``."""
    if isinstance(error_class, str) and error_class.strip():
        return error_class
    if status_code is not None:
        return str(status_code)
    return "_OTHER"


def record_tool_call_span(
    name: str,
    arguments: Any,
    *,
    outcome: str,
    status_code: int | None = None,
    error: str | None = None,
    error_class: str | None = None,
    latency_ms: float | None = None,
    end_time_ms: float | None = None,
    agent_name: str | None = None,
    rollout_id: str | None = None,
    turn_id: str | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    tool_call_id: str | None = None,
    execution_sha256: str | None = None,
    result: Any = None,
) -> None:
    """Project one recorded tool call into the customer's traces as an
    ``execute_tool {name}`` span.

    Its END is ``end_time_ms`` (default: now -- it is recorded when the call
    returns) and its START is ``latency_ms`` before that, so the span covers
    the call rather than the instant it was written down.

    A no-op when the API is absent, and it NEVER raises: this runs beside the
    trajectory write on the customer's own execution path, and a telemetry
    failure must cost us the span and nothing else.
    """
    tracer = _tracer()
    if tracer is None:
        return
    try:
        end = _finite_number(end_time_ms)
        if end is None:
            end = time.time() * 1000.0
        latency = _finite_number(latency_ms)
        start = end - latency if latency is not None and latency >= 0 else end
        # The attributes a sampler may decide on go in at CREATION, as the
        # convention asks: operation, tool name, tool type, agent.
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "execute_tool",
            "gen_ai.tool.name": str(name),
            # A client-side function the agent's own process ran.
            "gen_ai.tool.type": "function",
        }
        if agent_name:
            attributes["gen_ai.agent.name"] = str(agent_name)
        span = tracer.start_span(
            f"execute_tool {name}", kind=_span_kind_internal(),
            attributes=attributes, start_time=_ns(start),
        )
        try:
            if not getattr(span, "is_recording", lambda: False)():
                return
            if tool_call_id:
                # The model-assigned call id: the same value that reappears on
                # the tool response the model is sent next, so it joins the two.
                span.set_attribute("gen_ai.tool.call.id", str(tool_call_id))
            args_object = _as_object(arguments)
            if args_object is not None:
                text = _genai_json(args_object)
                if text is not None:
                    span.set_attribute("gen_ai.tool.call.arguments", text)
            # The result only when execution succeeded, and only as the object
            # the convention's schema requires; a text result reaches the next
            # model call as a tool response, not here.
            if outcome != "failed":
                result_object = _as_object(result)
                if result_object is not None:
                    text = _genai_json(result_object)
                    if text is not None:
                        span.set_attribute("gen_ai.tool.call.result", text)

            span.set_attribute("percepteye.tool.outcome", str(outcome))
            if status_code is not None:
                span.set_attribute("percepteye.tool.status_code", int(status_code))
            if latency is not None:
                span.set_attribute("percepteye.tool.latency_ms", latency)
            _set_correlation(
                span, rollout_id=rollout_id, turn_id=turn_id,
                conversation_id=conversation_id, task_id=task_id,
                execution_sha256=execution_sha256,
            )

            # Only a REPORTED failure sets the span to error. An `unknown`
            # outcome leaves the span status unset, because marking it OK would
            # assert a success nobody observed and marking it ERROR would
            # assert a failure nobody observed.
            if outcome == "failed":
                span.set_attribute("error.type", _error_type(error_class, status_code))
                _mark_error(span, error or "tool call failed")
        finally:
            try:
                span.end(end_time=_ns(end))
            except Exception:
                pass
    except Exception:
        return


def record_turn_span(
    *,
    system_instructions: list[Any] | None = None,
    input_messages: list[Any] | None = None,
    output_messages: list[Any] | None = None,
    input_text: str | None = None,
    output_text: str | None = None,
    turn_id: str | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    rollout_id: str | None = None,
    execution_sha256: str | None = None,
    start_time_ms: float | None = None,
    end_time_ms: float | None = None,
    error: str | None = None,
) -> None:
    """Emit one ``invoke_agent {agent}`` span carrying one whole TURN.

    THE CAPTURE SURFACE FOR THE FOUR TRAINABLE ALGORITHMS. SFT wants the
    messages; OPSD wants them plus a privileged reference; RAFT wants them plus
    a reward and a proven success; DPO wants two of them to pair.

    Content, all optional, in the convention's shapes:

    ``system_instructions``
        ``[{"type": "text", "content": ...}]``
    ``input_messages``
        the history the agent was given, in order
    ``output_messages``
        what it returned, one message per choice; a ``finish_reason`` on each
        is also collected into ``gen_ai.response.finish_reasons``
    ``input_text`` / ``output_text``
        a bare string for either side, when that is all the caller has;
        messages win when both are given

    ``error`` marks a turn the host reported as failed: ``error.type``
    ``_OTHER`` and ERROR status, since a failed invocation left UNSET claims it
    ran cleanly.

    OPT-IN, and the caller decides: nothing in this package can see a prompt
    or a completion on its own. With no content at all nothing is emitted: an
    empty turn span would assert a turn happened and carried nothing, which is
    a different claim from not having been given it. Never raises.
    """
    inputs = _genai_messages(input_messages)
    if inputs is None and input_text:
        inputs = [_text_message("user", input_text)]
    outputs = _genai_messages(output_messages)
    if outputs is None and output_text:
        outputs = [_text_message("assistant", output_text)]
    system = (list(system_instructions)
              if isinstance(system_instructions, (list, tuple))
              and system_instructions else None)
    if not inputs and not outputs and not system:
        return
    tracer = _tracer()
    if tracer is None:
        return
    try:
        end = _finite_number(end_time_ms)
        if end is None:
            end = time.time() * 1000.0
        begun = _finite_number(start_time_ms)
        start = begun if begun is not None and begun <= end else end
        attributes: dict[str, Any] = {"gen_ai.operation.name": "invoke_agent"}
        if agent_name:
            attributes["gen_ai.agent.name"] = str(agent_name)
        span = tracer.start_span(
            f"invoke_agent {agent_name}" if agent_name else "invoke_agent",
            kind=_span_kind_internal(), attributes=attributes,
            start_time=_ns(start),
        )
        try:
            if not getattr(span, "is_recording", lambda: False)():
                return
            if model:
                span.set_attribute("gen_ai.request.model", str(model))
            for key, value in (("gen_ai.system_instructions", system),
                               ("gen_ai.input.messages", inputs),
                               ("gen_ai.output.messages", outputs)):
                if value:
                    text = _genai_json(value)
                    if text is not None:
                        span.set_attribute(key, text)
            if outputs:
                # One reason per returned message, and only when every message
                # has one: a partial list would misalign reasons with messages.
                reasons = [m.get("finish_reason") if isinstance(m, dict) else None
                           for m in outputs]
                if all(isinstance(r, str) and r for r in reasons):
                    span.set_attribute("gen_ai.response.finish_reasons", reasons)
            _set_correlation(
                span, rollout_id=rollout_id, turn_id=turn_id,
                conversation_id=conversation_id, task_id=task_id,
                execution_sha256=execution_sha256,
            )
            if error:
                span.set_attribute("error.type", "_OTHER")
                _mark_error(span, error)
        finally:
            try:
                span.end(end_time=_ns(end))
            except Exception:
                pass
    except Exception:
        return
