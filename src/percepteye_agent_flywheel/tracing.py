"""OpenTelemetry egress: a projection out, never a source in.

Tool outcomes already land in the append-only trajectory, which is what the control
plane grades. This module additionally emits them as OpenTelemetry spans so
they show up in whatever the customer already runs -- Datadog, Phoenix, Grafana
-- alongside the rest of their agent's telemetry.

**Egress only. Nothing here is ever read back.** The trajectory stays the record of
what was trained on, for a reason that is a fact about the OTel spec rather
than about maturity: the GenAI semantic conventions define no attribute for
per-token logprobs or token ids. The string "logprob" appears in the whole
convention three times, every one a note telling instrumentors NOT to report
``top_logprobs`` as ``gen_ai.request.top_k``. So spans cannot carry the thing
training needs, and a system that ingested them would be quietly training on
less than it captured. Because we never read them back, the customer's
sampler, queue overflow and exporter failures cost us nothing.

**We never install a TracerProvider.** ``set_tracer_provider`` is once-only:
the second caller gets a log line, not an exception. So calling it would either
lose all our spans (if the customer configured theirs first) or silently kill
the customer's entire GenAI telemetry (if we won). We take whatever provider is
already registered and emit into it. With no provider configured, the API's
no-op tracer swallows everything, which is the correct outcome for a library.

Attributes follow ``gen_ai.*`` where the convention has a name for something,
and ``percepteye.*`` where it does not -- which notably includes the tri-state
outcome. The convention has no way to say "this tool call's result was never
observed", so the projection is documented as lossy rather than pretending
otherwise.
"""
from __future__ import annotations

from typing import Any

__all__ = ["record_tool_call_span", "tracing_available"]

#: Resolved once. ``None`` means the API is not installed, which is the normal
#: case: the ``otel`` extra is opt-in and the package ships with one runtime
#: dependency.
_TRACER: Any = None
_RESOLVED = False


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
        _TRACER = trace.get_tracer("percepteye-agent-flywheel")
    except Exception:
        _TRACER = None
    return _TRACER


def tracing_available() -> bool:
    """True when the OTel API is importable. Says nothing about whether the
    customer configured a provider -- with none, spans go to a no-op."""
    return _tracer() is not None


def record_tool_call_span(
    name: str,
    arguments: dict[str, Any] | None,
    *,
    outcome: str,
    status_code: int | None = None,
    error: str | None = None,
    latency_ms: float | None = None,
    agent_name: str | None = None,
    rollout_id: str | None = None,
    tool_call_id: str | None = None,
) -> None:
    """Project one recorded tool call into the customer's traces.

    A no-op when the API is absent, and it NEVER raises: this runs beside the
    trajectory write on the customer's own execution path, and a telemetry failure
    must cost us the span and nothing else.
    """
    tracer = _tracer()
    if tracer is None:
        return
    try:
        with tracer.start_as_current_span(f"execute_tool {name}") as span:
            if not getattr(span, "is_recording", lambda: False)():
                return
            # gen_ai.* where the convention names it.
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.tool.name", str(name))
            if tool_call_id:
                # The convention DOES name this one: the model-assigned call
                # id, the same value that reappears on the role="tool" message
                # of the next request. It is what lets a trace consumer join a
                # tool span to the completion that requested it.
                span.set_attribute("gen_ai.tool.call.id", str(tool_call_id))
            if arguments is not None:
                span.set_attribute("gen_ai.tool.call.arguments", _short(arguments))

            # percepteye.* where it does not. The GenAI conventions have no way
            # to express "the result of this call was never observed", and that
            # distinction is the whole contract -- so it is namespaced as ours
            # rather than bent into a standard attribute that means something
            # else to every other consumer.
            span.set_attribute("percepteye.tool.outcome", str(outcome))
            if status_code is not None:
                span.set_attribute("percepteye.tool.status_code", int(status_code))
            if rollout_id:
                span.set_attribute("percepteye.rollout.id", str(rollout_id))
            if agent_name:
                span.set_attribute("gen_ai.agent.name", str(agent_name))
            if latency_ms is not None:
                span.set_attribute("percepteye.tool.latency_ms", float(latency_ms))

            # Only a REPORTED failure sets the span to error. An `unknown`
            # outcome leaves the span status unset, because marking it OK would
            # assert a success nobody observed and marking it ERROR would
            # assert a failure nobody observed.
            if outcome == "failed":
                _mark_error(span, error or "tool call failed")
    except Exception:
        return


def _mark_error(span: Any, message: str) -> None:
    try:
        from opentelemetry.trace import Status, StatusCode  # type: ignore

        span.set_status(Status(StatusCode.ERROR, message[:500]))
    except Exception:
        # Older/partial API surfaces: the attribute still carries the verdict.
        pass


def _short(value: Any, limit: int = 2000) -> str:
    """Attributes are telemetry, not payloads. Truncated and stringified so a
    large tool argument cannot bloat every span in the customer's backend."""
    try:
        import json

        out = json.dumps(value, default=str, ensure_ascii=False)
    except Exception:
        out = str(value)
    return out if len(out) <= limit else out[:limit] + "...[truncated]"
