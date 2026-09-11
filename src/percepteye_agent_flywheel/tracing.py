"""OpenTelemetry egress: a projection out, never a source in.

Tool outcomes already land in the append-only trajectory, which is what the control
plane grades. This module additionally emits them as OpenTelemetry spans so
they show up in whatever the customer already runs -- Datadog, Phoenix, Grafana
-- alongside the rest of their agent's telemetry.

**Tool spans are egress only. THE TURN SPAN IS NOT** -- see below, because that
changed and the old reasoning does not survive the change.

The original rule was that nothing here is ever read back, justified by a fact
about the spec rather than about maturity: the GenAI semantic conventions define
no attribute for per-token logprobs or token ids. The string "logprob" appears
in the whole convention three times, every one a note telling instrumentors NOT
to report ``top_logprobs`` as ``gen_ai.request.top_k``. So a span could not carry
the thing training needed, and ingesting one would be training on less than we
captured.

THAT ARGUMENT WAS SCOPED TO ONE OBJECTIVE, and it only became visible when the
scope narrowed. A SAMPLED logprob -- the behaviour policy's own, as opposed to
one recomputed from the model at training time -- is read by exactly one family
of objective: those forming an importance ratio against the policy that
produced the text. Every other objective this SDK's evidence feeds either
recomputes what it needs by forward pass or has no logprob in its loss at all.
For those, "the thing training needs" is the prompt, the completion and a
verdict, all of which a span carries perfectly well.

So :func:`record_turn_span` is a capture surface, not a projection, and it is
OPT-IN for a reason the tool spans do not share: a tool name and an outcome are
telemetry, while a prompt and a completion are the user's own words.

The rest of the original rule still holds for tool spans: because we never read
those back, the customer's sampler, queue overflow and exporter failures cost us
nothing.

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

__all__ = ["record_tool_call_span", "record_turn_span", "tracing_available"]

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
    turn_id: str | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    tool_call_id: str | None = None,
    execution_sha256: str | None = None,
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
            if turn_id:
                # A production turn is NOT a rollout, and stamping its id under
                # `percepteye.rollout.id` made a customer's own telemetry unable
                # to answer "how much of my traffic was training" -- every
                # served turn looked like contributed work.
                span.set_attribute("percepteye.turn.id", str(turn_id))
            if conversation_id:
                # ON THE TOOL SPAN TOO, not only on the turn span. A consumer
                # grouping an export into conversations must put a call in the
                # same bucket as the turn that made it. The turn id can recover
                # that link -- but ONLY when the same export also carries a
                # chat span for that turn, and an export sliced by time or
                # sampled by the collector may not.
                span.set_attribute(
                    "percepteye.conversation.id", str(conversation_id))
            if task_id:
                # Caller-authored correlation evidence only.  This does not
                # assert success, policy identity or training eligibility; it
                # gives an ingest-side consumer an exact task join to validate.
                span.set_attribute("percepteye.task.id", str(task_id))
            if execution_sha256:
                span.set_attribute(
                    "percepteye.agent.execution_sha256", str(execution_sha256)
                )
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


#: Turn content is bounded before it leaves. A pasted stack trace or a long
#: answer must not make one span dominate the customer's export, and an
#: exporter that silently drops an oversized span costs us the whole turn
#: rather than its tail.
_MAX_TURN_CHARS = 20_000


def record_turn_span(
    *,
    input_text: str | None = None,
    output_text: str | None = None,
    turn_id: str | None = None,
    conversation_id: str | None = None,
    task_id: str | None = None,
    agent_name: str | None = None,
    model: str | None = None,
    rollout_id: str | None = None,
    execution_sha256: str | None = None,
) -> None:
    """Emit one span carrying what the agent was ASKED and what it ANSWERED.

    THE CAPTURE SURFACE FOR THE FOUR TRAINABLE ALGORITHMS. SFT wants the
    messages; OPSD wants them plus a privileged reference; RAFT wants them plus
    a reward and a proven success; DPO wants two of them to pair. None reads a
    sampled logprob, which is why a span suffices here where it would not for
    offline GRPO.

    OPT-IN, and the caller decides. Nothing in this package can see a prompt or
    a completion on its own -- the adapters observe tool calls, not turn content
    -- so this is only ever called with text the customer handed over
    deliberately. That asymmetry with :func:`record_tool_call_span` is the
    point: a tool name is telemetry, a prompt is the user's own words.

    Never raises, like everything else here.

    ``rollout_id`` is stamped when the turn IS a rollout, because that is the
    marker the ingest side reads to keep our own training traffic out of the
    customer's observed-traffic accounting. A production turn carries
    ``percepteye.turn.id`` instead and no rollout id, so it is correctly counted
    as the customer's own -- which is exactly what it is.
    """
    if not input_text and not output_text:
        # Nothing to say. An empty turn span would assert a turn happened and
        # carried no content, which is a different claim from not being given
        # the content.
        return
    tracer = _tracer()
    if tracer is None:
        return
    try:
        with tracer.start_as_current_span("chat") as span:
            if not getattr(span, "is_recording", lambda: False)():
                return
            span.set_attribute("gen_ai.operation.name", "chat")
            if model:
                span.set_attribute("gen_ai.request.model", str(model))
            if agent_name:
                span.set_attribute("gen_ai.agent.name", str(agent_name))
            # The convention's names for turn content. Carried as attributes
            # rather than events because the ingest side normalises attributes
            # into `raw_attributes` verbatim, and an event body does not
            # survive that path.
            if input_text:
                span.set_attribute(
                    "gen_ai.input.messages", str(input_text)[:_MAX_TURN_CHARS])
            if output_text:
                span.set_attribute(
                    "gen_ai.output.messages", str(output_text)[:_MAX_TURN_CHARS])
            if rollout_id:
                span.set_attribute("percepteye.rollout.id", str(rollout_id))
            if turn_id:
                span.set_attribute("percepteye.turn.id", str(turn_id))
            if conversation_id:
                span.set_attribute(
                    "percepteye.conversation.id", str(conversation_id))
            if task_id:
                span.set_attribute("percepteye.task.id", str(task_id))
            if execution_sha256:
                span.set_attribute(
                    "percepteye.agent.execution_sha256", str(execution_sha256)
                )
    except Exception:
        return
