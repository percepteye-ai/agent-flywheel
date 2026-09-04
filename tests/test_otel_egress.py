"""OpenTelemetry egress: a projection out, never a source in.

The trajectory is what the control plane grades. Spans are additionally emitted so
outcomes appear in whatever the customer already runs. Nothing here is ever
read back, which is why their sampler, queue overflow and exporter failures
cost us nothing.

The two properties worth defending: the tri-state survives the projection, and
we never touch the customer's TracerProvider.
"""
from __future__ import annotations

import pytest

from percepteye_agent_flywheel.outcomes import _RolloutScope, record_tool_call
from percepteye_agent_flywheel.tracing import (
    record_tool_call_span,
    tracing_available,
)

pytest.importorskip("opentelemetry.sdk")


@pytest.fixture
def spans(monkeypatch):
    """A real provider with an in-memory exporter, installed the way a CUSTOMER
    would — then handed to our module the same way theirs would be found."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    import percepteye_agent_flywheel.tracing as mod

    monkeypatch.setattr(mod, "_TRACER", provider.get_tracer("test"))
    monkeypatch.setattr(mod, "_RESOLVED", True)
    return exporter


def _only(exporter):
    got = exporter.get_finished_spans()
    assert len(got) == 1, f"expected exactly one span, got {len(got)}"
    return got[0]


# ── the tri-state survives the projection ──────────────────────────────────


@pytest.mark.parametrize("outcome", ["ok", "failed", "unknown"])
def test_the_outcome_rides_on_the_span_verbatim(spans, outcome):
    """THE reason this is a documented-lossy projection rather than a standard
    one: the GenAI conventions have no way to say 'the result of this call was
    never observed', so the tri-state gets our own namespace."""
    record_tool_call_span("charge_card", {"amount": 42}, outcome=outcome)
    span = _only(spans)
    assert span.attributes["percepteye.tool.outcome"] == outcome


def test_an_unknown_outcome_leaves_the_span_status_unset(spans):
    """Marking it OK would assert a success nobody observed; marking it ERROR
    would assert a failure nobody observed. Neither is available to us."""
    from opentelemetry.trace import StatusCode

    record_tool_call_span("charge_card", {}, outcome="unknown")
    assert _only(spans).status.status_code == StatusCode.UNSET


def test_a_reported_failure_marks_the_span_as_an_error(spans):
    from opentelemetry.trace import StatusCode

    record_tool_call_span("charge_card", {}, outcome="failed", error="declined")
    span = _only(spans)
    assert span.status.status_code == StatusCode.ERROR
    assert "declined" in (span.status.description or "")


def test_a_success_is_not_marked_error(spans):
    from opentelemetry.trace import StatusCode

    record_tool_call_span("lookup", {}, outcome="ok", status_code=200)
    assert _only(spans).status.status_code != StatusCode.ERROR


# ── conventions where they exist, our namespace where they do not ──────────


def test_it_uses_gen_ai_attributes_where_the_convention_names_them(spans):
    record_tool_call_span("lookup", {"q": "x"}, outcome="ok", agent_name="planner")
    a = _only(spans).attributes
    assert a["gen_ai.operation.name"] == "execute_tool"
    assert a["gen_ai.tool.name"] == "lookup"
    assert a["gen_ai.agent.name"] == "planner"
    assert "q" in a["gen_ai.tool.call.arguments"]


def test_percepteye_attributes_carry_what_the_convention_cannot(spans):
    record_tool_call_span(
        "lookup", {}, outcome="failed", status_code=200,
        latency_ms=12.5, rollout_id="rl_1",
    )
    a = _only(spans).attributes
    assert a["percepteye.tool.outcome"] == "failed"
    # An application-level denial the API returned as 200: both halves true,
    # both recorded, and neither inferable from the other.
    assert a["percepteye.tool.status_code"] == 200
    assert a["percepteye.rollout.id"] == "rl_1"
    assert a["percepteye.tool.latency_ms"] == pytest.approx(12.5)


def test_large_arguments_are_truncated(spans):
    """Attributes are telemetry, not payloads — one huge tool argument must not
    bloat every span in the customer's backend."""
    record_tool_call_span("upload", {"blob": "x" * 50_000}, outcome="ok")
    val = _only(spans).attributes["gen_ai.tool.call.arguments"]
    assert len(val) < 3_000 and val.endswith("...[truncated]")


# ── it never touches the customer's telemetry ──────────────────────────────


def test_we_never_install_a_tracer_provider(monkeypatch):
    """``set_tracer_provider`` is once-only: the loser gets a log line, not an
    exception. So calling it would either lose all our spans (if the customer
    configured theirs first) or silently kill the customer's entire GenAI
    telemetry (if we won).

    Asserted by BEHAVIOUR — the call is booby-trapped — rather than by grepping
    the source, which an earlier version of this test did and which failed on
    the comments explaining why we do not call it.
    """
    from opentelemetry import trace

    import percepteye_agent_flywheel.tracing as mod

    called: list[object] = []
    monkeypatch.setattr(
        trace, "set_tracer_provider",
        lambda *a, **kw: called.append(a),
        raising=True,
    )
    # Force a full, uncached resolution.
    monkeypatch.setattr(mod, "_TRACER", None)
    monkeypatch.setattr(mod, "_RESOLVED", False)
    mod.tracing_available()
    record_tool_call_span("lookup", {}, outcome="ok")

    assert called == [], "the customer's TracerProvider was replaced"


def test_the_sdk_is_never_imported_by_the_egress_module():
    """The extra is the API only. Importing the SDK would pull a provider
    implementation into a library that must not own one."""
    import inspect

    import percepteye_agent_flywheel.tracing as mod

    assert "opentelemetry.sdk" not in inspect.getsource(mod)


def test_with_no_provider_configured_nothing_raises(monkeypatch):
    """A library with no provider must be silent, not broken."""
    import percepteye_agent_flywheel.tracing as mod

    monkeypatch.setattr(mod, "_TRACER", None)
    monkeypatch.setattr(mod, "_RESOLVED", True)
    record_tool_call_span("lookup", {}, outcome="ok")   # no raise


def test_a_telemetry_failure_never_reaches_the_caller(spans, monkeypatch):
    """This runs on the customer's own execution path beside the trajectory write."""
    import percepteye_agent_flywheel.tracing as mod

    class _Exploding:
        def start_as_current_span(self, *a, **kw):
            raise RuntimeError("collector down")

    monkeypatch.setattr(mod, "_TRACER", _Exploding())
    record_tool_call_span("lookup", {}, outcome="ok")   # no raise


# ── the trajectory is unaffected ───────────────────────────────────────────────


def test_recording_an_outcome_still_writes_the_trajectory_and_also_emits(spans, tmp_path):
    """Egress is ADDITIVE. The trajectory stays the record of what was trained on,
    because spans cannot carry logprobs or token ids — the GenAI convention
    defines no attribute for either."""
    from percepteye_agent_flywheel.outcomes import Trajectory

    with _RolloutScope("rl_x", str(tmp_path)):
        record_tool_call("deploy", {"svc": "api"}, outcome="unknown", status_code=None)

    calls = Trajectory(str(tmp_path)).read()
    assert calls is not None and len(calls) == 1
    assert calls[0].outcome == "unknown"
    assert _only(spans).attributes["percepteye.tool.outcome"] == "unknown"


def test_tracing_available_reports_the_api_not_the_provider():
    assert isinstance(tracing_available(), bool)


# ── the turn span: a capture surface, not a projection ─────────────────────


def test_a_turn_span_carries_the_prompt_and_the_completion(spans):
    """THE CAPTURE the four trainable algorithms need. SFT wants the messages;
    OPSD wants them plus a reference; RAFT plus a reward and a proven success;
    DPO wants two to pair. None reads a sampled logprob, which is why a span
    suffices here where it would not for offline GRPO."""
    from percepteye_agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="summarise the incident",
                     output_text="Three services timed out.",
                     turn_id="t1", conversation_id="c1", model="gpt-4o")
    span = _only(spans)
    assert span.attributes["gen_ai.operation.name"] == "chat"
    assert span.attributes["gen_ai.input.messages"] == "summarise the incident"
    assert span.attributes["gen_ai.output.messages"] == "Three services timed out."
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"


def test_a_turn_with_no_content_emits_no_span(spans):
    """An empty turn span would assert a turn happened and carried no content,
    which is a different claim from not being given the content."""
    from percepteye_agent_flywheel.tracing import record_turn_span

    record_turn_span(turn_id="t2")
    assert spans.get_finished_spans() == ()


def test_half_a_turn_still_emits(spans):
    """The answer arrives later than the question; neither half waits for the
    other."""
    from percepteye_agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="just the question", turn_id="t3")
    span = _only(spans)
    assert span.attributes["gen_ai.input.messages"] == "just the question"
    assert "gen_ai.output.messages" not in span.attributes


def test_a_production_turn_is_not_stamped_as_a_rollout(spans):
    """THE SELF-FEEDING GUARD, from the emitting side. The ingest side keys on
    `percepteye.rollout.id` to keep our own training traffic out of the
    customer's observed-traffic accounting. A production turn must carry the
    turn id and NO rollout id, or every served turn looks like our work."""
    from percepteye_agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="q", output_text="a", turn_id="t4")
    span = _only(spans)
    assert span.attributes["percepteye.turn.id"] == "t4"
    assert "percepteye.rollout.id" not in span.attributes


def test_a_rollout_turn_IS_stamped_so_ingest_can_exclude_it(spans):
    from percepteye_agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="q", output_text="a", rollout_id="r9")
    span = _only(spans)
    assert span.attributes["percepteye.rollout.id"] == "r9"


def test_turn_content_is_bounded(spans):
    """One span must not dominate the customer's export, and an exporter that
    drops an oversized span costs the whole turn rather than its tail."""
    from percepteye_agent_flywheel.tracing import _MAX_TURN_CHARS, record_turn_span

    record_turn_span(input_text="x" * (_MAX_TURN_CHARS + 5_000), turn_id="t5")
    assert len(_only(spans).attributes["gen_ai.input.messages"]) == _MAX_TURN_CHARS


def test_recording_a_turn_emits_the_span_from_the_same_call(spans, tmp_path):
    """One call site for one fact. A turn that reached the trajectory but not
    the customer's traces is the drift nobody notices until they compare."""
    from percepteye_agent_flywheel.outcomes import record_turn_text

    record_turn_text(tmp_path, "task", "asked", turn_id="t6")
    span = _only(spans)
    assert span.attributes["gen_ai.input.messages"] == "asked"
    assert span.attributes["percepteye.turn.id"] == "t6"


# ── the same guard on the TOOL span ────────────────────────────────────────
#
# Found by a mis-aimed mutation: the two functions carry a byte-identical
# `if rollout_id: ... if turn_id:` block, so a mutation meant for the turn span
# landed on the tool span and EVERY test still passed. The turn span's guard was
# covered; this one was not, and it is the span that has been flowing all along.


def test_a_production_tool_call_is_not_stamped_as_a_rollout(spans):
    """`percepteye.rollout.id` is the marker the ingest side keys on to keep
    our own training traffic out of the customer's observed-traffic accounting.
    Stamping a served turn's tool call with it makes every one of them look
    like contributed work -- which is the exact failure the comment beside this
    code describes."""
    from percepteye_agent_flywheel.tracing import record_tool_call_span

    record_tool_call_span("shell", {"cmd": "ls"}, outcome="ok", turn_id="t9")
    span = _only(spans)
    assert span.attributes["percepteye.turn.id"] == "t9"
    assert "percepteye.rollout.id" not in span.attributes


def test_a_rollout_tool_call_IS_stamped_so_ingest_can_exclude_it(spans):
    from percepteye_agent_flywheel.tracing import record_tool_call_span

    record_tool_call_span("shell", {"cmd": "ls"}, outcome="ok", rollout_id="r9")
    span = _only(spans)
    assert span.attributes["percepteye.rollout.id"] == "r9"
    assert "percepteye.turn.id" not in span.attributes
