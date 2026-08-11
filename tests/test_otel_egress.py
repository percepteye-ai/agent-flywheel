"""OpenTelemetry egress: a projection out, never a source in.

The ledger is what the control plane grades. Spans are additionally emitted so
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
    """This runs on the customer's own execution path beside the ledger write."""
    import percepteye_agent_flywheel.tracing as mod

    class _Exploding:
        def start_as_current_span(self, *a, **kw):
            raise RuntimeError("collector down")

    monkeypatch.setattr(mod, "_TRACER", _Exploding())
    record_tool_call_span("lookup", {}, outcome="ok")   # no raise


# ── the ledger is unaffected ───────────────────────────────────────────────


def test_recording_an_outcome_still_writes_the_ledger_and_also_emits(spans, tmp_path):
    """Egress is ADDITIVE. The ledger stays the record of what was trained on,
    because spans cannot carry logprobs or token ids — the GenAI convention
    defines no attribute for either."""
    from percepteye_agent_flywheel.outcomes import OutcomeLedger

    with _RolloutScope("rl_x", str(tmp_path)):
        record_tool_call("deploy", {"svc": "api"}, outcome="unknown", status_code=None)

    calls = OutcomeLedger(str(tmp_path), "rl_x").read()
    assert calls is not None and len(calls) == 1
    assert calls[0].outcome == "unknown"
    assert _only(spans).attributes["percepteye.tool.outcome"] == "unknown"


def test_tracing_available_reports_the_api_not_the_provider():
    assert isinstance(tracing_available(), bool)
