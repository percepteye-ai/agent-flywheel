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

from agent_flywheel.outcomes import _RolloutScope, record_tool_call
from agent_flywheel.tracing import (
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

    import agent_flywheel.tracing as mod

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
    import json

    from opentelemetry.trace import SpanKind

    record_tool_call_span("lookup", {"q": "x"}, outcome="ok", agent_name="planner")
    span = _only(spans)
    a = span.attributes
    assert span.kind == SpanKind.INTERNAL
    assert a["gen_ai.operation.name"] == "execute_tool"
    assert a["gen_ai.tool.name"] == "lookup"
    assert a["gen_ai.tool.type"] == "function"
    assert a["gen_ai.agent.name"] == "planner"
    # An OBJECT, as JSON text -- the convention's shape for arguments.
    assert json.loads(a["gen_ai.tool.call.arguments"]) == {"q": "x"}


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


def test_large_arguments_are_carried_whole_as_valid_json(spans):
    """No size limitation. The 2,000-character cut that stood here ended the
    argument in ``...[truncated]``, which is not JSON at all -- a consumer
    parsing it got an error, not a shorter argument."""
    import json

    record_tool_call_span("upload", {"blob": "x" * 50_000}, outcome="ok")
    val = _only(spans).attributes["gen_ai.tool.call.arguments"]
    assert json.loads(val) == {"blob": "x" * 50_000}


def test_a_failure_carries_error_type_and_no_result(spans):
    """``error.type`` is Conditionally Required on a failed operation, and the
    result is recorded only when execution succeeded."""
    record_tool_call_span("charge_card", {}, outcome="failed", status_code=402,
                          error="declined", error_class="CardDeclined",
                          result={"code": "declined"})
    a = _only(spans).attributes
    assert a["error.type"] == "CardDeclined"
    assert "gen_ai.tool.call.result" not in a


def test_a_success_carries_its_result_as_an_object(spans):
    import json

    record_tool_call_span("lookup", {}, outcome="ok",
                          result='{"hits": 3, "next": null}')
    a = _only(spans).attributes
    assert json.loads(a["gen_ai.tool.call.result"]) == {"hits": 3, "next": None}


def test_the_span_covers_the_call(spans):
    """Its start is ``latency_ms`` before its end, not the instant it was
    written down."""
    record_tool_call_span("lookup", {}, outcome="ok", latency_ms=250,
                          end_time_ms=1_700_000_000_000)
    span = _only(spans)
    assert span.end_time == 1_700_000_000_000 * 1_000_000
    assert span.end_time - span.start_time == 250 * 1_000_000


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

    import agent_flywheel.tracing as mod

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

    import agent_flywheel.tracing as mod

    assert "opentelemetry.sdk" not in inspect.getsource(mod)


def test_with_no_provider_configured_nothing_raises(monkeypatch):
    """A library with no provider must be silent, not broken."""
    import agent_flywheel.tracing as mod

    monkeypatch.setattr(mod, "_TRACER", None)
    monkeypatch.setattr(mod, "_RESOLVED", True)
    record_tool_call_span("lookup", {}, outcome="ok")   # no raise


def test_a_telemetry_failure_never_reaches_the_caller(spans, monkeypatch):
    """This runs on the customer's own execution path beside the trajectory write."""
    import agent_flywheel.tracing as mod

    class _Exploding:
        def start_span(self, *a, **kw):
            raise RuntimeError("collector down")

    monkeypatch.setattr(mod, "_TRACER", _Exploding())
    record_tool_call_span("lookup", {}, outcome="ok")   # no raise


# ── the trajectory is unaffected ───────────────────────────────────────────────


def test_recording_an_outcome_still_writes_the_trajectory_and_also_emits(spans, tmp_path):
    """Egress is ADDITIVE. The trajectory stays the record of what was trained on,
    because spans cannot carry logprobs or token ids — the GenAI convention
    defines no attribute for either."""
    from agent_flywheel.outcomes import Trajectory

    with _RolloutScope("rl_x", str(tmp_path)):
        record_tool_call("deploy", {"svc": "api"}, outcome="unknown", status_code=None)

    calls = Trajectory(str(tmp_path)).read()
    assert calls is not None and len(calls) == 1
    assert calls[0].outcome == "unknown"
    assert _only(spans).attributes["percepteye.tool.outcome"] == "unknown"


def test_a_recorded_outcome_reaches_its_span_with_result_and_error_class(
    spans, tmp_path,
):
    """The wiring, not the renderer: ``record_tool_call`` hands the span the
    output and the classified error it recorded."""
    import json

    with _RolloutScope("rl_y", str(tmp_path)):
        record_tool_call("lookup", {"q": "x"}, outcome="ok", status_code=200,
                         output={"hits": 2})
        record_tool_call("charge", {}, outcome="failed", status_code=402,
                         error="declined", error_class="CardDeclined")
    ok, failed = spans.get_finished_spans()
    assert json.loads(ok.attributes["gen_ai.tool.call.result"]) == {"hits": 2}
    assert failed.attributes["error.type"] == "CardDeclined"


def test_turn_text_is_stored_whole(tmp_path):
    """No size limitation on the turn record either: the span is built from
    it, and the 20,000-character cut made both carry a different turn."""
    from agent_flywheel.outcomes import read_turn_text, record_turn_text

    text = "x" * 20_500 + "\U0001F4E6" * 5_000
    assert record_turn_text(tmp_path, "task", text)
    assert read_turn_text(tmp_path)["task"] == text


def test_tracing_available_reports_the_api_not_the_provider():
    assert isinstance(tracing_available(), bool)


# ── the turn span: a capture surface, not a projection ─────────────────────


def test_a_turn_span_carries_the_prompt_and_the_completion(spans):
    """THE CAPTURE the four trainable algorithms need. SFT wants the messages;
    OPSD wants them plus a reference; RAFT plus a reward and a proven success;
    DPO wants two to pair. None reads a sampled logprob, which is why a span
    suffices here where it would not for offline GRPO.

    In the convention's message shape, as JSON text -- plain text in these
    attributes is not what the schema says they hold."""
    import json

    from agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="summarise the incident",
                     output_text="Three services timed out.",
                     turn_id="t1", conversation_id="c1", model="gpt-4o")
    span = _only(spans)
    assert span.name == "invoke_agent"
    assert span.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert json.loads(span.attributes["gen_ai.input.messages"]) == [
        {"role": "user",
         "parts": [{"type": "text", "content": "summarise the incident"}]}]
    assert json.loads(span.attributes["gen_ai.output.messages"]) == [
        {"role": "assistant",
         "parts": [{"type": "text", "content": "Three services timed out."}]}]
    assert span.attributes["gen_ai.request.model"] == "gpt-4o"
    assert span.attributes["gen_ai.conversation.id"] == "c1"


def test_a_turn_with_no_content_emits_no_span(spans):
    """An empty turn span would assert a turn happened and carried no content,
    which is a different claim from not being given the content."""
    from agent_flywheel.tracing import record_turn_span

    record_turn_span(turn_id="t2")
    assert spans.get_finished_spans() == ()


def test_half_a_turn_still_emits(spans):
    """The answer arrives later than the question; neither half waits for the
    other."""
    from agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="just the question", turn_id="t3")
    span = _only(spans)
    assert "just the question" in span.attributes["gen_ai.input.messages"]
    assert "gen_ai.output.messages" not in span.attributes


def test_a_production_turn_is_not_stamped_as_a_rollout(spans):
    """THE SELF-FEEDING GUARD, from the emitting side. The ingest side keys on
    `percepteye.rollout.id` to keep our own training traffic out of the
    customer's observed-traffic accounting. A production turn must carry the
    turn id and NO rollout id, or every served turn looks like our work."""
    from agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="q", output_text="a", turn_id="t4")
    span = _only(spans)
    assert span.attributes["percepteye.turn.id"] == "t4"
    assert "percepteye.rollout.id" not in span.attributes


def test_a_rollout_turn_IS_stamped_so_ingest_can_exclude_it(spans):
    from agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="q", output_text="a", rollout_id="r9")
    span = _only(spans)
    assert span.attributes["percepteye.rollout.id"] == "r9"


def test_turn_content_is_carried_whole(spans):
    """No size limitation: the 20,000-character cut that stood here made the
    span carry a different turn from the one the agent served."""
    import json

    from agent_flywheel.tracing import record_turn_span

    record_turn_span(input_text="x" * 25_000, turn_id="t5")
    content = json.loads(_only(spans).attributes["gen_ai.input.messages"])
    assert content[0]["parts"][0]["content"] == "x" * 25_000


def test_writing_half_a_turn_emits_no_span_of_its_own(spans, tmp_path):
    """A span per half-turn described neither half. The turn's one span is
    emitted by ``record_answer``, when the turn is whole."""
    from agent_flywheel.outcomes import record_turn_text

    assert record_turn_text(tmp_path, "task", "asked", turn_id="t6")
    assert spans.get_finished_spans() == ()


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
    from agent_flywheel.tracing import record_tool_call_span

    record_tool_call_span("shell", {"cmd": "ls"}, outcome="ok", turn_id="t9")
    span = _only(spans)
    assert span.attributes["percepteye.turn.id"] == "t9"
    assert "percepteye.rollout.id" not in span.attributes


def test_a_rollout_tool_call_IS_stamped_so_ingest_can_exclude_it(spans):
    from agent_flywheel.tracing import record_tool_call_span

    record_tool_call_span("shell", {"cmd": "ls"}, outcome="ok", rollout_id="r9")
    span = _only(spans)
    assert span.attributes["percepteye.rollout.id"] == "r9"
    assert "percepteye.turn.id" not in span.attributes


# ── one span per TURN, carrying both halves ────────────────────────────────


def test_a_turn_is_one_invoke_agent_span_carrying_both_halves(spans, tmp_path):
    """The turn's ``invoke_agent`` span, emitted once the answer exists.

    It REPLACED two ``chat`` spans, one per half, which described neither half
    as what it was -- no model call happened on either -- and which a consumer
    had to pair by position. Now the turn is one span from the moment it
    opened to its answer, so its tool spans fall inside it in time, and all of
    them name the same turn and conversation.
    """
    import json

    from agent_flywheel.outcomes import record_tool_call
    from agent_flywheel.production import attach

    px = attach(agent_id="support-bot", trajectory_root=str(tmp_path),
                register=False)
    try:
        with px.turn(turn_id="t1", input_text="book a flight",
                     conversation_id="conv-1"):
            record_tool_call(name="book", arguments={}, outcome="ok",
                             status_code=201, latency_ms=0.0)
        px.record_answer("t1", "Booked.")
    finally:
        px.close()

    got = spans.get_finished_spans()
    turns = [s for s in got
             if s.attributes.get("gen_ai.operation.name") == "invoke_agent"]
    tools = [s for s in got
             if s.attributes.get("gen_ai.operation.name") == "execute_tool"]
    assert len(turns) == 1 and len(tools) == 1
    assert not [s for s in got
                if s.attributes.get("gen_ai.operation.name") == "chat"]
    turn, tool = turns[0], tools[0]
    assert turn.name == "invoke_agent support-bot"
    a = turn.attributes
    assert json.loads(a["gen_ai.input.messages"])[0]["parts"][0]["content"] == \
        "book a flight"
    assert json.loads(a["gen_ai.output.messages"])[0]["parts"][0]["content"] == \
        "Booked."
    # THE GAP THIS CLOSED: the answer half was recorded without its
    # conversation, so the span carrying it had none.
    assert a["gen_ai.conversation.id"] == "conv-1"
    assert a["percepteye.conversation.id"] == "conv-1"
    assert tool.attributes["gen_ai.conversation.id"] == "conv-1"
    # A served turn is the customer's own traffic, never a rollout.
    assert all("percepteye.rollout.id" not in s.attributes for s in got)
    assert {s.attributes["percepteye.turn.id"] for s in got} == {"t1"}
    # The turn spans its tool call.
    assert turn.start_time <= tool.start_time <= tool.end_time <= turn.end_time


def test_the_answer_record_names_its_conversation_like_the_task_record(tmp_path):
    """One format, two SDKs: Node writes the answer record with the turn's
    conversation id, so Python must too."""
    import json

    from agent_flywheel.production import attach

    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        with px.turn(turn_id="t1", input_text="q", conversation_id="conv-2"):
            pass
        px.record_answer("t1", "a")
    finally:
        px.close()
    records = [json.loads(line) for line in
               (tmp_path / "t1" / "turns.jsonl").read_text().splitlines()]
    assert [r.get("conversation_id") for r in records] == ["conv-2", "conv-2"]


def test_exact_attached_execution_identity_reaches_turn_and_tool_spans(
    spans, tmp_path, monkeypatch,
):
    import agent_flywheel as fw
    from agent_flywheel.introspection import (
        DiscoveredAgent,
        ModelConfig,
        ToolDefinition,
    )

    class _Agent:
        def invoke(self, task: str) -> str:
            return task

    monkeypatch.setattr(
        "agent_flywheel.introspection.introspect",
        lambda _entrypoint: DiscoveredAgent(
            agent_name="customer-agent",
            framework="customer-framework",
            system_prompt="Use tools and answer the user.",
            tool_definitions=[ToolDefinition(name="book", input_schema={})],
            discovered_model_config=ModelConfig(
                provider="gateway", model_name="policy",
            ),
        ),
    )
    px = fw.attach(
        agent_id="a",
        trajectory_root=str(tmp_path),
        register=False,
        adapters=[],
        execution_entrypoint=_Agent(),
        attested_execution_components={"customer.release": "a" * 64},
    )
    try:
        assert px.agent_fingerprint is not None
        digest = px.agent_fingerprint["execution_sha256"]
        with px.turn(turn_id="t1", input_text="book a flight"):
            record_tool_call(
                name="book", arguments={}, outcome="ok", status_code=201,
            )
        px.record_answer("t1", "Booked.")
    finally:
        px.close()

    got = spans.get_finished_spans()
    assert len(got) == 2
    assert {
        span.attributes.get("percepteye.agent.execution_sha256")
        for span in got
    } == {digest}


def test_task_correlation_reaches_turn_and_tool_spans(spans, tmp_path):
    """Upload and OTLP paths carry the same neutral exact-join candidate."""
    from agent_flywheel.outcomes import record_tool_call
    from agent_flywheel.production import attach

    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        with px.turn(turn_id="t1", task_id="case-7", input_text="book"):
            record_tool_call(
                name="book", arguments={}, outcome="ok", status_code=201,
            )
        px.record_answer("t1", "Booked.")
    finally:
        px.close()

    got = spans.get_finished_spans()
    assert len(got) == 2
    assert {
        span.attributes.get("percepteye.task.id") for span in got
    } == {"case-7"}


def test_absent_task_correlation_invents_no_otel_attribute(spans, tmp_path):
    from agent_flywheel.outcomes import record_tool_call
    from agent_flywheel.production import attach

    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        with px.turn(turn_id="t1", input_text="book"):
            record_tool_call(
                name="book", arguments={}, outcome="ok", status_code=201,
            )
    finally:
        px.close()

    assert all(
        "percepteye.task.id" not in span.attributes
        for span in spans.get_finished_spans()
    )


def test_conflicting_task_correlation_is_omitted_from_later_spans(
    spans, tmp_path,
):
    from agent_flywheel.outcomes import record_tool_call
    from agent_flywheel.production import attach

    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        with px.turn(turn_id="same", task_id="case-a", input_text="first"):
            pass
        px.record_answer("same", "one")
        with px.turn(turn_id="same", task_id="case-b", input_text="second"):
            record_tool_call(
                name="book", arguments={}, outcome="ok", status_code=201,
            )
        px.record_answer("same", "two")
    finally:
        px.close()

    got = spans.get_finished_spans()
    first = next(
        span for span in got
        if '"one"' in span.attributes.get("gen_ai.output.messages", "")
    )
    assert first.attributes["percepteye.task.id"] == "case-a"
    later = [span for span in got if span is not first]
    assert later
    assert all("percepteye.task.id" not in span.attributes for span in later)


def test_the_tool_span_carries_the_conversation_too(spans, tmp_path):
    """A consumer grouping an export into conversations must put a tool call in
    the same bucket as the turn that made it.

    The turn id can recover that link — a consumer can build turn -> conversation
    from the chat spans — but only when the SAME export also carries a chat span
    for that turn. An
    export sliced by time, or sampled by the collector, may carry the tool span
    and not its turn. Then the call has nothing to group by.

    This was claimed in a commit message before it was true: the edit failed
    and was not re-checked. The assertion is here so the claim cannot drift
    from the code again.
    """
    from agent_flywheel.outcomes import record_tool_call
    from agent_flywheel.production import attach

    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        with px.turn(turn_id="t1", input_text="hi", conversation_id="conv-7"):
            record_tool_call(name="book", arguments={}, outcome="ok",
                             status_code=201)
        px.record_answer("t1", "Booked.")
    finally:
        px.close()

    tool = [s for s in spans.get_finished_spans()
            if s.attributes.get("gen_ai.tool.name")]
    assert len(tool) == 1
    assert tool[0].attributes.get("percepteye.conversation.id") == "conv-7"
    # And under the convention's own name, which is what a consumer reads.
    assert tool[0].attributes.get("gen_ai.conversation.id") == "conv-7"
    # And the turn id is still there — this adds a key, it does not swap one.
    assert tool[0].attributes.get("percepteye.turn.id") == "t1"


def test_a_turn_with_no_conversation_sets_no_conversation_attribute(spans, tmp_path):
    """Absence stays absence. Stamping an empty string, or the turn id, would
    make an ungrouped call look like it belonged to a conversation."""
    from agent_flywheel.outcomes import record_tool_call
    from agent_flywheel.production import attach

    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        with px.turn(turn_id="t1", input_text="hi"):
            record_tool_call(name="book", arguments={}, outcome="ok",
                             status_code=201)
    finally:
        px.close()

    tool = [s for s in spans.get_finished_spans()
            if s.attributes.get("gen_ai.tool.name")]
    assert len(tool) == 1
    assert "percepteye.conversation.id" not in tool[0].attributes
    assert "gen_ai.conversation.id" not in tool[0].attributes
