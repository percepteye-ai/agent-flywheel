"""The model-boundary renderer: one production turn, one real OTel GenAI trace.

WHAT IS PINNED, AND AGAINST WHAT.

* ``tests/fixtures/genai_capture_golden.json`` is the contract both SDKs render
  to. Its ``mc_golden_*`` vectors carry Mission Control's own golden turns
  VERBATIM as the expected traces (that fixture is what Mission Control's
  intake, its capture export and its semantic-convention gate are tested on),
  so they are an oracle this renderer did not produce. The Node SDK holds a
  byte-identical copy and renders the same vectors in its own suite;
  ``scripts/cross_language_abi_check.sh`` runs both renderers and compares
  what they wrote.
* The rest are the properties the intake enforces and a golden would only
  show by example: no trace without an observed model call, no truncation, no
  marker, the turn's own conversation id on every span.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from agent_flywheel.genai_otlp import (
    SCOPE_NAME,
    content_json,
    new_span_id,
    new_trace_id,
    observation_from_lines,
    render_turn_trace,
)

GOLDEN_PATH = Path(__file__).resolve().parent / "fixtures" / "genai_capture_golden.json"
GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def _canonical(value) -> bytes:
    """The bytes Mission Control stores for a turn's trace, and so the input
    to its digest: sorted keys, two-space indent."""
    return json.dumps(value, sort_keys=True, default=str, indent=2).encode()


def _spans(trace):
    return [s for rs in trace["resourceSpans"] for ss in rs["scopeSpans"]
            for s in ss["spans"]]


def _attrs(span):
    return {a["key"]: a["value"] for a in span["attributes"]}


@pytest.mark.parametrize(
    "vector", GOLDEN["vectors"], ids=[v["name"] for v in GOLDEN["vectors"]])
def test_every_golden_observation_renders_to_its_trace(vector):
    got = render_turn_trace(
        copy.deepcopy(vector["observation"]),
        agent_name=vector["agent_name"],
        conversation_id=vector["conversation_id"],
    )
    assert got == vector["genai_trace"]
    if got is not None:
        assert _canonical(got) == _canonical(vector["genai_trace"])


def test_the_golden_carries_mission_controls_turns_and_a_refusal():
    """A golden that had drifted to one shape would still be green."""
    names = [v["name"] for v in GOLDEN["vectors"]]
    assert [n for n in names if n.startswith("mc_golden_")] == [
        "mc_golden_turn_0", "mc_golden_turn_1", "mc_golden_turn_2"]
    assert any(v["genai_trace"] is None for v in GOLDEN["vectors"])
    assert GOLDEN["observation_lines"], "the on-disk reader has no vector"


def test_this_runtime_renders_its_own_side_of_the_one_asymmetry():
    """JavaScript enumerates integer-like keys first and spells ``1.0`` as
    ``1``. Python renders what ITS runtime sent the model; the vector pins
    both sides, and that the two really differ, so a change on either side
    is a failure here rather than a surprise in a customer's corpus."""
    a = GOLDEN["runtime_asymmetry"]
    got = render_turn_trace(copy.deepcopy(a["observation"]),
                            agent_name=a["agent_name"],
                            conversation_id=a["conversation_id"])
    assert got == a["genai_trace_python"]
    assert got != a["genai_trace_node"]


@pytest.mark.parametrize(
    "vector", GOLDEN["observation_lines"],
    ids=[v["name"] for v in GOLDEN["observation_lines"]])
def test_the_on_disk_record_reads_back_as_its_observation(vector):
    assert observation_from_lines(vector["lines"]) == vector["observation"]


def _observation(**chat):
    op = {
        "operation": "chat", "span_id": new_span_id(),
        "start_time_unix_nano": "1000", "end_time_unix_nano": "2000",
        "provider_name": "openai", "request_model": "gpt-4o-mini",
        "input_messages": [{"role": "user", "parts": [
            {"type": "text", "content": "hi"}]}],
        "output_messages": [{"role": "assistant", "parts": [
            {"type": "text", "content": "hello"}], "finish_reason": "stop"}],
        **chat,
    }
    return {"trace_id": new_trace_id(), "span_id": new_span_id(),
            "start_time_unix_nano": "900", "end_time_unix_nano": "2100",
            "operations": [op]}


def test_a_model_call_with_no_provider_or_requested_model_is_skipped():
    """The conventions REQUIRE both on an inference span. Without either the
    call cannot be named, and a nameless call is not guessed at: it is
    skipped, and a turn with nothing else has no trace."""
    for missing in ("provider_name", "request_model"):
        obs = _observation()
        del obs["operations"][0][missing]
        assert render_turn_trace(obs, agent_name="a") is None, missing


def test_a_turn_with_no_observed_model_call_has_no_trace():
    """Never a trace rebuilt from the turn's text fields: with no chat span,
    nothing says what the model was sent."""
    obs = _observation()
    obs["operations"] = []
    assert render_turn_trace(obs, agent_name="a") is None
    assert render_turn_trace(None, agent_name="a") is None
    assert render_turn_trace(_observation(), agent_name="") is None


def test_a_megabyte_message_is_never_truncated():
    big = "x" * (1 << 20) + "é\U0001f4e6"
    trace = render_turn_trace(_observation(input_messages=[
        {"role": "user", "parts": [{"type": "text", "content": big}]}]),
        agent_name="a")
    chat = _spans(trace)[1]
    sent = json.loads(_attrs(chat)["gen_ai.input.messages"]["stringValue"])
    assert sent[0]["parts"][0]["content"] == big


def test_it_never_marks_a_decision_or_claims_to_be_training_traffic():
    """``percepteye.rollout.id`` makes the trace read as OUR rollout (dropped
    downstream); the decision marker is the synthetic dataset's label. A
    production trace carries neither -- nor any other ``percepteye.*`` key."""
    trace = render_turn_trace(_observation(), agent_name="a",
                              conversation_id="c")
    keys = {k for s in _spans(trace) for k in _attrs(s)}
    assert not {k for k in keys if k.startswith("percepteye.")}
    assert trace["resourceSpans"][0]["scopeSpans"][0]["scope"] == {
        "name": SCOPE_NAME}


def test_every_span_carries_exactly_the_turns_conversation_and_none_without_one():
    trace = render_turn_trace(GOLDEN["vectors"][0]["observation"],
                              agent_name="a", conversation_id="conv-7")
    assert {_attrs(s)["gen_ai.conversation.id"]["stringValue"]
            for s in _spans(trace)} == {"conv-7"}
    bare = render_turn_trace(GOLDEN["vectors"][0]["observation"],
                             agent_name="a", conversation_id=None)
    assert not any("gen_ai.conversation.id" in _attrs(s) for s in _spans(bare))


def test_the_root_names_the_first_model_calls_provider_and_owns_every_span():
    trace = render_turn_trace(GOLDEN["vectors"][0]["observation"],
                              agent_name="acme", conversation_id=None)
    root, *children = _spans(trace)
    assert "parentSpanId" not in root
    assert _attrs(root)["gen_ai.operation.name"] == {"stringValue": "invoke_agent"}
    assert _attrs(root)["gen_ai.provider.name"] == {"stringValue": "openai"}
    assert {c["parentSpanId"] for c in children} == {root["spanId"]}
    assert {c["traceId"] for c in children} == {root["traceId"]}


def test_an_error_is_a_status_and_a_type_never_a_result():
    obs = _observation()
    obs["operations"].append({
        "operation": "execute_tool", "span_id": new_span_id(),
        "start_time_unix_nano": "2001", "end_time_unix_nano": "2002",
        "tool_name": "t", "arguments": {}, "result": "should not appear",
        "error_type": "ValueError"})
    tool = _spans(render_turn_trace(obs, agent_name="a"))[-1]
    assert tool["status"] == {"code": 2}
    assert _attrs(tool)["error.type"] == {"stringValue": "ValueError"}
    assert "gen_ai.tool.call.result" not in _attrs(tool)


def test_content_is_compact_json_in_the_values_own_order():
    """Not sorted: the order is what the runtime serialized to the model, and
    a chat template renders it verbatim."""
    assert content_json({"b": 1, "a": [None, True, "é"]}) == (
        '{"b":1,"a":[null,true,"é"]}')


def test_fresh_ids_are_the_otlp_encodings():
    for _ in range(64):
        t, s = new_trace_id(), new_span_id()
        assert len(t) == 32 and len(s) == 16
        assert int(t, 16) and int(s, 16)
        assert t == t.lower() and s == s.lower()
