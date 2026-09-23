"""The spans this SDK emits ARE OpenTelemetry GenAI spans -- judged against the
published standard, never against our own reader.

Three checks, and each answers a different question:

1. THE GOLDEN FIXTURE. ``tests/fixtures/genai-golden.json`` holds a scripted
   scenario and the exact spans it must produce. The Node SDK carries the same
   file, byte for byte, and renders the same scenario in its own suite -- so a
   span name, attribute or content byte that differs between the two packages
   is a red test in one of them, not a drift someone notices months later.
2. THE SCHEMA GATE. Every content value this SDK emitted is validated against
   the convention's own JSON schemas (vendored, pinned; see
   ``tests/genai_schemas/README.md``). Those schemas end every part list in a
   ``GenericPart`` that requires only ``type``, so validating against the
   top-level schema alone accepts a tool call with no name. The gate
   therefore sends each KNOWN part type to its specific ``$defs`` entry.
3. THE GATE CAN FAIL. A gate that accepts everything proves nothing, so the
   mutations it exists to catch are applied to real output and must go red.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("opentelemetry.sdk")
jsonschema = pytest.importorskip("jsonschema")

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "genai-golden.json"
SCHEMAS = HERE / "genai_schemas"

#: Known part types, each to the ``$defs`` entry that actually constrains it.
_PART_DEFS = {
    "text": "TextPart",
    "tool_call": "ToolCallRequestPart",
    "tool_call_response": "ToolCallResponsePart",
    "server_tool_call": "ServerToolCallPart",
    "server_tool_call_response": "ServerToolCallResponsePart",
    "blob": "BlobPart",
    "file": "FilePart",
    "uri": "UriPart",
    "reasoning": "ReasoningPart",
    "compaction": "CompactionPart",
}

#: Content attribute -> the schema file it MUST follow.
_CONTENT = {
    "gen_ai.input.messages": "gen-ai-input-messages.json",
    "gen_ai.output.messages": "gen-ai-output-messages.json",
    "gen_ai.system_instructions": "gen-ai-system-instructions.json",
    "gen_ai.tool.call.arguments": "gen-ai-tool-call-arguments.json",
    "gen_ai.tool.call.result": "gen-ai-tool-call-result.json",
}


def _golden() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _schema(name: str) -> dict[str, Any]:
    return json.loads((SCHEMAS / name).read_text(encoding="utf-8"))


def _check(instance: Any, schema: dict[str, Any], ref: str | None = None) -> None:
    """Validate under Draft 2020-12, pinned: the schemas declare no draft."""
    if ref is not None:
        schema = {"$ref": f"#/$defs/{ref}", "$defs": schema["$defs"]}
    jsonschema.Draft202012Validator(schema).validate(instance)


def _check_parts(parts: list[Any], schema: dict[str, Any]) -> None:
    for part in parts:
        kind = part.get("type") if isinstance(part, dict) else None
        if kind not in _PART_DEFS:
            raise jsonschema.ValidationError(
                f"part type {kind!r} is not one this SDK emits")
        _check(part, schema, _PART_DEFS[kind])


def validate_content(key: str, text: str) -> None:
    """THE GATE for one content attribute, as emitted: JSON text first."""
    value = json.loads(text)
    schema = _schema(_CONTENT[key])
    _check(value, schema)
    if key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        roles = schema["$defs"]["Role"]["enum"]
        for message in value:
            if message["role"] not in roles:
                raise jsonschema.ValidationError(
                    f"role {message['role']!r} is not a convention role")
            _check_parts(message["parts"], schema)
            if "finish_reason" in message:
                _check(message["finish_reason"], schema, "FinishReason")
    elif key == "gen_ai.system_instructions":
        _check_parts(value, schema)


def _render(scenario: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run the scenario through THIS SDK's tracing and normalise what it
    emitted into the fixture's shape."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import agent_flywheel.tracing as mod

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    saved = (mod._TRACER, mod._RESOLVED)
    mod._TRACER, mod._RESOLVED = provider.get_tracer("golden"), True
    try:
        for step in copy.deepcopy(scenario):
            call = step.pop("call")
            if call == "tool_call":
                mod.record_tool_call_span(
                    step.pop("name"), step.pop("arguments"), **step)
            else:
                assert call == "turn", call
                mod.record_turn_span(**step)
    finally:
        mod._TRACER, mod._RESOLVED = saved

    def ms(ns: int) -> int | float:
        return ns // 1_000_000 if ns % 1_000_000 == 0 else ns / 1_000_000

    out = []
    for span in exporter.get_finished_spans():
        status: dict[str, Any] = {"code": span.status.status_code.name}
        if span.status.description:
            status["description"] = span.status.description
        out.append({
            "name": span.name,
            "kind": span.kind.name,
            "start_time_ms": ms(span.start_time),
            "end_time_ms": ms(span.end_time),
            "status": status,
            "attributes": {
                k: list(v) if isinstance(v, tuple) else v
                for k, v in span.attributes.items()
            },
        })
    return out


# ── 1. the golden fixture ──────────────────────────────────────────────────


def test_the_sdk_renders_the_golden_scenario_exactly():
    golden = _golden()
    got = _render(golden["scenario"])
    assert [s["name"] for s in got] == [s["name"] for s in golden["spans"]]
    for mine, want in zip(got, golden["spans"], strict=True):
        assert mine == want, mine["name"]


def test_content_attributes_match_the_fixture_byte_for_byte():
    """Equality of parsed JSON would pass ``{"a":1}`` against ``{"a": 1}``;
    the promise is identical BYTES across the two SDKs, so compare text."""
    golden = _golden()
    for mine, want in zip(_render(golden["scenario"]), golden["spans"], strict=True):
        for key in _CONTENT:
            assert mine["attributes"].get(key) == want["attributes"].get(key)
            if key in want["attributes"]:
                assert isinstance(mine["attributes"][key], str)


def test_the_fixture_is_the_node_sdks_fixture_byte_for_byte():
    """The fixture is only shared if the two copies are the same bytes."""
    from .node_sdk import node_sdk

    sdk, _named = node_sdk()
    if sdk is None:
        pytest.skip("the Node package is not checked out here")
    theirs = sdk / "test" / "fixtures" / "genai-golden.json"
    assert theirs.read_bytes() == FIXTURE.read_bytes(), (
        "the golden fixture differs between the Python and Node SDKs; copy "
        "the changed one over the other in the same change")


# ── 2. the schema gate ─────────────────────────────────────────────────────


def test_every_emitted_content_value_validates_against_the_official_schemas():
    spans = _render(_golden()["scenario"])
    checked = 0
    for span in spans:
        for key, value in span["attributes"].items():
            if key in _CONTENT:
                validate_content(key, value)
                checked += 1
    # Every content attribute the scenario produces, not a vacuous zero.
    assert checked == 10


def test_the_scenario_exercises_every_rule_it_claims_to():
    """The fixture is evidence only for what it contains."""
    spans = {(s["name"], s["attributes"].get("percepteye.turn.id")): s
             for s in _render(_golden()["scenario"])}
    ok = spans[("execute_tool get_weather", "turn-7")]["attributes"]
    failed = spans[("execute_tool book_table", "turn-7")]
    turn = spans[("invoke_agent concierge", "turn-7")]["attributes"]
    # Arguments and results are OBJECTS, a JSON string argument included.
    assert json.loads(ok["gen_ai.tool.call.arguments"]) == {
        "location": "Paris", "unit": "celsius"}
    assert json.loads(failed["attributes"]["gen_ai.tool.call.arguments"])[
        "party"] == 2
    # JSON null is data and survives.
    assert json.loads(ok["gen_ai.tool.call.result"])["next_cursor"] is None
    # A failure carries error.type and ERROR, and no result.
    assert failed["status"]["code"] == "ERROR"
    assert failed["attributes"]["error.type"] == "Conflict"
    assert "gen_ai.tool.call.result" not in failed["attributes"]
    # The turn carries system instructions, a tool call and its response.
    parts = [p for m in json.loads(turn["gen_ai.input.messages"])
             for p in m["parts"]]
    assert {p["type"] for p in parts} >= {"text", "tool_call", "tool_call_response"}
    assert all(isinstance(p["arguments"], dict)
               for p in parts if p["type"] == "tool_call")
    assert turn["gen_ai.response.finish_reasons"] == ["stop"]
    assert turn["gen_ai.conversation.id"] == turn["percepteye.conversation.id"]


# ── 3. the gate can fail ───────────────────────────────────────────────────


def _turn_input() -> str:
    for span in _render(_golden()["scenario"]):
        if "gen_ai.input.messages" in span["attributes"] and \
                "gen_ai.system_instructions" in span["attributes"]:
            return span["attributes"]["gen_ai.input.messages"]
    raise AssertionError("the scenario has no full turn")


@pytest.mark.parametrize("mutate", [
    # The mutation the top-level schema alone lets through.
    pytest.param(lambda ms: ms[1]["parts"][0].pop("name"), id="tool-call-no-name"),
    pytest.param(lambda ms: ms[2]["parts"][0].pop("response"),
                 id="tool-response-no-response"),
    pytest.param(lambda ms: ms[0]["parts"][0].__setitem__("content", 5),
                 id="text-content-not-a-string"),
    pytest.param(lambda ms: ms[0].__setitem__("role", "robot"), id="unknown-role"),
    pytest.param(lambda ms: ms[0].pop("parts"), id="message-without-parts"),
])
def test_the_gate_refuses_what_the_schema_forbids(mutate):
    messages = json.loads(_turn_input())
    validate_content("gen_ai.input.messages", json.dumps(messages))  # sane
    mutate(messages)
    with pytest.raises(jsonschema.ValidationError):
        validate_content("gen_ai.input.messages", json.dumps(messages))


def test_the_gate_refuses_a_non_object_tool_argument():
    with pytest.raises(jsonschema.ValidationError):
        validate_content("gen_ai.tool.call.arguments", '"{\\"q\\": 1}"')


def test_the_gate_refuses_an_unknown_finish_reason_type():
    with pytest.raises(jsonschema.ValidationError):
        validate_content(
            "gen_ai.output.messages",
            '[{"role":"assistant","parts":[],"finish_reason":7}]')


# ── the rules the renderer applies, stated once each ───────────────────────


def test_no_attribute_is_cut_whatever_its_size():
    """The owner's rule: no size limitations. The old 2,000-character cut
    ended a JSON argument in ``...[truncated]``, which is not JSON."""
    big = {"blob": "z" * 50_000, "emoji": "\U0001F4E6" * 10_000}
    step = {"call": "tool_call", "name": "upload", "arguments": big,
            "outcome": "ok", "result": big, "end_time_ms": 1}
    turn = {"call": "turn", "input_text": "q" * 100_000,
            "output_text": "\U0001F4E6" * 30_000, "end_time_ms": 1}
    tool_span, turn_span = _render([step, turn])
    assert json.loads(tool_span["attributes"]["gen_ai.tool.call.arguments"]) == big
    assert json.loads(tool_span["attributes"]["gen_ai.tool.call.result"]) == big
    out = json.loads(turn_span["attributes"]["gen_ai.output.messages"])
    assert out[0]["parts"][0]["content"] == "\U0001F4E6" * 30_000
    (failed,) = _render([{"call": "tool_call", "name": "f", "arguments": {},
                          "outcome": "failed", "error": "e" * 5_000,
                          "end_time_ms": 1}])
    assert failed["status"]["description"] == "e" * 5_000


def test_a_string_argument_holding_an_inexact_integer_stays_a_string():
    """Node cannot parse 2**53 + 1 exactly, so neither side deserializes it:
    the two SDKs must emit the same thing, and a changed number is worse than
    a string."""
    (span,) = _render([{
        "call": "turn", "end_time_ms": 1,
        "input_messages": [{"role": "assistant", "parts": [{
            "type": "tool_call", "name": "f",
            "arguments": '{"id": 9007199254740993}'}]}],
    }])
    part = json.loads(span["attributes"]["gen_ai.input.messages"])[0]["parts"][0]
    assert part["arguments"] == '{"id": 9007199254740993}'


def test_non_finite_numbers_serialize_as_valid_json():
    (span,) = _render([{
        "call": "tool_call", "name": "f", "arguments": {"x": float("nan")},
        "outcome": "ok", "end_time_ms": 1,
    }])
    assert span["attributes"]["gen_ai.tool.call.arguments"] == '{"x":null}'
