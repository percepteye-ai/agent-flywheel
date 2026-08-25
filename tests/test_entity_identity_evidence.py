"""Identity evidence: the entity a write produced, without the response body.

A tool outcome says whether a call WORKED. It cannot say whether the change
SURVIVED -- a create that returns 201 and then silently fails to persist is
indistinguishable, from inside the process, from one that landed. Only reading
the resource back settles it, and a read-back needs to know which entity to
look for.

That is the gap these tests close. The trajectory carried the response body or
nothing, and a body is both too much (it is customer production data) and too
little (nothing downstream knew which of its fields named the entity). The
identity is resolved at capture and the body is free to be dropped.
"""
from __future__ import annotations

import json

import pytest

from percepteye_agent_flywheel.contract import ToolCallOutcome
from percepteye_agent_flywheel.errors import ContractViolation
from percepteye_agent_flywheel.evidence import (
    IDENTITY_FIELDS,
    WEAK_ID_FIELDS,
    extract_entity_ids,
)
from percepteye_agent_flywheel.outcomes import Trajectory, record_tool_call

# ── extraction ───────────────────────────────────────────────────────────────

def test_a_creates_identity_is_filed_under_the_field_it_came_from():
    """Not a flat set. Per-field filing is what lets a later read-back compare
    a candidate's `key` to the write's `key` and never to the write's `id` --
    collapsing them is how an entity whose literal id equals another's key
    string gets falsely confirmed."""
    assert extract_entity_ids({"id": "10001", "key": "PROJ-42"}) == {
        "id": "10001", "key": "PROJ-42",
    }


def test_a_json_string_body_is_parsed():
    assert extract_entity_ids('{"id": 7}') == {"id": "7"}


def test_an_enveloped_identity_is_found():
    assert extract_entity_ids({"data": {"uuid": "abc-123"}}) == {"uuid": "abc-123"}


def test_prose_yields_nothing_rather_than_a_guess():
    """An id parsed out of free text is a guess, and a guessed identity
    confirms an entity nobody verified."""
    assert extract_entity_ids("created ticket PROJ-42 successfully") == {}


def test_the_shallowest_occurrence_wins():
    """The entity's own id sits at the top of its response; a deeper `id`
    belongs to some nested unrelated object."""
    assert extract_entity_ids({"id": "a", "items": [{"id": "b"}]}) == {"id": "a"}


def test_an_ambiguous_field_is_dropped_not_picked_between():
    """Two equally-shallow answers that disagree. We do not know which entity
    is THE entity, and the honest result of ambiguity is no evidence."""
    assert extract_entity_ids({"xs": [{"id": "a"}, {"id": "b"}]}) == {}


@pytest.mark.parametrize("weak", WEAK_ID_FIELDS)
def test_low_entropy_fields_are_never_identity(weak):
    """A created resource routinely echoes a human `name` ("Story 1") or a
    small positional `number` (7) that a DIFFERENT entity shares by
    coincidence. Matching on those confirms the wrong resource -- a false
    positive dressed as a verified success, worse than no evidence at all."""
    assert extract_entity_ids({weak: "Story 1"}) == {}


def test_a_boolean_is_not_an_identity():
    """`bool` is an `int` in Python, so this needs an explicit guard."""
    assert extract_entity_ids({"id": True}) == {}


def test_an_empty_or_blank_value_is_not_an_identity():
    assert extract_entity_ids({"id": "", "key": "   "}) == {}


def test_extraction_never_raises():
    class Hostile:
        def __getitem__(self, k):  # pragma: no cover - defensive
            raise RuntimeError("boom")

    for body in (Hostile(), object(), None, 3, [[[[["deep"]]]]]):
        assert isinstance(extract_entity_ids(body), dict)


def test_a_deeply_buried_id_is_not_reached():
    """Depth is bounded. Past an envelope or two, a match is more likely some
    unrelated nested object's id than the entity's own."""
    assert extract_entity_ids({"a": {"b": {"c": {"d": {"id": "x"}}}}}) == {}


# ── the contract ─────────────────────────────────────────────────────────────

def test_identity_survives_an_unknown_outcome():
    """THE point of the whole feature.

    `_scrub_status` strips a status from the body of an `unknown` call so an
    unobserved outcome cannot be recovered as a landed one through a side
    channel. An identity is not that: it claims nothing about whether the call
    worked, it names an entity so that something ELSE can go and observe one.
    Scrubbing it would suppress the only evidence capable of turning an honest
    `unknown` into an honest `ok`.
    """
    rec = ToolCallOutcome(
        name="POST /rest/api/2/issue", arguments={}, outcome="unknown",
        status_code=None, output={"id": "10001", "status_code": 201},
        entity_ids={"id": "10001"},
    )
    wire = rec.to_wire()
    assert wire["entity_ids"] == {"id": "10001"}
    assert "status_code" not in wire["output"]      # the 201 is gone
    assert wire["outcome"] == "unknown"             # and still unknown


@pytest.mark.parametrize("weak", WEAK_ID_FIELDS)
def test_the_contract_refuses_a_weak_field_by_name(weak):
    """Refused rather than discouraged: a caller who declares
    `entity_ids={"name": ...}` is handing the read-back a way to confirm the
    wrong entity."""
    with pytest.raises(ContractViolation):
        ToolCallOutcome(name="x", arguments={}, outcome="ok", status_code=None,
                        entity_ids={weak: "Story 1"})


@pytest.mark.parametrize("field", IDENTITY_FIELDS)
def test_every_strong_field_is_accepted(field):
    rec = ToolCallOutcome(name="x", arguments={}, outcome="ok",
                          status_code=None, entity_ids={field: "v"})
    assert rec.to_wire()["entity_ids"] == {field: "v"}


def test_an_empty_identity_is_absent_from_the_wire():
    """Absence means "no identity evidence", which downstream is an abstention
    and never a failure. An empty dict on the wire would be noise."""
    rec = ToolCallOutcome(name="x", arguments={}, outcome="ok", status_code=None)
    assert "entity_ids" not in rec.to_wire()


def test_a_non_string_value_is_refused():
    with pytest.raises(ContractViolation):
        ToolCallOutcome(name="x", arguments={}, outcome="ok", status_code=None,
                        entity_ids={"id": 10001})


# ── capture end to end ───────────────────────────────────────────────────────

def _rollout(monkeypatch, tmp_path):
    monkeypatch.setenv("PERCEPTEYE_ROLLOUT_ID", "rl_x")
    monkeypatch.setenv("PERCEPTEYE_TRAJECTORY_DIR", str(tmp_path))


def test_capture_records_identity_without_being_told(monkeypatch, tmp_path):
    _rollout(monkeypatch, tmp_path)
    record_tool_call("POST /rest/api/2/issue", {"p": "ENG"},
                     outcome="ok", status_code=201,
                     output={"id": "10001", "key": "ENG-7"})
    [call] = Trajectory(tmp_path).read()
    assert call.entity_ids == {"id": "10001", "key": "ENG-7"}


def test_a_declared_identity_beats_extraction(monkeypatch, tmp_path):
    """Extraction is best-effort acceleration over an explicit answer, exactly
    as the adapters are over `record_tool_call` itself."""
    _rollout(monkeypatch, tmp_path)
    record_tool_call("write", {}, outcome="ok", status_code=200,
                     output={"id": "guessed"}, entity_ids={"id": "declared"})
    [call] = Trajectory(tmp_path).read()
    assert call.entity_ids == {"id": "declared"}


def test_identity_round_trips_through_the_file(monkeypatch, tmp_path):
    _rollout(monkeypatch, tmp_path)
    record_tool_call("t", {}, outcome="unknown", status_code=None,
                     output={"key": "ENG-7"})
    raw = json.loads((tmp_path / "tool_calls.jsonl").read_text().strip())
    assert raw["entity_ids"] == {"key": "ENG-7"}
    assert Trajectory(tmp_path).read()[0].entity_ids == {"key": "ENG-7"}


def test_a_call_with_no_body_records_no_identity(monkeypatch, tmp_path):
    _rollout(monkeypatch, tmp_path)
    record_tool_call("read", {}, outcome="ok", status_code=200)
    assert Trajectory(tmp_path).read()[0].entity_ids == {}


def test_the_schema_and_the_extractor_agree_on_the_field_set():
    """One definition, never a twin that drifts. The published schema is what
    a non-Python writer builds against."""
    import pathlib

    schema = json.loads(
        (pathlib.Path(__file__).resolve().parent.parent
         / "schema" / "flywheel-1.json").read_text()
    )
    allowed = schema["$defs"]["ToolCallRecord"]["properties"]["entity_ids"][
        "propertyNames"]["enum"]
    assert tuple(allowed) == IDENTITY_FIELDS


# ── the paths identity could be silently lost on ─────────────────────────────

def test_identity_survives_the_child_to_parent_boundary():
    """The default isolation mode runs the agent in a child process, and an
    agent that reports its own tool calls resolved the identity THERE. The
    parent rebuilding the record from the wire is exactly where that evidence
    goes quietly missing -- it did, until this test."""
    from percepteye_agent_flywheel.runner import _output_from_wire

    out = _output_from_wire({
        "final_text": "done",
        "tool_calls": [{
            "name": "POST /rest/api/2/issue", "arguments": {},
            "outcome": "ok", "status_code": 201,
            "entity_ids": {"id": "10001", "key": "ENG-7"},
        }],
    })
    assert out.tool_calls[0].entity_ids == {"id": "10001", "key": "ENG-7"}


def test_a_wire_record_without_identity_is_still_readable():
    """Every producer predates this field. Absent must mean empty, not a
    KeyError that costs the whole rollout."""
    from percepteye_agent_flywheel.runner import _output_from_wire

    out = _output_from_wire({
        "final_text": "done",
        "tool_calls": [{"name": "t", "arguments": {}, "outcome": "ok",
                        "status_code": 200}],
    })
    assert out.tool_calls[0].entity_ids == {}


def test_a_cli_agents_event_stream_yields_identity():
    """The other capture path: a non-Python agent emitting JSON events. It has
    the structured result in hand, so it resolves identity the same way."""
    from percepteye_agent_flywheel.event_stream import parse_tool_calls

    calls = parse_tool_calls("\n".join([
        json.dumps({"type": "tool_execution_start", "toolCallId": "c1",
                    "toolName": "POST /issue", "args": {"p": "ENG"}}),
        json.dumps({"type": "tool_execution_end", "toolCallId": "c1",
                    "toolName": "POST /issue", "isError": False,
                    "result": {"id": "10001", "key": "ENG-7"}}),
    ]))
    assert calls is not None
    assert calls[0].entity_ids == {"id": "10001", "key": "ENG-7"}


# ── every framework path, not just one ───────────────────────────────────────
#
# Identity is resolved inside `record_tool_call`, which every adapter already
# routes through, so the framework paths needed no adapter change. That is a
# claim worth testing rather than reasoning about: "it should already work
# because of where I put it" is exactly how the child->parent boundary lost
# identity while looking correct.


def test_langchain_capture_yields_identity_with_no_instrumentation(tmp_path):
    """Zero Percepteye code in the tool, and the identity still lands.

    Note the outcome: a tool that RETURNS a value states no verdict, so it is
    honestly `unknown` -- and it carries the identity anyway. That pairing is
    the whole point of the feature.
    """
    pytest.importorskip("langchain_core")
    from langchain_core.tools import tool

    from percepteye_agent_flywheel.binding import bind_selected
    from percepteye_agent_flywheel.outcomes import _RolloutScope

    @tool
    def create_issue(project: str) -> dict:
        """Create an issue; returns what the API returned."""
        return {"id": "10001", "key": "ENG-7", "self": "https://jira/x/10001"}

    report, unbinds = bind_selected(["langchain"], llm_url="https://gw/x", api_key="k")
    assert "langchain" in report.bound
    try:
        with _RolloutScope("rl_lc", str(tmp_path)):
            create_issue.invoke({"project": "ENG"})
    finally:
        for u in unbinds:
            u()

    [call] = Trajectory(tmp_path).read() or []
    assert call.outcome == "unknown"
    assert call.entity_ids == {"id": "10001", "key": "ENG-7"}


def test_adk_capture_yields_identity_with_no_instrumentation(tmp_path):
    """The same claim for the second framework, through the real install path:
    constructing a PluginManager is what every ADK Runner does internally."""
    pytest.importorskip("google.adk")
    import asyncio

    from google.adk.plugins.plugin_manager import PluginManager

    from percepteye_agent_flywheel.binding import bind_selected
    from percepteye_agent_flywheel.outcomes import _RolloutScope

    class _Tool:
        def __init__(self, name):
            self.name = name

    class _Ctx:
        function_call_id, agent_name = "fc_1", "coder"

    report, unbinds = bind_selected(["adk"], llm_url="https://gw/x", api_key="k")
    assert "adk" in report.bound
    try:
        with _RolloutScope("rl_adk", str(tmp_path)):
            pm = PluginManager()
            create, ctx = _Tool("create_issue"), _Ctx()

            async def drive():
                await pm.run_before_tool_callback(
                    tool=create, tool_args={"p": "OPS"}, tool_context=ctx)
                await pm.run_after_tool_callback(
                    tool=create, tool_args={"p": "OPS"}, tool_context=ctx,
                    result={"id": "20002", "key": "OPS-3"})

            asyncio.run(drive())
    finally:
        for u in unbinds:
            u()

    [call] = Trajectory(tmp_path).read() or []
    assert call.entity_ids == {"id": "20002", "key": "OPS-3"}
    assert call.agent_name == "coder"


def test_a_cli_agent_writing_the_trajectory_itself_carries_identity(tmp_path):
    """The path a non-Python agent uses when it writes the file directly --
    dsh, or anything else that appends to the trajectory. The format is the
    interface, so identity is available to a writer in any language."""
    (tmp_path / "tool_calls.jsonl").write_text(
        json.dumps({
            "name": "POST /rest/api/2/issue", "arguments": {"p": "ENG"},
            "outcome": "unknown", "status_code": None,
            "entity_ids": {"key": "ENG-7"},
        }) + "\n",
        encoding="utf-8",
    )
    [call] = Trajectory(tmp_path).read()
    assert call.entity_ids == {"key": "ENG-7"}
