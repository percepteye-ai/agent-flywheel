"""The offline projection and the upload path must describe the SAME turn.

``export`` renders a captured turn into a corpus record; ``Attachment._turn_wire``
renders the same directory into what ``POST /turns`` sends. **No shipped code
path runs both**, so nothing except this file stops them drifting -- and the
drift that matters is not shape, it is EVIDENCE. A turn whose text, call
sequence or tri-state reads differently depending on which renderer looked at it
teaches two different lessons from one observation.

They are ALLOWED to differ in shape, and do: ``step_index`` exists only on the
corpus side, ``turn_index`` and ``input_text``/``final_text`` naming only on the
wire side. What is asserted here is the intersection -- the facts both claim to
carry about the same directory.
"""
from __future__ import annotations

import inspect

import pytest

import percepteye_agent_flywheel.introspection as introspection_module
from percepteye_agent_flywheel import export
from percepteye_agent_flywheel.export import turn_to_record
from percepteye_agent_flywheel.introspection import (
    DiscoveredAgent,
    ModelConfig,
    ToolDefinition,
)
from percepteye_agent_flywheel.outcomes import record_tool_call
from percepteye_agent_flywheel.production import attach


@pytest.fixture
def px(tmp_path):
    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        yield px
    finally:
        px.close()


CALLS = (
    {"name": "lookup", "arguments": {"id": "1"}, "outcome": "ok",
     "status_code": 200},
    {"name": "charge", "arguments": {}, "outcome": "failed",
     "status_code": 500},
    {"name": "refund", "arguments": {}, "outcome": "unknown",
     "status_code": None},
)


def _captured(px, tid="t1"):
    with px.turn(turn_id=tid, input_text="refund order 4417"):
        for c in CALLS:
            record_tool_call(**c)
    px.record_answer(tid, "Refunded.")
    return tid


def test_both_renderers_report_the_same_turn_id_and_text(px, tmp_path):
    tid = _captured(px)
    rec = turn_to_record(tmp_path / tid, turn_id=tid)
    wire = px._turn_wire(tmp_path / tid)

    assert rec["task_id"] == wire["turn_id"] == tid
    said = {m["role"]: m["content"] for m in rec["messages"]}
    assert said["user"] == wire["input_text"] == "refund order 4417"
    assert said["assistant"] == wire["final_text"] == "Refunded."


def test_both_renderers_report_the_same_calls_in_the_same_order(px, tmp_path):
    tid = _captured(px)
    rec = turn_to_record(tmp_path / tid, turn_id=tid)
    wire = px._turn_wire(tmp_path / tid)

    assert [c["name"] for c in rec["tool_calls"]] == \
        [c["name"] for c in wire["tool_calls"]] == \
        [c["name"] for c in CALLS]


def test_the_tri_state_survives_identically_on_both_paths(px, tmp_path):
    """The one divergence that would corrupt training rather than confuse a
    reader: an `unknown` rounded up on one path and not the other."""
    tid = _captured(px)
    rec = turn_to_record(tmp_path / tid, turn_id=tid)
    wire = px._turn_wire(tmp_path / tid)

    assert [c["percepteye_outcome"] for c in rec["tool_calls"]] == \
        [c["outcome"] for c in wire["tool_calls"]] == \
        ["ok", "failed", "unknown"]
    # And the corpus side's second spelling of it agrees with the first: absent
    # for `unknown`, because absent means unobserved and False would assert a
    # failure nobody saw.
    assert [c.get("success") for c in rec["tool_calls"]] == [True, False, None]
    assert "success" not in rec["tool_calls"][2]


def test_both_renderers_preserve_the_same_exact_execution_identity(
    tmp_path, monkeypatch,
):
    class _Agent:
        def invoke(self, task: str) -> str:
            return task

    monkeypatch.setattr(
        introspection_module,
        "introspect",
        lambda _entrypoint: DiscoveredAgent(
            agent_name="customer-agent",
            framework="customer-framework",
            system_prompt="Use tools and answer the user.",
            tool_definitions=[ToolDefinition(name="lookup", input_schema={})],
            discovered_model_config=ModelConfig(
                provider="gateway", model_name="policy",
            ),
        ),
    )
    attachment = attach(
        agent_id="a",
        trajectory_root=str(tmp_path),
        register=False,
        adapters=[],
        execution_entrypoint=_Agent(),
        attested_execution_components={"customer.release": "a" * 64},
    )
    try:
        tid = _captured(attachment)
        record = turn_to_record(tmp_path / tid, turn_id=tid)
        wire = attachment._turn_wire(tmp_path / tid)
        assert record is not None and wire is not None
        assert record["agent_fingerprint"] == wire["agent_fingerprint"]
    finally:
        attachment.close()


def test_a_turn_that_recorded_nothing_is_nothing_to_both(px, tmp_path):
    """Both must agree on the EMPTY case too -- one of them yielding a record
    for a directory the other calls empty is how an agent that answered from
    context ends up in a corpus as a turn that said nothing."""
    with px.turn(turn_id="t_empty", conversation_id="c1"):
        pass
    assert turn_to_record(tmp_path / "t_empty", turn_id="t_empty") is None
    assert px._turn_wire(tmp_path / "t_empty") is None


def test_all_lists_every_public_name_this_module_offers():
    """``__all__`` IS the supported surface here, because nothing re-exports
    this module from the package root -- a public function missing from it is a
    function no caller is told exists. ``conversation_to_record`` was.
    """
    public = {
        name for name, obj in vars(export).items()
        if not name.startswith("_")
        and inspect.isfunction(obj)
        and obj.__module__ == export.__name__
    }
    assert public == set(export.__all__)
