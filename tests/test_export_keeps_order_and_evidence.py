"""The offline export orders turns by when they started, keeps tool-only
conversations, carries every field the upload path carries, and replaces its
output whole."""
from __future__ import annotations

import json
import os

import pytest

from agent_flywheel import export
from agent_flywheel.export import iter_records, write_records
from agent_flywheel.outcomes import CONVERSATION_ID_FILENAME, record_tool_call
from agent_flywheel.production import attach


@pytest.fixture
def px(tmp_path):
    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        yield px
    finally:
        px.close()


def test_a_late_answer_does_not_reorder_its_conversation(px, tmp_path):
    for tid in ("first", "second"):
        with px.turn(turn_id=tid, conversation_id="c1", input_text=tid):
            pass
    px.record_answer("second", "two")
    px.record_answer("first", "one")         # answered LAST
    # The directory of "first" was written last; its turn still began first.
    for n, tid in enumerate(("first", "second")):
        os.utime(tmp_path / tid / CONVERSATION_ID_FILENAME, (100 + n, 100 + n))
    os.utime(tmp_path / "first", (300, 300))
    os.utime(tmp_path / "second", (200, 200))
    (record,) = iter_records(tmp_path)
    assert [m["content"] for m in record["messages"]] == [
        "first", "one", "second", "two"]


def test_a_conversation_of_tool_calls_alone_is_kept(px, tmp_path):
    for tid in ("t1", "t2"):
        with px.turn(turn_id=tid, conversation_id="c1"):
            record_tool_call(name="lookup", arguments={}, outcome="ok",
                             status_code=200)
    records = list(iter_records(tmp_path))
    assert [[c["name"] for c in r["tool_calls"]] for r in records] == [
        ["lookup"], ["lookup"]]


def test_every_field_the_upload_path_carries_survives(px, tmp_path):
    with px.turn(turn_id="t1", input_text="make it", task_id="ticket-9"):
        record_tool_call(name="create", arguments={}, outcome="failed",
                         status_code=500, error="boom",
                         error_class="TimeoutError", agent_name="planner",
                         entity_ids={"id": "f81d4fae7dec11d0a76500a0c91e6bf6"})
    px.record_answer("t1", "done")
    (record,) = iter_records(tmp_path)
    (call,) = record["tool_calls"]
    wire = px._turn_wire(tmp_path / "t1")["tool_calls"][0]
    for key in ("error_class", "agent_name", "entity_ids"):
        assert call[key] == wire[key]
    assert record["percepteye_task_id"] == "ticket-9"
    assert record["task_id"] == "t1", "the record's identity is unchanged"


def test_an_export_that_fails_leaves_the_previous_one_whole(
        px, tmp_path, monkeypatch):
    out = tmp_path / "out" / "turns.jsonl"
    out.parent.mkdir()
    out.write_text('{"previous": true}\n', encoding="utf-8")

    def broken(root):
        yield {"messages": []}
        raise OSError("disk went away")

    monkeypatch.setattr(export, "iter_records", broken)
    with pytest.raises(OSError):
        write_records(tmp_path, out)
    assert out.read_text(encoding="utf-8") == '{"previous": true}\n'
    assert [p.name for p in out.parent.iterdir()] == ["turns.jsonl"]
    assert json.loads(out.read_text())["previous"] is True
