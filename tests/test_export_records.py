"""A captured production turn renders into the shape the control plane ingests.

This is the seam that was missing. Everything downstream already existed -- a
post-hoc pipeline that ingests recorded trajectories, compiles a reward suite,
scores them with no environment to re-execute against, and emits training rows.
What it could not receive was a turn, because nothing rendered one into a record
it accepts.

Nothing new is invented: the ingest side already accepts a "unified messages
log" -- one record with an OpenAI-style `messages` list and an optional
`tool_calls` list -- and a production turn IS that shape.
"""
from __future__ import annotations

import json

import pytest

from percepteye_agent_flywheel.export import (
    iter_records,
    turn_to_record,
    write_records,
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


def _turn(px, tid, *, ask=None, answer=None, calls=()):
    with px.turn(turn_id=tid, input_text=ask):
        for c in calls:
            record_tool_call(**c)
    if answer:
        px.record_answer(tid, answer)


OK = {"name": "lookup", "arguments": {"id": "1"}, "outcome": "ok",
      "status_code": 200}
FAILED = {"name": "charge", "arguments": {}, "outcome": "failed",
          "status_code": 500}
UNKNOWN = {"name": "refund", "arguments": {}, "outcome": "unknown",
           "status_code": None}


# ── the shape ──────────────────────────────────────────────────────────────


def test_a_turn_becomes_an_openai_style_messages_record(px, tmp_path):
    _turn(px, "t1", ask="refund order 4417", answer="Refunded.")
    rec = turn_to_record(tmp_path / "t1", turn_id="t1")
    assert rec["messages"] == [
        {"role": "user", "content": "refund order 4417"},
        {"role": "assistant", "content": "Refunded."},
    ]
    assert rec["task_id"] == "t1" and rec["rollout_id"] == "t1"


def test_tool_calls_ride_along(px, tmp_path):
    _turn(px, "t2", ask="q", answer="a", calls=[OK])
    rec = turn_to_record(tmp_path / "t2", turn_id="t2")
    assert [c["name"] for c in rec["tool_calls"]] == ["lookup"]
    assert rec["tool_calls"][0]["status_code"] == 200


# ── THE TRI-STATE, which is the whole point of this SDK ────────────────────


def test_ok_and_failed_map_onto_the_key_ingest_reads(px, tmp_path):
    _turn(px, "t3", ask="q", calls=[OK, FAILED])
    by_name = {c["name"]: c for c in turn_to_record(tmp_path / "t3")["tool_calls"]}
    assert by_name["lookup"]["success"] is True
    assert by_name["charge"]["success"] is False


def test_unknown_is_OMITTED_rather_than_sent_as_a_failure(px, tmp_path):
    """Absent means unobserved, which is the truth. `False` would assert a
    failure nobody saw -- and the ingest side maps a missing key through
    `_maybe_bool` to None precisely so a silent source is not laundered."""
    _turn(px, "t4", ask="q", calls=[UNKNOWN])
    call = turn_to_record(tmp_path / "t4")["tool_calls"][0]
    assert "success" not in call
    assert call["percepteye_outcome"] == "unknown"


def test_the_verbatim_outcome_travels_too(px, tmp_path):
    """So nothing downstream has to invert the boolean mapping to recover
    which of the three states it was."""
    _turn(px, "t5", ask="q", calls=[OK, FAILED, UNKNOWN])
    got = {c["name"]: c["percepteye_outcome"]
           for c in turn_to_record(tmp_path / "t5")["tool_calls"]}
    assert got == {"lookup": "ok", "charge": "failed", "refund": "unknown"}


# ── what must NOT be invented ──────────────────────────────────────────────


def test_no_reward_and_no_success_are_invented(px, tmp_path):
    """A production turn has no verdict -- nobody graded it. Inventing one is
    the exact defect the emitter's `success is None` rule exists to prevent:
    an ungraded turn cloned as a demonstration of good work."""
    _turn(px, "t6", ask="q", answer="a", calls=[OK])
    rec = turn_to_record(tmp_path / "t6")
    assert "reward" not in rec
    assert "success" not in rec
    assert "final_return" not in rec


# ── absence ────────────────────────────────────────────────────────────────


def test_a_turn_that_recorded_nothing_is_not_a_record(px, tmp_path):
    with px.turn(turn_id="t7"):
        pass
    assert turn_to_record(tmp_path / "t7") is None


def test_a_turn_with_tools_but_no_text_IS_a_record(px, tmp_path):
    """Its tool sequence is real evidence, and several verifier kinds score
    exactly that."""
    _turn(px, "t8", calls=[OK])
    rec = turn_to_record(tmp_path / "t8")
    assert rec is not None
    assert rec["messages"] == []
    assert len(rec["tool_calls"]) == 1


def test_a_partly_unreadable_trajectory_says_so(px, tmp_path):
    """Non-zero forbids the training grade downstream, so it must travel --
    a partly unreadable trajectory must not be graded as a complete one."""
    _turn(px, "t9", ask="q", calls=[OK])
    (tmp_path / "t9" / "tool_calls.jsonl").open("a").write("{truncated\n")
    assert turn_to_record(tmp_path / "t9")["tool_calls_omitted_count"] == 1


# ── the whole root ─────────────────────────────────────────────────────────


def test_turns_come_back_in_the_order_they_happened(px, tmp_path):
    """The ingest side chunks by assistant turn; an out-of-order stream would
    interleave two conversations."""
    for i in range(3):
        _turn(px, f"turn{i}", ask=f"q{i}", answer=f"a{i}")
    ids = [r["task_id"] for r in iter_records(tmp_path)]
    assert ids == ["turn0", "turn1", "turn2"]


def test_write_records_emits_one_json_object_per_line(px, tmp_path):
    _turn(px, "t10", ask="q", answer="a")
    out = tmp_path / "out" / "export.jsonl"
    assert write_records(tmp_path, out) == 1
    lines = [json.loads(x) for x in out.read_text().splitlines() if x.strip()]
    assert len(lines) == 1 and lines[0]["task_id"] == "t10"


def test_an_empty_root_writes_an_empty_file_rather_than_failing(tmp_path):
    out = tmp_path / "empty.jsonl"
    assert write_records(tmp_path / "nothing-here", out) == 0
    assert out.read_text() == ""


# ── conversations: what makes a privileged reference obtainable ────────────


def test_a_conversation_exports_as_ONE_record(px, tmp_path):
    """The ingest side chunks a messages list into one transition per
    assistant turn, and the messages that follow a turn are its observed next
    state. A lone turn exported by itself has nothing after it, so it can never
    carry one -- and that observation is the only privileged reference
    obtainable without a verifier, a passing sibling or a grader."""
    for i, (ask, ans) in enumerate([("q0", "a0"), ("q1", "a1"), ("q2", "a2")]):
        with px.turn(turn_id=f"t{i}", conversation_id="c1", input_text=ask):
            pass
        px.record_answer(f"t{i}", ans)
    recs = list(iter_records(tmp_path))
    assert len(recs) == 1
    assert [m["content"] for m in recs[0]["messages"]] == \
        ["q0", "a0", "q1", "a1", "q2", "a2"]
    assert recs[0]["conversation_id"] == "c1"


def test_turns_from_DIFFERENT_conversations_do_not_merge(px, tmp_path):
    """Merging them would attach one user's outcome to another's answer."""
    for cid in ("c1", "c2"):
        with px.turn(turn_id=f"{cid}-t", conversation_id=cid, input_text="q"):
            pass
        px.record_answer(f"{cid}-t", "a")
    recs = list(iter_records(tmp_path))
    assert len(recs) == 2
    assert {r["conversation_id"] for r in recs} == {"c1", "c2"}


def test_a_turn_with_no_conversation_id_stays_on_its_own(px, tmp_path):
    """A turn that never declared which conversation it belonged to cannot be
    placed in one, and guessing is how two users' turns end up in one record."""
    with px.turn(turn_id="lonely", input_text="q"):
        pass
    px.record_answer("lonely", "a")
    recs = list(iter_records(tmp_path))
    assert len(recs) == 1 and recs[0]["task_id"] == "lonely"


def test_grouping_can_be_turned_off(px, tmp_path):
    for i in range(2):
        with px.turn(turn_id=f"t{i}", conversation_id="c1", input_text="q"):
            pass
        px.record_answer(f"t{i}", "a")
    assert len(list(iter_records(tmp_path, group_conversations=False))) == 2


def test_conversation_order_follows_the_turns_not_the_dict(px, tmp_path):
    """Out of order, one turn's outcome attaches to another's answer."""
    for i in range(3):
        with px.turn(turn_id=f"t{i}", conversation_id="c1", input_text=f"q{i}"):
            pass
        px.record_answer(f"t{i}", f"a{i}")
    rec = next(iter(iter_records(tmp_path)))
    msgs = [m["content"] for m in rec["messages"]]
    assert msgs.index("q0") < msgs.index("q1") < msgs.index("q2")
