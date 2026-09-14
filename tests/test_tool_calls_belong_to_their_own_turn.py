"""A turn's tool calls must not be attributed to another turn's answer.

``conversation_to_record`` emits several turns as one record: an OpenAI-style
``messages`` list plus a TOP-LEVEL ``tool_calls`` list. The control plane's
ingest chunks that messages list into one transition per assistant message and
matches the top-level list by ``step_index``, treating a MISSING ``step_index``
as "belongs to every step".

``_tool_call`` emitted no ``step_index``. So on a 2-turn conversation making one
call per turn there were **4 attributions instead of 2** — turn 0 carrying turn
1's failure and turn 1 carrying turn 0's success. That is not cosmetic
duplication: ``success`` on a tool call is what the outcome verifiers score, so
both rows were graded against outcomes their own turn never produced.
``conversation_to_record``'s own docstring names this as the thing the caller's
turn ordering exists to prevent — "getting it wrong would attach one turn's
outcome to another's answer."

SCOPE OF THIS FILE. It asserts only what this SDK EMITS. The consumer's side of
the same contract is asserted in the consumer's own suite, against the wire
shape. This package imports nothing from the control plane's internals — in
production code or in tests — and talks to its HTTP API and nothing else; an
import here would establish exactly the coupling that boundary exists to
prevent, and would not survive a fresh install either, since this package
declares one runtime dependency.
"""
from __future__ import annotations

import pytest

from agent_flywheel.export import conversation_to_record
from agent_flywheel.outcomes import record_tool_call
from agent_flywheel.production import attach


@pytest.fixture
def px(tmp_path):
    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        yield px
    finally:
        px.close()


def _turn(px, tid, *, ask, answer=None, calls=()):
    with px.turn(turn_id=tid, input_text=ask, conversation_id="conv-1"):
        for c in calls:
            record_tool_call(**c)
    if answer:
        px.record_answer(tid, answer)


A_OK = {"name": "create_issue", "arguments": {"title": "A"},
        "outcome": "ok", "status_code": 201}
B_FAILED = {"name": "create_issue", "arguments": {"title": "B"},
            "outcome": "failed", "status_code": 500}


def _by_title(rec):
    return {tc["arguments"]["title"]: tc for tc in rec.get("tool_calls", [])}


# ── every emitted call says which assistant turn it belongs to ────────────


def test_each_call_carries_its_own_assistant_turn_index(px, tmp_path):
    """THE defect. Without an index the consumer matches every call to every
    turn, because absent means "any"."""
    _turn(px, "t0", ask="create issue A", answer="made A", calls=[A_OK])
    _turn(px, "t1", ask="create issue B", answer="made B", calls=[B_FAILED])

    rec = conversation_to_record([tmp_path / "t0", tmp_path / "t1"],
                                 conversation_id="conv-1")
    by_title = _by_title(rec)
    assert by_title["A"]["step_index"] == 0
    assert by_title["B"]["step_index"] == 1


def test_no_emitted_call_is_ever_missing_an_index(px, tmp_path):
    """The property, not the two values. A call without ``step_index`` is the
    match-everything case, so the invariant is that none escapes unindexed —
    which stays true however many turns a conversation grows."""
    for i in range(4):
        _turn(px, f"t{i}", ask=f"ask {i}", answer=f"ans {i}",
              calls=[{**A_OK, "arguments": {"title": str(i)}}])

    rec = conversation_to_record([tmp_path / f"t{i}" for i in range(4)],
                                 conversation_id="conv-1")
    assert len(rec["tool_calls"]) == 4
    assert all("step_index" in tc for tc in rec["tool_calls"])
    assert sorted(tc["step_index"] for tc in rec["tool_calls"]) == [0, 1, 2, 3]


def test_a_single_turn_record_is_unaffected(px, tmp_path):
    """One assistant turn cannot cross-attribute; the common case must not
    change shape."""
    _turn(px, "t0", ask="create issue A", answer="made A", calls=[A_OK])
    rec = conversation_to_record([tmp_path / "t0"], conversation_id="conv-1")
    assert len(rec["tool_calls"]) == 1
    assert rec["tool_calls"][0]["step_index"] == 0


def test_several_calls_in_one_turn_share_that_turn_index(px, tmp_path):
    """The index is per assistant TURN, not per call."""
    _turn(px, "t0", ask="two things", answer="did both",
          calls=[A_OK, {**B_FAILED, "arguments": {"title": "B"}}])
    _turn(px, "t1", ask="one more", answer="did it",
          calls=[{**A_OK, "arguments": {"title": "C"}}])

    rec = conversation_to_record([tmp_path / "t0", tmp_path / "t1"],
                                 conversation_id="conv-1")
    idx = {t: tc["step_index"] for t, tc in _by_title(rec).items()}
    assert idx == {"A": 0, "B": 0, "C": 1}, idx


# ── a turn with no answer has no step to belong to ────────────────────────


def test_calls_from_an_answerless_turn_are_counted_not_misattributed(px, tmp_path):
    """``record_answer`` is documented as callable after the scope closes, so a
    turn with calls and no answer is an ordinary state — and it produces no
    assistant message, hence no step. Those calls must not be attached to a
    neighbouring turn's answer, and must not be emitted unindexed, which would
    re-create the match-every-step bug for exactly these rows."""
    _turn(px, "t0", ask="create issue A", calls=[A_OK])          # no answer
    _turn(px, "t1", ask="create issue B", answer="made B", calls=[B_FAILED])

    rec = conversation_to_record([tmp_path / "t0", tmp_path / "t1"],
                                 conversation_id="conv-1")
    assert list(_by_title(rec)) == ["B"], "the answerless turn's call leaked"
    assert rec["tool_calls_omitted_count"] == 1, (
        "a call this record does not carry must be COUNTED, or a partly "
        "carried trajectory is graded as a complete one"
    )
    assert rec["tool_calls"][0]["step_index"] == 0


def test_an_answerless_turn_does_not_shift_later_indices(px, tmp_path):
    """The index counts ASSISTANT MESSAGES, not turn directories. Counting
    directories would offset every later call by one and reproduce the same
    misattribution one step over — a fix that looks right on the happy path and
    is wrong the moment a turn goes unanswered."""
    _turn(px, "t0", ask="A", calls=[A_OK])                        # no answer
    _turn(px, "t1", ask="B", answer="made B", calls=[B_FAILED])
    _turn(px, "t2", ask="C", answer="made C",
          calls=[{**A_OK, "arguments": {"title": "C"}}])

    rec = conversation_to_record(
        [tmp_path / "t0", tmp_path / "t1", tmp_path / "t2"],
        conversation_id="conv-1")
    idx = {t: tc["step_index"] for t, tc in _by_title(rec).items()}
    assert idx == {"B": 0, "C": 1}, idx
    assert rec["tool_calls_omitted_count"] == 1


def test_the_index_matches_the_assistant_messages_actually_emitted(px, tmp_path):
    """Binds the two halves of the record to each other rather than to a
    literal: every emitted index must address a real assistant message in the
    same record. This is what would catch an off-by-one introduced later by a
    change to how messages are appended."""
    _turn(px, "t0", ask="A", calls=[A_OK])                        # no answer
    _turn(px, "t1", ask="B", answer="made B", calls=[B_FAILED])
    _turn(px, "t2", ask="C", answer="made C",
          calls=[{**A_OK, "arguments": {"title": "C"}}])

    rec = conversation_to_record(
        [tmp_path / "t0", tmp_path / "t1", tmp_path / "t2"],
        conversation_id="conv-1")
    n_assistant = sum(1 for m in rec["messages"] if m["role"] == "assistant")
    for tc in rec["tool_calls"]:
        assert 0 <= tc["step_index"] < n_assistant, (
            f"step_index {tc['step_index']} addresses no assistant message "
            f"(there are {n_assistant})"
        )
