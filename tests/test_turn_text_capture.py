"""The turn itself is capturable, and both producers write one format.

Nothing on the production path recorded what the agent was ASKED or what it
ANSWERED. attach()'s trajectory held tool-call outcomes only, and the OTel
egress emitted tool spans only, so the loosest trainable shape in the tree --
SFT, which wants messages or prompt+completion -- could not be fed from
production at all. The gap was never logprobs; it was the turn.

`turns.jsonl` already existed and was already written by the CLI recorder from
its host's turn events. It had no reader. This adds the reader, and gives
attach() a way to write the same format, so there is ONE format with two
producers rather than two formats.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from percepteye_agent_flywheel.outcomes import (
    TURNS_FILENAME,
    read_turn_text,
    record_turn_text,
)
from percepteye_agent_flywheel.production import attach


@pytest.fixture
def px(tmp_path):
    # CLOSED on teardown. attach() binds the adapters GLOBALLY, so an
    # attachment left open leaks that binding into every later test -- which
    # is exactly how this file first broke
    # `test_binding_off_captures_nothing_rather_than_guessing`, a test whose
    # whole point is that nothing is captured when binding is off.
    px = attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    try:
        yield px
    finally:
        px.close()


# ── the round trip ─────────────────────────────────────────────────────────


def test_a_turn_records_what_was_asked_and_answered(px, tmp_path):
    with px.turn(turn_id="t1", input_text="summarise the incident") as tid:
        pass
    px.record_answer(tid, "Three services timed out at 14:02.")
    got = px.read_turn("t1")
    assert got == {"task": "summarise the incident",
                   "answer": "Three services timed out at 14:02."}


def test_the_answer_can_arrive_after_the_scope_closed(px):
    """The same reason turn() yields its id: an outcome can arrive later than
    the turn does -- the draft kept an hour afterwards."""
    with px.turn(turn_id="t2", input_text="draft it") as tid:
        pass
    assert px.record_answer(tid, "here is the draft") is True
    assert px.read_turn("t2")["answer"] == "here is the draft"


def test_a_turn_with_no_text_records_none_and_leaves_nothing(px, tmp_path):
    """Turn text is OPT-IN. Passing nothing must behave exactly as before --
    tool outcomes stay zero-instrumentation."""
    with px.turn(turn_id="t3"):
        pass
    assert px.read_turn("t3") is None
    assert not (tmp_path / "t3").exists()


# ── absence is never collapsed ─────────────────────────────────────────────


def test_empty_text_writes_nothing_rather_than_an_empty_record(px):
    """"The agent answered with silence" and "we were not given the answer"
    are different claims, and only the second is true."""
    with px.turn(turn_id="t4", input_text="   "):
        pass
    assert px.read_turn("t4") is None
    assert px.record_answer("t4", "") is False


def test_only_the_half_that_was_recorded_comes_back(px):
    with px.turn(turn_id="t5", input_text="ask"):
        pass
    assert px.read_turn("t5") == {"task": "ask"}


def test_the_last_answer_wins(px, tmp_path):
    """A turn answered twice -- a retry, a correction -- ends with the answer
    that was actually returned."""
    with px.turn(turn_id="t6", input_text="q"):
        pass
    px.record_answer("t6", "first")
    px.record_answer("t6", "second")
    assert px.read_turn("t6")["answer"] == "second"


# ── never breaks the turn ──────────────────────────────────────────────────


def test_an_unwritable_trajectory_does_not_raise(tmp_path):
    """A capture bug costs the observation and never the turn."""
    assert record_turn_text("/proc/nope-xyz", "task", "hi") is False


def test_an_unknown_kind_is_a_programming_error_not_a_silent_drop():
    """The two kinds are the contract. A typo'd kind that silently wrote
    nothing would be a capture hole nobody could see."""
    with pytest.raises(ValueError):
        record_turn_text("/tmp", "reply", "hi")


def test_a_truncated_final_line_costs_that_record_and_nothing_else(tmp_path):
    d = tmp_path / "t7"
    d.mkdir()
    record_turn_text(d, "task", "good")
    (d / TURNS_FILENAME).open("a").write('{"kind":"answer","text":"trunc')
    assert read_turn_text(d) == {"task": "good"}


# ── ONE format, TWO producers ──────────────────────────────────────────────


def test_the_cli_recorders_records_read_back_through_the_same_reader(tmp_path):
    """THE ANTI-DRIFT TEST, and the reason it is not a comment.

    The CLI recorder is stdlib-only by contract and cannot import outcomes, so
    it writes this format with its own code. Two producers of one structure
    drift unless something reads BOTH -- which is exactly how a previous seam
    authored zero on every real handoff while its whole test file passed.

    These lines are the recorder's shape, built the way `_turn_record` builds
    them: kind + text, plus the host ids it carries.
    """
    d = tmp_path / "from_cli"
    d.mkdir()
    with (d / TURNS_FILENAME).open("w", encoding="utf-8") as fh:
        for rec in (
            {"kind": "task", "text": "fix the flaky test",
             "session_id": "s1", "working_dir": "/w"},
            {"kind": "answer", "text": "retried on a fixed seed",
             "session_id": "s1", "working_dir": "/w"},
        ):
            fh.write(json.dumps(rec) + "\n")
    assert read_turn_text(d) == {"task": "fix the flaky test",
                                 "answer": "retried on a fixed seed"}


def test_both_producers_agree_on_the_wire_keys(tmp_path):
    """Not just readable -- the SAME keys. A reader tolerant enough to accept
    two shapes would hide the drift it exists to catch."""
    d = tmp_path / "ours"
    d.mkdir()
    record_turn_text(d, "task", "x", session_id="s1", working_dir="/w")
    ours = json.loads((d / TURNS_FILENAME).read_text().splitlines()[0])
    assert set(ours) == {"kind", "text", "session_id", "working_dir"}
    assert ours["kind"] == "task"
