"""A customer can say how a turn or a conversation actually turned out.

``turn()`` has always yielded its id as "the join key for any outcome that
arrives later" -- and nothing took the outcome. ``report_outcome`` does, and
these tests pin what makes it safe to call from a customer's own code:

* IT IS VALIDATED WHERE IT IS CALLED, against the closed vocabulary, with a
  named error -- whether or not capture is on, so a typo is not a silently
  missing label.
* IT IS A NO-OP WITHOUT CONSENT. An outcome is a fact about the agent's end
  users' traffic: nothing is queued, written or sent for an agent whose
  production capture is not enabled.
* IT RETRIES LIKE A TURN. It waits on disk until the control plane
  acknowledges it, survives a restart, stops on a 403, and is resent with the
  SAME idempotency key.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

import agent_flywheel as fw
from agent_flywheel import production
from agent_flywheel.contract import (
    _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS,
    _PRODUCTION_OUTCOME_NOTE_MAX_CHARS,
    _RFC3339_PATTERN,
    PRODUCTION_OUTCOMES,
    production_outcome,
)
from agent_flywheel.errors import (
    ContractViolation,
    InvalidOutcome,
    TransportError,
)

from .test_production_turn_upload import _attached, _Transport, _turn

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json")
    .read_text(encoding="utf-8"))


class _Outcomes(_Transport):
    """The turn fake, plus the outcome route."""

    def __init__(self, *, outcomes_raise=None, refuse=(), **kw):
        super().__init__(**kw)
        self.outcome_batches: list[list[dict]] = []
        self.outcomes_raise = outcomes_raise
        self._refuse = set(refuse)

    def report_outcomes(self, agent_id, outcomes):
        if self.outcomes_raise is not None:
            raise self.outcomes_raise
        self.outcome_batches.append([dict(o) for o in outcomes])
        return {"agent_id": agent_id, "outcomes": [
            {"outcome_id": o["outcome_id"], "accepted": False,
             "duplicate": False, "error": "outcome_conflict"}
            if o["outcome_id"] in self._refuse else
            {"outcome_id": o["outcome_id"], "accepted": True,
             "duplicate": False}
            for o in outcomes
        ]}

    @property
    def outcomes(self):
        return [o for batch in self.outcome_batches for o in batch]


def _pending(root) -> list[Path]:
    return sorted(
        p for p in Path(root).iterdir() if p.name.startswith(".outcome."))


# ── the happy path ─────────────────────────────────────────────────────────


def test_the_turn_id_turn_yields_is_what_an_outcome_joins_on(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(input_text="where is my order?") as turn_id:
            pass
        px.record_answer(turn_id, "It ships tomorrow.")
        outcome_id = px.report_outcome(
            "resolved", turn_id=turn_id, value=129.5, note="confirmed by mail",
            occurred_at=datetime(2026, 11, 10, 10, 0, tzinfo=timezone(
                timedelta(hours=1))),
        )
        assert outcome_id and outcome_id.startswith("o_")
        assert quiet == [1], "queued outcomes are uploaded off-thread at once"
        counts = px.flush()
    assert counts["outcomes_sent"] == 1 and counts["outcomes_failed"] == 0
    assert tr.outcomes == [{
        "outcome_id": outcome_id, "outcome": "resolved", "turn_id": turn_id,
        "value": 129.5, "note": "confirmed by mail",
        # UTC, milliseconds, Z: byte for byte what the Node SDK writes.
        "occurred_at": "2026-11-10T09:00:00.000Z",
    }]
    assert _pending(tmp_path) == [], "an acknowledged outcome is not resent"


def test_every_word_of_the_vocabulary_is_accepted(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        for word in PRODUCTION_OUTCOMES:
            assert px.report_outcome(word, conversation_id="c1")
        px.flush()
    assert sorted(o["outcome"] for o in tr.outcomes) == sorted((
        "resolved", "escalated", "reopened", "abandoned",
        "thumbs_up", "thumbs_down"))
    assert all("turn_id" not in o for o in tr.outcomes)
    assert all(o["occurred_at"].endswith("Z") for o in tr.outcomes)


def test_several_outcomes_of_one_conversation_are_each_sent(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        first = px.report_outcome("resolved", conversation_id="c1")
        second = px.report_outcome("reopened", conversation_id="c1")
        px.flush()
    assert first != second
    assert {o["outcome_id"]: o["outcome"] for o in tr.outcomes} == {
        first: "resolved", second: "reopened"}


def test_a_callers_own_outcome_id_is_the_idempotency_key(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.report_outcome(
            "thumbs_up", turn_id="t1", outcome_id="ticket-9-closed",
        ) == "ticket-9-closed"
        # The same report again is the same file, not a second outcome.
        px.report_outcome("thumbs_up", turn_id="t1", outcome_id="ticket-9-closed")
        assert len(_pending(tmp_path)) == 1
        px.flush()
    assert [o["outcome_id"] for o in tr.outcomes] == ["ticket-9-closed"]


# ── validated where it is called ───────────────────────────────────────────


def test_a_word_outside_the_vocabulary_is_a_named_error(
    monkeypatch, tmp_path, quiet,
):
    with _attached(monkeypatch, tmp_path, _Outcomes()) as px:
        with pytest.raises(InvalidOutcome, match="thumbs_down") as raised:
            px.report_outcome("delighted", turn_id="t1")
        assert '"delighted"' in str(raised.value)
        assert isinstance(raised.value, ContractViolation)
        assert _pending(tmp_path) == []
    assert fw.InvalidOutcome is InvalidOutcome


def test_it_is_validated_even_while_capture_is_off(
    monkeypatch, tmp_path, quiet,
):
    """A typo must not hide behind the consent no-op and surface the day an
    operator enables capture."""
    with _attached(monkeypatch, tmp_path, _Outcomes(capture=False)) as px:
        with pytest.raises(InvalidOutcome):
            px.report_outcome("delighted", turn_id="t1")


@pytest.mark.parametrize(("kwargs", "problem"), [
    ({}, "conversation_id or the episode_id it is the outcome of"),
    ({"turn_id": "../escape"}, "turn_id"),
    ({"turn_id": 7}, "turn_id"),
    ({"conversation_id": " padded"}, "conversation_id"),
    ({"conversation_id": "c" * 256}, "conversation_id"),
    ({"turn_id": "t1", "value": float("nan")}, "finite number"),
    ({"turn_id": "t1", "value": float("inf")}, "finite number"),
    ({"turn_id": "t1", "value": True}, "finite number"),
    ({"turn_id": "t1", "value": "12"}, "finite number"),
    ({"turn_id": "t1", "note": ""}, "note"),
    ({"turn_id": "t1", "note": 5}, "note"),
    ({"turn_id": "t1",
      "note": "x" * (_PRODUCTION_OUTCOME_NOTE_MAX_CHARS + 1)}, "1000"),
    ({"turn_id": "t1", "occurred_at": datetime(2026, 11, 10, 9)}, "timezone"),
    ({"turn_id": "t1", "occurred_at": "yesterday"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "2026-11-10T09:00:00"}, "RFC 3339"),
    # The shape of a time is not a time: each of these used to pass.
    ({"turn_id": "t1", "occurred_at": "2026-02-30T09:00:00Z"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "2025-02-29T09:00:00Z"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "2026-11-10T24:00:00Z"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "2026-11-10T09:60:00Z"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "2026-11-10T23:59:60Z"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "2026-13-10T09:00:00Z"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "0000-11-10T09:00:00Z"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "2026-11-10T09:00:00+24:00"}, "RFC 3339"),
    # Digits of another script: Python's ``\d`` matched them.
    ({"turn_id": "t1", "occurred_at": "\u0662026-11-10T09:00:00Z"}, "RFC 3339"),
    ({"turn_id": "t1", "occurred_at": "\uff12\uff10\uff12\uff16-11-10T09:00:00Z"}, "RFC 3339"),
    # An int the wire's float cannot hold.
    ({"turn_id": "t1", "value": 10 ** 400}, "finite number"),
    ({"turn_id": "t1", "outcome_id": "a/b"}, "outcome_id"),
    ({"turn_id": "t1", "outcome_id": ""}, "outcome_id"),
])
def test_what_the_wire_cannot_carry_is_refused_by_name(
    monkeypatch, tmp_path, quiet, kwargs, problem,
):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with pytest.raises(InvalidOutcome, match=problem):
            px.report_outcome("resolved", **kwargs)
        assert _pending(tmp_path) == []
        px.flush()
    assert tr.outcomes == []


def test_a_note_of_exactly_the_limit_and_an_rfc3339_string_are_carried(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes()
    note = "é" * _PRODUCTION_OUTCOME_NOTE_MAX_CHARS
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("abandoned", turn_id="t1", note=note, value=3,
                          occurred_at="2026-11-10T09:00:00+01:00")
        px.flush()
    (sent,) = tr.outcomes
    assert sent["note"] == note
    assert sent["value"] == 3
    assert sent["occurred_at"] == "2026-11-10T09:00:00+01:00"


# ── a no-op without consent ────────────────────────────────────────────────


def test_it_is_a_no_op_until_production_capture_is_enabled(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes(capture=False, reason="opt-in per agent")
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is False
        assert px.report_outcome("resolved", turn_id="t1", note="hi") is None
        assert _pending(tmp_path) == [], "nothing is kept for an agent that may not send it"
        assert quiet == []
        px.flush()
    assert tr.outcomes == []


def test_it_is_a_no_op_with_no_control_plane_key(monkeypatch, tmp_path, quiet):
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    try:
        assert px.report_outcome("resolved", turn_id="t1") is None
        assert _pending(tmp_path) == []
    finally:
        px.close()


class _Unanswered(_Outcomes):
    """A registration that failed on a blip, so consent is UNKNOWN; the
    capture read answers ``later`` (``None``: still nothing) when re-asked."""

    def __init__(self, later=None, **kw):
        super().__init__(**kw)
        self.later = later
        self.asked = 0

    def register(self, spec):
        raise TransportError("connection reset")

    def capture_current(self, agent_id):
        self.asked += 1
        return {} if self.later is None else {"capture": self.later}


def test_an_outcome_reported_before_consent_is_known_is_kept_and_asks_again(
    monkeypatch, tmp_path, quiet,
):
    """``may_upload`` is False both after an explicit no and after a
    registration that never got its answer. Read as a no, the second dropped
    every outcome reported until a turn's flush happened to re-ask -- and
    nothing re-asked for an outcome. It is kept, and reporting it starts the
    flush that asks."""
    tr = _Unanswered()
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is False
        outcome_id = px.report_outcome("resolved", turn_id="t1")
        assert outcome_id is not None
        assert len(_pending(tmp_path)) == 1
        assert quiet == [1], "reporting it starts the flush that re-asks"
        # Still no answer: kept, not sent.
        px.flush()
        assert tr.asked >= 1 and tr.outcomes == []
        assert len(_pending(tmp_path)) == 1
        # The operator's yes arrives: the next flush asks, learns it, sends.
        tr.later = {"turns": True}
        counts = px.flush()
        assert px.may_upload is True
        assert counts["outcomes_sent"] == 1
    assert [o["outcome_id"] for o in tr.outcomes] == [outcome_id]
    assert _pending(tmp_path) == []


def test_an_explicit_no_learned_later_makes_reporting_a_no_op_again(
    monkeypatch, tmp_path, quiet,
):
    """Only the control plane's answered no ends the waiting: what was kept
    for that answer is deleted, since nothing will ever send it, and nothing
    new is kept -- the terms ``report_outcome`` keeps after a no. An outcome
    naming only an episode never waited on consent and stays."""
    tr = _Unanswered(later={"turns": False, "reason": "opt-in per agent"})
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.report_outcome("resolved", turn_id="t1") is not None
        episode = px.report_outcome("resolved", episode_id="rl_ep1")
        assert episode is not None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            px.flush()
        assert px.may_upload is False and px.upload_refusal == "opt-in per agent"
        assert px.report_outcome("reopened", turn_id="t1") is None
        assert _pending(tmp_path) == [], "the consent-gated outcome is purged"
        assert any("1 reported outcome(s)" in str(w.message)
                   and "opt-in per agent" in str(w.message) for w in caught)
    assert [o["outcome_id"] for o in tr.outcomes] == [episode]


@pytest.mark.parametrize("answer", [
    (401, {"detail": "invalid key"}),
    (403, {"detail": "this key may not read capture"}),
    (404, {"detail": "Not Found"}),
    (200, {"agent_id": "support"}),
], ids=["401", "403", "404", "no-capture-block"])
def test_a_definitive_answer_on_the_verdict_read_ends_the_waiting(
    monkeypatch, tmp_path, quiet, answer,
):
    """A 401, 403 or 404 on ``GET /agents/{id}/capture``, or an answer with no
    verdict in it, will be said again to every read: it is an answer, and it
    is no. Read as "unknown", every outcome reported was kept on disk for
    good and every report cost another read. Through the REAL client."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _over_http(monkeypatch, tmp_path, {
            # A registration that did not answer: consent starts UNKNOWN.
            "POST /agents": (200, {"agent_id": "support"}),
            "GET /agents/support/capture": answer,
        }) as (px, seen):
            assert px.may_upload is False
            assert px.report_outcome("resolved", turn_id="t1") is not None
            px.flush()
            reads = seen.count("GET /agents/support/capture")
            assert reads >= 1
            assert px.report_outcome("reopened", turn_id="t1") is None, (
                "a definitive answer is no longer 'unknown'")
            assert "did not grant" in px.upload_refusal
            if answer[0] != 200:
                assert f"HTTP {answer[0]}" in px.upload_refusal
            assert _pending(tmp_path) == [], (
                "kept for an answer that came back no: nothing will send it")


def test_an_outcome_kept_while_consent_is_unknown_still_expires(
    monkeypatch, tmp_path, quiet,
):
    """The week's expiry ran inside the upload, which runs only once consent
    is yes: an outcome kept while the answer never came stayed forever."""
    tr = _Unanswered()
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1")
        (stale,) = _pending(tmp_path)
        long_ago = time.time() - production._OUTCOME_MAX_AGE_S - 60
        os.utime(stale, (long_ago, long_ago))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            px.flush()
        assert px.may_upload is False
        assert _pending(tmp_path) == []
        assert any("for a week" in str(w.message) for w in caught)
    assert tr.outcomes == []


def test_at_most_so_many_outcomes_are_kept_while_consent_is_unknown(
    monkeypatch, tmp_path, quiet,
):
    """Bounded by count as well as by age: the oldest go, out loud."""
    monkeypatch.setattr(production, "_OUTCOME_MAX_KEPT_UNANSWERED", 3)
    tr = _Unanswered()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for n in range(5):
                px.report_outcome("resolved", turn_id="t1", outcome_id=f"o{n}")
                written = production._outcome_file(tmp_path, f"o{n}")
                when = time.time() - 1000 + n
                os.utime(written, (when, when))
        assert _pending(tmp_path) == sorted(
            production._outcome_file(tmp_path, f"o{n}") for n in (2, 3, 4))
        assert any("oldest first" in str(w.message) for w in caught)
        # Once consent is yes there is no cap: it is the upload's queue.
        tr.later = {"turns": True}
        px.flush()
    assert sorted(o["outcome_id"] for o in tr.outcomes) == ["o2", "o3", "o4"]


def test_the_bound_while_consent_is_unknown_holds_no_episode_outcome(
    monkeypatch, tmp_path, quiet,
):
    """An outcome that names only an episode does not wait on consent, so the
    count bound neither counts nor drops it -- it once did both, and said
    consent was the cause. The warning names the actual one: no answer."""
    monkeypatch.setattr(production, "_OUTCOME_MAX_KEPT_UNANSWERED", 3)
    tr = _Unanswered()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for n in range(5):
                px.report_outcome("resolved", episode_id=f"rl_{n}",
                                  outcome_id=f"e{n}")
            assert len(_pending(tmp_path)) == 5, "episodes are not bounded"
            assert not caught, "nothing was dropped"
            for n in range(5):
                px.report_outcome("resolved", turn_id="t1", outcome_id=f"o{n}")
                written = production._outcome_file(tmp_path, f"o{n}")
                when = time.time() - 1000 + n
                os.utime(written, (when, when))
        kept = sorted(json.loads(p.read_text())["outcome_id"]
                      for p in _pending(tmp_path))
        assert kept == ["e0", "e1", "e2", "e3", "e4", "o2", "o3", "o4"]
        said = [str(w.message) for w in caught if "oldest first" in str(w.message)]
        assert said and "has not answered" in said[-1]
        assert "has not yet said" not in said[-1]


def test_the_unanswered_drop_is_said_in_one_sentence_by_both_sdks():
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    theirs = run_node(sdk, node, (
        "import {unansweredOutcomesDropped} from './src/capture.js';\n"
        "process.stdout.write(unansweredOutcomesDropped(7));\n"))
    ours = production._unanswered_outcomes_dropped(7)
    assert theirs.removeprefix("[agent-flywheel] ") == ours.removeprefix(
        "agent-flywheel: ")


def test_nothing_waiting_is_sent_once_consent_is_withdrawn(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1")
        px.may_upload = False
        monkeypatch.setattr(px, "_recheck_capture", lambda: None)
        assert "outcomes_sent" not in px.flush()
        assert tr.outcomes == []
        assert len(_pending(tmp_path)) == 1


# ── the turns' retry and offline behaviour ─────────────────────────────────


def test_an_unreachable_control_plane_keeps_the_outcome_and_resends_the_same_id(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes(outcomes_raise=TransportError("connection reset"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        outcome_id = px.report_outcome("escalated", turn_id="t1")
        counts = px.flush()
        assert counts["outcomes_sent"] == 0 and counts["outcomes_failed"] == 1
        assert len(_pending(tmp_path)) == 1
        assert px.may_upload is True, "a blip is not a refusal"

        tr.outcomes_raise = None
        assert px.flush()["outcomes_sent"] == 1
    assert [o["outcome_id"] for o in tr.outcomes] == [outcome_id]
    assert _pending(tmp_path) == []


def test_an_undelivered_outcome_survives_a_restart(
    monkeypatch, tmp_path, quiet,
):
    down = _Outcomes(outcomes_raise=TransportError("timed out"))
    with _attached(monkeypatch, tmp_path, down) as px:
        outcome_id = px.report_outcome("thumbs_down", conversation_id="c1")
    assert len(_pending(tmp_path)) == 1

    up = _Outcomes()
    with _attached(monkeypatch, tmp_path, up) as px:
        px.flush()
    assert [o["outcome_id"] for o in up.outcomes] == [outcome_id]


def test_close_sends_what_is_waiting(monkeypatch, tmp_path, quiet):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1")
    assert len(tr.outcomes) == 1


def test_a_403_stops_sending_and_keeps_the_outcome(
    monkeypatch, tmp_path, quiet,
):
    """The turn upload's rule: a refusal is a decision, not a blip."""
    tr = _Outcomes(outcomes_raise=TransportError(
        "POST /outcomes -> HTTP 403: production capture is opt-in per agent",
        status_code=403))
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        assert counts["outcomes_failed"] == 1
        assert px.may_upload is False
        assert "opt-in per agent" in px.upload_refusal
        assert any("refused" in str(w.message) for w in caught)
        assert len(_pending(tmp_path)) == 1
        # ...and from here it is the no-op again.
        assert px.report_outcome("reopened", turn_id="t1") is None
        tr.outcomes_raise = None


def test_an_outcome_undelivered_for_a_week_is_dropped_out_loud(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="old")
        (stale,) = _pending(tmp_path)
        long_ago = time.time() - production._OUTCOME_MAX_AGE_S - 60
        os.utime(stale, (long_ago, long_ago))
        fresh = px.report_outcome("resolved", turn_id="fresh")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            px.flush()
    assert [o["outcome_id"] for o in tr.outcomes] == [fresh]
    assert any("for a week" in str(w.message) for w in caught)
    assert _pending(tmp_path) == []


def test_an_outcome_the_control_plane_refuses_by_name_is_not_resent(
    monkeypatch, tmp_path, quiet,
):
    """And it is said BY ITS ID, with the control plane's word, and counted
    as dropped: the warning used to name the word alone, and the count said
    the outcome was sent."""
    tr = _Outcomes(refuse={"dup"})
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1", outcome_id="dup")
        px.report_outcome("resolved", turn_id="t1", outcome_id="fine")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        said = [str(w.message) for w in caught
                if "did not record" in str(w.message)]
        assert len(said) == 1, "said once"
        assert "dup (outcome_conflict)" in said[0]
        assert "fine" not in said[0]
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (1, 0, 1)
        assert _pending(tmp_path) == []
        px.flush()
    assert len(tr.outcome_batches) == 1


def test_an_outcome_recorded_without_its_note_is_said_by_its_id(
    monkeypatch, tmp_path, quiet,
):
    """The control plane records an outcome whose note it could not mask and
    says so in the reply (``note.stored: false``). Nothing read it."""
    class _Noteless(_Outcomes):
        def report_outcomes(self, agent_id, outcomes):
            reply = super().report_outcomes(agent_id, outcomes)
            for result in reply["outcomes"]:
                result["note"] = (
                    {"stored": False, "error": "note_unmaskable"}
                    if result["outcome_id"] == "o-bare" else {"stored": True})
            return reply

    tr = _Noteless()
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome(
            "resolved", turn_id="t1", outcome_id="o-bare", note="a note")
        px.report_outcome(
            "resolved", turn_id="t1", outcome_id="o-whole", note="a note")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        said = [str(w.message) for w in caught
                if "without the note" in str(w.message)]
        assert len(said) == 1
        assert "o-bare (note_unmaskable)" in said[0]
        assert "o-whole" not in said[0]
        assert counts["outcomes_sent"] == 2, "it was recorded"


def test_more_than_a_batch_goes_in_batches(monkeypatch, tmp_path, quiet):
    tr = _Outcomes()
    total = _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS + 3
    with _attached(monkeypatch, tmp_path, tr) as px:
        for index in range(total):
            px.report_outcome("resolved", turn_id=f"t{index}")
        assert px.flush()["outcomes_sent"] == total
    assert [len(batch) for batch in tr.outcome_batches] == [
        _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS, 3]


class _Refusing(_Outcomes):
    """The outcome route answering by status: ``answer(ids)`` is ``None``
    (acknowledged) or the ``(status, detail)`` the control plane answers
    that batch with. Every batch it was sent is kept, refused or not."""

    def __init__(self, answer, **kw):
        super().__init__(**kw)
        self.answer = answer
        self.tried: list[list[str]] = []
        self.records: dict[str, dict] = {}

    def report_outcomes(self, agent_id, outcomes):
        ids = [o["outcome_id"] for o in outcomes]
        self.tried.append(ids)
        self.records.update((o["outcome_id"], dict(o)) for o in outcomes)
        refused = self.answer(ids)
        if refused is not None:
            status, detail = refused
            raise TransportError(
                f"POST /outcomes -> HTTP {status}: {detail}",
                status_code=status)
        return super().report_outcomes(agent_id, outcomes)


def _report(px, *ids):
    for outcome_id in ids:
        px.report_outcome("resolved", turn_id="t1", outcome_id=outcome_id)


def test_one_refused_outcome_never_holds_back_the_good_ones(
    monkeypatch, tmp_path, quiet,
):
    """A 4xx about the batch is final for it. It used to be retried every
    flush for a week, then the whole batch deleted: one bad record cost up to
    63 good ones. The batch is split until the refused record stands alone;
    the rest are sent at once and the refused one is dropped, by name."""
    ids = [f"o{n}" for n in range(9)]
    tr = _Refusing(lambda batch: (
        (422, "value: Input should be a finite number")
        if "o5" in batch else None))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, *ids)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (8, 0, 1)
        assert _pending(tmp_path) == [], "nothing is left to retry forever"
        assert any("HTTP 422" in str(w.message)
                   and "finite number" in str(w.message) for w in caught)
        tried = len(tr.tried)
        px.flush()
        assert len(tr.tried) == tried, "a refused record is not sent again"
    assert sorted(o["outcome_id"] for o in tr.outcomes) == sorted(
        set(ids) - {"o5"})
    assert ["o5"] in tr.tried, "the refused record was isolated, not guessed"


def test_a_refusal_of_the_whole_batch_keeps_every_outcome(
    monkeypatch, tmp_path, quiet,
):
    """Every part of the batch refused alike -- the same status, the same
    words -- is a refusal of THE BATCH: a 422 from a control plane on another
    contract minor, a proxy's 400. It used to be bisected to singles
    and every outcome dropped, 64 outcomes costing 127 requests, each charged
    against the control plane's rate limit. Now the two halves agreeing ends
    it: three requests, every outcome kept for a later flush (the week still
    applies), and the cause said once, not once per flush."""
    ids = [f"o{n:02}" for n in range(_PRODUCTION_OUTCOME_BATCH_MAX_ITEMS)]
    tr = _Refusing(lambda batch: (422, "outcomes.0.note: Extra inputs are "
                                       "not permitted"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, *ids)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
            again = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (0, len(ids), 0)
        assert again["outcomes_failed"] == len(ids)
        assert len(_pending(tmp_path)) == len(ids), "nothing is dropped"
        batch = tr.tried[0]
        assert sorted(batch) == ids
        half = len(ids) // 2
        assert tr.tried == [batch, batch[:half], batch[half:]] * 2, (
            "the batch and its two halves, per flush -- not a bisection")
        said = [str(w.message) for w in caught
                if "whole batches" in str(w.message)]
        assert len(said) == 1 and "HTTP 422" in said[0], said
        assert not any("were dropped" in str(w.message) for w in caught)
    assert tr.outcomes == []


def test_a_413_is_about_size_so_the_split_goes_on_until_it_fits(
    monkeypatch, tmp_path, quiet,
):
    """A 413 is about the BODY'S SIZE, so both halves of a batch too large
    for a proxy answer the same words -- and halves too large still is not a
    refusal of the batch: quarters may fit. It used to stop the flush there,
    and the outcomes waited a week and expired. A 413 always splits on."""
    ids = [f"o{n:02}" for n in range(_PRODUCTION_OUTCOME_BATCH_MAX_ITEMS)]
    quarter = len(ids) // 4
    tr = _Refusing(lambda batch: (
        (413, "Request Entity Too Large") if len(batch) > quarter else None))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, *ids)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (len(ids), 0, 0)
        assert _pending(tmp_path) == []
        assert len(tr.tried) == 1 + 2 + 4, "halves, then quarters"
        assert not any("whole batches" in str(w.message) for w in caught)
    assert sorted(o["outcome_id"] for o in tr.outcomes) == ids


def test_one_outcome_too_large_alone_is_dropped_naming_413_and_its_size(
    monkeypatch, tmp_path, quiet,
):
    """A record that still draws a 413 standing alone is too large on its
    own: it is the record's fault, so it is dropped -- said with the status
    and the record's size -- and the rest of its batch is sent."""
    ids = [f"o{n}" for n in range(8)]
    tr = _Refusing(lambda batch: (
        (413, "Request Entity Too Large") if "o5" in batch else None))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, *ids)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (7, 0, 1)
        # The size as the wire spells the record: compact JSON, UTF-8 -- the
        # figure the Node SDK's ``Buffer.byteLength(JSON.stringify(wire))``
        # gives for the same record.
        size = len(json.dumps(tr.records["o5"], separators=(",", ":"),
                              ensure_ascii=False).encode("utf-8"))
        said = [str(w.message) for w in caught
                if "were dropped" in str(w.message)]
        assert len(said) == 1, said
        assert "HTTP 413" in said[0] and f"{size} bytes" in said[0], said


def test_two_records_refused_for_different_causes_are_still_isolated(
    monkeypatch, tmp_path, quiet,
):
    """The whole-batch rule asks for the SAME status and the SAME words: a
    record refused in each half, each for its own cause, is two bad records,
    and the good ones beside them still go."""
    ids = [f"o{n}" for n in range(8)]

    def answer(batch):
        if "o1" in batch and "o6" in batch:
            return (422, "outcomes: two records are invalid")
        if "o1" in batch:
            return (422, "o1: value must be finite")
        if "o6" in batch:
            return (409, "o6: replayed with different content")
        return None

    tr = _Refusing(answer)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, *ids)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            counts = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (6, 0, 2)
        assert _pending(tmp_path) == []
    assert sorted(o["outcome_id"] for o in tr.outcomes) == sorted(
        set(ids) - {"o1", "o6"})


def test_a_stop_counts_the_half_refused_beside_it(
    monkeypatch, tmp_path, quiet,
):
    """One half refused for a record, the other answered with a 403: sending
    stops, every outcome is still on disk, and each is counted once --
    including the refused half, which was neither sent nor dropped."""
    ids = [f"o{n}" for n in range(8)]

    def answer(batch):
        if "o1" in batch:
            return (422, "o1: value must be finite")
        return (403, "production capture is opt-in per agent")

    tr = _Refusing(answer)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, *ids)
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            counts = px.flush()
        assert px.may_upload is False
        assert len(_pending(tmp_path)) == len(ids)
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (0, len(ids), 0)


def test_splitting_is_bounded_per_flush(monkeypatch, tmp_path, quiet):
    """A refusal that names each batch differently is never recognised as a
    whole-batch one, and splitting it to singles would cost two requests per
    outcome. Each flush spends at most ``_OUTCOME_SPLIT_MAX_REQUESTS`` on
    splitting; what that leaves unsent stays for the next flush."""
    ids = [f"o{n:02}" for n in range(_PRODUCTION_OUTCOME_BATCH_MAX_ITEMS)]
    tr = _Refusing(lambda batch: (422, f"{batch[0]}: refused"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, *ids)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        assert len(tr.tried) <= 1 + production._OUTCOME_SPLIT_MAX_REQUESTS
        assert counts["outcomes_sent"] == 0
        assert (counts["outcomes_failed"] + counts["outcomes_dropped"]
                == len(ids))
        assert counts["outcomes_failed"] > 0
        assert len(_pending(tmp_path)) == counts["outcomes_failed"]
        assert any(f"{production._OUTCOME_SPLIT_MAX_REQUESTS} requests"
                   in str(w.message) for w in caught)


@contextmanager
def _over_http(monkeypatch, root, routes):
    """An attachment whose every request goes through the REAL
    ``ControlPlaneClient`` and ``AttachTransport``, answered in-process:
    ``routes`` maps ``"METHOD /path"`` to ``(status, json body)`` (or
    ``(status, json body, headers)``) or to a
    callable taking the request and returning one. Anything else is 200
    ``{}``. Every request is recorded as ``"METHOD /path"``."""
    import httpx

    from agent_flywheel import _http

    seen: list[str] = []

    def handler(request):
        key = f"{request.method} {request.url.path}"
        seen.append(key)
        answer = routes.get(key, (200, {}))
        if callable(answer):
            answer = answer(request)
        status, body, *headers = answer
        return httpx.Response(status, json=body,
                              headers=headers[0] if headers else None)

    real = _http.ControlPlaneClient
    monkeypatch.setattr(_http, "ControlPlaneClient", lambda url, key, **kw: real(
        url, key, client=httpx.Client(transport=httpx.MockTransport(handler))))
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_x")
    px = fw.attach(agent_id="support", trajectory_root=str(root))
    try:
        yield px, seen
    finally:
        px.close()


#: A control plane that granted capture, at registration and on the verdict
#: read the flush re-asks while model content is withheld.
GRANTED = {"POST /agents": (200, {"capture": {"turns": True}}),
           "GET /agents/support/capture": (200, {"capture": {"turns": True}})}


@pytest.mark.parametrize("status", [401, 404, 405, 410])
def test_a_refusal_of_the_route_drops_nothing(
    monkeypatch, tmp_path, quiet, status,
):
    """A control plane with no ``/outcomes`` route yet, or a key it does not
    take, refuses every batch for a reason that is not the records': bisecting
    would drop every good outcome. They stay, the cause is said, and the
    batches after it are not tried.

    THROUGH THE REAL HTTP CLIENT: a stub raising ``TransportError(status_code=
    410)`` passed while the client raised ``LeaseLost`` with no status for a
    410, so in production a 410 was a blip and every batch was tried."""
    total = _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS + 3
    with warnings.catch_warnings(record=True) as caught, _over_http(
        monkeypatch, tmp_path,
        {**GRANTED, "POST /outcomes": (status, {"detail": "Not Found"})},
    ) as (px, seen):
        warnings.simplefilter("always")
        _report(px, *(f"o{n}" for n in range(total)))
        counts = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (0, total, 0)
        assert len(_pending(tmp_path)) == total
        assert seen.count("POST /outcomes") == 1, "one try, not a bisection"
        assert any(f"HTTP {status}" in str(w.message)
                   and "did not take" in str(w.message) for w in caught)
        assert px.may_upload is True, "not a consent refusal"


def test_a_later_403_counts_each_unsent_outcome_once(
    monkeypatch, tmp_path, quiet,
):
    """A blip on the first batch and a 403 on the second: every outcome is
    still on disk, and is counted once. ``failed += len(pending) - sent`` on
    top of the first batch's own count made it 131 of 67."""
    total = _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS + 3
    calls = []

    class _BlipThenRefuse(_Outcomes):
        def report_outcomes(self, agent_id, outcomes):
            calls.append(len(outcomes))
            if len(calls) == 1:
                raise TransportError("connection reset")
            raise TransportError(
                "POST /outcomes -> HTTP 403: capture is off", status_code=403)

    with _attached(monkeypatch, tmp_path, _BlipThenRefuse()) as px:
        _report(px, *(f"o{n}" for n in range(total)))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            counts = px.flush()
        assert calls == [_PRODUCTION_OUTCOME_BATCH_MAX_ITEMS, 3]
        assert (counts["outcomes_sent"], counts["outcomes_failed"]) == (0, total)
        assert len(_pending(tmp_path)) == total


def test_a_waiting_outcome_is_never_read_as_a_turn(
    monkeypatch, tmp_path, quiet,
):
    """It is a FILE in the trajectory root because every directory there is a
    turn: the turn upload and the pruner must both walk past it."""
    tr = _Outcomes(outcomes_raise=TransportError("down"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="hi", answer="there")
        px.report_outcome("resolved", turn_id="t1")
        counts = px.flush()
        assert (counts["sent"], counts["failed"]) == (1, 0)
        assert [t["turn_id"] for t in tr.sent] == ["t1"]
        px.prune(keep_last=0)
        assert len(_pending(tmp_path)) == 1


def test_only_a_finished_outcome_file_is_read_as_an_outcome(
    monkeypatch, tmp_path, quiet,
):
    """The queue is the files this package finished writing: not a write
    still in progress, and not a customer's own JSON in a directory they
    named."""
    tr = _Outcomes()
    record = json.dumps(GOOD)
    with _attached(monkeypatch, tmp_path, tr) as px:
        half_written = tmp_path / ".outcome.0123.json.tmp"
        theirs = tmp_path / "notes.json"
        half_written.write_text(record, encoding="utf-8")
        theirs.write_text(record, encoding="utf-8")
        assert "outcomes_sent" not in px.flush()
        assert half_written.exists() and theirs.exists()
    assert tr.outcomes == []


def test_a_flush_with_no_outcomes_reports_only_the_turn_counts(
    monkeypatch, tmp_path, quiet,
):
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="hi", answer="there")
        assert px.flush() == {"sent": 1, "skipped": 0, "failed": 0}
    assert tr.outcome_batches == []


def test_a_disk_that_cannot_take_the_outcome_costs_the_outcome_not_the_caller(
    monkeypatch, tmp_path, quiet,
):
    with _attached(monkeypatch, tmp_path, _Outcomes()) as px:
        def refuse(*a, **kw):
            raise OSError("read-only file system")
        monkeypatch.setattr(production.os, "replace", refuse)
        assert px.report_outcome("resolved", turn_id="t1") is None
        assert quiet == []


def test_reporting_starts_a_background_flush(monkeypatch, tmp_path):
    """The real ``_flush_soon``: the caller's thread never waits on the
    network, and the outcome still goes."""
    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1")
        deadline = time.time() + 5
        while not tr.outcomes and time.time() < deadline:
            time.sleep(0.01)
        assert len(tr.outcomes) == 1


# ── the schema and the code are one vocabulary ─────────────────────────────


def test_schema_and_code_agree_on_the_outcome_wire():
    outcome = SCHEMA["$defs"]["ProductionOutcome"]
    batch = SCHEMA["$defs"]["ProductionOutcomeBatch"]
    assert tuple(outcome["properties"]["outcome"]["enum"]) == PRODUCTION_OUTCOMES
    assert outcome["properties"]["note"]["maxLength"] == (
        _PRODUCTION_OUTCOME_NOTE_MAX_CHARS)
    assert outcome["properties"]["occurred_at"]["pattern"] == _RFC3339_PATTERN
    assert batch["properties"]["outcomes"]["maxItems"] == (
        _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS)
    assert outcome["additionalProperties"] is False
    turn = SCHEMA["$defs"]["ProductionTurn"]["properties"]
    assert outcome["properties"]["turn_id"] == {
        **turn["turn_id"],
        "description": outcome["properties"]["turn_id"]["description"]}
    assert outcome["properties"]["conversation_id"]["pattern"] == (
        turn["conversation_id"]["pattern"])


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator({
        "$schema": SCHEMA["$schema"], "$defs": SCHEMA["$defs"],
        "$ref": f"#/$defs/{name}",
    }, format_checker=FormatChecker())


GOOD = {"outcome_id": "o1", "outcome": "resolved", "turn_id": "t1",
        "occurred_at": "2026-11-10T09:00:00Z"}


@pytest.mark.parametrize("wire", [
    {**GOOD, "outcome": "delighted"},
    {k: v for k, v in GOOD.items() if k != "turn_id"},
    {**GOOD, "turn_id": "../x"},
    {**GOOD, "conversation_id": " c"},
    {**GOOD, "value": "12"},
    {**GOOD, "note": ""},
    {**GOOD, "note": "x" * 1001},
    {**GOOD, "occurred_at": "2026-11-10T09:00:00"},
    {**GOOD, "occurred_at": "\u0662026-11-10T09:00:00Z"},
    {**GOOD, "outcome_id": "a/b"},
    {**GOOD, "extra": 1},
    {k: v for k, v in GOOD.items() if k != "occurred_at"},
])
def test_the_validator_and_the_published_schema_refuse_the_same_outcomes(wire):
    assert list(_validator("ProductionOutcome").iter_errors(wire))
    with pytest.raises(InvalidOutcome):
        production_outcome(wire)


@pytest.mark.parametrize("wire", [
    GOOD,
    {"outcome_id": "o2", "outcome": "thumbs_down", "conversation_id": "c 1",
     "value": -2.5, "note": "ñ", "occurred_at": "2026-11-10T09:00:00.250+05:30"},
    {**GOOD, "conversation_id": "c1", "value": 0},
])
def test_the_validator_and_the_published_schema_accept_the_same_outcomes(wire):
    _validator("ProductionOutcome").validate(wire)
    assert production_outcome(wire) == wire


# ── the wire call itself ───────────────────────────────────────────────────


class _Client:
    """Records requests instead of making them."""

    def __init__(self):
        self.calls: list[dict] = []

    def request(self, method, path, **kw):
        self.calls.append({"method": method, "path": path, **kw})
        return {}


#: Awkward on purpose, like the turn wire's fixture: non-ASCII text and an
#: astral character, where ``JSON.stringify`` and ``json.dumps`` diverge.
#: Integers only -- a non-integral number is the documented gap in the two
#: SDKs' shared digest (the Node SDK's ``stableStringify``).
BATCH = [
    GOOD,
    {"outcome_id": "o2", "outcome": "thumbs_down", "conversation_id": "c é",
     "value": 12, "note": "mañana \U0001f4e6",
     "occurred_at": "2026-11-10T09:00:00.250+05:30"},
]


def _sent(agent_id="agent-1", outcomes=BATCH):
    client = _Client()
    fw.AttachTransport(client, agent_id).report_outcomes(agent_id, outcomes)
    return client.calls[0]


def test_it_posts_the_batch_to_the_route_the_intake_serves():
    call = _sent()
    assert (call["method"], call["path"]) == ("POST", "/outcomes")
    assert call["json_body"] == {"agent_id": "agent-1", "outcomes": BATCH}
    _validator("ProductionOutcomeBatch").validate(call["json_body"])
    assert call.get("retry_if_ambiguous") is not False


def test_the_batch_carries_a_stable_idempotency_key():
    import re

    key = _sent()["idempotency_key"]
    assert re.fullmatch(r"outcomes-agent-1-[0-9a-f]{32}", key), key
    assert _sent()["idempotency_key"] == key
    assert _sent(outcomes=[{**GOOD, "outcome": "reopened"}])[
        "idempotency_key"] != key


@pytest.mark.parametrize(("agent_id", "outcomes", "problem"), [
    ("customer/team", BATCH, "agent_id"),
    ("agent-1", [], "between 1 and 64"),
    ("agent-1", [GOOD] * 65, "between 1 and 64"),
    ("agent-1", [{**GOOD, "outcome": "delighted"}], "outcome must be one of"),
    ("agent-1", [{**GOOD, "extra": 1}], "no field"),
])
def test_the_transport_refuses_what_the_published_schema_forbids(
    agent_id, outcomes, problem,
):
    client = _Client()
    with pytest.raises(ContractViolation, match=problem):
        fw.AttachTransport(client, agent_id).report_outcomes(agent_id, outcomes)
    assert client.calls == []


# ── the cross-SDK half ─────────────────────────────────────────────────────


def test_both_sdks_derive_the_same_outcome_batch_key():
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import {AttachTransport} from './src/transport.js';\n"
        "const calls=[];\n"
        "const tx=new AttachTransport({request:(m,p,o)=>{calls.push([m,p,o]);}},"
        "{agentId:'agent-1'});\n"
        f"tx.reportOutcomes('agent-1', {json.dumps(BATCH)});\n"
        "console.log(JSON.stringify({method:calls[0][0],path:calls[0][1],"
        "key:calls[0][2].idempotencyKey,body:calls[0][2].jsonBody}));\n"
    )
    theirs = json.loads(run_node(sdk, node, script))
    ours = _sent()
    assert theirs == {
        "method": ours["method"], "path": ours["path"],
        "key": ours["idempotency_key"], "body": ours["json_body"],
    }


#: Wires both validators must judge alike, word for word.
_VERDICT_CASES = [
    GOOD,
    BATCH[1],
    {**GOOD, "outcome": "delighted"},
    {**GOOD, "outcome": 7},
    {k: v for k, v in GOOD.items() if k != "outcome"},
    {k: v for k, v in GOOD.items() if k != "turn_id"},
    {**GOOD, "turn_id": "../x"},
    {**GOOD, "conversation_id": " c"},
    {**GOOD, "value": "12"},
    {**GOOD, "value": True},
    {**GOOD, "note": ""},
    {**GOOD, "note": "\U0001f4e6" * 1000},
    {**GOOD, "note": "x" * 1001},
    {**GOOD, "occurred_at": "2026-11-10T09:00:00"},
    {**GOOD, "occurred_at": "2026-02-30T09:00:00Z"},
    {**GOOD, "occurred_at": "2024-02-29T09:00:00Z"},
    {**GOOD, "occurred_at": "1900-02-29T09:00:00Z"},
    {**GOOD, "occurred_at": "2000-02-29T23:59:59.999-23:59"},
    {**GOOD, "occurred_at": "2026-11-10T24:00:00Z"},
    {**GOOD, "occurred_at": "2026-11-10T23:59:60Z"},
    {**GOOD, "occurred_at": "2026-11-10T09:00:00+24:00"},
    {**GOOD, "occurred_at": "0000-01-01T00:00:00Z"},
    {**GOOD, "occurred_at": "\u0662026-11-10T09:00:00Z"},
    {**GOOD, "value": 10 ** 400},
    {**GOOD, "value": 1e308},
    {k: v for k, v in GOOD.items() if k != "occurred_at"},
    {**GOOD, "outcome_id": "a/b"},
    {**GOOD, "zeta": 1, "extra": 1},
    ["not", "an", "object"],
    None,
]


def _verdict(wire):
    try:
        return {"ok": production_outcome(wire)}
    except InvalidOutcome as exc:
        return {"refused": str(exc)}


def test_both_sdks_validate_an_outcome_alike_word_for_word():
    """Same vocabulary, same rules, same refusal text: a customer who moves
    between the two packages meets one contract."""
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import {productionOutcome, InvalidOutcome, PRODUCTION_OUTCOMES} "
        "from './src/wire.js';\n"
        "let raw='';for await (const c of process.stdin) raw+=c;\n"
        "const out=JSON.parse(raw).map((w)=>{try{return {ok:productionOutcome(w)};}"
        "catch(e){if(!(e instanceof InvalidOutcome)) throw e;"
        "return {refused:e.message};}});\n"
        "console.log(JSON.stringify({vocabulary:PRODUCTION_OUTCOMES,out}));\n"
    )
    theirs = json.loads(run_node(
        sdk, node, script, stdin=json.dumps(_VERDICT_CASES)))
    assert tuple(theirs["vocabulary"]) == PRODUCTION_OUTCOMES
    assert theirs["out"] == [_verdict(wire) for wire in _VERDICT_CASES]
    refused = [v for v in theirs["out"] if "refused" in v]
    assert len(refused) == len(_VERDICT_CASES) - 6, "the table is not vacuous"


def test_an_outcome_the_node_sdk_queued_is_one_this_sdk_uploads(
    monkeypatch, tmp_path, quiet,
):
    """One on-disk queue, two writers: the same file name for the same id and
    a record either package reads, so a harness migration strands nothing."""
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import {createTurnUploader} from './src/capture.js';\n"
        "const u=createTurnUploader({transport:{reportOutcomes:async()=>{"
        "throw new Error('down');},reportTurns:async()=>({})},agentId:'support',"
        f"root:{json.dumps(str(tmp_path))}}});\n"
        "u.setVerdict({mayUpload:true});\n"
        "const id=u.reportOutcome('escalated',{turnId:'run-1',"
        "outcomeId:'ticket-9',note:'ma\\u00f1ana',value:2,"
        "occurredAt:new Date('2026-11-10T09:00:00Z')});\n"
        "await new Promise((r)=>setTimeout(r,50));\n"
        "console.log(id);\n"
    )
    assert run_node(sdk, node, script).strip() == "ticket-9"
    assert _pending(tmp_path) == [production._outcome_file(tmp_path, "ticket-9")]

    tr = _Outcomes()
    with _attached(monkeypatch, tmp_path, tr) as px:
        # The same report from this side lands on the same file.
        px.report_outcome(
            "escalated", turn_id="run-1", outcome_id="ticket-9",
            note="mañana", value=2,
            occurred_at=datetime(2026, 11, 10, 9, tzinfo=timezone.utc))
        assert len(_pending(tmp_path)) == 1
        px.flush()
    assert tr.outcomes == [{
        "outcome_id": "ticket-9", "outcome": "escalated", "turn_id": "run-1",
        "note": "mañana", "value": 2,
        "occurred_at": "2026-11-10T09:00:00.000Z",
    }]


#: The parity table: what the control plane answers each batch with. A dict of
#: refused ids -> (status, detail), the first of them in a batch deciding its
#: answer; ``"*"`` refuses every batch; ``">N"`` refuses every batch of more
#: than N records (a proxy's size limit).
REFUSAL_TABLE = {
    "one record, 422": {"o5": (422, "refused")},
    "one record, 400": {"o0": (400, "refused")},
    "one record, 409": {"o3": (409, "refused")},
    "two records, two causes": {"o1": (422, "o1: value must be finite"),
                                "o6": (409, "o6: replayed")},
    "the whole batch, 422": {"*": (422, "contract minor mismatch")},
    "a proxy's size limit, 413, cleared by quarters": {
        ">3": (413, "Request Entity Too Large")},
    "every request, 413, so each record alone": {
        "*": (413, "Request Entity Too Large")},
    "the route, 404": {"*": (404, "Not Found")},
    "the route, 401": {"*": (401, "invalid key")},
    "the route, 410": {"*": (410, "gone")},
}


def _table_answer(table, ids):
    if "*" in table:
        return table["*"]
    for key, answer in table.items():
        if key.startswith(">"):
            return answer if len(ids) > int(key[1:]) else None
    for refused_id, answer in table.items():
        if refused_id in ids:
            return answer
    return None


@pytest.mark.parametrize("row", list(REFUSAL_TABLE))
def test_both_sdks_isolate_a_refused_outcome_alike(
    monkeypatch, tmp_path, quiet, row,
):
    """The same queue on disk, the same answers from the control plane: both
    packages try the same batches in the same order, send the same outcomes,
    drop the same ones, keep the same ones and count alike.

    BOTH THROUGH THEIR REAL HTTP CLIENTS, answered in-process: a stub that
    raises a ready-made error skips the one place the two packages differed
    (the Python client raised a 410 with no status)."""
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    table = REFUSAL_TABLE[row]
    ids = [f"o{n}" for n in range(11)]

    def seed(root):
        root.mkdir(parents=True, exist_ok=True)
        for outcome_id in ids:
            production._outcome_file(root, outcome_id).write_text(json.dumps(
                {"outcome_id": outcome_id, "outcome": "resolved",
                 "turn_id": "t1", "occurred_at": "2026-11-10T09:00:00Z"}),
                encoding="utf-8")

    tried: list[list[str]] = []
    sent: list[str] = []

    def outcomes_route(request):
        batch = [o["outcome_id"]
                 for o in json.loads(request.content)["outcomes"]]
        tried.append(batch)
        refused = _table_answer(table, batch)
        if refused is not None:
            return refused[0], {"detail": refused[1]}
        sent.extend(batch)
        return 200, {"outcomes": [
            {"outcome_id": i, "accepted": True} for i in batch]}

    seed(tmp_path / "py")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _over_http(monkeypatch, tmp_path / "py", {
            **GRANTED, "POST /outcomes": outcomes_route,
        }) as (px, _seen):
            counts = px.flush()
            # Copied now: closing the attachment flushes again.
            ours = {
                "tried": [list(batch) for batch in tried],
                "sent": sorted(sent),
                "left": len(_pending(tmp_path / "py")),
                "counts": [counts["outcomes_sent"], counts["outcomes_failed"],
                           counts["outcomes_dropped"]],
            }

    seed(tmp_path / "node")
    script = (
        "import {createTurnUploader} from './src/capture.js';\n"
        "import {ControlPlaneClient} from './src/http.js';\n"
        "import {AttachTransport} from './src/transport.js';\n"
        "import {readdirSync} from 'node:fs';\n"
        f"const table={json.dumps(table)};\n"
        f"const root={json.dumps(str(tmp_path / 'node'))};\n"
        "const tried=[], sent=[];\n"
        "const answer=(ids)=>{if(table['*'])return table['*'];"
        "for(const [k,a] of Object.entries(table))"
        "if(k.startsWith('>'))return ids.length>Number(k.slice(1))?a:null;"
        "for(const [id,a] of Object.entries(table))"
        "if(ids.includes(id))return a;return null;};\n"
        "const reply=(status,body)=>new Response(JSON.stringify(body),"
        "{status,headers:{'content-type':'application/json'}});\n"
        "const fetchImpl=async(url,init)=>{"
        "if(!String(url).endsWith('/outcomes'))return reply(200,{});"
        "const ids=JSON.parse(init.body).outcomes.map((o)=>o.outcome_id);"
        "tried.push(ids);const a=answer(ids);"
        "if(a)return reply(a[0],{detail:a[1]});"
        "sent.push(...ids);return reply(200,{outcomes:ids.map((id)=>("
        "{outcome_id:id,accepted:true}))});};\n"
        "const client=new ControlPlaneClient('https://cp.invalid','pek_x',"
        "{fetchImpl});\n"
        "const transport=new AttachTransport(client,{agentId:'a1'});\n"
        "const u=createTurnUploader({transport,agentId:'a1',root});\n"
        "u.setVerdict({mayUpload:true});\n"
        "const c=await u.flush();\n"
        "const left=readdirSync(root).filter((n)=>n.startsWith('.outcome.')).length;\n"
        "console.log(JSON.stringify({tried,sent:sent.sort(),left,"
        "counts:[c.outcomesSent,c.outcomesFailed,c.outcomesDropped]}));\n"
    )
    theirs = json.loads(run_node(sdk, node, script))
    assert theirs == ours
    assert ours["tried"], "the table is not vacuous"
    if row.startswith("a proxy's size limit"):
        assert ours["sent"] == sorted(ids) and ours["left"] == 0, ours


def test_the_table_reaches_every_way_a_flush_ends():
    """A table that drifted to one shape would still be green."""
    kinds = {next(iter(t)) == "*" for t in REFUSAL_TABLE.values()}
    statuses = {a[0] for t in REFUSAL_TABLE.values() for a in t.values()}
    assert kinds == {True, False}
    assert {410, 404, 401, 422, 413, 409, 400} <= statuses


def test_both_sdks_bound_the_outcome_queue_alike():
    """The two bounds the flush applies are one figure in both packages."""
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import {OUTCOME_MAX_KEPT_UNANSWERED, OUTCOME_SPLIT_MAX_REQUESTS,"
        "OUTCOME_MAX_AGE_MS} from './src/capture.js';\n"
        "import {CAPTURE_READ_REFUSALS, captureReadRefusal}"
        " from './src/transport.js';\n"
        "console.log(JSON.stringify([OUTCOME_MAX_KEPT_UNANSWERED,"
        "OUTCOME_SPLIT_MAX_REQUESTS, OUTCOME_MAX_AGE_MS,"
        "[...CAPTURE_READ_REFUSALS].sort(), captureReadRefusal('x')]));\n"
    )
    from agent_flywheel import transport

    assert json.loads(run_node(sdk, node, script)) == [
        production._OUTCOME_MAX_KEPT_UNANSWERED,
        production._OUTCOME_SPLIT_MAX_REQUESTS,
        production._OUTCOME_MAX_AGE_S * 1000,
        sorted(transport.CAPTURE_READ_REFUSALS),
        transport.capture_read_refusal("x"),
    ]


class _BareRefusing(_Refusing):
    """Refuses with the control plane's words alone: the status rides only
    on the error, as a client that does not spell it into the message."""

    def report_outcomes(self, agent_id, outcomes):
        ids = [o["outcome_id"] for o in outcomes]
        refused = self.answer(ids)
        if refused is not None:
            self.tried.append(ids)
            status, detail = refused
            raise TransportError(detail, status_code=status)
        return super().report_outcomes(agent_id, outcomes)


def test_the_same_words_at_two_statuses_are_no_refusal_of_the_batch(
    monkeypatch, tmp_path, quiet,
):
    """Whole-batch means the same STATUS and the same words, in unison with
    the Node SDK: halves refused in the same words at 400 and 422 are two
    answers, so the split goes on. Mutation: compare the words alone ->
    stopped as a whole-batch refusal."""
    ids = [f"o{n}" for n in range(8)]
    # The two halves of a split are tried one after the other, so
    # alternating answers refuse every pair of halves at two statuses.
    calls = iter(range(10_000))
    tr = _BareRefusing(lambda batch: (
        (400 if next(calls) % 2 else 422, "refused")))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, *ids)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
    assert not any("whole batches" in str(w.message) for w in caught)
    assert counts["outcomes_dropped"] == len(ids)
    assert len(tr.tried) > 3, "split on, not stopped at the halves"


def test_a_batch_of_one_refused_is_kept_not_dropped(
    monkeypatch, tmp_path, quiet,
):
    """Every ``report_outcome`` flushes, so its outcome usually travels
    alone. A 4xx on a batch of one that was never split cannot be told from a
    refusal of the batch -- a control plane on another contract minor refuses
    every batch alike -- and reading it as the record's dropped each outcome
    as it was reported. It stays on disk, said once, and goes with the next."""
    tr = _Outcomes(outcomes_raise=TransportError(
        "POST /outcomes -> HTTP 422: unknown field", status_code=422))
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (0, 1, 0)
        assert len(_pending(tmp_path)) == 1
        assert any("batch of one" in str(w.message) for w in caught)
        tr.outcomes_raise = None
        assert px.flush()["outcomes_sent"] == 1
