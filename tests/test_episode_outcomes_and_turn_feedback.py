"""An outcome can name the episode it is the outcome of, and one turn of an
episode can be given feedback.

``report_outcome`` joined a turn or a conversation; an episode the platform
ran had no way to be told how it turned out. ``episode_id`` is that join, and
``report_turn_feedback`` says what the user did with one turn of it. These
tests pin what makes both safe in a customer's own code:

* THE WIRE IS THE PUBLISHED ONE, in both SDKs, byte for byte.
* NEITHER IS ABOUT END USERS' TRAFFIC, so neither waits for production
  capture consent -- and a turn's or a conversation's outcome still does.
* A CONTROL PLANE THAT PREDATES THEM never reaches the caller: it is said
  once and the process stops sending.
"""
from __future__ import annotations

import json
import re
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import agent_flywheel as fw
from agent_flywheel import production
from agent_flywheel.contract import (
    _TURN_FEEDBACK_BATCH_MAX_ITEMS,
    _TURN_FEEDBACK_TURN_INDEX_MAX,
    TURN_FEEDBACK_SIGNALS,
    names_only_an_episode,
    production_outcome,
    turn_feedback,
)
from agent_flywheel.errors import (
    ContractViolation,
    InvalidFeedback,
    InvalidOutcome,
    TransportError,
)

from .test_production_turn_upload import _attached
from .test_report_outcome import (
    GRANTED,
    SCHEMA,
    _Client,
    _Outcomes,
    _over_http,
    _pending,
    _Unanswered,
    _validator,
)

EPISODE = "run-7:rollout-0003"
AT = "2026-11-17T09:00:00Z"


class _Feedback(_Outcomes):
    """The outcome fake, plus the feedback route."""

    def __init__(self, *, feedback_raise=None, **kw):
        super().__init__(**kw)
        self.feedback_batches: list[list[dict]] = []
        self.feedback_raise = feedback_raise
        self.feedback_tried = 0

    def report_turn_feedback(self, agent_id, feedback):
        self.feedback_tried += 1
        if self.feedback_raise is not None:
            raise self.feedback_raise
        self.feedback_batches.append([dict(f) for f in feedback])
        return {"agent_id": agent_id, "feedback": [
            {"feedback_id": f["feedback_id"], "accepted": True,
             "duplicate": False} for f in feedback]}

    @property
    def feedback(self):
        return [f for batch in self.feedback_batches for f in batch]


def _waiting(root) -> list[Path]:
    return sorted(
        p for p in Path(root).iterdir() if p.name.startswith(".feedback."))


def _said(caught, *words) -> list[str]:
    return [str(w.message) for w in caught
            if all(word in str(w.message) for word in words)]


# ── an outcome joined to an episode ────────────────────────────────────────


def test_an_outcome_carries_the_episode_it_is_the_outcome_of(
    monkeypatch, tmp_path, quiet,
):
    tr = _Feedback()
    with _attached(monkeypatch, tmp_path, tr) as px:
        outcome_id = px.report_outcome(
            "resolved", episode_id=EPISODE, turn_id="t1", occurred_at=AT)
        px.flush()
    assert tr.outcomes == [{
        "outcome_id": outcome_id, "outcome": "resolved", "turn_id": "t1",
        "episode_id": EPISODE, "occurred_at": AT,
    }]


def test_an_outcome_without_an_episode_is_the_wire_it_always_was(
    monkeypatch, tmp_path, quiet,
):
    tr = _Feedback()
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1", occurred_at=AT)
        px.flush()
    (sent,) = tr.outcomes
    assert sorted(sent) == ["occurred_at", "outcome", "outcome_id", "turn_id"]


@pytest.mark.parametrize("episode_id", [" padded", "e" * 256, 7, ""])
def test_an_episode_id_the_wire_cannot_carry_is_refused_by_name(
    monkeypatch, tmp_path, quiet, episode_id,
):
    with _attached(monkeypatch, tmp_path, _Feedback()) as px:
        with pytest.raises(InvalidOutcome, match="episode_id"):
            px.report_outcome("resolved", episode_id=episode_id)
        assert _pending(tmp_path) == []


def test_an_outcome_of_an_episode_alone_needs_no_capture_consent(
    monkeypatch, tmp_path, quiet,
):
    """It says nothing about the agent's end users' traffic. A turn's outcome
    still does, and is still the no-op."""
    tr = _Feedback(capture=False, reason="opt-in per agent")
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is False
        assert px.report_outcome("resolved", turn_id="t1") is None
        assert px.report_outcome(
            "resolved", turn_id="t1", episode_id=EPISODE) is None, (
            "an outcome that also names a turn is about that turn")
        outcome_id = px.report_outcome("resolved", episode_id=EPISODE)
        assert outcome_id and len(_pending(tmp_path)) == 1
        counts = px.flush()
        assert counts["outcomes_sent"] == 1
        assert _pending(tmp_path) == []
        assert px.may_upload is False, "sending it grants nothing"
    assert [o["outcome_id"] for o in tr.outcomes] == [outcome_id]


def test_only_the_episode_outcomes_go_while_consent_is_unknown(
    monkeypatch, tmp_path, quiet,
):
    """A turn's outcome kept while the answer is unknown must not ride along
    with the episode's."""
    tr = _Unanswered()
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is False
        kept = px.report_outcome("resolved", turn_id="t1")
        sent = px.report_outcome("escalated", episode_id=EPISODE)
        assert kept and sent
        px.flush()
        assert [o["outcome_id"] for o in tr.outcomes] == [sent]
        (still,) = _pending(tmp_path)
        assert json.loads(still.read_text())["outcome_id"] == kept


def test_a_control_plane_that_predates_the_join_is_said_once_and_kept(
    monkeypatch, tmp_path, quiet,
):
    """An older control plane answers an episode-only batch with its consent
    403. That is not this agent's capture being switched off: the refusal the
    customer reads stays what it was, the outcome stays on disk, and no more
    are queued on those terms."""
    tr = _Feedback(capture=False, reason="opt-in per agent",
                   outcomes_raise=TransportError(
                       "POST /outcomes -> HTTP 403: production capture is "
                       "opt-in per agent", status_code=403))
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.report_outcome("resolved", episode_id=EPISODE)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
            px.flush()
        assert counts["outcomes_failed"] == 1
        assert len(_said(caught, "reported for an episode", "predate")) == 1
        assert not _said(caught, "captured turns"), (
            "the turns' refusal is not what happened")
        assert px.upload_refusal == "opt-in per agent"
        assert len(_pending(tmp_path)) == 1
        assert px.report_outcome("reopened", episode_id=EPISODE) is None
        assert len(_pending(tmp_path)) == 1
        tr.outcomes_raise = None
        px.flush()
    assert tr.outcomes == [], "nothing is sent on those terms again"


def test_names_only_an_episode_is_about_the_two_other_joins():
    base = {"outcome_id": "o1", "outcome": "resolved", "occurred_at": AT}
    assert names_only_an_episode({**base, "episode_id": EPISODE})
    assert not names_only_an_episode(
        {**base, "episode_id": EPISODE, "turn_id": "t1"})
    assert not names_only_an_episode(
        {**base, "episode_id": EPISODE, "conversation_id": "c1"})
    assert not names_only_an_episode({**base, "turn_id": "t1"})


# ── per-turn feedback ──────────────────────────────────────────────────────


def test_feedback_is_sent_joined_to_the_episode_and_the_turn(
    monkeypatch, tmp_path, quiet,
):
    tr = _Feedback()
    with _attached(monkeypatch, tmp_path, tr) as px:
        feedback_id = px.report_turn_feedback(
            "corrected", episode_id=EPISODE, turn_index=3,
            occurred_at=datetime(2026, 11, 17, 10, 0, tzinfo=timezone(
                timedelta(hours=1))))
        assert feedback_id and feedback_id.startswith("f_")
        assert quiet == [1], "queued feedback is uploaded off-thread at once"
        assert len(_waiting(tmp_path)) == 1
        counts = px.flush()
    assert (counts["feedback_sent"], counts["feedback_failed"],
            counts["feedback_dropped"]) == (1, 0, 0)
    assert tr.feedback == [{
        "feedback_id": feedback_id, "episode_id": EPISODE, "turn_index": 3,
        "signal": "corrected", "occurred_at": "2026-11-17T09:00:00.000Z",
    }]
    assert _waiting(tmp_path) == [], "acknowledged feedback is not resent"
    assert tr.outcomes == [], "feedback is not an outcome"


def test_every_word_of_the_feedback_vocabulary_is_accepted(
    monkeypatch, tmp_path, quiet,
):
    tr = _Feedback()
    with _attached(monkeypatch, tmp_path, tr) as px:
        for index, word in enumerate(TURN_FEEDBACK_SIGNALS):
            assert px.report_turn_feedback(
                word, episode_id=EPISODE, turn_index=index)
        px.flush()
    assert sorted(f["signal"] for f in tr.feedback) == sorted((
        "accepted", "corrected", "repeated", "escalated", "abandoned"))


def test_a_word_outside_the_feedback_vocabulary_names_the_allowed_ones(
    monkeypatch, tmp_path, quiet,
):
    with _attached(monkeypatch, tmp_path, _Feedback()) as px:
        with pytest.raises(InvalidFeedback) as raised:
            px.report_turn_feedback("liked", episode_id=EPISODE, turn_index=0)
        assert str(raised.value) == (
            "signal must be one of accepted, corrected, repeated, escalated, "
            'abandoned; got "liked"')
        assert isinstance(raised.value, ContractViolation)
        assert not isinstance(raised.value, InvalidOutcome)
        assert _waiting(tmp_path) == []
    assert fw.InvalidFeedback is InvalidFeedback
    assert fw.TURN_FEEDBACK_SIGNALS is TURN_FEEDBACK_SIGNALS


@pytest.mark.parametrize(("kwargs", "problem"), [
    ({"episode_id": None, "turn_index": 0}, "episode_id"),
    ({"episode_id": " padded", "turn_index": 0}, "episode_id"),
    ({"episode_id": EPISODE, "turn_index": -1}, "whole number from 0"),
    ({"episode_id": EPISODE, "turn_index": 1.5}, "whole number from 0"),
    ({"episode_id": EPISODE, "turn_index": True}, "whole number from 0"),
    ({"episode_id": EPISODE, "turn_index": "3"}, "whole number from 0"),
    ({"episode_id": EPISODE, "turn_index": None}, "whole number from 0"),
    ({"episode_id": EPISODE,
      "turn_index": _TURN_FEEDBACK_TURN_INDEX_MAX + 1}, "100000"),
    ({"episode_id": EPISODE, "turn_index": 0, "feedback_id": "a/b"},
     "feedback_id"),
    ({"episode_id": EPISODE, "turn_index": 0,
      "occurred_at": datetime(2026, 11, 17, 9)}, "timezone"),
    ({"episode_id": EPISODE, "turn_index": 0, "occurred_at": "yesterday"},
     "RFC 3339"),
])
def test_feedback_the_wire_cannot_carry_is_refused_by_name(
    monkeypatch, tmp_path, quiet, kwargs, problem,
):
    tr = _Feedback()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with pytest.raises(InvalidFeedback, match=problem):
            px.report_turn_feedback("accepted", **kwargs)
        assert _waiting(tmp_path) == []
        px.flush()
    assert tr.feedback == []


def test_feedback_is_validated_with_nowhere_to_send_it(
    monkeypatch, tmp_path, quiet,
):
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    try:
        assert px.report_turn_feedback(
            "accepted", episode_id=EPISODE, turn_index=0) is None
        assert _waiting(tmp_path) == []
        with pytest.raises(InvalidFeedback):
            px.report_turn_feedback("liked", episode_id=EPISODE, turn_index=0)
    finally:
        px.close()


def test_feedback_needs_no_capture_consent(monkeypatch, tmp_path, quiet):
    tr = _Feedback(capture=False, reason="opt-in per agent")
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is False
        feedback_id = px.report_turn_feedback(
            "accepted", episode_id=EPISODE, turn_index=0)
        px.flush()
        assert px.may_upload is False
    assert [f["feedback_id"] for f in tr.feedback] == [feedback_id]


def test_a_callers_own_feedback_id_is_resent_unchanged_after_a_blip(
    monkeypatch, tmp_path, quiet,
):
    tr = _Feedback(feedback_raise=TransportError("connection reset"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.report_turn_feedback(
            "repeated", episode_id=EPISODE, turn_index=2,
            feedback_id="fb-1") == "fb-1"
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush()
        assert (counts["feedback_sent"], counts["feedback_failed"],
                counts["feedback_dropped"]) == (0, 1, 0)
        assert len(_waiting(tmp_path)) == 1, "a blip keeps it"
        assert not _said(caught, "per-turn feedback"), "a blip is not a no"
        tr.feedback_raise = None
        assert px.flush()["feedback_sent"] == 1
    assert [f["feedback_id"] for f in tr.feedback] == ["fb-1"]


@pytest.mark.parametrize("status", [408, 429, 500, 503])
def test_a_blip_with_a_status_keeps_the_feedback(
    monkeypatch, tmp_path, quiet, status,
):
    tr = _Feedback(feedback_raise=TransportError(
        f"POST /turn-feedback -> HTTP {status}: later", status_code=status))
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_turn_feedback("accepted", episode_id=EPISODE, turn_index=0)
        px.flush()
        assert len(_waiting(tmp_path)) == 1
        assert px.report_turn_feedback(
            "accepted", episode_id=EPISODE, turn_index=1) is not None
        tr.feedback_raise = None


@pytest.mark.parametrize("status", [404, 405])
def test_a_control_plane_with_no_feedback_route_never_reaches_the_caller(
    monkeypatch, tmp_path, quiet, status,
):
    """THROUGH THE REAL HTTP CLIENT. A control plane that predates the route
    answers 404 (405 where another method has the path); nothing raises into
    the customer's loop, it is said once, what was waiting is dropped, and
    the process stops asking."""
    total = _TURN_FEEDBACK_BATCH_MAX_ITEMS + 3
    with warnings.catch_warnings(record=True) as caught, _over_http(
        monkeypatch, tmp_path,
        {**GRANTED, "POST /turn-feedback": (status, {"detail": "Not Found"})},
    ) as (px, seen):
        warnings.simplefilter("always")
        for index in range(total):
            assert px.report_turn_feedback(
                "accepted", episode_id=EPISODE, turn_index=index)
        counts = px.flush()
        assert (counts["feedback_sent"], counts["feedback_failed"],
                counts["feedback_dropped"]) == (0, 0, total)
        assert _waiting(tmp_path) == []
        assert seen.count("POST /turn-feedback") == 1, "one try, then it stops"
        assert px.report_turn_feedback(
            "accepted", episode_id=EPISODE, turn_index=0) is None
        assert _waiting(tmp_path) == [], "nothing more is kept"
        px.flush()
        assert seen.count("POST /turn-feedback") == 1
        assert len(_said(caught, "per-turn feedback", "predate")) == 1
        # Still validated: a typo is a typo against any control plane.
        with pytest.raises(InvalidFeedback):
            px.report_turn_feedback("liked", episode_id=EPISODE, turn_index=0)
        assert px.may_upload is True, "not a consent refusal"


def test_a_switched_off_intake_keeps_every_report_for_later(
    monkeypatch, tmp_path, quiet,
):
    """A control plane whose episode intake is switched off answers 503 with
    a Retry-After: not a missing route (which would drop what is waiting),
    not a refusal of the key (which would stop sending). THROUGH THE REAL
    HTTP CLIENT: the request is retried as the header says, both kinds of
    report stay on disk, nothing is said as a no, and they go once it is on."""
    off = (503, {"detail": "episode reports are switched off"},
           {"Retry-After": "0"})
    routes = {**GRANTED, "POST /turn-feedback": off, "POST /outcomes": off}
    with warnings.catch_warnings(record=True) as caught, _over_http(
        monkeypatch, tmp_path, routes,
    ) as (px, seen):
        warnings.simplefilter("always")
        px.report_turn_feedback("accepted", episode_id=EPISODE, turn_index=0)
        px.report_outcome("resolved", episode_id=EPISODE)
        px.flush()
        assert len(_waiting(tmp_path)) == 1, "the feedback is kept"
        assert len(px._outcome_files()) == 1, "the outcome is kept"
        assert seen.count("POST /turn-feedback") > 1, "retried as told"
        assert seen.count("POST /outcomes") > 1, "retried as told"
        assert not _said(caught, "predate"), "not a missing route"
        assert not _said(caught, "refused"), "not a refusal"
        assert px.report_turn_feedback(
            "accepted", episode_id=EPISODE, turn_index=1) is not None
        routes["POST /turn-feedback"] = (200, {})
        routes["POST /outcomes"] = (200, {})
        counts = px.flush()
        assert counts["feedback_sent"] == 2
        assert counts["outcomes_sent"] == 1
        assert _waiting(tmp_path) == [] and px._outcome_files() == []


def test_a_403_keeps_the_feedback_and_stops_sending(
    monkeypatch, tmp_path, quiet,
):
    """A 403 is the control plane's decision about THIS KEY (it lacks the
    scope, it is bound elsewhere), not a missing route. It was read as "the
    route is not there" and the whole queue was deleted for it. The outcomes'
    rule: sending stops for the process, said once, and what is waiting stays
    on disk for the process that holds a key the route takes."""
    total = _TURN_FEEDBACK_BATCH_MAX_ITEMS + 3
    with warnings.catch_warnings(record=True) as caught, _over_http(
        monkeypatch, tmp_path,
        {**GRANTED, "POST /turn-feedback": (
            403, {"detail": "this key lacks the turn.report scope"})},
    ) as (px, seen):
        warnings.simplefilter("always")
        for index in range(total):
            assert px.report_turn_feedback(
                "accepted", episode_id=EPISODE, turn_index=index)
        counts = px.flush()
        assert (counts["feedback_sent"], counts["feedback_failed"],
                counts["feedback_dropped"]) == (0, total, 0)
        assert len(_waiting(tmp_path)) == total, "a 403 deletes nothing"
        assert seen.count("POST /turn-feedback") == 1, "one try, then it stops"
        assert px.report_turn_feedback(
            "accepted", episode_id=EPISODE, turn_index=0) is None
        px.flush()
        assert seen.count("POST /turn-feedback") == 1
        assert len(_waiting(tmp_path)) == total
        said = _said(caught, "refused per-turn feedback")
        assert len(said) == 1 and "turn.report scope" in said[0]
        assert not _said(caught, "predate"), "not the absent route's words"
        assert px.may_upload is True, "feedback is not the consent question"


@pytest.mark.parametrize("status", [401, 410])
def test_a_refusal_of_the_key_keeps_the_feedback_for_a_later_flush(
    monkeypatch, tmp_path, quiet, status,
):
    """401 and 410 are about the key or the resource, as on ``/outcomes``:
    sending stops for THIS FLUSH, nothing is dropped, and a later flush
    tries again."""
    total = _TURN_FEEDBACK_BATCH_MAX_ITEMS + 3
    with warnings.catch_warnings(record=True) as caught, _over_http(
        monkeypatch, tmp_path,
        {**GRANTED, "POST /turn-feedback": (status, {"detail": "no"})},
    ) as (px, seen):
        warnings.simplefilter("always")
        for index in range(total):
            px.report_turn_feedback(
                "accepted", episode_id=EPISODE, turn_index=index)
        counts = px.flush()
        assert (counts["feedback_sent"], counts["feedback_failed"],
                counts["feedback_dropped"]) == (0, total, 0)
        assert len(_waiting(tmp_path)) == total
        assert seen.count("POST /turn-feedback") == 1, "not batch after batch"
        assert _said(caught, f"HTTP {status}", "did not take")
        assert px.report_turn_feedback(
            "accepted", episode_id=EPISODE, turn_index=0) is not None
        px.flush()
        assert seen.count("POST /turn-feedback") == 2, "asked again later"


@pytest.mark.parametrize("status", [400, 409, 413, 422])
def test_one_refused_feedback_report_never_holds_back_the_rest(
    monkeypatch, tmp_path, quiet, status,
):
    """A 4xx about the records is final for THAT record: the batch is split
    as an outcome batch is, the good reports are sent and only the refused
    one is dropped. It used to delete every report waiting and switch
    feedback off for the process."""
    ids = [f"fb{n}" for n in range(9)]
    sent: list[str] = []

    def route(request):
        got = [f["feedback_id"]
               for f in json.loads(request.content)["feedback"]]
        if "fb5" in got:
            return status, {"detail": "turn_index: not this one"}
        sent.extend(got)
        return 200, {"agent_id": "support", "feedback": [
            {"feedback_id": i, "accepted": True, "duplicate": False}
            for i in got]}

    with warnings.catch_warnings(record=True) as caught, _over_http(
        monkeypatch, tmp_path, {**GRANTED, "POST /turn-feedback": route},
    ) as (px, _seen):
        warnings.simplefilter("always")
        for index, feedback_id in enumerate(ids):
            px.report_turn_feedback(
                "accepted", episode_id=EPISODE, turn_index=index,
                feedback_id=feedback_id)
        counts = px.flush()
        assert (counts["feedback_sent"], counts["feedback_failed"],
                counts["feedback_dropped"]) == (8, 0, 1)
        assert _waiting(tmp_path) == []
        assert _said(caught, f"HTTP {status}", "per-turn feedback report(s)")
        assert sorted(sent) == sorted(set(ids) - {"fb5"})
        assert px.report_turn_feedback(
            "accepted", episode_id=EPISODE, turn_index=0) is not None, (
                "one bad report does not switch feedback off")


def test_a_feedback_report_refused_by_name_is_said_by_its_id_and_not_resent(
    monkeypatch, tmp_path, quiet,
):
    """A 200 can still refuse a report (``accepted: false`` and why). The
    reply was never read: the report counted as sent and nothing said it was
    not recorded."""
    def route(request):
        got = json.loads(request.content)["feedback"]
        return 200, {"agent_id": "support", "feedback": [
            {"feedback_id": f["feedback_id"], "accepted": False,
             "duplicate": False, "error": "episode_not_found",
             "detail": "no such episode"}
            if f["feedback_id"] == "fb-lost" else
            {"feedback_id": f["feedback_id"], "accepted": True,
             "duplicate": False} for f in got]}

    with warnings.catch_warnings(record=True) as caught, _over_http(
        monkeypatch, tmp_path, {**GRANTED, "POST /turn-feedback": route},
    ) as (px, seen):
        warnings.simplefilter("always")
        for feedback_id in ("fb-kept", "fb-lost"):
            px.report_turn_feedback(
                "accepted", episode_id=EPISODE, turn_index=0,
                feedback_id=feedback_id)
        counts = px.flush()
        assert (counts["feedback_sent"], counts["feedback_failed"],
                counts["feedback_dropped"]) == (1, 0, 1)
        said = _said(caught, "did not record")
        assert len(said) == 1, "said once"
        assert "fb-lost (episode_not_found)" in said[0]
        assert "fb-kept" not in said[0]
        assert _waiting(tmp_path) == []
        px.flush()
        assert seen.count("POST /turn-feedback") == 1, "not sent again"


def test_feedback_undelivered_for_a_week_is_dropped(
    monkeypatch, tmp_path, quiet,
):
    import os
    import time

    tr = _Feedback(feedback_raise=TransportError("connection reset"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_turn_feedback("accepted", episode_id=EPISODE, turn_index=0)
        (path,) = _waiting(tmp_path)
        old = time.time() - production._OUTCOME_MAX_AGE_S - 60
        os.utime(path, (old, old))
        assert px.flush()["feedback_dropped"] == 1
        assert _waiting(tmp_path) == []
        assert tr.feedback_tried == 0
        tr.feedback_raise = None


def test_more_feedback_than_a_batch_goes_in_batches(
    monkeypatch, tmp_path, quiet,
):
    total = _TURN_FEEDBACK_BATCH_MAX_ITEMS + 3
    tr = _Feedback()
    with _attached(monkeypatch, tmp_path, tr) as px:
        for index in range(total):
            px.report_turn_feedback(
                "accepted", episode_id=EPISODE, turn_index=index)
        assert px.flush()["feedback_sent"] == total
    assert sorted(len(b) for b in tr.feedback_batches) == [
        3, _TURN_FEEDBACK_BATCH_MAX_ITEMS]


def test_a_flush_with_no_feedback_reports_no_feedback_counts(
    monkeypatch, tmp_path, quiet,
):
    tr = _Feedback()
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert not [k for k in px.flush() if k.startswith("feedback")]
    assert tr.feedback_tried == 0


# ── the schema and the code are one vocabulary ─────────────────────────────


def test_schema_and_code_agree_on_the_feedback_wire():
    feedback = SCHEMA["$defs"]["TurnFeedback"]
    batch = SCHEMA["$defs"]["TurnFeedbackBatch"]
    outcome = SCHEMA["$defs"]["ProductionOutcome"]
    assert tuple(feedback["properties"]["signal"]["enum"]) == (
        TURN_FEEDBACK_SIGNALS)
    assert feedback["properties"]["turn_index"]["maximum"] == (
        _TURN_FEEDBACK_TURN_INDEX_MAX)
    assert batch["properties"]["feedback"]["maxItems"] == (
        _TURN_FEEDBACK_BATCH_MAX_ITEMS)
    assert feedback["additionalProperties"] is False
    assert set(feedback["required"]) == set(feedback["properties"])
    # One episode id, spelled one way on both wires.
    for name in ("pattern", "minLength", "maxLength", "type"):
        assert feedback["properties"]["episode_id"][name] == (
            outcome["properties"]["episode_id"][name])
    assert {"required": ["episode_id"]} in outcome["anyOf"]


FEEDBACK = {"feedback_id": "f1", "episode_id": EPISODE, "turn_index": 2,
            "signal": "accepted", "occurred_at": AT}
EPISODE_OUTCOME = {"outcome_id": "o1", "outcome": "resolved",
                   "episode_id": EPISODE, "occurred_at": AT}

#: Wires both validators, and both SDKs, must judge alike, word for word.
_FEEDBACK_CASES = [
    FEEDBACK,
    {**FEEDBACK, "turn_index": 0, "signal": "abandoned",
     "episode_id": "é \U0001f4e6",
     "occurred_at": "2026-11-17T09:00:00.250+05:30"},
    {**FEEDBACK, "turn_index": 100000},
    {**FEEDBACK, "signal": "liked"},
    {**FEEDBACK, "signal": 7},
    {**FEEDBACK, "signal": "resolved"},
    {k: v for k, v in FEEDBACK.items() if k != "signal"},
    {k: v for k, v in FEEDBACK.items() if k != "episode_id"},
    {k: v for k, v in FEEDBACK.items() if k != "turn_index"},
    {k: v for k, v in FEEDBACK.items() if k != "feedback_id"},
    {k: v for k, v in FEEDBACK.items() if k != "occurred_at"},
    {**FEEDBACK, "episode_id": " e"},
    {**FEEDBACK, "feedback_id": "a/b"},
    {**FEEDBACK, "turn_index": -1},
    {**FEEDBACK, "turn_index": 1.5},
    {**FEEDBACK, "turn_index": "2"},
    {**FEEDBACK, "turn_index": True},
    {**FEEDBACK, "turn_index": 100001},
    {**FEEDBACK, "occurred_at": "2026-11-17T09:00:00"},
    {**FEEDBACK, "zeta": 1, "extra": 1},
    ["not", "an", "object"],
    None,
]
_FEEDBACK_ACCEPTED = 3

_EPISODE_OUTCOME_CASES = [
    EPISODE_OUTCOME,
    {**EPISODE_OUTCOME, "turn_id": "t1", "conversation_id": "c1"},
    {**EPISODE_OUTCOME, "episode_id": " e"},
    {**EPISODE_OUTCOME, "episode_id": "e" * 256},
    {**EPISODE_OUTCOME, "episode_id": 7},
    {k: v for k, v in EPISODE_OUTCOME.items() if k != "episode_id"},
]
_EPISODE_OUTCOME_ACCEPTED = 2


def _judged(validate, refusal, wire):
    try:
        return {"ok": validate(wire)}
    except refusal as exc:
        return {"refused": str(exc)}


@pytest.mark.parametrize("wire", _FEEDBACK_CASES[:_FEEDBACK_ACCEPTED])
def test_the_validator_and_the_schema_accept_the_same_feedback(wire):
    _validator("TurnFeedback").validate(wire)
    assert turn_feedback(wire) == wire


@pytest.mark.parametrize("wire", _FEEDBACK_CASES[_FEEDBACK_ACCEPTED:])
def test_the_validator_and_the_schema_refuse_the_same_feedback(wire):
    assert list(_validator("TurnFeedback").iter_errors(wire))
    with pytest.raises(InvalidFeedback):
        turn_feedback(wire)


@pytest.mark.parametrize(
    "wire", _EPISODE_OUTCOME_CASES[:_EPISODE_OUTCOME_ACCEPTED])
def test_the_validator_and_the_schema_accept_the_same_episode_outcomes(wire):
    _validator("ProductionOutcome").validate(wire)
    assert production_outcome(wire) == wire


@pytest.mark.parametrize(
    "wire", _EPISODE_OUTCOME_CASES[_EPISODE_OUTCOME_ACCEPTED:])
def test_the_validator_and_the_schema_refuse_the_same_episode_outcomes(wire):
    assert list(_validator("ProductionOutcome").iter_errors(wire))
    with pytest.raises(InvalidOutcome):
        production_outcome(wire)


def test_a_whole_number_a_json_reader_cannot_tell_from_an_integer_is_one():
    """``2.0`` is what the Node SDK reads for ``2``; both send ``2``."""
    assert turn_feedback({**FEEDBACK, "turn_index": 2.0})["turn_index"] == 2
    assert isinstance(
        turn_feedback({**FEEDBACK, "turn_index": 2.0})["turn_index"], int)


# ── the wire call itself ───────────────────────────────────────────────────

BATCH = [FEEDBACK, _FEEDBACK_CASES[1]]


def _sent(agent_id="agent-1", feedback=BATCH):
    client = _Client()
    fw.AttachTransport(client, agent_id).report_turn_feedback(
        agent_id, feedback)
    return client.calls[0]


def test_it_posts_the_feedback_to_the_route_the_intake_serves():
    call = _sent()
    assert (call["method"], call["path"]) == ("POST", "/turn-feedback")
    assert call["json_body"] == {"agent_id": "agent-1", "feedback": BATCH}
    _validator("TurnFeedbackBatch").validate(call["json_body"])
    assert call.get("retry_if_ambiguous") is not False


def test_the_feedback_batch_carries_a_stable_idempotency_key():
    key = _sent()["idempotency_key"]
    assert re.fullmatch(r"turn-feedback-agent-1-[0-9a-f]{32}", key), key
    assert _sent()["idempotency_key"] == key
    assert _sent(feedback=[{**FEEDBACK, "signal": "repeated"}])[
        "idempotency_key"] != key


@pytest.mark.parametrize(("agent_id", "feedback", "problem"), [
    ("customer/team", BATCH, "agent_id"),
    ("agent-1", [], "between 1 and 64"),
    ("agent-1", [FEEDBACK] * 65, "between 1 and 64"),
    ("agent-1", [{**FEEDBACK, "signal": "liked"}], "signal must be one of"),
    ("agent-1", [{**FEEDBACK, "extra": 1}], "no field"),
])
def test_the_transport_refuses_feedback_the_published_schema_forbids(
    agent_id, feedback, problem,
):
    client = _Client()
    with pytest.raises(ContractViolation, match=problem):
        fw.AttachTransport(client, agent_id).report_turn_feedback(
            agent_id, feedback)
    assert client.calls == []


# ── the cross-SDK half ─────────────────────────────────────────────────────


def test_both_sdks_send_the_same_feedback_batch_under_the_same_key():
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import {AttachTransport} from './src/transport.js';\n"
        "import {feedbackFile} from './src/capture.js';\n"
        "import {basename} from 'node:path';\n"
        "const calls=[];\n"
        "const tx=new AttachTransport({request:(m,p,o)=>{calls.push([m,p,o]);}},"
        "{agentId:'agent-1'});\n"
        f"tx.reportTurnFeedback('agent-1', {json.dumps(BATCH)});\n"
        "console.log(JSON.stringify({method:calls[0][0],path:calls[0][1],"
        "key:calls[0][2].idempotencyKey,body:calls[0][2].jsonBody,"
        "file:basename(feedbackFile('/r','f \\u00e9'))}));\n"
    )
    theirs = json.loads(run_node(sdk, node, script))
    ours = _sent()
    assert theirs == {
        "method": ours["method"], "path": ours["path"],
        "key": ours["idempotency_key"], "body": ours["json_body"],
        "file": production._feedback_file("/r", "f é").name,
    }


def test_both_sdks_validate_feedback_and_episode_outcomes_alike():
    """Same vocabulary, same rules, same refusal text."""
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import {productionOutcome, turnFeedback, InvalidOutcome, "
        "InvalidFeedback, TURN_FEEDBACK_SIGNALS, namesOnlyAnEpisode} "
        "from './src/wire.js';\n"
        "let raw='';for await (const c of process.stdin) raw+=c;\n"
        "const judge=(f,E)=>(w)=>{try{return {ok:f(w)};}"
        "catch(e){if(!(e instanceof E)) throw e;return {refused:e.message};}};\n"
        "const cases=JSON.parse(raw);\n"
        "console.log(JSON.stringify({vocabulary:TURN_FEEDBACK_SIGNALS,"
        "feedback:cases.feedback.map(judge(turnFeedback,InvalidFeedback)),"
        "outcomes:cases.outcomes.map(judge(productionOutcome,InvalidOutcome)),"
        "only:cases.outcomes.slice(0,2).map(namesOnlyAnEpisode)}));\n"
    )
    theirs = json.loads(run_node(sdk, node, script, stdin=json.dumps({
        "feedback": _FEEDBACK_CASES, "outcomes": _EPISODE_OUTCOME_CASES})))
    assert tuple(theirs["vocabulary"]) == TURN_FEEDBACK_SIGNALS
    assert theirs["feedback"] == [
        _judged(turn_feedback, InvalidFeedback, w) for w in _FEEDBACK_CASES]
    assert theirs["outcomes"] == [
        _judged(production_outcome, InvalidOutcome, w)
        for w in _EPISODE_OUTCOME_CASES]
    assert theirs["only"] == [
        names_only_an_episode(w) for w in _EPISODE_OUTCOME_CASES[:2]
    ] == [True, False]
    assert len([v for v in theirs["feedback"] if "refused" in v]) == (
        len(_FEEDBACK_CASES) - _FEEDBACK_ACCEPTED), "the table is not vacuous"
