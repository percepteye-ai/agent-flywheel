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


@pytest.fixture
def quiet(monkeypatch):
    """No background thread: each test flushes when it means to. Returns the
    list of times a flush was asked for."""
    asked: list[int] = []
    monkeypatch.setattr(
        fw.Attachment, "_flush_soon", lambda self: asked.append(1))
    return asked


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
    ({}, "turn_id or the conversation_id"),
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
    tr = _Outcomes(refuse={"dup"})
    with _attached(monkeypatch, tmp_path, tr) as px:
        px.report_outcome("resolved", turn_id="t1", outcome_id="dup")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            px.flush()
        assert any("outcome_conflict" in str(w.message) for w in caught)
        assert _pending(tmp_path) == []
        px.flush()
    assert len(tr.outcome_batches) == 1


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

    def report_outcomes(self, agent_id, outcomes):
        ids = [o["outcome_id"] for o in outcomes]
        self.tried.append(ids)
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


def test_a_refusal_of_every_record_drops_each_and_sends_nothing(
    monkeypatch, tmp_path, quiet,
):
    tr = _Refusing(lambda batch: (400, "bad body"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _report(px, "a", "b", "c")
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            counts = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (0, 0, 3)
        assert _pending(tmp_path) == []
    assert tr.outcomes == []


@pytest.mark.parametrize("status", [401, 404, 405, 410])
def test_a_refusal_of_the_route_drops_nothing(
    monkeypatch, tmp_path, quiet, status,
):
    """A control plane with no ``/outcomes`` route yet, or a key it does not
    take, refuses every batch for a reason that is not the records': bisecting
    would drop every good outcome. They stay, and the cause is said."""
    tr = _Refusing(lambda batch: (status, "Not Found"))
    with warnings.catch_warnings(record=True) as caught, \
            _attached(monkeypatch, tmp_path, tr) as px:
        warnings.simplefilter("always")
        _report(px, "a", "b", "c")
        counts = px.flush()
        assert (counts["outcomes_sent"], counts["outcomes_failed"],
                counts["outcomes_dropped"]) == (0, 3, 0)
        assert len(_pending(tmp_path)) == 3
        assert len(tr.tried) == 1, "one try, not a bisection"
        assert any(f"HTTP {status}" in str(w.message) for w in caught)
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


@pytest.mark.parametrize(("refused_id", "status"), [
    ("o5", 422), ("o0", 400), (None, 404), (None, 401), ("o3", 409),
])
def test_both_sdks_isolate_a_refused_outcome_alike(
    monkeypatch, tmp_path, quiet, refused_id, status,
):
    """The same queue on disk, the same answers from the control plane: both
    packages try the same batches in the same order, send the same outcomes,
    drop the same ones and count alike."""
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    ids = [f"o{n}" for n in range(11)]

    def answer(batch):
        if refused_id is None:
            return (status, "refused")
        return (status, "refused") if refused_id in batch else None

    def seed(root):
        root.mkdir(parents=True, exist_ok=True)
        for outcome_id in ids:
            production._outcome_file(root, outcome_id).write_text(json.dumps(
                {"outcome_id": outcome_id, "outcome": "resolved",
                 "turn_id": "t1", "occurred_at": "2026-11-10T09:00:00Z"}),
                encoding="utf-8")

    seed(tmp_path / "py")
    tr = _Refusing(answer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with _attached(monkeypatch, tmp_path / "py", tr) as px:
            counts = px.flush()
            # Copied now: closing the attachment flushes again.
            ours = {
                "tried": [list(batch) for batch in tr.tried],
                "sent": sorted(o["outcome_id"] for o in tr.outcomes),
                "left": len(_pending(tmp_path / "py")),
                "counts": [counts["outcomes_sent"], counts["outcomes_failed"],
                           counts["outcomes_dropped"]],
            }

    seed(tmp_path / "node")
    script = (
        "import {createTurnUploader} from './src/capture.js';\n"
        "import {TransportError} from './src/errors.js';\n"
        "import {readdirSync} from 'node:fs';\n"
        f"const refusedId={json.dumps(refused_id)}, status={status};\n"
        f"const root={json.dumps(str(tmp_path / 'node'))};\n"
        "const tried=[], sent=[];\n"
        "const transport={captureCurrent:async()=>({}),"
        "reportTurns:async()=>({}),"
        "reportOutcomes:async(_a,outcomes)=>{"
        "const ids=outcomes.map((o)=>o.outcome_id);tried.push(ids);"
        "if(refusedId===null||ids.includes(refusedId))"
        "throw new TransportError(`POST /outcomes -> HTTP ${status}: refused`,"
        "{statusCode:status});"
        "sent.push(...ids);return {outcomes:ids.map((id)=>({outcome_id:id,"
        "accepted:true}))};}};\n"
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
