"""Captured turns reach the control plane — and only when they are allowed to.

Until now ``attach()`` recorded production turns to disk and shipped nothing:
the contract had no route for a turn, and the module said so in as many words.
It has one now, and these tests pin the properties that decide whether opening
that path was safe.

TIMING IS THE ONE THAT BIT. An earlier design uploaded at turn-scope EXIT.
``record_answer`` is documented as callable after the scope closes, and this
package's own README records the answer after the ``with`` block — so uploading
on exit shipped a turn whose answer did not exist yet, for the documented usage.
Every turn, silently, with the field that matters empty.

CONSENT IS THE OTHER. A turn carries this agent's END USERS' own messages, so
capture is opt-in per agent. The SDK learns its permission from the
registration response rather than by uploading — finding out by sending is the
wrong order when the thing being asked about is whether to send.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel.outcomes import record_tool_call


class _Transport:
    """Records turn batches. ``capture`` is what the server would say."""

    def __init__(self, *, capture=True, reason=None, turns_raise=None):
        self.registered = None
        self.batches: list[list[dict]] = []
        self._capture = capture
        self._reason = reason
        self._turns_raise = turns_raise

    def register(self, spec):
        self.registered = dict(spec)
        cap: dict = {"turns": self._capture}
        if not self._capture and self._reason:
            cap["reason"] = self._reason
        return {"capture": cap}

    def policy_current(self):
        return {}

    def report_turns(self, agent_id, turns):
        if self._turns_raise is not None:
            raise self._turns_raise
        self.batches.append(list(turns))
        return {"agent_id": agent_id,
                "turns": [{"turn_id": t["turn_id"], "accepted": True}
                          for t in turns]}

    # Every turn this transport was handed, flattened.
    @property
    def sent(self):
        return [t for b in self.batches for t in b]


@contextmanager
def _attached(monkeypatch, tmp_path, transport):
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_x")
    monkeypatch.setattr(
        "percepteye_agent_flywheel.transport.AttachTransport",
        lambda client, agent_id: transport,
    )
    monkeypatch.setattr(
        "percepteye_agent_flywheel._http.ControlPlaneClient.close",
        lambda self: None,
    )
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    try:
        yield px
    finally:
        px.close()


def _turn(px, tid, *, ask, answer=None, calls=()):
    with px.turn(turn_id=tid, input_text=ask, conversation_id="c1"):
        for c in calls:
            record_tool_call(**c)
    if answer:
        px.record_answer(tid, answer)


OK = {"name": "lookup", "arguments": {"id": "1"}, "outcome": "ok",
      "status_code": 200}


# ── consent ───────────────────────────────────────────────────────────────


def test_without_permission_nothing_is_uploaded(monkeypatch, tmp_path):
    tr = _Transport(capture=False, reason="opt-in per agent")
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is False
        assert "opt-in" in (px.upload_refusal or "")
        _turn(px, "t1", ask="hi", answer="there")
        assert px.flush() == {"sent": 0, "skipped": 0, "failed": 0,
                              "not_permitted": 1}
    assert tr.sent == []


def test_permission_comes_from_the_server_not_from_us(monkeypatch, tmp_path):
    """The default is off, and it is the registration response that changes
    it. A client that assumed permission would send first and ask later."""
    tr = _Transport(capture=True)
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is True
        assert px.upload_refusal is None


def test_a_registration_that_says_nothing_means_no(monkeypatch, tmp_path):
    """An older control plane returns no ``capture`` block at all. Absent is
    not consent."""
    class _Old(_Transport):
        def register(self, spec):
            self.registered = dict(spec)
            return {"agent_id": "support"}

    tr = _Old()
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is False
        _turn(px, "t1", ask="hi", answer="there")
        px.flush()
    assert tr.sent == []


# ── timing: the defect ────────────────────────────────────────────────────


def test_a_turn_is_not_uploaded_before_its_answer_exists(monkeypatch, tmp_path):
    """THE defect. Uploading at scope exit shipped every turn answerless."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="t1", input_text="hi"):
            record_tool_call(**OK)
        # The scope has closed and the answer has NOT been recorded yet.
        assert tr.sent == [], "uploaded before record_answer could run"
        px.record_answer("t1", "there")
        px.flush()

    sent = {t["turn_id"]: t for t in tr.sent}
    assert sent["t1"]["final_text"] == "there"


def test_flush_holds_back_a_turn_with_no_answer(monkeypatch, tmp_path):
    """The answer may still be coming; a flush mid-process must not decide it
    never will."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "answered", ask="a", answer="A")
        _turn(px, "pending", ask="b", calls=[OK])
        counts = px.flush()
        assert counts["sent"] == 1
        assert counts["skipped"] == 1
        assert [t["turn_id"] for t in tr.sent] == ["answered"]
        # close() is the last chance, so the pending one goes then.
    assert sorted(t["turn_id"] for t in tr.sent) == ["answered", "pending"]


def test_close_sends_the_unanswered_turn_with_its_calls(monkeypatch, tmp_path):
    """At shutdown nothing more is coming. A turn that made real tool calls
    and recorded no answer is honest data, and the intake stores it."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="hi", calls=[OK])
    sent = {t["turn_id"]: t for t in tr.sent}
    assert sent["t1"]["final_text"] is None
    assert len(sent["t1"]["tool_calls"]) == 1


# ── the wire shape ────────────────────────────────────────────────────────


def test_the_wire_shape_is_turn_native(monkeypatch, tmp_path):
    """What crosses the boundary is what the agent OBSERVED, not a
    training-row projection. The corpus shape is the far side's business, and
    sending it would also have meant the intake's tool-call rules -- written
    against outcome/status_code -- could not be reused."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="refund 4417", answer="Refunded.", calls=[OK])
        px.flush()

    t = tr.sent[0]
    assert t["turn_id"] == "t1"
    assert t["conversation_id"] == "c1"
    assert t["input_text"] == "refund 4417"
    assert t["final_text"] == "Refunded."
    assert t["tool_calls"][0]["outcome"] == "ok"
    assert t["tool_calls"][0]["status_code"] == 200
    # Not the corpus projection.
    assert "messages" not in t
    assert "success" not in t["tool_calls"][0]


def test_an_unobserved_outcome_survives_as_unknown(monkeypatch, tmp_path):
    """The tri-state this SDK exists to preserve, on the upload path too."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="a", answer="A", calls=[
            {"name": "refund", "arguments": {}, "outcome": "unknown",
             "status_code": None}])
        px.flush()
    call = tr.sent[0]["tool_calls"][0]
    assert call["outcome"] == "unknown"
    assert call["status_code"] is None


# ── idempotency and failure ───────────────────────────────────────────────


def test_a_turn_is_not_uploaded_twice(monkeypatch, tmp_path):
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="a", answer="A")
        assert px.flush()["sent"] == 1
        assert px.flush()["sent"] == 0
    assert len(tr.sent) == 1


def test_a_failed_batch_is_retried_not_dropped(monkeypatch, tmp_path):
    """The marker is written only after the server acknowledged. Writing it
    first would lose every turn in a batch that timed out."""
    tr = _Transport(turns_raise=RuntimeError("network"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="a", answer="A")
        counts = px.flush()
        assert counts["sent"] == 0 and counts["failed"] == 1
        # Now the network comes back.
        tr._turns_raise = None
        assert px.flush()["sent"] == 1
    assert [t["turn_id"] for t in tr.sent] == ["t1"]


def test_an_unreachable_control_plane_never_raises(monkeypatch, tmp_path):
    """A process that is serving users must not be taken down by our upload."""
    tr = _Transport(turns_raise=RuntimeError("boom"))
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="a", answer="A")
        px.flush()          # must not raise
        # And close(), which flushes again, must not raise either.


def test_close_is_still_idempotent(monkeypatch, tmp_path):
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="a", answer="A")
        px.close()
        px.close()
    assert len(tr.sent) == 1


# ── batching ──────────────────────────────────────────────────────────────


def test_turns_are_batched_not_sent_one_by_one(monkeypatch, tmp_path):
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        for i in range(5):
            _turn(px, f"t{i}", ask=str(i), answer=str(i))
        px.flush(batch_size=2)
    assert [len(b) for b in tr.batches] == [2, 2, 1]
    assert len(tr.sent) == 5


def test_an_empty_turn_directory_is_not_uploaded(monkeypatch, tmp_path):
    """A turn that captured nothing is not data."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="empty"):
            pass
        px.flush(include_unanswered=True)
    assert tr.sent == []


# ── uploading must not disturb the tree it reads ──────────────────────────


def test_flushing_does_not_reorder_a_conversation(monkeypatch, tmp_path):
    """THE defect this file's bookkeeping used to cause.

    Upload state started as a marker FILE written inside the turn directory.
    Creating a file inside a directory bumps that DIRECTORY's mtime, and mtime
    is the ordering key ``export.iter_records`` sorts by — so marking a turn
    uploaded floated it to the end of its own conversation, and the record
    handed to the ingest had its turns out of order.

    THE IDS BELOW ARE DELIBERATELY OUT OF ALPHABETICAL ORDER, and that is what
    makes this test able to fail. It used to create ``t0..t3``, whose NAME
    order is also their creation order -- and ``flush`` scans the root sorted
    by name, so it touched the directories in exactly the order they were
    already in. Restoring the original bug (a marker file inside each turn
    directory, bumping its mtime) left this GREEN: the mutation reordered
    nothing, because the two orders were the same. A test whose fixture cannot
    distinguish the defect from correct behaviour certifies nothing.

    Real turn ids are ``t_<uuid>``, which do not sort in creation order, so
    the shape below is the ordinary case and ``t0..t3`` was the special one.
    """
    from percepteye_agent_flywheel.export import iter_records

    # Created oldest-first; NAME order is the reverse.
    ids = ["zz_first", "mm_second", "bb_third", "aa_fourth"]
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        for i, tid in enumerate(ids):
            _turn(px, tid, ask=f"ask {i}", answer=f"ans {i}")
        before = [r for r in iter_records(str(tmp_path), group_conversations=False)]
        px.flush()
        after = [r for r in iter_records(str(tmp_path), group_conversations=False)]

    assert [r["rollout_id"] for r in before] == ids, (
        f"the fixture itself is not in creation order: {before}"
    )
    assert [r["rollout_id"] for r in after] == [r["rollout_id"] for r in before], (
        "flushing reordered the conversation"
    )


def test_flushing_does_not_change_what_retention_keeps(monkeypatch, tmp_path):
    """The same mtime, read by the other consumer. ``prune(keep_last=N)`` keeps
    the NEWEST N; a turn that looked newest because we had just marked it
    uploaded would evict a genuinely newer one."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        for i in range(4):
            _turn(px, f"t{i}", ask=str(i), answer=str(i))
        # Upload the OLDEST two only.
        px.flush(batch_size=2)
        px.prune(keep_last=2)
        kept = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())

    assert kept == ["t2", "t3"], f"retention kept the wrong turns: {kept}"


def test_the_upload_index_is_a_file_not_a_scanned_directory(monkeypatch, tmp_path):
    """Neither reader filters dotfiles — they take everything that is_dir() —
    so a `.uploaded/` DIRECTORY would have been scanned as a turn."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t1", ask="a", answer="A")
        px.flush()
    index = tmp_path / ".uploaded"
    assert index.is_file(), "upload state must not be a directory under the root"
    assert "t1" in index.read_text()
    assert [d.name for d in tmp_path.iterdir() if d.is_dir()] == ["t1"]


def test_pruned_turns_are_forgotten_from_the_index(monkeypatch, tmp_path):
    """The index describes this tree, so it is bounded by this tree. Left
    alone it grows for the life of the process while the directory it
    describes stays capped."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        for i in range(4):
            _turn(px, f"t{i}", ask=str(i), answer=str(i))
        px.flush()
        assert len(px._uploaded_ids()) == 4
        px.prune(keep_last=1)
        assert px._uploaded_ids() == {"t3"}, px._uploaded_ids()


# ── robustness: this runs inside a customer's process ─────────────────────


def test_an_unreadable_turn_costs_that_turn_and_nothing_else(monkeypatch, tmp_path):
    """A truncated trajectory is the EXPECTED case, not an exotic one — a
    deadline kill cuts the final line, and this package says so. The scan used
    to sit outside the try, so one such turn raised out of a method documented
    as never raising, on a customer's shutdown path."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "good", ask="a", answer="A")
        _turn(px, "bad", ask="b", answer="B")

        real = px._turn_wire

        def _boom(d):
            if d.name == "bad":
                raise OSError("truncated")
            return real(d)

        monkeypatch.setattr(px, "_turn_wire", _boom)
        counts = px.flush()          # must not raise

    assert counts["sent"] == 1
    assert counts["failed"] == 1
    assert [t["turn_id"] for t in tr.sent] == ["good"]


def test_tool_calls_none_vs_empty_comes_from_the_FILE(monkeypatch, tmp_path):
    """``None`` = nothing was recorded for this turn; ``[]`` = it ran and
    called nothing. Deriving it from whether an adapter bound at ATTACH time
    answered a different question, and would have uploaded a turn with no
    trajectory as a positive assertion that the agent made zero calls."""
    from percepteye_agent_flywheel.outcomes import TOOL_CALLS_FILENAME

    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "called", ask="a", answer="A", calls=[OK])
        _turn(px, "silent", ask="b", answer="B")
        # A turn whose trajectory file never appeared at all.
        (tmp_path / "silent" / TOOL_CALLS_FILENAME).unlink(missing_ok=True)
        px.flush()

    sent = {t["turn_id"]: t for t in tr.sent}
    assert len(sent["called"]["tool_calls"]) == 1
    assert sent["silent"]["tool_calls"] is None, (
        "no trajectory file means nothing was recorded — not 'zero calls'"
    )


def test_turn_index_orders_the_conversation(monkeypatch, tmp_path):
    """MC indexes on (org, agent, conversation_id, turn_index). Sending 0 for
    every turn — which is what happened when the SDK never set the field —
    makes a conversation unorderable, which is the one thing that index exists
    to prevent."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        for i in range(4):
            _turn(px, f"t{i}", ask=f"ask {i}", answer=f"ans {i}")
        px.flush()
    idx = {t["turn_id"]: t["turn_index"] for t in tr.sent}
    assert idx == {"t0": 0, "t1": 1, "t2": 2, "t3": 3}, idx


def test_a_turn_sent_in_a_later_flush_keeps_its_position(monkeypatch, tmp_path):
    """A counter would restart at zero on the second flush; the position is
    derived from the turns on disk, so it does not."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _turn(px, "t0", ask="a", answer="A")
        _turn(px, "t1", ask="b", answer="B")
        px.flush()
        _turn(px, "t2", ask="c", answer="C")
        px.flush()
    idx = {t["turn_id"]: t["turn_index"] for t in tr.sent}
    assert idx == {"t0": 0, "t1": 1, "t2": 2}, idx


def test_a_late_answer_does_not_reorder_the_conversation(monkeypatch, tmp_path):
    """THE DEFECT: the position came from the turn DIRECTORY's mtime.

    A directory's mtime advances every time a file is CREATED inside it, and
    ``record_answer`` creates ``turns.jsonl`` whenever ``input_text`` was not
    passed -- both are documented optional. So answering turn 0 after turn 1
    had already happened re-dated turn 0's directory to a time AFTER turn 1's,
    and the conversation was uploaded in reverse: MC stores ``turn_index``
    verbatim, ``_body_of`` excludes it from the dedupe body, and the wrong
    index therefore sticks even if the turn is sent again.

    The two existing ordering tests never saw it because both record the
    answer immediately, in the same order the turns happened.
    """
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="t0", conversation_id="c1"):
            pass
        time.sleep(0.02)          # two turns that really did happen apart
        with px.turn(turn_id="t1", conversation_id="c1"):
            pass
        px.record_answer("t1", "second")
        px.record_answer("t0", "first")     # the late answer
        px.flush()
    idx = {t["turn_id"]: t["turn_index"] for t in tr.sent}
    assert idx == {"t0": 0, "t1": 1}, (
        f"a late answer inverted the conversation: {idx}"
    )
