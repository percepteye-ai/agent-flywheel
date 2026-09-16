"""A rollout that ran must leave a record, or say why it could not.

THE FAILURE THIS FILE EXISTS FOR. Every rollout on the OPSD queue from
2026-09-10 came back ``status=abandoned`` / ``exclusion_reason=deadline_exceeded``
with zero inference calls. Only the control plane's reaper writes ``abandoned``;
its report route sets ``status=reported`` unconditionally for an ordinary
rollout, even over a row it already swept and even when the payload carries an
exclusion. So on those rows THE SDK'S REPORT NEVER LANDED -- and the SDK had
four terminal paths out of ``_serve_one_inner``, three of which returned
without reporting, each emitting an event into an ``on_event`` hook production
had never wired, after which a ``finally`` deleted the trajectory directory.
Nothing on the row, nothing on disk, nothing in the log.

WHICH of those paths fired was never determined, and these tests deliberately
do not depend on knowing. They assert the property instead: whatever goes
wrong, the outcome is DELIVERED or it is DURABLY HELD AND LOUDLY NAMED. A test
that asserted "report was called" would have passed on the paths that returned
before ever calling it.

Every test here drives ``serve_forever`` -- the real loop, real threads, real
child processes -- because the defect lived in the loop AROUND one rollout.
"""
from __future__ import annotations

import json
import pathlib
import stat
import threading
import time
import warnings

from agent_flywheel import RolloutRequest
from agent_flywheel.contract import RolloutOutput, ToolCallOutcome
from agent_flywheel.errors import LeaseLost, TransportError
from agent_flywheel.runner import RolloutRunner, _output_from_wire

# ── the world ──────────────────────────────────────────────────────────────


def _req(tmp_path, rid="rl_1", seconds=60.0, task="do the thing") -> RolloutRequest:
    return RolloutRequest(
        rollout_id=rid, turn_input=task,
        llm_url=f"https://gw.invalid/llm/v1/rollouts/{rid}", api_key="perk_x",
        artifacts_dir=str(tmp_path), deadline_ts=time.time() + seconds,
        lease_token=f"lt_{rid}",
    )


class ScriptedTransport:
    """Claims a fixed script, and answers ``report`` from a queue of verdicts.

    A verdict is None (accept) or an exception to raise. Records the exact wire
    payload of every accepted report, because "a report happened" is not the
    property under test -- "the agent's result arrived intact" is.
    """

    accepts_delivery_deadline = True
    delivery_envelope_s = 20.0
    agent_id = "agent-under-test"
    lease_ms = 1_800_000

    def __init__(self, batches, verdicts=(), *, heartbeat_raises=None):
        self._batches = list(batches)
        self._verdicts = list(verdicts)
        self._heartbeat_raises = heartbeat_raises
        self.runner: RolloutRunner | None = None
        self.handed_out: list[str] = []
        self.reported: list[str] = []
        self.payloads: list[dict] = []
        self.report_deadlines: list[float | None] = []
        #: Set by a test to look at the outbox from INSIDE the report call --
        #: the only moment at which "spooled before the first network call"
        #: can be observed at all.
        self.spool_probe = None
        self.spool_present_at_send: list = []
        #: Ordered log of what the control plane saw, so "a held result is
        #: delivered BEFORE more work is claimed" is assertable.
        self.order: list[str] = []
        self.abandoned: list[tuple[str, str]] = []
        self.heartbeats: list[str] = []
        self._lock = threading.Lock()

    def claim(self, max_rollouts):
        with self._lock:
            self.order.append("claim")
            if not self._batches:
                if self.runner is not None:
                    self.runner.stop()
                return [], 10
            batch = self._batches.pop(0)
            self.handed_out.extend(r.rollout_id for r in batch)
            return batch, 10

    def heartbeat(self, rollout, *, progress=None):
        with self._lock:
            self.heartbeats.append(rollout.rollout_id)
        if self._heartbeat_raises is not None:
            raise self._heartbeat_raises

    def report(self, rollout, output, *, deadline=None):
        with self._lock:
            self.report_deadlines.append(deadline)
            self.spool_present_at_send.append(
                self.spool_probe() if self.spool_probe else None
            )
            verdict = self._verdicts.pop(0) if self._verdicts else None
            if verdict is not None:
                raise verdict
            self.reported.append(rollout.rollout_id)
            self.order.append(f"report:{rollout.rollout_id}")
            self.payloads.append(output.to_wire())
        return {"grade": "advisory", "duplicate": False}

    def abandon(self, rollout, *, reason, detail=""):
        with self._lock:
            self.abandoned.append((rollout.rollout_id, reason))


def _script(tmp_path, body: str, name="agent.sh") -> str:
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _runner(tmp_path, transport, argv, **kw) -> RolloutRunner:
    kwargs = dict(
        transport=transport, input_shape="text", artifacts_dir=str(tmp_path),
        rollout_timeout_s=60.0,
    )
    kwargs.update(kw)
    runner = RolloutRunner(argv, **kwargs)
    transport.runner = runner
    return runner


def _outbox(tmp_path) -> pathlib.Path:
    return pathlib.Path(tmp_path) / "_percepteye" / "_outbox" / "agent-under-test"


def _spooled(tmp_path) -> list[dict]:
    root = _outbox(tmp_path)
    if not root.is_dir():
        return []
    return [json.loads(p.read_text()) for p in sorted(root.glob("*.json"))]


# ── the gap itself ─────────────────────────────────────────────────────────


def test_a_report_the_control_plane_refuses_once_is_redelivered(tmp_path):
    """A transient refusal used to LOSE the rollout, not delay it.

    The old code emitted ``report_failed`` and returned; the comment beside it
    claimed the server "will hand the work to someone else, so this costs a
    repeat rather than a hole". It does not: the control plane refuses to claim
    a rollout whose budget has elapsed, and the report is attempted at the end
    of the budget by construction. It was always a hole.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1")]],
        verdicts=[TransportError("503 overloaded")],   # first attempt fails
    )
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "the answer"\n')])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        delivered = runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"], (
        f"the result was produced and then dropped: reported={tr.reported}"
    )
    assert delivered == 1, "a delivered outcome was not counted as delivered"
    assert tr.payloads[0]["final_text"] == "the answer", (
        "the redelivered payload is not the agent's own result"
    )
    assert _spooled(tmp_path) == [], "a delivered outcome was left in the outbox"
    assert any("could not be delivered" in str(w.message) for w in caught), (
        "the failed attempt was silent; production had no on_event hook wired, "
        "so an event is not loudness"
    )


def test_the_runner_hands_the_report_the_deadline_it_reserved(tmp_path):
    """The producer/consumer half of the reserve, at the seam it crosses.

    Reserving time and spending it are only one number if the runner actually
    PASSES the reserve to the call that spends it. A transport that declares
    ``accepts_delivery_deadline`` and never receives one has a reserve on paper
    and an unbounded retry loop in fact.
    """
    tr = ScriptedTransport([[_req(tmp_path, "rl_1", seconds=894.0)]])
    runner = _runner(
        tmp_path, tr, ["sh", _script(tmp_path, 'echo "hi"\n')],
        rollout_timeout_s=900.0,
    )
    before = time.time()
    runner.serve_forever(max_rollouts=1)

    assert tr.report_deadlines and tr.report_deadlines[0] is not None, (
        "the report was sent with no deadline; the reserve bounds nothing"
    )
    reserved = tr.report_deadlines[0] - before
    # ScriptedTransport declares a 20s envelope; an 894s episode is far above
    # the fraction cap and 20s is above the floor, so the reserve IS that
    # envelope. Asserted as a RANGE around the declared cost, not "> 0": a
    # runner that fell back to the 10s floor, or to some unrelated constant,
    # must fail here.
    assert 18.0 <= reserved <= 22.0, (
        f"the report was given {reserved:.1f}s against a declared 20s cost"
    )


def test_an_undelivered_result_survives_the_process_that_produced_it(tmp_path):
    """THE ONE THAT CLOSES THE GAP: a restart delivers what a crash could not.

    The result is spooled BEFORE the first network call, so a flywheel that
    dies between producing a result and delivering it still delivers it -- and
    because the control plane accepts a late report and overwrites the reaper's
    verdict, that delivery REPAIRS an already-abandoned row rather than merely
    recording the loss.
    """
    # Process one: produces a result, and every delivery is refused.
    dead = ScriptedTransport(
        [[_req(tmp_path, "rl_1")]],
        verdicts=[TransportError("connection reset"),
                  TransportError("connection reset")],
    )
    first = _runner(tmp_path, dead, ["sh", _script(tmp_path, 'echo "the answer"\n')])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first.serve_forever(max_rollouts=1)

    assert dead.reported == [], "this test needs the first process to fail"
    held = _spooled(tmp_path)
    assert len(held) == 1, f"nothing durable survived the first process: {held}"
    assert held[0]["result"]["final_text"] == "the answer"
    assert held[0]["rollout_id"] == "rl_1"
    assert "api_key" not in json.dumps(held[0]), (
        "the outbox must never write the per-rollout inference credential"
    )

    # Process two: same artifacts_dir, a control plane that answers. It claims
    # nothing at all -- the recovery must not depend on new work arriving.
    alive = ScriptedTransport([])
    second = _runner(tmp_path, alive, ["sh", _script(tmp_path, "echo x\n")])
    delivered = second.serve_forever()

    assert alive.reported == ["rl_1"], (
        "the result produced by the dead process was never delivered"
    )
    assert alive.payloads[0]["final_text"] == "the answer", (
        "the redelivered result is not the one the agent produced"
    )
    assert delivered == 1
    assert _spooled(tmp_path) == []


def test_the_result_is_on_disk_BEFORE_the_first_network_call(tmp_path):
    """The ordering is the whole mechanism, and it is invisible from outside.

    Spooling only AFTER a send fails still recovers a refusal, but loses
    everything if the process dies DURING the send -- a SIGKILL, an OOM, a
    container eviction mid-POST. Observed from inside the report call, which is
    the one moment the question can be asked.
    """
    tr = ScriptedTransport([[_req(tmp_path, "rl_1")]])
    tr.spool_probe = lambda: [r["result"]["final_text"] for r in _spooled(tmp_path)]
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "the answer"\n')])

    runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"]
    assert tr.spool_present_at_send == [["the answer"]], (
        f"the result was not durable when the first POST went out: "
        f"{tr.spool_present_at_send}. A process killed mid-report loses it."
    )
    assert _spooled(tmp_path) == [], "and it must not survive a successful send"


def test_a_held_result_is_delivered_BEFORE_more_work_is_claimed(tmp_path):
    """Ordering, because it is what makes a restart recover rather than forget.

    Draining only on the way out means a flywheel that is killed and restarted
    claims a fresh batch first and delivers the old result only if it happens
    to shut down cleanly -- which, in the incident this exists for, it did not.
    """
    dead = ScriptedTransport(
        [[_req(tmp_path, "rl_old")]], verdicts=[TransportError("reset")] * 4,
    )
    first = _runner(tmp_path, dead, ["sh", _script(tmp_path, 'echo "old work"\n')])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        first.serve_forever(max_rollouts=1)
    assert _spooled(tmp_path), "this test needs an undelivered result to exist"

    alive = ScriptedTransport([[_req(tmp_path, "rl_new")]])
    second = _runner(tmp_path, alive, ["sh", _script(tmp_path, 'echo "new work"\n')])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        second.serve_forever(max_rollouts=1)

    assert alive.order[0] == "report:rl_old", (
        f"the flywheel took on new work before delivering what it already had: "
        f"{alive.order}"
    )
    assert "report:rl_new" in alive.order


def test_a_payload_the_wire_cannot_carry_does_not_strand_the_other_leases(tmp_path):
    """``to_wire`` raises ``ContractViolation``, which is NOT a TransportError.

    It escaped ``_serve_one_inner``'s two excepts, escaped ``_serve_one``
    (which had only a ``finally``), and was re-raised by ``fut.result()`` --
    past a comment reading "_serve_one never raises". ``serve_forever``'s
    ``finally`` then stopped the heartbeat thread, so ONE malformed payload
    ended lease renewal for every concurrent rollout and left them all to be
    reaped. The detector is the SECOND rollout.
    """
    tr = ScriptedTransport([[_req(tmp_path, "rl_1")], [_req(tmp_path, "rl_2")]])
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "hi"\n')])

    original = runner.run_one
    def poisoned(rollout):
        out = original(rollout)
        if rollout.rollout_id == "rl_1":
            # Unrepresentable on the flywheel/1 wire, exactly as a buggy
            # adapter would produce it.
            out.llm_call_count = -7
            out.reward = 0.75
            out.tool_calls = [ToolCallOutcome(
                name="grep", arguments={"q": "x"}, status_code=None,
                outcome="ok",
            )]
        return out
    runner.run_one = poisoned

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=2)

    assert "rl_2" in tr.reported, (
        f"the second rollout never reported: one bad payload stopped the loop "
        f"({tr.reported})"
    )
    assert "rl_1" in tr.reported, "the unrepresentable rollout left no record"
    bad = tr.payloads[tr.reported.index("rl_1")]
    assert bad["exclusion"]["reason"] == "flywheel_crash"
    assert "could not represent" in bad["exclusion"]["detail"], bad
    # AND SAYS WHAT WENT WITH IT. The salvage rebuilds from final_text and the
    # exclusion alone, so ONE unrepresentable field takes every other field
    # down with it. Silently, the row reads as a rollout that produced nothing
    # -- indistinguishable from an agent that did nothing -- and the tool calls
    # and reward that WERE captured are gone with no trace that they existed.
    detail = bad["exclusion"]["detail"]
    assert "dropped by the salvage" in detail, detail
    for lost in ("tool_calls", "reward"):
        assert lost in detail.split("dropped by the salvage:")[1], (
            f"{lost} was populated, was discarded, and is not named: {detail}"
        )
    assert bad["tool_calls"] is None and not bad.get("reward"), (
        "this test no longer reproduces the salvage path"
    )


def test_a_permanent_refusal_is_named_and_stops_the_flywheel_claiming(tmp_path):
    """Six silent days become one rollout and a sentence.

    A flywheel key holding ``rollout.claim`` but not ``rollout.report`` claims
    work forever and 403s every report from the day it was issued -- which
    reproduces "every rollout since 2026-09-10, zero inference calls" exactly,
    with no trace anywhere. The refusal must be repeated back verbatim, the
    result kept (a reissued key can still deliver it), and claiming must stop.
    """
    refusal = TransportError(
        "POST /rollouts/rl_1/result -> HTTP 403: Flywheel key is missing "
        "scope(s): rollout.report",
        status_code=403,
    )
    batches = [[_req(tmp_path, f"rl_{i}")] for i in range(1, 6)]
    tr = ScriptedTransport(batches, verdicts=[refusal] * 12)
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "hi"\n')])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.serve_forever(max_rollouts=5)

    messages = [str(w.message) for w in caught]
    assert any("rollout.report" in m for m in messages), (
        f"the control plane's own refusal never reached anyone: {messages}"
    )
    braked = [m for m in messages if "stopped claiming" in m]
    assert braked, f"the flywheel kept claiming into a permanent refusal: {messages}"
    assert len(tr.reported) == 0
    # The brake is what bounds the damage. Asserted on WORK CLAIMED, not on
    # attempts: the loop is allowed to retry results it already holds, and
    # must not keep taking on NEW rollouts it will refuse to deliver.
    assert len(tr.handed_out) < 5, (
        f"the flywheel claimed every rollout it was offered into a permanent "
        f"refusal: {tr.handed_out}"
    )
    kept = _spooled(tmp_path)
    assert kept, "a 403 is fixable by reissuing the key; the result must be kept"
    assert "403" in (kept[0]["last_error"] or ""), kept[0]


def test_a_refusal_resending_cannot_fix_retires_the_file_and_keeps_the_work(
    tmp_path,
):
    """The other half of the predicate: STOP TRYING, do not shred.

    A 409 means a different result for this rollout already landed; resending
    the same bytes cannot change that, so the file leaves the ``*.json`` retry
    set. It used to leave by being UNLINKED, which for the refusal that matters
    -- the 410 an identity-sealed rollout answers forever once its lease is
    reaped -- destroyed the only copy of work a customer's agent actually did.
    It leaves by being RENAMED now, with a sentence naming the refusal and the
    path, so the outbox still cannot grow without bound and nothing is lost.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1")]],
        verdicts=[TransportError(
            "POST /rollouts/rl_1/result -> HTTP 409: result_conflict",
            status_code=409,
        )],
    )
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "hi"\n')])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.serve_forever(max_rollouts=1)

    messages = [str(w.message) for w in caught]
    assert any("409" in m and "will not be retried" in m for m in messages), messages
    # Out of the retry set...
    assert _spooled(tmp_path) == []
    # ...and still on disk, under a name that says what refused it.
    retired = sorted(_outbox(tmp_path).glob("*.refused-409"))
    assert len(retired) == 1, f"the result was destroyed: {list(_outbox(tmp_path).iterdir())}"
    kept = json.loads(retired[0].read_text())
    assert "hi" in kept["result"]["final_text"]
    assert "409" in (kept["last_error"] or "")
    assert any(str(retired[0]) in m for m in messages), (
        f"the operator was not told where the result is: {messages}"
    )


def test_a_sealed_rollout_the_control_plane_refuses_forever_keeps_everything(
    tmp_path,
):
    """THE OPSD CASE, END TO END. A 410 is permanent and must not be a shredder.

    An identity-sealed rollout's report route requires a live, MATCHING lease.
    Mission Control's reaper clears ``lease_token`` and writes ``ABANDONED``,
    so from that moment every report -- ours and anyone's -- is refused with
    ``PermissionError`` -> HTTP 410, forever. 410 is in
    ``_REDELIVERY_CANNOT_FIX``, so the SDK rightly stops resending. It then
    treated the record as settled and deleted it, together with the trajectory
    directory behind it -- and for a sealed rollout there is no second copy
    anywhere: the row is lost by construction and re-running is the only
    repair, which needs to be able to see what the first run produced.

    Both halves are asserted, because the previous fix reached the result file
    and left the evidence to a predicate ("no file at the spool path") that a
    renamed record satisfies instantly.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1")]],
        verdicts=[LeaseLost(
            "POST /rollouts/rl_1/result -> HTTP 410: rollout worker identity "
            "or lease lost"
        )],
    )
    runner = _runner(
        tmp_path, tr,
        ["sh", _script(tmp_path, 'echo solved > answer.txt; echo "done"\n')],
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.serve_forever(max_rollouts=1)

    assert tr.reported == []
    assert _spooled(tmp_path) == [], "a refusal nothing can fix stayed in the retry set"

    retired = sorted(_outbox(tmp_path).glob("*.refused-lease-lost"))
    assert len(retired) == 1, (
        f"the sealed rollout's result was destroyed: "
        f"{sorted(p.name for p in _outbox(tmp_path).iterdir())}"
    )
    kept = json.loads(retired[0].read_text())
    assert "done" in kept["result"]["final_text"]
    assert kept["result"]["artifacts"]["answer.txt"]["content"] == "solved\n"

    trajectory = tmp_path / "_percepteye" / "rl_1"
    assert (trajectory / "answer.txt").read_text() == "solved\n", (
        "the evidence was purged behind a result that can never be delivered"
    )

    said = [str(w.message) for w in caught]
    assert any(str(retired[0]) in m and str(trajectory) in m for m in said), (
        f"the operator was not told where the result and the evidence are: {said}"
    )


def test_the_drain_keeps_the_evidence_of_what_it_could_not_deliver_either(
    tmp_path,
):
    """THE SECOND WRITER. ``_purge_trajectory`` is called from two places.

    The in-episode path is ``_serve_one``'s ``finally``; this one is
    ``_drain_one``, and it had the same wrong predicate -- "the file at the
    spool path is gone" -- which a RETIRED record satisfies the instant it is
    renamed. Fixing only the path a test happens to drive leaves the other
    writer shipping the defect, so both are driven: here the first attempt
    fails transiently (the result is held, the episode ends) and the refusal
    that retires it arrives on the drain.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1")]],
        verdicts=[
            TransportError("connection reset"),          # in-episode: held
            LeaseLost("HTTP 410: rollout worker identity or lease lost"),
        ],
    )
    runner = _runner(
        tmp_path, tr,
        ["sh", _script(tmp_path, 'echo solved > answer.txt; echo "done"\n')],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=1)

    assert tr.reported == []
    retired = sorted(_outbox(tmp_path).glob("*.refused-lease-lost"))
    assert len(retired) == 1, (
        f"the drain destroyed the result: "
        f"{sorted(p.name for p in _outbox(tmp_path).iterdir())}"
    )
    assert (tmp_path / "_percepteye" / "rl_1" / "answer.txt").read_text() == "solved\n", (
        "the drain purged the evidence behind a result it could not deliver"
    )


def test_a_lost_lease_holds_the_result_rather_than_racing_the_new_holder(tmp_path):
    """Lease loss must not mean SILENCE, and must not mean a race either.

    The old code returned before collecting artifacts and before reporting, so
    final text, artifacts, tool calls and timing were all discarded on the one
    path whose whole justification was "the server already knows". It did not
    know what the agent had produced.

    But delivering immediately would be worse: for an ordinary rollout the
    control plane's report route performs no ownership check, so a displaced
    holder's result would overwrite a live holder's. The result is therefore
    HELD until the rollout's deadline, past which no new claim is possible.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1", seconds=3600.0)]],
        heartbeat_raises=LeaseLost("gone"),
    )
    runner = _runner(
        tmp_path, tr, ["sh", _script(tmp_path, 'echo "work"; sleep 30\n')],
        rollout_timeout_s=60.0,
    )
    # A short lease so the beat -- and the 410 -- lands quickly.
    import agent_flywheel.runner as _rmod
    original = _rmod.RolloutRunner._beat_interval_for
    _rmod.RolloutRunner._beat_interval_for = lambda self, r: 1.0
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner.serve_forever(max_rollouts=1)
    finally:
        _rmod.RolloutRunner._beat_interval_for = original

    assert tr.reported == [], "a live holder was raced"
    held = _spooled(tmp_path)
    assert len(held) == 1, "the outcome was dropped on the lease-lost path"
    assert held[0]["lease_lost"] is True
    assert "work" in held[0]["result"]["final_text"], (
        "the agent's work was discarded, which is what the old short-circuit did"
    )
    said = [str(w.message) for w in caught]
    assert any("will be attempted once" in m for m in said), (
        f"the operator was not told the result is held: {said}"
    )
    # And told what it CANNOT do. This used to promise delivery flatly, which
    # is false for an identity-sealed rollout: that report route requires the
    # very lease that was just lost, so the deferred send is refused by
    # construction. Promising a repair that cannot happen is how a lost row
    # goes uninvestigated.
    assert any("SEALED" in m for m in said), (
        f"the hold promised a repair it cannot make for a sealed rollout: {said}"
    )


def test_the_trajectory_is_kept_while_its_outcome_is_undelivered(tmp_path):
    """``_purge_trajectory`` ran in a ``finally``, on every drop path.

    The trajectory directory is the only local copy of the evidence, so
    deleting it on exactly the paths where the report did not land made the
    loss irrecoverable by construction.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1")]], verdicts=[TransportError("reset"),
                                              TransportError("reset")],
    )
    runner = _runner(
        tmp_path, tr,
        ["sh", _script(tmp_path, 'echo hi > result.txt; echo "done"\n')],
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=1)

    assert tr.reported == []
    assert (tmp_path / "_percepteye" / "rl_1").is_dir(), (
        "the evidence was deleted while the result was still undelivered"
    )

    # And it IS cleaned up once delivery settles -- this must not become a leak.
    alive = ScriptedTransport([])
    _runner(tmp_path, alive, ["sh", _script(tmp_path, "echo x\n")]).serve_forever()
    assert alive.reported == ["rl_1"]
    assert not (tmp_path / "_percepteye" / "rl_1").exists()


def test_a_crash_the_control_plane_never_heard_about_is_still_delivered(tmp_path):
    """``except (TransportError, LeaseLost): pass`` around ``abandon``.

    A crashed rollout whose abandon ALSO failed left the control plane knowing
    nothing at all, and the reaper then wrote its own reason over the row. The
    crash is now reported as an exclusion when it cannot be handed back.
    """
    class NoAbandon(ScriptedTransport):
        def abandon(self, rollout, *, reason, detail=""):
            raise TransportError("abandon route down")

    tr = NoAbandon([[_req(tmp_path, "rl_1")]])
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, "echo x\n")])
    runner.run_one = lambda rollout: (_ for _ in ()).throw(OSError("disk gone"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"], "the crash left no record at all"
    payload = tr.payloads[0]
    assert payload["exclusion"]["reason"] == "flywheel_crash"
    assert "disk gone" in payload["exclusion"]["detail"]


# ── the spooled payload is the contract payload, for every harness ─────────


def test_a_transport_that_raises_something_unexpected_keeps_the_result(tmp_path):
    """``FlywheelTransport`` is a Protocol: a customer transport raises anything.

    Treating an unrecognised exception as permanent would delete a customer's
    work on the strength of not understanding the error, and letting it escape
    would stop the drain for every OTHER result the outbox still holds.
    """
    class Rude(ScriptedTransport):
        def report(self, rollout, output, *, deadline=None):
            raise ValueError("my transport does its own thing")

    tr = Rude([[_req(tmp_path, "rl_1")]])
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "hi"\n')])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.serve_forever(max_rollouts=1)

    held = _spooled(tmp_path)
    assert len(held) == 1, "an unrecognised transport error deleted the result"
    assert "ValueError" in (held[0]["last_error"] or "")
    assert any("ValueError" in str(w.message) for w in caught)


def test_a_spooled_outcome_round_trips_byte_for_byte():
    """Redelivery is safe only if what is resent is what was produced.

    The outbox stores ``to_wire()`` -- the flywheel/1 shape EVERY adapter
    produces -- which is why the drain needs to know nothing about the harness.
    That only holds if the round trip is lossless, so it is checked rather than
    assumed.
    """
    rich = RolloutOutput(
        final_text="done",
        tool_calls=[ToolCallOutcome(
            name="run", arguments={"cmd": "ls"}, outcome="ok", status_code=200,
            output="a\nb", latency_ms=12, agent_name="sub", tool_call_id="tc1",
            entity_ids={"id": "o-1"},
        )],
        tool_calls_omitted_count=2,
        llm_call_count=3,
        artifacts={"a.txt": {"sha256": "0" * 64, "size_bytes": 3, "content": "abc"},
                   "b.bin": "1" * 64},
        reward=0.5, success=True,
        exclusion_reason="deadline_exceeded", exclusion_detail="clock",
        events=[{"k": "v"}], timing={"elapsed_s": 1.25},
        agent_fingerprint={"execution_sha256": "2" * 64},
    )
    wire = rich.to_wire()
    assert _output_from_wire(wire).to_wire() == wire, (
        "a spooled outcome does not survive redelivery unchanged"
    )


def test_a_second_refusal_never_overwrites_the_first_retired_record(tmp_path):
    """Retiring twice keeps BOTH copies.

    ``_move_aside`` used ``os.replace`` onto a bare ``<id>.json.<why>``, which
    silently clobbers an existing target. A rollout refused a second time --
    a retried task, or a restart that re-spools the same id -- therefore
    destroyed the first retired record: the outbox reported success (the file
    is still there) while the earlier result was gone. That is the exact
    silent loss this whole mechanism exists to end, so it is asserted on the
    CONTENTS, not on the filename: a test that globbed for the suffix passed
    with the clobber in place, because one file always survives.
    """
    from agent_flywheel.runner import _move_aside

    outbox = tmp_path / "outbox"
    outbox.mkdir()

    kept = []
    for attempt in ("first", "second", "third"):
        spool = outbox / "rl_1.json"
        spool.write_text(json.dumps({"attempt": attempt}))
        kept.append(_move_aside(spool, "refused-lease-lost"))

    assert len({p.name for p in kept}) == 3, (
        f"retirements collapsed onto the same name: {[p.name for p in kept]}"
    )
    survived = sorted(
        json.loads(p.read_text())["attempt"]
        for p in outbox.glob("rl_1.json.refused-lease-lost*")
    )
    assert survived == ["first", "second", "third"], (
        f"a later refusal destroyed an earlier retired record: {survived}"
    )
