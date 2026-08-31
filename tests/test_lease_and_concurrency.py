"""The serving loop's obligations to the lease it holds.

Every test here failed before the fix it guards. The previous suite exercised
``run_one`` in isolation, which is exactly why it was silent about all of
this: the defects live in the loop AROUND one rollout, not inside it.
"""
from __future__ import annotations

import os
import pathlib
import threading
import time
from dataclasses import replace

import httpx
import pytest

from percepteye_agent_flywheel import RolloutRequest
from percepteye_agent_flywheel._http import ControlPlaneClient
from percepteye_agent_flywheel.errors import LeaseLost, TransportError
from percepteye_agent_flywheel.runner import RolloutRunner

TESTS_DIR = str(pathlib.Path(__file__).resolve().parent)


@pytest.fixture(autouse=True)
def _child_can_import(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", TESTS_DIR + (os.pathsep + existing if existing else "")
    )


def _req(tmp_path, rid="rl_1", seconds=60.0, task="do the thing") -> RolloutRequest:
    return RolloutRequest(
        rollout_id=rid,
        turn_input=task,
        llm_url=f"https://gw.example.invalid/llm/v1/rollouts/{rid}",
        api_key="perk_single_use",
        artifacts_dir=str(tmp_path),
        deadline_ts=time.time() + seconds,
        lease_token=f"lt_{rid}",
    )


class FakeTransport:
    """Serves a fixed script of claims, then nothing. Records every call."""

    def __init__(self, batches, *, lease_ms=1_800_000, heartbeat_raises=None):
        self._batches = list(batches)
        self.lease_ms = lease_ms
        self._heartbeat_raises = heartbeat_raises
        #: Set by the test after construction. serve_forever is a server loop
        #: and correctly polls forever on an empty queue, so a test that does
        #: not reach max_rollouts has to end it.
        self.runner: RolloutRunner | None = None
        self.claim_sizes: list[int] = []
        self.heartbeats: list[str] = []
        self.reported: list[str] = []
        self.abandoned: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def claim(self, max_rollouts):
        with self._lock:
            self.claim_sizes.append(max_rollouts)
            if not self._batches:
                if self.runner is not None:
                    self.runner.stop()
                return [], 10
            return self._batches.pop(0), 10

    def heartbeat(self, rollout, *, progress=None):
        with self._lock:
            self.heartbeats.append(rollout.rollout_id)
        if self._heartbeat_raises is not None:
            raise self._heartbeat_raises

    def report(self, rollout, output):
        with self._lock:
            self.reported.append(rollout.rollout_id)
        return {"grade": "advisory", "duplicate": False}

    def abandon(self, rollout, *, reason, detail=""):
        with self._lock:
            self.abandoned.append((rollout.rollout_id, reason))


def _runner(tmp_path, transport, spec="fixture_agent:build", **kw):
    kwargs = dict(
        transport=transport,
        input_shape="text",
        artifacts_dir=str(tmp_path),
        isolation="subprocess",
        concurrency=1,
        rollout_timeout_s=60.0,
    )
    kwargs.update(kw)
    runner = RolloutRunner(spec, **kwargs)
    transport.runner = runner
    return runner


# ── the lease is kept alive ────────────────────────────────────────────────


def test_a_long_rollout_heartbeats_its_lease(tmp_path):
    """The server reaps a lease that stops being renewed. Before this, nothing
    ever called heartbeat, so any rollout outliving its lease was requeued
    WHILE STILL RUNNING and reported into a lease that no longer existed."""
    tr = FakeTransport([[_req(tmp_path)]], lease_ms=3_000)  # -> beat every 1s
    runner = _runner(tmp_path, tr, spec="fixture_agent:build_medium")

    runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"]
    # AT LEAST ONE, not at least two. The bug this guards was that nothing
    # called heartbeat AT ALL, so one beat already proves the thread runs and
    # the lease is renewed. Asserting a COUNT asserted how many times a
    # background thread happened to wake inside a 2s window, which is a fact
    # about the machine rather than about this code -- it failed on six of
    # twelve CI jobs, on 3.10 and 3.11 while 3.12 and 3.13 passed on the same
    # OS, which is scheduling noise wearing a version number.
    assert tr.heartbeats.count("rl_1") >= 1, (
        f"a rollout outliving its beat interval must renew its lease, "
        f"got {tr.heartbeats}"
    )


def test_losing_the_lease_kills_the_child_and_reports_nothing(tmp_path):
    """A lost lease means the server already handed the work to someone else.
    Running on burns the customer's compute for a result nothing will take,
    and reporting into it only earns a 410."""
    tr = FakeTransport(
        [[_req(tmp_path)]], lease_ms=3_000, heartbeat_raises=LeaseLost("gone")
    )
    events: list[str] = []
    runner = _runner(
        tmp_path, tr, spec="fixture_agent:build_slow",   # sleeps 120s
        rollout_timeout_s=60.0,
        on_event=lambda k, p: events.append(k),
    )

    started = time.time()
    runner.serve_forever(max_rollouts=1)
    elapsed = time.time() - started

    assert elapsed < 20, f"the child was not killed on lease loss ({elapsed:.1f}s)"
    assert tr.reported == []
    assert tr.abandoned == [], "we do not hold the lease; there is nothing to hand back"
    assert "lease_lost" in events


def test_heartbeat_failure_is_transient_and_does_not_kill_the_rollout(tmp_path):
    """A 500 on the beat is not a lost lease. There are two more beats before
    it lapses, so the rollout must run on."""
    tr = FakeTransport(
        [[_req(tmp_path)]], lease_ms=3_000, heartbeat_raises=TransportError("boom")
    )
    runner = _runner(tmp_path, tr, spec="fixture_agent:build_medium")

    runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"]


# ── claiming N means running N ─────────────────────────────────────────────


def test_a_claimed_batch_runs_concurrently(tmp_path, monkeypatch):
    """Claiming N and serving them one after another held N-1 leases on work
    nobody was doing.

    Proved by OVERLAP rather than by elapsed time. "Three 2s rollouts must
    finish in under 4.5s" is a claim about the machine as much as the code:
    three subprocess spawns on a loaded runner can eat the margin, and then a
    correct implementation fails. If the last rollout STARTED before the first
    one FINISHED, all three were in flight at once, and that is true on any
    machine at any speed."""
    spans = tmp_path / "spans"
    spans.mkdir()
    monkeypatch.setenv("AGENT_SPAN_DIR", str(spans))

    batch = [_req(tmp_path, rid=f"rl_{i}") for i in range(3)]
    tr = FakeTransport([batch])
    runner = _runner(
        tmp_path, tr, spec="fixture_agent:build_medium", concurrency=3
    )

    runner.serve_forever(max_rollouts=3)

    assert sorted(tr.reported) == ["rl_0", "rl_1", "rl_2"]
    seen = [tuple(float(x) for x in p.read_text().split())
            for p in spans.glob("*.span")]
    assert len(seen) == 3, f"expected three spans, got {len(seen)}"
    last_start = max(s for s, _ in seen)
    first_end = min(e for _, e in seen)
    assert last_start < first_end, (
        f"the last rollout started {last_start - first_end:.2f}s AFTER the "
        f"first finished -- they ran sequentially"
    )


def test_the_loop_never_claims_more_than_it_will_run(tmp_path):
    """Claiming a full batch when one rollout remained left the rest leased to
    a runner that was about to exit."""
    tr = FakeTransport([[_req(tmp_path, rid="rl_a")]])
    runner = _runner(tmp_path, tr, concurrency=8)

    runner.serve_forever(max_rollouts=2)

    assert tr.claim_sizes[0] == 2, f"asked for {tr.claim_sizes[0]} to run 2"


# ── the budget the server actually gave us ─────────────────────────────────


def test_an_already_expired_rollout_is_handed_back_immediately(tmp_path):
    """It used to fall back to the FULL local timeout -- spending 60s of the
    customer's compute on work the server had already given up on."""
    tr = FakeTransport([])
    runner = _runner(
        tmp_path, tr, spec="fixture_agent:build_slow", rollout_timeout_s=60.0
    )

    started = time.time()
    out = runner.run_one(_req(tmp_path, seconds=-5.0))
    elapsed = time.time() - started

    assert out.exclusion_reason == "deadline_exceeded"
    assert "already passed" in (out.exclusion_detail or "")
    assert elapsed < 2.0, f"ran for {elapsed:.1f}s on an expired deadline"


# ── the child runs what the parent validated ───────────────────────────────


def test_method_reaches_the_child(tmp_path):
    """``method=`` was honoured at startup and hardcoded to None on the child
    payload, so every rollout re-probed and could run a different method. The
    fixture's decoy ``invoke`` makes that show up as a wrong value."""
    tr = FakeTransport([])
    runner = _runner(
        tmp_path, tr, spec="fixture_agent:build_named_method", method="handle_task"
    )

    out = runner.run_one(_req(tmp_path))

    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.final_text == "named:do the thing", (
        "the child probed instead of using the method the parent resolved"
    )


def test_a_class_entrypoint_is_instantiated(tmp_path):
    """Pointing at a class found ``Cls.invoke`` -- an unbound function -- and
    called it with the turn input as ``self``."""
    tr = FakeTransport([])
    runner = _runner(tmp_path, tr, spec="fixture_agent:EchoAgent")

    out = runner.run_one(_req(tmp_path))

    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.final_text == "handled:do the thing"


# ── housekeeping ───────────────────────────────────────────────────────────


def test_the_trajectory_directory_is_purged_after_serving(tmp_path):
    """One directory per rollout accumulated for the life of the process."""
    tr = FakeTransport([[_req(tmp_path)]])
    runner = _runner(tmp_path, tr)

    runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"]
    assert not (tmp_path / "_percepteye" / "rl_1").exists()


# ── retry safety on a route with no replay protection ──────────────────────


def _client(handler, **kw) -> ControlPlaneClient:
    return ControlPlaneClient(
        "https://cp.example.invalid", "pehk_test",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        **kw,
    )


def test_an_ambiguous_failure_is_not_replayed_when_the_route_is_not_replayable():
    """The control plane does not honour Idempotency-Key on claim, so a resend
    after a READ timeout -- where the server may already have leased a batch --
    leases a second one that nobody holds."""
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ReadTimeout("timed out reading", request=request)

    with pytest.raises(TransportError):
        _client(handler).request(
            "POST", "/rollouts/claim", json_body={}, retry_if_ambiguous=False
        )

    assert attempts["n"] == 1, "resent a claim that may have been processed"


def test_a_failure_that_never_reached_the_server_is_still_retried():
    """Narrowing retries must not disable them: a refused connection provably
    did nothing, so backing off and trying again is free."""
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ConnectError("connection refused", request=request)

    with pytest.raises(TransportError):
        _client(handler, max_attempts=3).request(
            "POST", "/rollouts/claim", json_body={}, retry_if_ambiguous=False
        )

    assert attempts["n"] == 3


def test_a_5xx_is_not_replayed_on_a_non_replayable_route():
    """The server may well have committed the lease before failing to answer."""
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        return httpx.Response(503, json={"detail": "overloaded"})

    with pytest.raises(TransportError, match="503"):
        _client(handler).request(
            "POST", "/rollouts/claim", json_body={}, retry_if_ambiguous=False
        )

    assert attempts["n"] == 1


def test_the_claim_call_site_actually_asks_for_the_narrow_retry():
    """Guards the CALL SITE, not just the capability. The client can support
    narrow retries and claim can still forget to ask for them."""
    from percepteye_agent_flywheel.transport import AttachTransport

    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ReadTimeout("timed out reading", request=request)

    tr = AttachTransport(_client(handler), "agent-1")
    with pytest.raises(TransportError):
        tr.claim(4)

    assert attempts["n"] == 1, "claim replayed a request that may have leased work"


def test_a_429_is_still_retried_because_nothing_was_leased():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={})
        return httpx.Response(200, json={"rollouts": [], "poll_after_ms": 1000})

    body = _client(handler).request(
        "POST", "/rollouts/claim", json_body={}, retry_if_ambiguous=False
    )

    assert attempts["n"] == 3
    assert body["poll_after_ms"] == 1000


# ── the heartbeat paces itself against the lease it was GRANTED ────────────


def test_the_beat_follows_the_granted_lease_not_the_requested_one(tmp_path):
    """The server clamps what we ask for. Pacing against our own request means
    a deployment with a lower ceiling reaps every long rollout underneath a
    heartbeat that thought it had plenty of time."""
    granted = 3.0
    req = RolloutRequest(
        rollout_id="rl_1", turn_input="x",
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_1", api_key="k",
        artifacts_dir=str(tmp_path), deadline_ts=time.time() + 600,
        lease_expires_ts=time.time() + granted, lease_token="lt",
    )
    # We asked for half an hour; the server granted three seconds.
    tr = FakeTransport([[req]], lease_ms=1_800_000)
    runner = _runner(tmp_path, tr, spec="fixture_agent:build_medium")

    # Asserted where the decision is MADE, not inferred from how many beats a
    # loaded CI runner managed to fit into a 2.5s window. The count was the
    # original assertion and it was a proxy: macOS/3.10 fitted exactly one
    # beat and failed a run whose pacing was entirely correct. A test that
    # goes red on a slow machine teaches you to re-run it, not to read it.
    assert runner._beat_interval_for(req) < 600.0, (
        "the beat paced itself against the 30min we REQUESTED"
    )
    # A third of what was granted. Stated with a lease long enough that the
    # division is visible -- at the 3s above, a third is 1.0s, which is also
    # the floor, so that case alone cannot tell arithmetic from clamping.
    longer = replace(req, lease_expires_ts=time.time() + 12.0)
    assert runner._beat_interval_for(longer) == pytest.approx(4.0, abs=0.2)

    runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"]
    # Liveness only: that the loop beat at all. How MANY beats fit is a
    # property of the machine, not of the code under test.
    assert tr.heartbeats, "the rollout held a 3s lease and never beat"


def test_the_beat_falls_back_when_the_server_reports_no_lease(tmp_path):
    """An older control plane says nothing about what it granted. Falling back
    to the requested lease is the only estimate available, and is better than
    not beating at all."""
    req = _req(tmp_path)
    assert req.lease_expires_ts is None, "the fallback case needs no granted lease"
    tr = FakeTransport([[req]], lease_ms=3_000)
    runner = _runner(tmp_path, tr, spec="fixture_agent:build_medium")

    # Falls back to the transport's requested lease: a third of 3s, floored.
    assert runner._beat_interval_for(req) == pytest.approx(1.0, abs=0.05)
    # And it is genuinely READING that value rather than defaulting: a
    # transport that asked for half an hour must pace at 600s, not 1s.
    assert RolloutRunner._beat_interval_for(
        _runner(tmp_path, FakeTransport([[req]], lease_ms=1_800_000)), req
    ) == pytest.approx(600.0, abs=1.0)

    runner.serve_forever(max_rollouts=1)
    assert tr.heartbeats, "no granted lease is not a reason to stop beating"


def test_the_work_deadline_is_not_the_lease(tmp_path):
    """The budget is what bounds the run. A lease shorter than the work is
    normal -- the heartbeat renews it -- and must not shorten the rollout."""
    req = RolloutRequest(
        rollout_id="rl_1", turn_input="x",
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_1", api_key="k",
        artifacts_dir=str(tmp_path), deadline_ts=time.time() + 600,
        lease_expires_ts=time.time() + 1.0, lease_token="lt",
    )
    assert req.seconds_remaining(time.time()) > 500, (
        "the work budget followed the lease"
    )


# ── artifacts: for a coding agent, the artifact IS the reward ───────────────


class ArtifactTransport(FakeTransport):
    def __init__(self, *a, fail=False, **kw):
        super().__init__(*a, **kw)
        self.artifacts: list[dict] = []
        self._fail = fail

    def register_artifact(self, rollout, *, name, size_bytes, sha256,
                          content_type="application/octet-stream", role="output"):
        if self._fail:
            raise TransportError("registry down")
        self.artifacts.append({"name": name, "size": size_bytes, "sha256": sha256})
        return {}


def test_files_the_agent_wrote_are_registered_before_the_report(tmp_path):
    """`register_artifact` was fully implemented with NO caller, so an agent
    that produced a diff or a built binary reported only its commentary."""
    tr = ArtifactTransport([[_req(tmp_path)]])
    runner = _runner(tmp_path, tr, spec="fixture_agent:build_artifact_writer")

    runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"]
    names = sorted(a["name"] for a in tr.artifacts)
    assert names == ["out/patch.diff", "result.txt"], names
    for a in tr.artifacts:
        assert len(a["sha256"]) == 64 and a["size"] > 0


def test_the_trajectory_is_not_registered_as_an_artifact(tmp_path):
    """It is our bookkeeping, sitting in the same directory. Registering it
    would hand the customer's own capture back to them as a deliverable."""
    tr = ArtifactTransport([[_req(tmp_path)]])
    _runner(tmp_path, tr, spec="fixture_agent:build_artifact_writer").serve_forever(
        max_rollouts=1)
    assert all("tool_calls.jsonl" not in a["name"] for a in tr.artifacts)


def test_a_failed_registration_does_not_lose_the_rollout(tmp_path):
    """The expensive half already ran. An artifact we could not register is
    worth strictly less than the result it belongs to."""
    tr = ArtifactTransport([[_req(tmp_path)]], fail=True)
    events: list[str] = []
    runner = _runner(tmp_path, tr, spec="fixture_agent:build_artifact_writer",
                     on_event=lambda k, p: events.append(k))

    runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"], "the result was lost with the artifact"
    assert "artifact_failed" in events


def test_a_rollout_with_no_files_registers_nothing(tmp_path):
    tr = ArtifactTransport([[_req(tmp_path)]])
    _runner(tmp_path, tr).serve_forever(max_rollouts=1)
    assert tr.artifacts == []


def test_an_empty_claim_is_announced_rather_than_silent(tmp_path):
    """An empty queue and a broken claim route used to look identical.

    Both were a silent sleep; the only event in the loop, `claim_failed`, fires
    on the failing one alone. So an operator watching a flywheel that had
    claimed nothing for an hour could not tell "there is no work" from "my
    claim route is 404ing" without a packet capture. `claim_empty` gives the
    successful-but-empty case its own signal, carrying the server's own
    backoff so the operator can also see a control plane stalling them.
    """
    events: list[tuple[str, dict]] = []
    tr = FakeTransport([])
    runner = _runner(
        tmp_path, tr, on_event=lambda kind, ev: events.append((kind, ev))
    )
    runner.serve_forever()

    empties = [ev for kind, ev in events if kind == "claim_empty"]
    assert empties, f"no claim_empty emitted; saw {[k for k, _ in events]}"
    assert empties[0]["poll_after_ms"] == 10
    assert not [k for k, _ in events if k == "claim_failed"], (
        "an empty queue is not a failure and must not be reported as one"
    )


def test_a_raising_on_event_cannot_stop_lease_renewal(tmp_path):
    """The sharpest edge an observability hook could have.

    `lease_lost` and `heartbeat_failed` are emitted from inside the heartbeat
    thread's own exception handlers, and `_emit` had no guard. A callback that
    raised there exited the for, the while, and the THREAD -- after which no
    in-flight lease was renewed again, every later rollout collected a 410 at
    report time, and the exception reached nobody but the thread excepthook.

    The detector is the SECOND rollout: it can only be heartbeated by a thread
    that survived the first rollout's callback. Asserting on the first proves
    nothing, because the thread dies after emitting.
    """
    tr = FakeTransport(
        [[_req(tmp_path, "rl_1")], [_req(tmp_path, "rl_2")]],
        lease_ms=3_000,                       # -> a beat every second
        heartbeat_raises=TransportError("beat boom"),
    )
    raised: list[str] = []

    def explode(kind, ev):
        if kind == "heartbeat_failed":
            raised.append(ev["rollout_id"])
            raise RuntimeError("callback blew up inside the heartbeat thread")

    runner = _runner(tmp_path, tr, spec="fixture_agent:build_medium")
    runner._on_event = explode
    runner.serve_forever(max_rollouts=2)

    assert raised, "the callback never fired; this test proved nothing"
    assert tr.heartbeats.count("rl_2") >= 1, (
        f"the heartbeat thread did not survive the first raising callback, "
        f"so rl_2 was never renewed: heartbeats={tr.heartbeats}"
    )


def test_the_beat_interval_is_computed_once_and_does_not_decay(tmp_path):
    """`lease_expires_ts` is an ABSOLUTE timestamp, frozen at claim time.

    The heartbeat loop used to recompute `(granted - now) / 3` on every beat,
    which paced against the shrinking REMAINDER rather than the lease: the
    interval halved, then quartered, then pinned itself to the 1.0s floor. A
    300s grant on a 900s rollout became roughly 600 heartbeat POSTs where the
    design intends nine. Nothing was reaped -- the server extends on every beat
    -- so it was pure load, invisible from the client and paid for by the
    control plane.

    Before the schema declared `lease_expires_ts`, `additionalProperties:
    false` meant a conforming server could not send it, `granted` was always
    None, and the constant `lease_ms / 3` was used. Declaring the field is what
    made the decaying branch reachable, so the guard belongs with it.

    Asserted structurally rather than by counting beats in a wall-clock window,
    which this file already learned is a fact about the machine rather than
    about this code.
    """
    tr = FakeTransport([[_req(tmp_path, "rl_1")]], lease_ms=3_000)
    runner = _runner(tmp_path, tr, spec="fixture_agent:build_medium")

    seen: list[str] = []
    original = runner._beat_interval_for

    def spy(rollout):
        seen.append(rollout.rollout_id)
        return original(rollout)

    runner._beat_interval_for = spy
    runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"]
    assert seen.count("rl_1") == 1, (
        f"the interval was recomputed {seen.count('rl_1')} times; recomputing "
        f"it per beat is what makes it decay toward the 1s floor"
    )


def test_a_granted_lease_sets_the_interval_to_a_third_of_it(tmp_path):
    """The value itself, since the test above only pins how often it is taken."""
    tr = FakeTransport([], lease_ms=1_800_000)          # 600s if unused
    runner = _runner(tmp_path, tr)
    granted = replace(_req(tmp_path, "rl_1"), lease_expires_ts=time.time() + 90.0)

    interval = runner._beat_interval_for(granted)

    assert 28.0 <= interval <= 31.0, interval          # a third of 90s
    # and the requested lease is NOT what was used, which is the whole point
    assert interval < 600.0
