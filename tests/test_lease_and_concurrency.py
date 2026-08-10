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

import httpx
import pytest

from percepteye_flywheel import RolloutRequest
from percepteye_flywheel._http import ControlPlaneClient
from percepteye_flywheel.errors import LeaseLost, TransportError
from percepteye_flywheel.runner import RolloutRunner

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
    assert len(tr.heartbeats) >= 2, (
        f"a ~2.5s rollout under a 3s lease must beat at least twice, "
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


def test_a_claimed_batch_runs_concurrently(tmp_path):
    """Claiming N and serving them one after another held N-1 leases on work
    nobody was doing. Three 2s rollouts must not take 6s."""
    batch = [_req(tmp_path, rid=f"rl_{i}") for i in range(3)]
    tr = FakeTransport([batch])
    runner = _runner(
        tmp_path, tr, spec="fixture_agent:build_medium", concurrency=3
    )

    started = time.time()
    runner.serve_forever(max_rollouts=3)
    elapsed = time.time() - started

    assert sorted(tr.reported) == ["rl_0", "rl_1", "rl_2"]
    assert elapsed < 4.5, (
        f"three 2s rollouts took {elapsed:.1f}s -- they ran sequentially"
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


def test_the_ledger_directory_is_purged_after_serving(tmp_path):
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
    from percepteye_flywheel.transport import AttachTransport

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
