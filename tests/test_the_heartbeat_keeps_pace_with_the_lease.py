"""The heartbeat paces itself against the extension it asks for and the
expiry the control plane answers (PY-C14), on its own short budget, and one
slow beat does not hold up the others (PY-H25)."""
from __future__ import annotations

import threading
import time

from agent_flywheel import RolloutRequest
from agent_flywheel.runner import RolloutRunner, _InFlight
from agent_flywheel.transport import AttachTransport


def _req(rid, lease_s=1800.0, token=None):
    now = time.time()
    return RolloutRequest(rollout_id=rid, turn_input="t", llm_url="https://gw.invalid/v1",
                          api_key="k", artifacts_dir="/tmp", deadline_ts=now + 3600,
                          lease_expires_ts=now + lease_s, lease_token=token or f"lt_{rid}")


class _Client:
    def __init__(self, body=None):
        self.calls = []
        self.body = body if body is not None else {}

    def request(self, method, path, **kw):
        self.calls.append((method, path, kw))
        return self.body


def _runner(transport, concurrency=2):
    return RolloutRunner(["true"], transport=transport, input_shape="text",
                         artifacts_dir="/tmp", concurrency=concurrency)


def test_a_full_lease_is_paced_against_the_extension_not_the_grant():
    """1800s granted, 600s per extension: pacing at 600s put every beat after
    the first at the very expiry it exists to pre-empt."""
    tr = AttachTransport(_Client(), agent_id="a", lease_ms=1_800_000)
    interval = _runner(tr)._beat_interval_for(_req("rl_1", lease_s=1800.0))
    assert 199.0 <= interval <= 200.5, interval


def test_the_beat_returns_the_new_expiry_and_has_its_own_short_budget():
    expires = time.time() + 600
    client = _Client({"lease_expires_ts": expires})
    tr = AttachTransport(client, agent_id="a")
    before = time.time()
    assert tr.heartbeat(_req("rl_1")) == expires
    _, path, kw = client.calls[-1]
    assert path == "/rollouts/rl_1/heartbeat"
    assert kw["deadline"] - before <= 10.5, "a beat ran the report's two-minute envelope"
    assert kw["timeout"] <= 5.0


def test_the_next_beat_is_scheduled_from_the_expiry_the_server_returned():
    """A server that answers a shorter lease than asked is followed: the next
    beat comes with two beats of margin before THAT expiry."""
    class _Short:
        lease_ms = 1_800_000
        heartbeat_extend_ms = 600_000
        def heartbeat(self, r, *, progress=None):
            return time.time() + 6.0

    runner = _runner(_Short())
    req = _req("rl_1")
    entry = _InFlight(req, runner._beat_interval_for(req), time.time())
    runner._beat_one(entry)
    assert entry.next_beat_at - time.time() <= 2.5, (
        f"next beat in {entry.next_beat_at - time.time():.0f}s against a lease "
        f"that ends in 6s")


def test_one_slow_beat_does_not_starve_the_other_leases():
    gate = threading.Event()
    beaten: dict[str, float] = {}

    class _OneSlow:
        lease_ms = 30_000
        def heartbeat(self, r, *, progress=None):
            if r.rollout_id == "rl_slow":
                gate.wait(10)
            beaten.setdefault(r.rollout_id, time.time())
            return None

    runner = _runner(_OneSlow())
    now = time.time()
    for rid in ("rl_slow", "rl_fast"):
        req = _req(rid, lease_s=30.0)
        runner._inflight[rid] = _InFlight(req, 10.0, now)
    t = threading.Thread(target=runner._heartbeat_loop, daemon=True)
    started = time.time()
    t.start()
    try:
        deadline = time.time() + 6
        while "rl_fast" not in beaten and time.time() < deadline:
            time.sleep(0.05)
    finally:
        gate.set()
        runner._beat_stop.set()
        t.join(5)
    assert "rl_fast" in beaten and beaten["rl_fast"] - started < 4.0, (
        "a slow beat for one lease held up the beat for another")
