"""A command rollout whose lease is lost is killed and reported as
``credential_expired`` (PY-H23), including a child spawned just after the beat
that lost it (PY-H24)."""
from __future__ import annotations

import os
import threading
import time

import pytest

from agent_flywheel import RolloutRequest
from agent_flywheel.errors import LeaseLost
from agent_flywheel.runner import RolloutRunner, _attempt_key, _InFlight

pytestmark = pytest.mark.skipif(os.name == "nt", reason="sh")


class _LosesTheLease:
    agent_id = "a"
    def claim(self, n): return [], 1000
    def heartbeat(self, r, *, progress=None): raise LeaseLost("reaped")
    def report(self, r, o): return {}
    def abandon(self, r, *, reason, detail=""): return None


def _req(tmp_path):
    return RolloutRequest(rollout_id="rl_ll", turn_input="t",
                          llm_url="https://gw.invalid/v1", api_key="k",
                          artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60,
                          lease_token="lt")


def _runner(tmp_path):
    return RolloutRunner(["sh", "-c", "echo started; sleep 30"],
                         transport=_LosesTheLease(), input_shape="text",
                         artifacts_dir=str(tmp_path))


def test_a_command_killed_for_a_lost_lease_is_credential_expired(tmp_path):
    runner = _runner(tmp_path)
    req = _req(tmp_path)
    runner._inflight[_attempt_key(req)] = _InFlight(req, 1.0, time.time())
    beat = threading.Thread(target=runner._heartbeat_loop, daemon=True)
    beat.start()
    try:
        started = time.monotonic()
        out = runner.run_one(req)
    finally:
        runner._beat_stop.set()
    assert time.monotonic() - started < 20, "the lease-lost kill never landed"
    assert out.exclusion_reason == "credential_expired", (
        f"a lease kill was scored as the agent's own failure: "
        f"{out.exclusion_reason}: {out.exclusion_detail}")


def test_a_child_spawned_after_the_lease_was_lost_is_killed_at_once(tmp_path):
    runner = _runner(tmp_path)
    req = _req(tmp_path)
    entry = _InFlight(req, 3600.0, time.time() + 3600)
    entry.lease_lost = True          # the beat already lost it; no beat is due
    runner._inflight[_attempt_key(req)] = entry
    started = time.monotonic()
    out = runner.run_one(req)
    assert time.monotonic() - started < 10, "the child ran on against a lost lease"
    assert out.exclusion_reason == "credential_expired"
