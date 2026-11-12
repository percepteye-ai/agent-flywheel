"""A free slot is refilled at once: the runner never waits for a whole batch.

tau3 7456644c (2026-11-12): the serve loop claimed a batch, then waited for
EVERY rollout in it before claiming again, so an agent with free slots sat
behind its slowest rollout while work queued -- rollouts waited 1-2 minutes to
be claimed, and two agents with 32 slots between them held 2 in flight with 6
queued. Mutation: restore the batch barrier (wait for all claimed rollouts
before the next claim) -- red.
"""
from __future__ import annotations

import threading
import time

from agent_flywheel.contract import RolloutRequest
from agent_flywheel.runner import RolloutRunner


def _req(i: int) -> RolloutRequest:
    return RolloutRequest(
        rollout_id=f"rl_{i}", turn_input="go",
        llm_url=f"https://gw.invalid/llm/v1/rollouts/rl_{i}", api_key="perk_x",
        artifacts_dir=f"rollouts/rl_{i}", deadline_ts=time.time() + 60,
        lease_token="lt",
    )


class _OneAtATime:
    """A control plane that hands out one queued rollout per claim, as a
    long-poll does when work arrives one at a time."""

    lease_ms = 1_800_000

    def __init__(self, n: int) -> None:
        self.queue = [_req(i) for i in range(n)]
        self.lock = threading.Lock()

    def claim(self, n):
        with self.lock:
            return ([self.queue.pop(0)] if self.queue else []), 5

    def heartbeat(self, r, *, progress=None):
        return None

    def report(self, r, o):
        return {}

    def abandon(self, r, *, reason, detail=""):
        pass


def test_a_slow_rollout_does_not_hold_the_free_slots(tmp_path, monkeypatch):
    tr = _OneAtATime(8)
    runner = RolloutRunner(["true"], transport=tr, input_shape="text",
                           artifacts_dir=str(tmp_path), rollout_timeout_s=30.0,
                           concurrency=4)
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    ended: dict[str, float] = {}

    def serve(r):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        time.sleep(0.6 if r.rollout_id == "rl_0" else 0.05)
        with lock:
            state["active"] -= 1
            ended[r.rollout_id] = time.monotonic()
        return True

    monkeypatch.setattr(runner, "_serve_one", serve)
    delivered = runner.serve_forever(max_rollouts=8)
    assert delivered == 8
    assert state["peak"] >= 3
    # Every fast rollout finished while the slow one still ran.
    assert max(t for k, t in ended.items() if k != "rl_0") < ended["rl_0"]
