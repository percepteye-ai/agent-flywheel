"""Two claims of ONE rollout id, running at once in one process (PY-C8).

A lapsed lease re-queues the same rollout id with a new lease token, so this
process can claim an id it is still running (Mission Control 2026-11-20: 5 of
80). Keyed by the bare id, the first attempt's cleanup popped the second's
in-flight entry -- ending its heartbeats -- and deleted the directory it was
running in.
"""
from __future__ import annotations

import threading
import time

import pytest

from agent_flywheel import RolloutRequest
from agent_flywheel.runner import RolloutRunner, _attempt_key


def _req(tmp_path, token, task, lease_s=60.0):
    now = time.time()
    return RolloutRequest(
        rollout_id="rl_same", turn_input=task,
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_same", api_key="k",
        artifacts_dir=str(tmp_path), deadline_ts=now + 60,
        lease_expires_ts=now + lease_s, lease_token=token,
    )


class _Transport:
    agent_id = "a"

    def __init__(self, batch):
        self.batch = batch
        self.reports: dict[str, object] = {}
        self.beats: list[tuple[str, float]] = []
        self.lock = threading.Lock()

    def claim(self, n):
        batch, self.batch = self.batch, []
        return batch, 50

    def heartbeat(self, r, *, progress=None):
        with self.lock:
            self.beats.append((r.lease_token, time.time()))
        return None

    def report(self, r, out):
        with self.lock:
            self.reports[r.lease_token] = out
        return {}

    def abandon(self, r, *, reason, detail=""):
        return None


@pytest.mark.skipif(__import__("os").name == "nt", reason="sh")
def test_two_attempts_of_one_id_keep_their_own_directory_and_entry(tmp_path):
    slow = _req(tmp_path, "lt_second", "slow", lease_s=3.0)
    fast = _req(tmp_path, "lt_first", "fast")
    tr = _Transport([slow, fast])
    # The fast attempt finishes (and is purged) while the slow one is still
    # writing into its working directory.
    script = (
        'if [ "$PERCEPTEYE_TURN_INPUT" = fast ]; then echo fast; exit 0; fi; '
        'sleep 2.5; echo ok > out.txt && cat out.txt'
    )
    runner = RolloutRunner(["sh", "-c", script], transport=tr, input_shape="text",
                           artifacts_dir=str(tmp_path), concurrency=2)
    finished = {}
    real = runner._serve_one

    def spy(r):
        try:
            return real(r)
        finally:
            finished[r.lease_token] = time.time()

    runner._serve_one = spy
    runner.serve_forever(max_rollouts=2)

    assert _attempt_key(slow) != _attempt_key(fast)
    out = tr.reports["lt_second"]
    assert out.exclusion_reason is None, (
        f"the second attempt's directory was deleted under it: {out.exclusion_detail}")
    assert out.final_text == "ok"
    late = [t for tok, t in tr.beats
            if tok == "lt_second" and t > finished["lt_first"]]
    assert late, "the first attempt's cleanup ended the second attempt's heartbeats"
