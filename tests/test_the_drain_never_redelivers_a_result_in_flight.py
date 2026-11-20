"""The drain and a worker never deliver the same spooled result twice.

The drain read a spool file, THEN checked whether its rollout was still in
flight. A worker that reported, removed the file and left ``_inflight`` in
between passed that check, and the drain sent the same result again (seen
intermittently as ``rl_0`` reported twice in test_a_claimed_batch_runs_
concurrently).
"""
from __future__ import annotations

import threading
import time
import warnings

from agent_flywheel import RolloutRequest
from agent_flywheel import runner as _rmod
from agent_flywheel.runner import RolloutRunner


def test_a_result_the_worker_is_delivering_is_not_also_drained(tmp_path, monkeypatch):
    spooled, drain_read, served = (threading.Event() for _ in range(3))
    req = RolloutRequest(rollout_id="rl_0", turn_input="t", llm_url="u", api_key="k",
                         artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60,
                         lease_token="lt_0")

    class _T:
        agent_id = "a"
        reported: list[str] = []  # noqa: RUF012

        def __init__(self):
            self.batches = [[req]]

        def claim(self, n):
            return (self.batches.pop(0) if self.batches else []), 10

        def heartbeat(self, r, *, progress=None):
            return None

        def report(self, r, out):
            if not _T.reported:
                spooled.set()
                drain_read.wait(2.0)
            _T.reported.append(r.rollout_id)
            return {}

        def abandon(self, r, *, reason, detail=""):
            return None

    real_read = _rmod._read_spool

    def read_spool(path):
        record = real_read(path)
        if threading.current_thread() is threading.main_thread() and spooled.is_set():
            drain_read.set()
            served.wait(3.0)
        return record

    monkeypatch.setattr(_rmod, "_read_spool", read_spool)
    runner = RolloutRunner(["true"], transport=_T(), input_shape="text",
                           artifacts_dir=str(tmp_path), concurrency=2)
    real_serve = runner._serve_one

    def serve_one(r):
        try:
            return real_serve(r)
        finally:
            served.set()
            runner.stop()

    runner._serve_one = serve_one
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=2)
    assert _T.reported == ["rl_0"], f"delivered {len(_T.reported)} times"
