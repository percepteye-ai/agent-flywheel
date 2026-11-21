"""A result that failed to send is not resent on every slot completion
(PY-H22): it waits its jittered backoff, and the drain skips it unread."""
from __future__ import annotations

import time

from agent_flywheel import RolloutRequest
from agent_flywheel.errors import TransportError
from agent_flywheel.runner import RolloutRunner


def _req(tmp_path, rid):
    return RolloutRequest(rollout_id=rid, turn_input="t", llm_url="https://gw.invalid/v1",
                          api_key="k", artifacts_dir=str(tmp_path),
                          deadline_ts=time.time() + 60, lease_token=f"lt_{rid}")


class _Down:
    agent_id = "a"

    def __init__(self, tmp_path, n):
        self.queue = [_req(tmp_path, f"rl_{i}") for i in range(n)]
        self.reports: list[str] = []

    def claim(self, n):
        batch, self.queue = self.queue[:1], self.queue[1:]
        return batch, 10

    def heartbeat(self, r, *, progress=None):
        return None

    def report(self, r, out):
        self.reports.append(r.rollout_id)
        raise TransportError("HTTP 503: down", status_code=503)

    def abandon(self, r, *, reason, detail=""):
        return None


def test_a_failed_delivery_is_not_retried_on_every_slot_completion(tmp_path):
    import warnings

    tr = _Down(tmp_path, 5)
    runner = RolloutRunner(["true"], transport=tr, input_shape="text",
                           artifacts_dir=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=5)
    # Once in the episode, once in the forced drain on the way out. Without
    # the backoff the four later slot completions each resent it too.
    assert tr.reports.count("rl_0") == 2, tr.reports
