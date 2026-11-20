"""The control plane's episode budget bounds the agent, and every exit keeps
the policy calls it counted.

τ³ banking, 2026-11-19: the control plane minted ~3455s episodes and the SDK
killed the agent at its own 900s default on every long task -- 200 of 562
rollouts. Those timeouts (and crashes) returned with no ``llm_call_count``,
so the control plane's reconciliation held them out as
``llm_calls_unobserved`` -- undoing its own ruling that a policy that ran
out of time is a scored failure.
"""
from __future__ import annotations

import time

import pytest

from agent_flywheel.contract import RolloutRequest
from agent_flywheel.runner import (
    DEFAULT_ROLLOUT_TIMEOUT_S,
    episode_plan,
    resolve_entrypoint,
    run_rollout_in_process,
)


def test_with_no_ceiling_the_budget_is_the_bound():
    """Mutation: restore a 900s default -- red."""
    assert DEFAULT_ROLLOUT_TIMEOUT_S is None
    plan = episode_plan(3455.0, envelope_s=10.0)
    assert plan.child_s == pytest.approx(3455.0 - 10.0 - plan.teardown_s)


def test_an_explicit_ceiling_still_caps_it():
    assert episode_plan(3455.0, rollout_timeout_s=600.0,
                        envelope_s=10.0).child_s == 600.0


class _Calls:
    """Counts as the SDK counter does: one completed policy POST each."""

    def __init__(self, url: str, n: int) -> None:
        self.url, self.n = url, n

    def make(self) -> None:
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"ok": True})

        with httpx.Client(transport=httpx.MockTransport(handler)) as c:
            for _ in range(self.n):
                c.post(self.url + "chat/completions", json={})


def _request(url: str, artifacts: str) -> RolloutRequest:
    return RolloutRequest(rollout_id="rl_1", turn_input="go", llm_url=url,
                          api_key="k", artifacts_dir=artifacts,
                          deadline_ts=time.time() + 60)


@pytest.mark.parametrize("ending,reason", [
    ("timeout", "deadline_exceeded"),
    ("raise", "entrypoint_error"),
])
def test_a_timed_out_or_crashed_rollout_keeps_its_count(tmp_path, ending, reason):
    """Mutation: drop ``llm_call_count`` from either exit -- red."""
    url = "http://gw/llm/v1/o/org/rollouts/rl_1/"
    calls = _Calls(url, 3)

    def agent(_text: str) -> str:
        calls.make()
        if ending == "raise":
            raise RuntimeError("context length exceeded")
        time.sleep(5)
        return "late"

    out = run_rollout_in_process(
        resolve_entrypoint(agent, input_shape="text"), _request(url, str(tmp_path)),
        trajectory_dir=str(tmp_path), timeout_s=1.0)
    assert out.exclusion_reason == reason
    assert out.llm_call_count == 3
