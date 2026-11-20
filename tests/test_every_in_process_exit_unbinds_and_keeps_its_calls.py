"""Every exit of an in-process rollout unbinds its adapters and keeps the tool
calls it recorded (PY-H20); a timed-out async agent is a deadline, not the
agent's error, on every supported Python (PY-H26)."""
from __future__ import annotations

import asyncio
import time

import pytest

from agent_flywheel import RolloutRequest, record_tool_call
from agent_flywheel import binding as _binding
from agent_flywheel.entrypoint import resolve_entrypoint
from agent_flywheel.runner import run_rollout_in_process


def _rollout(tmp_path):
    return RolloutRequest(
        rollout_id="rl_ip_1", turn_input="go",
        llm_url="https://gw.example.invalid/llm/v1/rollouts/rl_ip_1",
        api_key="perk_x", artifacts_dir=str(tmp_path),
        deadline_ts=time.time() + 60,
    )


class _CallsThenRaises:
    def invoke(self, task):
        record_tool_call("lookup", {"q": task}, outcome="ok", status_code=200)
        raise RuntimeError("agent broke after acting")


class _CallsThenHangs:
    def invoke(self, task):
        record_tool_call("lookup", {"q": task}, outcome="ok", status_code=200)
        time.sleep(5)
        return "late"


class _SlowAsync:
    async def invoke(self, task):
        await asyncio.sleep(5)
        return "late"


def _run(agent, tmp_path, timeout_s=0.5, adapters=None):
    resolved = resolve_entrypoint(agent, input_shape="text")
    traj = tmp_path / "traj"
    traj.mkdir()
    return run_rollout_in_process(resolved, _rollout(tmp_path),
                                  trajectory_dir=str(traj),
                                  timeout_s=timeout_s, adapters=adapters)


@pytest.mark.parametrize("agent,reason", [
    (_CallsThenRaises(), "entrypoint_error"),
    (_CallsThenHangs(), "deadline_exceeded"),
])
def test_the_calls_made_before_a_crash_or_timeout_are_kept(tmp_path, agent, reason):
    out = _run(agent, tmp_path)
    assert out.exclusion_reason == reason, out.exclusion_detail
    assert out.tool_calls is not None and [c.name for c in out.tool_calls] == [
        "lookup"], "the calls that explain the failure were dropped"


@pytest.mark.parametrize("agent", [_CallsThenRaises(), _CallsThenHangs()])
def test_adapters_are_unbound_on_a_crash_or_timeout(tmp_path, monkeypatch, agent):
    unbound: list[str] = []
    monkeypatch.setattr(_binding, "bind_selected",
                        lambda *a, **k: (None, [lambda: unbound.append("x")]))
    _run(agent, tmp_path, adapters="auto")
    assert unbound == ["x"], "the rollout's framework patches outlived it"


def test_a_timed_out_async_agent_is_a_deadline_not_its_own_error(tmp_path):
    """On 3.10 ``asyncio.wait_for`` raises ``asyncio.TimeoutError``, which is
    not ``concurrent.futures.TimeoutError`` there; it fell through to
    ``entrypoint_error`` with ``success=False`` -- scored as the policy's own
    failure."""
    out = _run(_SlowAsync(), tmp_path, timeout_s=0.3)
    assert out.exclusion_reason == "deadline_exceeded", out.exclusion_detail
    assert out.success is None
