"""A child whose agent outlives its budget still delivers its result.

τ³ banking, 2026-11-20 (prompt evolution 4c4ce5f1): the synchronous agent
was still running when the child's timeout fired. The child wrote its
``deadline_exceeded`` result -- carrying the policy calls it counted -- and
then could not exit: the interpreter waits for the agent's worker thread at
shutdown. The parent killed it 15 s later and discarded what it had already
written, so the report had no ``llm_call_count`` and Mission Control held
the policy's own timeout out as ``llm_calls_unobserved``.
"""
from __future__ import annotations

import os
import pathlib
import time

import pytest

from agent_flywheel import RolloutRequest
from agent_flywheel.runner import RolloutRunner

TESTS_DIR = str(pathlib.Path(__file__).resolve().parent)


class _NullTransport:
    def claim(self, n): return [], 1000
    def heartbeat(self, r, progress=None): return None
    def report(self, r, o): return {}
    def abandon(self, r, reason, detail=""): return None


@pytest.fixture(autouse=True)
def _child_can_import(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", TESTS_DIR + (os.pathsep + existing if existing else ""))


def test_the_timed_out_childs_count_reaches_the_report(tmp_path):
    """Mutation: drop the hard exit after the child writes its result --
    the parent kills it and the count is lost, red."""
    runner = RolloutRunner(
        "fixture_agent:build_slow_after_policy_calls",
        transport=_NullTransport(), input_shape="text",
        artifacts_dir=str(tmp_path), isolation="subprocess", concurrency=1,
        rollout_timeout_s=3.0)
    rollout = RolloutRequest(
        rollout_id="rl_slow", turn_input="go",
        llm_url="https://gw.example.invalid/llm/v1/rollouts/rl_slow",
        api_key="perk_x", artifacts_dir=str(tmp_path),
        deadline_ts=time.time() + 60)
    started = time.time()
    out = runner.run_one(rollout)
    assert out.exclusion_reason == "deadline_exceeded", out.exclusion_detail
    assert out.llm_call_count == 3, out.exclusion_detail
    assert time.time() - started < 15, "the child outlived its own result"
