"""An agent backed by its own environment can tell which task it was handed.

A customer that registered its own tasks (a benchmark, a simulator, a test
suite) must start exactly the task the control plane claimed, by that task's
id. The free-text turn input cannot carry it reliably. The claim's ``task_id``
reaches the agent through ``current_task_id()`` on every serving path: a Python
entrypoint in a child, the same in-process, and a command (``PERCEPTEYE_TASK_ID``).
"""
from __future__ import annotations

import os
import pathlib
import stat
import time

import pytest

from agent_flywheel import RolloutRequest, current_task_id
from agent_flywheel.isolation import build_child_env
from agent_flywheel.runner import RolloutRunner

TESTS_DIR = str(pathlib.Path(__file__).resolve().parent)


class _T:
    lease_ms = 1_800_000

    def claim(self, n): return [], 10
    def heartbeat(self, r, *, progress=None): return None
    def report(self, r, o): return {}
    def abandon(self, r, *, reason, detail=""): return None


def _req(tmp_path, task_id):
    meta = {} if task_id is None else {"task_id": task_id}
    return RolloutRequest(
        rollout_id="rl_t", turn_input="do it",
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_t", api_key="perk_x",
        artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60, metadata=meta,
    )


@pytest.fixture(autouse=True)
def _child_can_import(monkeypatch):
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv("PYTHONPATH", TESTS_DIR + (os.pathsep + existing if existing else ""))
    monkeypatch.delenv("PERCEPTEYE_TASK_ID", raising=False)
    monkeypatch.delenv("PERCEPTEYE_ROLLOUT_ID", raising=False)


@pytest.mark.parametrize("isolation", ["subprocess", "inprocess"])
def test_a_python_agent_reads_the_claimed_task_id(tmp_path, isolation):
    out = RolloutRunner(
        "fixture_agent:build_task_probe", transport=_T(), input_shape="text",
        artifacts_dir=str(tmp_path), isolation=isolation, concurrency=1,
        rollout_timeout_s=60.0,
    ).run_one(_req(tmp_path, "banking_knowledge:task_017"))
    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.final_text == "task_id=banking_knowledge:task_017"


def test_a_command_agent_reads_the_claimed_task_id(tmp_path):
    script = tmp_path / "agent.sh"
    script.write_text('#!/bin/sh\necho "task_id=$PERCEPTEYE_TASK_ID"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    out = RolloutRunner(
        ["sh", str(script)], transport=_T(), input_shape="text",
        artifacts_dir=str(tmp_path), rollout_timeout_s=30.0,
    ).run_one(_req(tmp_path, "task_017"))
    assert out.final_text == "task_id=task_017"


def test_no_task_named_means_none_not_a_guess(tmp_path):
    out = RolloutRunner(
        "fixture_agent:build_task_probe", transport=_T(), input_shape="text",
        artifacts_dir=str(tmp_path), isolation="inprocess", concurrency=1,
        rollout_timeout_s=60.0,
    ).run_one(_req(tmp_path, None))
    assert out.final_text == "task_id=None"


def test_an_unsafe_id_never_crosses_into_the_child_environment():
    env = build_child_env(
        rollout_id="rl_t", llm_url="u", api_key="k", trajectory_dir="/tmp/x",
        parent_env={}, task_id="bad\nid",
    )
    assert "PERCEPTEYE_TASK_ID" not in env


def test_outside_a_rollout_there_is_no_task():
    assert current_task_id() is None


@pytest.mark.parametrize("isolation", ["subprocess", "inprocess"])
def test_an_agent_reaches_its_auxiliary_models_on_the_rollout_route(tmp_path, isolation):
    """A user simulator or judge must not be the policy: the rollout's own
    route serves other models at ``<llm_url>/hosted/<provider>``."""
    out = RolloutRunner(
        "fixture_agent:build_hosted_probe", transport=_T(), input_shape="text",
        artifacts_dir=str(tmp_path), isolation=isolation, concurrency=1,
        rollout_timeout_s=60.0,
    ).run_one(_req(tmp_path, None))
    assert out.final_text == "hosted=https://gw.invalid/llm/v1/rollouts/rl_t/hosted/openai"


def test_a_command_agent_gets_the_auxiliary_base(tmp_path):
    script = tmp_path / "agent.sh"
    script.write_text('#!/bin/sh\necho "$PERCEPTEYE_HOSTED_URL"\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    out = RolloutRunner(
        ["sh", str(script)], transport=_T(), input_shape="text",
        artifacts_dir=str(tmp_path), rollout_timeout_s=30.0,
    ).run_one(_req(tmp_path, None))
    assert out.final_text == "https://gw.invalid/llm/v1/rollouts/rl_t/hosted"


def test_outside_a_rollout_there_is_no_auxiliary_route():
    from agent_flywheel import hosted_model_url

    assert hosted_model_url("openai") is None


def test_a_command_agent_reports_its_own_grade(tmp_path):
    """An environment-backed CLI harness grades the episode itself. Mutation:
    unwire ``_attach_reported_outcome`` -> no reward reaches the report."""
    script = tmp_path / "agent.sh"
    script.write_text(
        '#!/bin/sh\nprintf \'{"reward": 1.0, "success": true}\' '
        '> "$PERCEPTEYE_TRAJECTORY_DIR/outcome.json"\necho done\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    out = RolloutRunner(
        ["sh", str(script)], transport=_T(), input_shape="text",
        artifacts_dir=str(tmp_path), rollout_timeout_s=30.0,
    ).run_one(_req(tmp_path, "task_017"))
    assert (out.reward, out.success) == (1.0, True)


def test_a_malformed_grade_is_ignored_not_coerced(tmp_path):
    script = tmp_path / "agent.sh"
    script.write_text(
        '#!/bin/sh\nprintf \'{"reward": "1", "success": "yes"}\' '
        '> "$PERCEPTEYE_TRAJECTORY_DIR/outcome.json"\necho done\n')
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    out = RolloutRunner(
        ["sh", str(script)], transport=_T(), input_shape="text",
        artifacts_dir=str(tmp_path), rollout_timeout_s=30.0,
    ).run_one(_req(tmp_path, None))
    assert (out.reward, out.success) == (None, None)
