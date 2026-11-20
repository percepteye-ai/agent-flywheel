"""The per-rollout directory is created, run in, collected and rmtree'd; it
must be a real directory directly inside ``_percepteye`` (PY-C6, runner side)."""
from __future__ import annotations

import os
import time

import pytest

from agent_flywheel import RolloutRequest
from agent_flywheel.contract import RolloutOutput
from agent_flywheel.errors import ContractViolation
from agent_flywheel.runner import RolloutRunner, _attempt_key

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX symlinks")


class _T:
    agent_id = "a"
    def claim(self, n): return [], 1000
    def heartbeat(self, r, *, progress=None): return None
    def report(self, r, o): return {}
    def abandon(self, r, *, reason, detail=""): return None


def _req(tmp_path, rid="rl_x"):
    return RolloutRequest(rollout_id=rid, turn_input="t", llm_url="https://gw.invalid/v1",
                          api_key="k", artifacts_dir=str(tmp_path),
                          deadline_ts=time.time() + 60, lease_token="lt")


def _runner(tmp_path):
    return RolloutRunner(["true"], transport=_T(), input_shape="text",
                         artifacts_dir=str(tmp_path / "art"))


@pytest.mark.parametrize("rid", ["../../escape", "/etc", "_outbox", ".."])
def test_an_id_that_escapes_the_base_is_refused_before_anything_runs(tmp_path, rid):
    runner = _runner(tmp_path)
    req = _req(tmp_path, rid)
    req = RolloutRequest(**{**req.__dict__, "lease_token": None})
    with pytest.raises(ContractViolation, match="refusing to run"):
        runner.run_one(req)


def test_a_planted_symlink_is_neither_collected_nor_purged_through(tmp_path):
    """Another rollout's agent (cwd = a sibling slot) replaces this rollout's
    slot with a link to a host directory."""
    secret_dir = tmp_path / "host"
    secret_dir.mkdir()
    (secret_dir / "credentials").write_text("aws_secret=hunter2\n")
    runner = _runner(tmp_path)
    req = _req(tmp_path)
    slot = tmp_path / "art" / "_percepteye" / _attempt_key(req)
    slot.parent.mkdir(parents=True)
    slot.symlink_to(secret_dir, target_is_directory=True)

    out = RolloutOutput(final_text="")
    runner._collect_artifacts(req, out)
    assert "hunter2" not in repr(out.artifacts) and not out.artifacts

    runner._purge_trajectory(_attempt_key(req))
    assert (secret_dir / "credentials").exists(), "the purge followed the link"
