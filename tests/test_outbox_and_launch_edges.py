"""Outbox and launch edges (PY-M11, PY-M12)."""
from __future__ import annotations

import json
import os
import time

import pytest

from agent_flywheel import RolloutRequest
from agent_flywheel.runner import RolloutRunner, _move_aside


class _T:
    agent_id = "a"
    def claim(self, n): return [], 1000
    def heartbeat(self, r, *, progress=None): return None
    def report(self, r, o): return {}
    def abandon(self, r, *, reason, detail=""): return None


def _runner(tmp_path, argv=("true",)):
    return RolloutRunner(list(argv), transport=_T(), input_shape="text",
                         artifacts_dir=str(tmp_path))


def test_a_spool_file_from_a_newer_sdk_is_left_where_it_is(tmp_path):
    """The schema tag exists so a future SDK's file is left alone; the drain
    retired it as 'unreadable' instead."""
    runner = _runner(tmp_path)
    root = runner._outbox_dir()
    root.mkdir(parents=True)
    future = root / "rl_future.json"
    future.write_text(json.dumps({"schema": "percepteye.outbox/v2", "x": 1}))
    runner._drain_outbox()
    assert future.is_file(), sorted(p.name for p in root.iterdir())
    assert not list(root.glob("*.unreadable*"))


def test_a_failed_retirement_leaves_no_empty_reservation(tmp_path, monkeypatch):
    spool = tmp_path / "rl_1.json"
    spool.write_text("{}")

    def _boom(*a, **k):
        raise OSError("read-only")

    monkeypatch.setattr(os, "replace", _boom)
    assert _move_aside(spool, "refused-409") == spool
    assert sorted(p.name for p in tmp_path.iterdir()) == ["rl_1.json"]


@pytest.mark.skipif(os.name == "nt", reason="sh")
def test_a_task_too_big_for_the_environment_still_launches(tmp_path):
    """Linux refuses exec with one env string over 128 KiB (E2BIG); macOS
    caps the whole environment. Either way the launch used to fail."""
    task = "x" * (1536 * 1024)   # over both limits
    runner = _runner(tmp_path, ["sh", "-c", 'wc -c | tr -d " "'])
    req = RolloutRequest(rollout_id="rl_big", turn_input=task,
                         llm_url="https://gw.invalid/v1", api_key="k",
                         artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60)
    out = runner.run_one(req)
    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.final_text == str(len(task))
