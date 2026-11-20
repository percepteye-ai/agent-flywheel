"""The child's result travels on a descriptor the agent cannot write (PY-C7),
and nothing the agent started outlives the child (PY-H21)."""
from __future__ import annotations

import os
import pathlib
import signal
import time

import pytest

from agent_flywheel import RolloutRequest
from agent_flywheel.runner import RolloutRunner

TESTS_DIR = str(pathlib.Path(__file__).resolve().parent)

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX process groups")


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


def _rollout(tmp_path, seconds=60.0):
    return RolloutRequest(
        rollout_id="rl_io_1", turn_input="go",
        llm_url="https://gw.example.invalid/llm/v1/rollouts/rl_io_1",
        api_key="perk_x", artifacts_dir=str(tmp_path),
        deadline_ts=time.time() + seconds,
    )


def _runner(tmp_path, spec, **kw):
    return RolloutRunner(spec, transport=_NullTransport(), input_shape="text",
                         artifacts_dir=str(tmp_path), isolation="subprocess",
                         concurrency=1, **kw)


def test_agent_writes_to_stdout_cannot_touch_the_result(tmp_path):
    """``print(..., end="")`` glued itself to the result line, and a raw write
    to fd 1 could replace it outright with a payload the agent authored."""
    out = _runner(tmp_path, "fixture_child_io:build_noisy").run_one(_rollout(tmp_path))
    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.final_text == "the real answer"


def test_a_grandchild_holding_stderr_neither_delays_nor_survives_the_result(
        tmp_path, monkeypatch):
    """The child returned at once, but a process it started held stderr, so
    ``communicate()`` sat out the whole timeout and then reported the rollout
    as ``deadline_exceeded`` -- the result it had already received thrown
    away -- and the grandchild lived on."""
    pid_file = tmp_path / "grandchild.pid"
    monkeypatch.setenv("FIXTURE_GRANDCHILD_PID_FILE", str(pid_file))
    runner = _runner(tmp_path, "fixture_child_io:build_leaves_a_grandchild",
                     rollout_timeout_s=10.0)
    started = time.monotonic()
    out = runner.run_one(_rollout(tmp_path))
    elapsed = time.monotonic() - started

    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.final_text == "done early"
    assert elapsed < 10.0, f"the parent waited {elapsed:.1f}s on a grandchild"
    pid = int(pid_file.read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    else:
        os.kill(pid, signal.SIGKILL)
        pytest.fail("the agent's grandchild outlived the rollout")


def test_a_kill_whose_pipes_stay_held_still_reaps_the_child(tmp_path):
    """PY-M12. When a descendant outside the group kept the pipes, the drain
    closed them and returned -- and the killed child stayed a zombie."""
    import subprocess
    import sys

    from agent_flywheel.runner import _kill_and_drain

    pid_file = tmp_path / "escaped.pid"
    escape = (
        "import os, sys, time\n"
        "if os.fork() == 0:\n"
        "    os.setsid()\n"
        f"    open({str(pid_file)!r}, 'w').write(str(os.getpid()))\n"
        "    time.sleep(60)\n"
        "    os._exit(0)\n"
        "time.sleep(60)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", escape], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            start_new_session=True)
    for _ in range(100):
        if pid_file.exists() and pid_file.read_text():
            break
        time.sleep(0.05)
    try:
        _, _, drained = _kill_and_drain(proc, 0.5)
        assert not drained
        assert proc.returncode is not None, "the killed child was left a zombie"
    finally:
        os.kill(int(pid_file.read_text()), signal.SIGKILL)
