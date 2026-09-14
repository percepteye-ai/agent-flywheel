"""What capture costs the process it runs in.

Five measured problems were fixed together, and these pin the properties rather
than the numbers. Timing assertions are deliberately absent: this suite already
learned once, in test_lease_and_concurrency, that asserting how fast something
ran is a fact about the machine rather than about the code. Every test here
asserts a STRUCTURAL consequence of the fix -- a file that is not created, a
handle that is closed, a module that is not imported -- which is what actually
regresses if someone undoes one.

The measured effects, on an M4 Pro / CPython 3.14, for the record:

    import                53.2 ms, 19.3 MiB  ->  27.1 ms, 10.9 MiB
    record_tool_call      38.3 us            ->  13.0 us
    throughput            ~20k/s             ->  ~77k/s
    latency at 8 threads  8.7x inflation     ->  1.1x
    1M empty turns        0.33 GB, 1M inodes ->  0
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import agent_flywheel as fw
from agent_flywheel.outcomes import _RolloutScope, _sinks


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for n in ("PERCEPTEYE_ROLLOUT_ID", "PERCEPTEYE_TRAJECTORY_DIR",
              "PERCEPTEYE_AGENT_MODE"):
        monkeypatch.delenv(n, raising=False)


def _count(root) -> tuple[int, int]:
    dirs = sum(len(d) for _, d, _ in os.walk(root))
    files = sum(len(f) for _, _, f in os.walk(root))
    return dirs, files


# ── a turn that records nothing leaves nothing ─────────────────────────────


def test_a_turn_that_records_nothing_creates_nothing(tmp_path):
    """The directory used to be created on turn ENTRY, unconditionally.

    An agent that answers from context without calling a tool still cost one
    inode. At a million turns that was ~0.33 GB and a million directories
    holding nothing at all.
    """
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    with px:
        for _ in range(200):
            with px.turn():
                pass
    assert _count(tmp_path) == (0, 0)


def test_a_turn_that_records_creates_exactly_what_it_needs(tmp_path):
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    with px:
        with px.turn():
            fw.record_tool_call("t", {}, outcome="ok", status_code=200)
    assert _count(tmp_path) == (1, 1)


def test_a_conversation_id_still_gets_its_directory(tmp_path):
    """The one thing that legitimately needs the directory up front: it writes
    a file whether or not a tool is ever called."""
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    with px, px.turn(conversation_id="c-1") as tid:
        pass
    assert (tmp_path / tid / "conversation_id").read_text() == "c-1"


# ── the handle is held, and released ───────────────────────────────────────


def test_the_write_handle_is_released_when_the_scope_exits(tmp_path):
    """Holding the handle is what made recording 3x cheaper. Not releasing it
    would trade that speed for one leaked file handle per turn."""
    before = len(_sinks)
    with _RolloutScope("rl_1", str(tmp_path)):
        fw.record_tool_call("t", {}, outcome="ok", status_code=200)
        assert len(_sinks) == before + 1, "no sink was opened"
    assert len(_sinks) == before, f"sink not released: {_sinks}"


def test_many_turns_do_not_accumulate_handles(tmp_path):
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    with px:
        for _ in range(300):
            with px.turn():
                fw.record_tool_call("t", {}, outcome="ok", status_code=200)
    assert _sinks == {}, f"{len(_sinks)} handles left open"


def test_concurrent_trajectories_get_independent_sinks(tmp_path):
    """One process-global lock held across open/write/close made every
    recording thread queue behind every other one, even writing to different
    rollouts -- ~20k records/second flat from 1 to 8 threads, with per-call
    latency 8.7x worse at 8. A lock per file makes them independent, which is
    what they always were."""
    a, b = tmp_path / "a", tmp_path / "b"
    with _RolloutScope("rl_a", str(a)):
        fw.record_tool_call("t", {}, outcome="ok", status_code=200)
        with _RolloutScope("rl_b", str(b)):
            fw.record_tool_call("t", {}, outcome="ok", status_code=200)
            sinks = list(_sinks.values())
            assert len(sinks) == 2
            assert sinks[0]._lock is not sinks[1]._lock


# ── capture never raises into the caller ───────────────────────────────────


def test_an_unwritable_trajectory_does_not_break_the_turn(tmp_path):
    """Creating the directory moved to the first write, and the failure moved
    with it. It must not surface inside a customer's tool."""
    root = tmp_path / "root"
    root.mkdir()
    px = fw.attach(agent_id="a", trajectory_root=str(root), register=False)
    root.chmod(0o500)                      # readable, not writable
    ran = False
    try:
        with px, pytest.warns(RuntimeWarning, match="could not write the trajectory"):
            with px.turn():
                fw.record_tool_call("t", {}, outcome="ok", status_code=200)
                ran = True
    finally:
        root.chmod(0o700)                  # so tmp_path cleanup can run
    assert ran, "the turn body did not run"


def test_a_broken_trajectory_warns_once_not_per_call(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    px = fw.attach(agent_id="a", trajectory_root=str(root), register=False)
    root.chmod(0o500)
    try:
        with px, pytest.warns(RuntimeWarning) as rec:
            with px.turn():
                for _ in range(50):
                    fw.record_tool_call("t", {}, outcome="ok", status_code=200)
    finally:
        root.chmod(0o700)
    assert len([w for w in rec if "could not write" in str(w.message)]) == 1


# ── retention ──────────────────────────────────────────────────────────────


def test_prune_with_no_argument_deletes_nothing(tmp_path):
    """An unqualified "prune" would be a request to delete everything, and
    that is not a thing to infer."""
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    with px:
        for _ in range(5):
            with px.turn():
                fw.record_tool_call("t", {}, outcome="ok", status_code=200)
        assert px.prune() == 0
        assert _count(tmp_path)[0] == 5


def test_prune_keeps_the_most_recent(tmp_path):
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    with px:
        for _ in range(20):
            with px.turn():
                fw.record_tool_call("t", {}, outcome="ok", status_code=200)
        assert px.prune(keep_last=5) == 15
        assert _count(tmp_path)[0] == 5


def test_prune_never_raises_on_a_missing_root(tmp_path):
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path / "gone"),
                   register=False)
    with px:
        assert px.prune(keep_last=1) == 0


def test_retention_turns_bounds_the_directory(tmp_path):
    """Unset, the trajectory grows without bound -- that is deliberate, since
    it is the customer's data in a directory they named. Set, it is capped."""
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path),
                   register=False, retention_turns=20)
    with px:
        for _ in range(900):
            with px.turn():
                fw.record_tool_call("t", {}, outcome="ok", status_code=200)
    # The prune runs on an interval, so the ceiling is retention + interval.
    assert _count(tmp_path)[0] <= 20 + 256


def test_without_retention_nothing_is_deleted(tmp_path):
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    with px:
        for _ in range(30):
            with px.turn():
                fw.record_tool_call("t", {}, outcome="ok", status_code=200)
    assert _count(tmp_path)[0] == 30


# ── import cost ────────────────────────────────────────────────────────────


def test_importing_the_package_does_not_import_httpx():
    """httpx was 83-87% of this package's import time and ~88% of its resident
    memory, imported eagerly for a transport that a production-only customer --
    one that reads the promoted policy and never serves -- never opens.

    Measured: 53.2 ms / 19.3 MiB -> 27.1 ms / 10.9 MiB, 101 fewer modules.

    Run in a subprocess because this process has almost certainly imported
    httpx already, through some other test.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "import sys, agent_flywheel;"
         "print('httpx' in sys.modules, 'ssl' in sys.modules)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "False False", f"httpx/ssl imported eagerly: {out}"


def test_the_transport_still_works_when_it_is_finally_needed():
    """Deferring an import is only safe if the deferred thing still resolves."""
    from agent_flywheel._http import ControlPlaneClient

    c = ControlPlaneClient("https://cp.invalid", "pek_x")
    assert c is not None
    c.close()


def test_importing_the_package_registers_no_atexit_hook():
    """The sink's cleanup hook is registered on the FIRST WRITE, not at import,
    so a process that imports this package and never records stays inert.

    Measured against a baseline that has already imported `logging`, because
    the stdlib's own `logging.shutdown` hook is not ours and would otherwise be
    counted as though it were.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "import atexit, logging; b = atexit._ncallbacks();"
         "import agent_flywheel;"
         "print(atexit._ncallbacks() - b)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "0", f"import registered {out} atexit hook(s) of our own"


def test_recording_registers_the_cleanup_hook_exactly_once(tmp_path):
    out = subprocess.run(
        [sys.executable, "-c",
         "import atexit, os, tempfile, logging;"
         "import agent_flywheel as fw;"
         "b = atexit._ncallbacks();"
         "d = tempfile.mkdtemp();"
         "os.environ['PERCEPTEYE_ROLLOUT_ID'] = 'rl_1';"
         "os.environ['PERCEPTEYE_TRAJECTORY_DIR'] = d;"
         "[fw.record_tool_call('t', {}, outcome='ok', status_code=200)"
         " for _ in range(20)];"
         "print(atexit._ncallbacks() - b)"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == "1", f"expected exactly one hook, got {out}"


# ── the records themselves are unchanged ───────────────────────────────────


def test_every_record_still_lands_and_reads_back(tmp_path):
    """The point of all of the above is that it changes cost, not content."""
    with _RolloutScope("rl_1", str(tmp_path)):
        for i in range(500):
            fw.record_tool_call(f"t{i}", {"i": i}, outcome="ok", status_code=200)
    calls = fw.Trajectory(Path(tmp_path)).read()
    assert len(calls) == 500
    assert [c.name for c in calls] == [f"t{i}" for i in range(500)]
    assert all(c.outcome == "ok" for c in calls)
