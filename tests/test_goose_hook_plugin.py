"""The goose/Gemini hook plugin, exercised as the hosts actually invoke it.

This file exists because the plugin's contract is a SUBPROCESS and a FILE, not a
function: goose spawns `python3 scripts/percepteye_record.py`, pipes a JSON event
to its stdin, and never looks at the return value. Importing the module and
calling `_record()` would test none of the three properties that matter, so every
test here goes through the real script the way a host does, and reads the result
back with the SDK's own `Trajectory` rather than by parsing the file itself.

The tri-state is the load-bearing part. goose chooses PostToolUse vs
PostToolUseFailure with its own `is_error` check, so `ok` and `failed` are
transcribed verdicts; anything we do not recognise must land on `unknown` and
must never be rounded up.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from percepteye_agent_flywheel.outcomes import Trajectory

PLUGIN = Path(__file__).resolve().parent.parent / "packages" / "percepteye-agent-flywheel-hooks"
SCRIPT = PLUGIN / "scripts" / "percepteye_record.py"


def run_hook(event: dict | str, trajectory_dir: Path | None) -> subprocess.CompletedProcess:
    """Invoke the recorder exactly as a host does: JSON on stdin, env, no args."""
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin"}
    if trajectory_dir is not None:
        env["PERCEPTEYE_TRAJECTORY_DIR"] = str(trajectory_dir)
    payload = event if isinstance(event, str) else json.dumps(event)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        input=payload, capture_output=True, text=True, env=env, timeout=30,
    )


def goose_event(name: str, *, failed: bool = False, **kw) -> dict:
    return {
        "event": "PostToolUseFailure" if failed else "PostToolUse",
        "session_id": "s1", "tool_name": name, "tool_input": {"command": "ls"},
        "tool_call_id": "call_1", "working_dir": "/tmp", **kw,
    }


def gemini_event(name: str, *, error: str | None = None, **kw) -> dict:
    response: dict = {"llmContent": "done", "returnDisplay": "done"}
    if error is not None:
        response["error"] = error
    return {
        "hook_event_name": "AfterTool", "session_id": "s1", "cwd": "/tmp",
        "tool_name": name, "tool_input": {"path": "a.txt"},
        "tool_response": response, **kw,
    }


@pytest.mark.parametrize("event,expected", [
    (goose_event("developer__shell"), "ok"),
    (goose_event("developer__shell", failed=True), "failed"),
    (gemini_event("write_file"), "ok"),
    (gemini_event("run_shell_command", error="exit status 1"), "failed"),
])
def test_outcome_is_the_hosts_verdict(tmp_path, event, expected):
    """Both vocabularies, one recorder, and the SDK reads what it wrote."""
    assert run_hook(event, tmp_path).returncode == 0
    calls = Trajectory(tmp_path).read()
    assert calls is not None and len(calls) == 1
    assert calls[0].outcome == expected
    # A local tool has no HTTP status; inventing one is the failure the
    # tri-state exists to prevent.
    assert calls[0].status_code is None


def test_unrecognised_event_is_unknown_never_ok(tmp_path):
    """An event we do not know means we did not observe the outcome."""
    run_hook({"event": "SomeFutureEvent", "tool_name": "mystery"}, tmp_path)
    calls = Trajectory(tmp_path).read()
    assert calls is not None and [c.outcome for c in calls] == ["unknown"]


def test_inert_without_trajectory_dir(tmp_path):
    """Unset, the plugin does nothing at all -- so it stays installed in prod."""
    proc = run_hook(goose_event("developer__shell"), None)
    assert proc.returncode == 0
    assert proc.stdout == "" and proc.stderr == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("payload", ["not json at all{{{", "", "[]", "null"])
def test_never_fails_the_agent(tmp_path, payload):
    """Telemetry that can break someone's tool call is worse than none."""
    proc = run_hook(payload, tmp_path)
    assert proc.returncode == 0
    assert proc.stdout == ""


def test_call_without_a_name_is_dropped_not_invented(tmp_path):
    """Retry detection keys on name + arguments; a guessed name corrupts it."""
    run_hook({"event": "PostToolUse", "tool_input": {"a": 1}}, tmp_path)
    assert Trajectory(tmp_path).read() is None


def test_writes_nothing_to_stdout(tmp_path):
    """Observing must not be able to speak to the host."""
    proc = run_hook(goose_event("developer__shell"), tmp_path)
    assert proc.stdout == ""


def test_appends_rather_than_truncates(tmp_path):
    """A killed rollout keeps every call up to the kill."""
    for _ in range(3):
        run_hook(goose_event("developer__shell"), tmp_path)
    calls = Trajectory(tmp_path).read()
    assert calls is not None and len(calls) == 3


def test_hooks_json_registers_both_hosts():
    """One file, three events: goose reads two of them, Gemini the third.

    Each host therefore only ever encounters its OWN root-path substitution
    variable, which is what makes a single hooks.json portable.
    """
    hooks = json.loads((PLUGIN / "hooks" / "hooks.json").read_text())["hooks"]
    assert set(hooks) == {"PostToolUse", "PostToolUseFailure", "AfterTool"}
    for event, expected_var in [("PostToolUse", "${PLUGIN_ROOT}"),
                                ("PostToolUseFailure", "${PLUGIN_ROOT}"),
                                ("AfterTool", "${extensionPath}")]:
        commands = [h["command"] for entry in hooks[event] for h in entry["hooks"]]
        assert commands and all(expected_var in c for c in commands), event


def test_both_manifests_ship_and_agree():
    """goose resolves open-plugins before gemini, so the pair is unambiguous."""
    op = json.loads((PLUGIN / "plugin.json").read_text())
    gemini = json.loads((PLUGIN / "gemini-extension.json").read_text())
    assert op["name"] == gemini["name"]
    assert op["version"] == gemini["version"]


# --- the local report ------------------------------------------------------
#
# The report is the plugin's day-one payoff: it answers "which of my agent's
# tool calls actually worked" from the same file the recorder writes, with no
# account, no network and no SDK. These tests run it as a user does -- as a
# subprocess against a real file on disk.

REPORT = PLUGIN / "scripts" / "percepteye_report.py"


def run_report(target) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPORT), str(target)],
        capture_output=True, text=True, env={"PATH": "/usr/bin:/bin", "NO_COLOR": "1"},
        timeout=30,
    )


def write_trajectory(tmp_path, records):
    (tmp_path / "tool_calls.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return tmp_path


def call(name, outcome, **kw):
    return {"name": name, "arguments": {}, "outcome": outcome, "status_code": None, **kw}


def test_failure_rate_excludes_unknown_from_the_denominator(tmp_path):
    """The invariant the tri-state exists for.

    3 ok, 1 failed, 6 unknown. Over OBSERVED calls the rate is 1/4 = 25%.
    Counting the unknowns as successes would report 10% and read as "things are
    fine" for precisely the wrong reason -- nobody watched those six.
    """
    write_trajectory(tmp_path, [call("a", "ok")] * 3 + [call("a", "failed")]
                     + [call("b", "unknown")] * 6)
    out = run_report(tmp_path).stdout
    assert "Real failure rate: 25%" in out
    assert "1 of 4 observed calls" in out
    assert "6 call(s) where nothing observed the outcome" in out


def test_reports_where_the_failures_are(tmp_path):
    write_trajectory(tmp_path, [call("text_editor", "failed")] * 4
                     + [call("text_editor", "ok")] * 5
                     + [call("shell", "ok")] * 10)
    out = run_report(tmp_path).stdout
    assert "text_editor" in out and "4 failed /   9 calls" in out
    # A tool that never failed is not listed as a failure site.
    assert "shell" not in out.split("Where the failures are")[1]


def test_missing_trajectory_explains_how_to_make_one(tmp_path):
    proc = run_report(tmp_path / "nope")
    assert proc.returncode == 1
    assert "PERCEPTEYE_TRAJECTORY_DIR" in proc.stdout


def test_one_unreadable_line_costs_that_line_only(tmp_path):
    """A rollout killed on its deadline truncates the final write."""
    path = write_trajectory(tmp_path, [call("a", "ok")] * 3) / "tool_calls.jsonl"
    with path.open("a", encoding="utf-8") as fh:
        fh.write('{"name": "truncated"')
    out = run_report(tmp_path).stdout
    assert "3 tool calls" in out
    assert "1 unreadable line(s) skipped" in out


def test_accepts_a_file_path_as_well_as_a_directory(tmp_path):
    write_trajectory(tmp_path, [call("a", "ok")])
    assert "1 tool call" in run_report(tmp_path / "tool_calls.jsonl").stdout


# --- bounds and atomicity --------------------------------------------------
#
# A file-writing tool puts the whole file in `tool_input`, so an uncapped
# record is both enormous and unsafe: goose runs tool calls concurrently, and a
# record too large for one write() lets a second hook process interleave and
# corrupt both lines.

MAX_RECORD_BYTES = 7000


def test_a_huge_argument_is_clipped_and_says_so(tmp_path):
    """200KB of file_text must not become 200KB of trajectory."""
    run_hook(goose_event("text_editor", tool_input={"path": "a.py",
                                                    "file_text": "x" * 200_000}), tmp_path)
    line = (tmp_path / "tool_calls.jsonl").read_text()
    assert len(line) < MAX_RECORD_BYTES
    args = json.loads(line)["arguments"]
    # The small argument survives intact; the huge one is clipped VISIBLY, so a
    # reader can tell a big argument from a missing one.
    assert args["path"] == "a.py"
    assert "truncated, 200000 chars total" in args["file_text"]


def test_a_record_with_many_fat_keys_still_fits(tmp_path):
    run_hook(goose_event("wide", tool_input={f"k{i}": "y" * 900 for i in range(400)}),
             tmp_path)
    line = (tmp_path / "tool_calls.jsonl").read_text()
    assert len(line) < MAX_RECORD_BYTES
    assert "dropped" in json.dumps(json.loads(line)["arguments"])


def test_concurrent_hooks_never_shred_each_other(tmp_path):
    """goose runs tools concurrently, so hook processes append concurrently."""
    n = 12
    procs = []
    env = {"PATH": "/usr/bin:/bin", "PERCEPTEYE_TRAJECTORY_DIR": str(tmp_path)}
    for i in range(n):
        event = goose_event(f"tool_{i}", tool_input={"blob": "z" * 900, "i": i})
        proc = subprocess.Popen([sys.executable, str(SCRIPT)], stdin=subprocess.PIPE,
                                text=True, env=env)
        proc.stdin.write(json.dumps(event))
        proc.stdin.close()
        procs.append(proc)
    for proc in procs:
        proc.wait(timeout=30)

    lines = [ln for ln in (tmp_path / "tool_calls.jsonl").read_text().splitlines() if ln]
    assert len(lines) == n
    for ln in lines:
        json.loads(ln)  # every line whole; a torn write raises here
    traj = Trajectory(tmp_path)
    calls = traj.read()
    assert calls is not None and len(calls) == n
    assert traj.malformed == 0


def test_a_nonzero_count_never_renders_as_an_empty_bar(tmp_path):
    """1 in 100 rounds to zero blocks; the row the report exists for."""
    write_trajectory(tmp_path, [call("a", "failed")] + [call("a", "ok")] * 99)
    failed_row = next(ln for ln in run_report(tmp_path).stdout.splitlines()
                      if ln.strip().startswith("failed"))
    assert "█" in failed_row, failed_row


# --- the serve-side preflight ----------------------------------------------

SERVE = PLUGIN.parent.parent / "examples" / "goose" / "serve_goose.py"


def run_serve(tmp_path, *, mode: str, workspace) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({"HOME": str(tmp_path / "home"), "GOOSE_WORKSPACE": str(workspace),
                "PERCEPTEYE_AGENT_MODE": mode})
    return subprocess.run([sys.executable, str(SERVE)], capture_output=True,
                          text=True, env=env, timeout=60)


def test_serving_without_the_plugin_refuses_rather_than_downgrading(tmp_path):
    """`reports_tool_calls=True` is an assertion; this is what makes it true.

    Without the check, rollouts run and route perfectly while the trajectory
    stays empty -- a silently smaller reward surface, discovered a run later.
    """
    proc = run_serve(tmp_path, mode="training", workspace=tmp_path / "ws")
    assert proc.returncode != 0
    assert "hook plugin is not installed" in proc.stderr


def test_production_mode_does_not_raise_over_a_missing_plugin(tmp_path):
    """A switch that crashes the service it was meant to quiet is not a switch."""
    proc = run_serve(tmp_path, mode="production", workspace=tmp_path / "ws")
    assert proc.returncode == 0
    assert "not claiming rollouts" in proc.stdout


def test_preflight_finds_a_project_scope_plugin(tmp_path):
    workspace = tmp_path / "ws"
    dest = workspace / ".agents" / "plugins" / "percepteye-agent-flywheel"
    dest.parent.mkdir(parents=True)
    shutil.copytree(PLUGIN, dest)
    proc = run_serve(tmp_path, mode="production", workspace=workspace)
    assert proc.returncode == 0
    assert "hook plugin:" in proc.stdout
