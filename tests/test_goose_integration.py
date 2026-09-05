"""The goose integration's two halves, both of which now live in THIS repository.

THE SPLIT IS GONE. The plugin used to live in its own repository because both
marketplaces install from a repository ROOT and neither can install a
subdirectory -- so the customer needed two installs. The wheel now ships the
recorder itself (`percepteye_agent_flywheel/hosts/scripts/`) and `percepteye
install-hooks` renders the plugin directory pointed at it, which is what
collapsed two installs into one.

That matters for what this file can PROVE, not just for how it reads:

1.  The WIRE FORMAT. The recorder appends to `$PERCEPTEYE_TRAJECTORY_DIR/
    tool_calls.jsonl` and this SDK reads it. The fixtures below are verbatim
    lines captured from the real plugin driven by a live goose 1.48.0, so this
    fails if the READER drifts from what the recorder emits. Unlike before, the
    other direction is now checkable here too -- producer and consumer are one
    artifact -- so a drift on either side is this suite's problem and not
    another repository's.

2.  `examples/goose/serve_goose.py`, the CAPACITY-DONATION path, whose
    `reports_tool_calls=True` is an assertion only the recorder makes true. It
    is not the only path: a customer running goose themselves uses
    `percepteye install-hooks` and no worker process at all.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from percepteye_agent_flywheel.outcomes import Trajectory

SERVE = Path(__file__).resolve().parent.parent / "examples" / "goose" / "serve_goose.py"

#: Verbatim output of the hook plugin. The first two were produced by a live
#: `goose run` against a stub OpenAI endpoint -- one succeeding shell call and
#: one failing one, which is how goose's PostToolUse/PostToolUseFailure split
#: shows up on the wire. The rest cover the Gemini shape and the tri-state.
PLUGIN_OUTPUT = [
    '{"name": "shell", "arguments": {"command": "echo percepteye-hook-probe"},'
    ' "outcome": "ok", "status_code": null, "tool_call_id": "call_probe_1"}',
    '{"name": "shell", "arguments": {"command": "cat /definitely/no/such/file"},'
    ' "outcome": "failed", "status_code": null, "tool_call_id": "call_probe_1"}',
    '{"name": "write_file", "arguments": {"path": "a.txt"}, "outcome": "ok",'
    ' "status_code": null, "output": "wrote 12 bytes"}',
    '{"name": "run_shell_command", "arguments": {"command": "false"},'
    ' "outcome": "failed", "status_code": null, "error": "exit status 1"}',
    '{"name": "mystery_tool", "arguments": {}, "outcome": "unknown", "status_code": null}',
]


def write_plugin_output(tmp_path: Path, lines=PLUGIN_OUTPUT) -> Path:
    (tmp_path / "tool_calls.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def test_the_reader_accepts_what_the_plugin_writes(tmp_path):
    """One ABI, two repositories. This is the seam."""
    traj = Trajectory(write_plugin_output(tmp_path))
    calls = traj.read()
    assert traj.malformed == 0
    assert [c.outcome for c in calls] == ["ok", "failed", "ok", "failed", "unknown"]
    assert [c.name for c in calls][:2] == ["shell", "shell"]
    # A local tool has no HTTP status, and the reader must not manufacture one.
    assert all(c.status_code is None for c in calls)
    assert calls[3].error == "exit status 1"
    assert calls[0].tool_call_id == "call_probe_1"


def test_unknown_survives_the_round_trip_as_unknown(tmp_path):
    """Never rounded up to ok -- verifiers abstain on it rather than scoring."""
    calls = Trajectory(write_plugin_output(tmp_path)).read()
    assert calls[-1].outcome == "unknown"


def test_a_truncated_final_line_costs_that_line_only(tmp_path):
    """What a deadline kill mid-write actually produces."""
    write_plugin_output(tmp_path)
    with (tmp_path / "tool_calls.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"name": "truncated"')
    traj = Trajectory(tmp_path)
    calls = traj.read()
    assert len(calls) == len(PLUGIN_OUTPUT)
    assert traj.malformed == 1


def test_no_trajectory_is_not_a_claim_of_zero_calls(tmp_path):
    """None and [] mean different things and the difference is load-bearing."""
    assert Trajectory(tmp_path).read() is None


# --- examples/goose/serve_goose.py -----------------------------------------


def run_serve(tmp_path, *, mode: str, workspace: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.update({"HOME": str(tmp_path / "home"), "GOOSE_WORKSPACE": str(workspace),
                "PERCEPTEYE_AGENT_MODE": mode})
    return subprocess.run([sys.executable, str(SERVE)], capture_output=True,
                          text=True, env=env, timeout=60)


def install_stub_plugin(root: Path) -> Path:
    """The layout the preflight looks for, without the plugin repository.

    Deliberately a stub: this asserts serve_goose.py finds a plugin where goose
    would look, which is our behaviour to get right. Whether the real plugin
    records correctly is the plugin repository's suite to prove.
    """
    dest = root / ".agents" / "plugins" / "percepteye-agent-flywheel" / "scripts"
    dest.mkdir(parents=True)
    (dest / "percepteye_record.py").write_text("# stub\n", encoding="utf-8")
    return dest


def test_serving_without_the_plugin_refuses_rather_than_downgrading(tmp_path):
    """`reports_tool_calls=True` asserts a per-call outcome for every call.

    Without the plugin the rollouts still run and still route, so the run looks
    healthy while the trajectory stays empty and every verifier that grades
    "did this call land" abstains for the whole run. Checked before the first
    rollout, like every other thing that can be misconfigured.
    """
    proc = run_serve(tmp_path, mode="training", workspace=tmp_path / "ws")
    assert proc.returncode != 0
    assert "hook plugin is not installed" in proc.stderr
    # THE REMEDY, not just the diagnosis -- and it must be the remedy that
    # actually works today. This used to say `goose plugin install <url>`,
    # sending the operator to a second install that the wheel now ships; a
    # refusal naming a stale fix costs more than one naming none.
    assert "percepteye install-hooks" in proc.stderr
    assert "goose plugin install" not in proc.stderr


def test_production_mode_does_not_raise_over_a_missing_plugin(tmp_path):
    """A switch that crashes the service it was meant to quiet is not a switch."""
    proc = run_serve(tmp_path, mode="production", workspace=tmp_path / "ws")
    assert proc.returncode == 0
    assert "not claiming rollouts" in proc.stdout


@pytest.mark.parametrize("scope", ["user", "project"])
def test_preflight_finds_the_plugin_in_either_scope(tmp_path, scope):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    install_stub_plugin(tmp_path / "home" if scope == "user" else workspace)
    proc = run_serve(tmp_path, mode="production", workspace=workspace)
    assert proc.returncode == 0
    assert "hook plugin:" in proc.stdout


def test_routing_overrides_that_would_bypass_the_gateway_are_scrubbed(tmp_path):
    """OPENAI_HOST outranks OPENAI_BASE_URL in goose and the SDK does not scrub it.

    Left in place, every completion goes to the operator's own host while the
    control plane believes it is sampling the policy under training.
    """
    workspace = tmp_path / "ws"
    workspace.mkdir()
    install_stub_plugin(workspace)
    env = dict(os.environ)
    env.update({"HOME": str(tmp_path / "home"), "GOOSE_WORKSPACE": str(workspace),
                "PERCEPTEYE_AGENT_MODE": "production",
                "OPENAI_HOST": "https://operator.example.com",
                "GOOSE_PROVIDER": "anthropic", "GOOSE_MODEL": "claude-3"})
    proc = subprocess.run([sys.executable, str(SERVE)], capture_output=True,
                          text=True, env=env, timeout=60)
    assert proc.returncode == 0
    for name in ("OPENAI_HOST", "GOOSE_PROVIDER", "GOOSE_MODEL"):
        assert name in proc.stdout, name


def test_argv_pins_the_routed_provider_and_bounds_the_episode(tmp_path):
    """goose's --max-turns default is 1000, which is not a bound."""
    spec = json.loads(subprocess.run(
        [sys.executable, "-c",
         "import json,importlib.util;"
         f"s=importlib.util.spec_from_file_location('sg', {str(SERVE)!r});"
         "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
         "print(json.dumps(m.ARGV))"],
        capture_output=True, text=True,
        env={**os.environ, "PERCEPTEYE_AGENT_MODE": "production",
             "HOME": str(tmp_path)}, timeout=60).stdout)
    assert spec[:2] == ["goose", "run"]
    assert "--provider" in spec and spec[spec.index("--provider") + 1] == "openai"
    assert "--max-turns" in spec
    # Task on stdin.
    assert "-i" in spec and spec[spec.index("-i") + 1] == "-"
    # A NAMED session, and deliberately not `--no-session` (they are mutually
    # exclusive). `maybe_update_name` returns early on `user_set_name`, which
    # `--name` sets — and that is the only way to stop goose spending a
    # COMPLETION on a 3-5 word session title. Under a rollout that title
    # authenticates as the rollout, lands as inference_calls seq 0, passes the
    # logprobs-only trainable predicate, and is emitted as a GRPO row carrying
    # the episode's advantage sign while diluting every real decision to
    # adv/(K+1). Measured: 2 completions per run before this flag, 1 after.
    assert "--name" in spec and spec[spec.index("--name") + 1]
    assert "--no-session" not in spec, (
        "`--no-session` reintroduces the session-naming completion")
    # The tool catalogue is a property of the rollout, not of whatever
    # `goose configure` last wrote: 0 tools before this pair, 5 after.
    assert "--no-profile" in spec
    assert "--with-builtin" in spec and spec[spec.index("--with-builtin") + 1]


# ── the recorder as a SUBPROCESS, which is what the contract actually is ────
#
# Restored with the payload. When the plugin lived in its own repository this
# coverage was lost: the deleted suite's docstring put it exactly right — "the
# plugin's contract is a SUBPROCESS and a FILE, not a function: goose spawns
# `python3 scripts/percepteye_record.py`, pipes a JSON event to its stdin". A
# test that imports and calls `_record` proves something the host never does.


def _fire(event, env_extra, script=None):
    """Drive the real recorder the way goose does: argv, JSON on stdin."""
    from percepteye_agent_flywheel.hosts import RECORDER

    return subprocess.run(
        [sys.executable, str(script or RECORDER)],
        input=json.dumps(event), text=True, capture_output=True,
        env={**os.environ, **env_extra},
    )


def _goose_turn(traj_dir):
    """The four event shapes one goose turn actually emits."""
    env = {"PERCEPTEYE_TRAJECTORY_DIR": str(traj_dir)}
    _fire({"event": "UserPromptSubmit", "session_id": "s1",
           "message": "Write solve.py"}, env)
    _fire({"event": "PostToolUse", "session_id": "s1", "tool_name": "write",
           "tool_input": {"path": "solve.py"}, "tool_call_id": "c1"}, env)
    _fire({"event": "PostToolUseFailure", "session_id": "s1",
           "tool_name": "shell", "tool_input": {"command": "false"}}, env)
    _fire({"event": "Stop", "session_id": "s1",
           "last_assistant_message": "Created solve.py",
           "working_dir": "/w"}, env)


def test_turn_records_never_corrupt_the_tool_call_file(tmp_path):
    """THE reason turns get their own file.

    ``Trajectory.read`` builds rows with REQUIRED subscripts and counts anything
    else as ``malformed``; that becomes ``tool_calls_omitted_count``, and
    non-zero forbids the training grade. So a task or answer record in
    ``tool_calls.jsonl`` would not be politely ignored by an older reader — it
    would silently demote the rollout out of training.
    """
    _goose_turn(tmp_path)
    traj = Trajectory(tmp_path)
    calls = traj.read()
    assert traj.malformed == 0, "turn records leaked into tool_calls.jsonl"
    assert [(c.name, c.outcome) for c in calls] == [("write", "ok"), ("shell", "failed")]


def test_the_turn_file_carries_the_task_and_the_answer(tmp_path):
    """Without these the trajectory is tool calls with no task and no answer."""
    _goose_turn(tmp_path)
    rows = [json.loads(line) for line in
            (tmp_path / "turns.jsonl").read_text().splitlines() if line.strip()]
    assert [r["kind"] for r in rows] == ["task", "answer"]
    assert rows[0]["text"] == "Write solve.py"
    assert rows[1]["text"] == "Created solve.py"
    assert rows[1]["working_dir"] == "/w"


def test_capture_needs_no_rollout_and_no_worker(tmp_path):
    """The line that made zero-instrumentation impossible.

    The recorder used to return 0 unless ``PERCEPTEYE_TRAJECTORY_DIR`` was set,
    and the only thing that ever set it was an external worker — so the plugin's
    own day-one promise ("your agent's real tool failure rate", no account, no
    network) was unreachable unless the customer exported it by hand, which is
    what the report tool's help text still tells them to do.
    """
    env = {k: v for k, v in os.environ.items()
           if k != "PERCEPTEYE_TRAJECTORY_DIR"}
    env["PERCEPTEYE_HOME"] = str(tmp_path)
    _fire({"event": "PostToolUse", "session_id": "sess-42",
           "tool_name": "shell", "tool_input": {}}, env)
    assert (tmp_path / "turns" / "sess-42" / "tool_calls.jsonl").is_file()


def test_capture_has_a_kill_switch(tmp_path):
    """Default ON, per the standing rule — but a switch that turns it off."""
    env = {k: v for k, v in os.environ.items()
           if k != "PERCEPTEYE_TRAJECTORY_DIR"}
    env["PERCEPTEYE_HOME"] = str(tmp_path)
    env["PERCEPTEYE_CAPTURE"] = "0"
    _fire({"event": "PostToolUse", "session_id": "sess-99",
           "tool_name": "shell", "tool_input": {}}, env)
    assert not (tmp_path / "turns").exists()


def test_a_session_with_no_id_is_not_guessed_at(tmp_path):
    """Two concurrent sessions sharing one file are two half-turns and no
    trajectory. With nothing to key on, record nothing."""
    env = {k: v for k, v in os.environ.items()
           if k != "PERCEPTEYE_TRAJECTORY_DIR"}
    env["PERCEPTEYE_HOME"] = str(tmp_path)
    _fire({"event": "PostToolUse", "tool_name": "shell", "tool_input": {}}, env)
    assert not (tmp_path / "turns").exists()


def test_the_recorder_never_fails_the_agent_and_says_nothing(tmp_path):
    """Both invariants, and ``Stop`` is a BLOCKING event whose chain can deny —
    goose reads a denial from exit code 2 or a JSON verdict on stdout, so these
    two properties are what make subscribing it safe."""
    for event in ({}, {"event": "Stop"}, {"garbage": True},
                  {"event": "PostToolUse", "tool_name": None}):
        r = _fire(event, {"PERCEPTEYE_TRAJECTORY_DIR": str(tmp_path)})
        assert r.returncode == 0, (event, r.stderr)
        assert r.stdout == "", (event, r.stdout)


# ── the trained prompt goose is actually handed ───────────────────────────


def _prompt_argv(tmp_path: Path, served: object) -> list[str]:
    """``_trained_prompt_argv`` with the control plane's answer substituted.

    A child process because importing ``serve_goose`` runs its module body,
    which scrubs routing variables out of the environment it finds itself in.
    """
    out = subprocess.run(
        [sys.executable, "-c",
         "import json,importlib.util;"
         f"s=importlib.util.spec_from_file_location('sg', {str(SERVE)!r});"
         "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
         f"m.fw.current_prompt=lambda **kw: json.loads({json.dumps(json.dumps(served))});"
         "print(json.dumps(m._trained_prompt_argv()))"],
        capture_output=True, text=True,
        env={**os.environ, "PERCEPTEYE_AGENT_MODE": "production",
             "HOME": str(tmp_path)}, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_goose_is_handed_an_approved_prompt():
    """``--system`` is ADDITIVE in goose, which is what makes applying the
    approved text safe without knowing what goose put in its own prompt."""
    argv = _prompt_argv(Path.cwd(), {
        "text": "You triage carefully.", "status": "approved", "version": 3,
        "sha256": "c" * 64, "model_prompt_checksum": "c" * 64,
    })
    assert argv == ["--system", "You triage carefully."]


@pytest.mark.parametrize("served,why", [
    ({"text": "t", "status": "draft"}, "an unapproved status"),
    ({"text": "t", "status": "approved", "stale_baseline": True},
     "a baseline that moved after certification"),
    ({"text": "t", "status": "approved", "sha256": "a" * 64,
      "model_prompt_checksum": "b" * 64}, "a bundle that disagrees with itself"),
    ({}, "nothing served"),
])
def test_goose_is_NOT_handed_a_prompt_that_did_not_clear_the_gate(served, why):
    """The whole predicate, not the half this file used to ask.

    It checked for non-blank text and warned about a stale baseline; a bundle
    whose status nobody approved, or whose prompt identity disagrees with the
    checkpoint it was certified against, went onto goose's command line
    anyway. ``fw.prompt_decision`` is the one answer, and this asserts the
    integration takes it rather than deriving its own.
    """
    assert _prompt_argv(Path.cwd(), served) == [], why
