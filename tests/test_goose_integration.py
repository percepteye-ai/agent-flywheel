"""The goose integration's two halves that live in THIS repository.

The hook plugin itself moved to its own repository, because both marketplaces
install from a repository ROOT and neither can install a subdirectory:

    https://github.com/percepteye-ai/agent-flywheel-hooks

What stays here is everything that is ours rather than the plugin's:

1.  The WIRE FORMAT. The plugin appends to `$PERCEPTEYE_TRAJECTORY_DIR/
    tool_calls.jsonl` and this SDK reads it -- two ends of one ABI, now in two
    repositories. The fixtures below are verbatim lines captured from the real
    plugin driven by a live goose 1.48.0, so this fails if the READER drifts
    away from what the plugin emits. It cannot catch the plugin drifting away
    from the reader; that direction is covered by the plugin's own suite, and
    the split is the cost of the two repositories.

2.  `examples/goose/serve_goose.py`, which is this repository's code and whose
    `reports_tool_calls=True` is an assertion only the plugin makes true.
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
    assert "goose plugin install" in proc.stderr


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
    # Task on stdin, and no session file between rollouts.
    assert "-i" in spec and spec[spec.index("-i") + 1] == "-"
    assert "--no-session" in spec
