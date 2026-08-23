"""The agent needs a way to know it is finished.

prime-agent's ``--autonomous`` is documented as "continue until gates pass or a
limit is reached". With no gate registered, a LIMIT is the only exit that
exists — so every rollout on runs b6b4388d and b60067db ended on maxTurns or
maxTokens, carried an exclusion, and was held out of training by
GRPOAdvantage even when the artifact it produced was correct.

Two independent facts, deliberately kept apart:

* ``gate_flag`` — how THIS binary spells the option. Serve configuration,
  because it is a fact about the CLI and cannot be guessed; passing an unknown
  flag to an arbitrary agent would break it outright.
* the gate COMMAND — a property of the TASK, arriving per rollout in metadata.

Either side missing leaves the argv untouched, which is exactly the pre-gate
behaviour.
"""
from __future__ import annotations

import time

from percepteye_agent_flywheel.contract import RolloutRequest
from percepteye_agent_flywheel.runner import _argv_with_gate

ARGV = ["prime-agent", "-p", "--autonomous"]


def _req(metadata=None) -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_g", turn_input="do it",
        llm_url="https://gw.invalid/v1", api_key="k",
        artifacts_dir="rollouts/rl_g", deadline_ts=time.time() + 300,
        lease_token="lt", metadata=metadata or {},
    )


def test_the_gate_is_appended_when_both_sides_declare_one():
    out = _argv_with_gate(
        ARGV, _req({"completion_gate": "test -s top_ips.sh"}), "--autonomous-gate")
    assert out == [*ARGV, "--autonomous-gate", "test -s top_ips.sh"]


def test_no_flag_means_the_argv_is_untouched():
    """An agent whose CLI has no gate concept must not be handed an unknown
    option — that breaks it outright rather than helping."""
    out = _argv_with_gate(ARGV, _req({"completion_gate": "test -s x"}), None)
    assert out == ARGV


def test_no_gate_in_metadata_means_the_argv_is_untouched():
    """A task whose deliverable could not be identified degrades to exactly
    the previous behaviour."""
    assert _argv_with_gate(ARGV, _req({}), "--autonomous-gate") == ARGV
    assert _argv_with_gate(ARGV, _req(None), "--autonomous-gate") == ARGV


def test_a_blank_gate_is_treated_as_absent():
    out = _argv_with_gate(ARGV, _req({"completion_gate": "   "}), "--autonomous-gate")
    assert out == ARGV


def test_a_non_string_gate_is_ignored_rather_than_stringified():
    """Metadata is data from another system. Coercing a dict into a shell
    command is how you execute something nobody wrote."""
    out = _argv_with_gate(ARGV, _req({"completion_gate": {"cmd": "rm -rf /"}}),
                          "--autonomous-gate")
    assert out == ARGV


def test_the_original_argv_is_not_mutated():
    """The runner's argv is shared across every rollout in the process — one
    mutation would accumulate a gate per rollout."""
    original = list(ARGV)
    _argv_with_gate(ARGV, _req({"completion_gate": "test -s a"}), "--autonomous-gate")
    _argv_with_gate(ARGV, _req({"completion_gate": "test -s b"}), "--autonomous-gate")
    assert ARGV == original


def test_serve_prime_declares_the_flag():
    import pathlib
    src = pathlib.Path("examples/prime_agent/serve_prime.py").read_text()
    assert 'gate_flag="--autonomous-gate"' in src


def test_the_gate_reaches_the_spawned_process_argv(tmp_path):
    """The whole point, driven through a real subprocess rather than the pure
    function: the agent binary has to actually RECEIVE the flag and command.

    A script that echoes its own argv stands in for prime-agent.
    """
    import os
    import stat as _stat
    from pathlib import Path

    from percepteye_agent_flywheel.runner import run_rollout_as_command

    script = tmp_path / "agent.sh"
    script.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > argv.txt\n')
    script.chmod(script.stat().st_mode | _stat.S_IEXEC)

    req = _req({"completion_gate": "test -s top_ips.sh"})
    run_rollout_as_command(
        ["sh", str(script), "--autonomous"], req, trajectory_dir=str(tmp_path),
        timeout_s=30.0, env=dict(os.environ), cwd=str(tmp_path),
        gate_flag="--autonomous-gate",
    )

    argv = (Path(tmp_path) / "argv.txt").read_text().splitlines()
    assert argv[-2:] == ["--autonomous-gate", "test -s top_ips.sh"], argv


def test_without_a_flag_the_spawned_process_sees_no_extra_args(tmp_path):
    import os
    import stat as _stat
    from pathlib import Path

    from percepteye_agent_flywheel.runner import run_rollout_as_command

    script = tmp_path / "agent.sh"
    script.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > argv.txt\n')
    script.chmod(script.stat().st_mode | _stat.S_IEXEC)

    req = _req({"completion_gate": "test -s top_ips.sh"})
    run_rollout_as_command(
        ["sh", str(script), "--autonomous"], req, trajectory_dir=str(tmp_path),
        timeout_s=30.0, env=dict(os.environ), cwd=str(tmp_path),
    )

    # `$@` excludes the script path, which is `$0`.
    argv = (Path(tmp_path) / "argv.txt").read_text().splitlines()
    assert argv == ["--autonomous"], argv
