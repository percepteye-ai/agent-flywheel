"""An agent that is a PROCESS, not an importable Python object.

The importable-entrypoint requirement was the single biggest limit on who could
be served: a compiled binary, a Node CLI, anything not written in Python had no
way in at all. A command has the same contract as every other entrypoint — it
is handed the rollout through the environment, it writes artifacts, and its
outcomes come from the trajectory — so an existing CLI needs no changes.
"""
from __future__ import annotations

import stat
import time

from percepteye_agent_flywheel import RolloutRequest
from percepteye_agent_flywheel.runner import RolloutRunner


class _T:
    lease_ms = 1_800_000

    def __init__(self):
        self.reported: list[str] = []
        self.abandoned: list[str] = []
        self.artifacts: list[str] = []

    def claim(self, n):
        return [], 10

    def heartbeat(self, r, *, progress=None):
        return None

    def report(self, r, o):
        self.reported.append(r.rollout_id)
        return {}

    def abandon(self, r, *, reason, detail=""):
        self.abandoned.append(reason)

    def register_artifact(self, r, *, name, size_bytes, sha256, **kw):
        self.artifacts.append(name)
        return {}


def _script(tmp_path, body: str) -> str:
    p = tmp_path / "agent.sh"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _req(tmp_path, task="ship it", seconds=60.0) -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_c", turn_input=task,
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_c", api_key="perk_x",
        artifacts_dir="rollouts/rl_c", deadline_ts=time.time() + seconds,
        lease_token="lt",
    )


def _runner(tmp_path, tr, argv):
    return RolloutRunner(argv, transport=tr, input_shape="text",
                         artifacts_dir=str(tmp_path), rollout_timeout_s=30.0)


def test_a_shell_agent_is_served_without_any_python(tmp_path):
    """No import, no entrypoint, no SDK inside the agent."""
    tr = _T()
    argv = ["sh", _script(tmp_path, 'echo "handled: $PERCEPTEYE_TURN_INPUT"\n')]
    out = _runner(tmp_path, tr, argv).run_one(_req(tmp_path))
    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.final_text == "handled: ship it"


def test_the_turn_input_arrives_on_stdin_too(tmp_path):
    """CLIs differ and guessing wrongly is silent, so it is offered both ways."""
    tr = _T()
    out = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'read line; echo "stdin:$line"\n')]
                  ).run_one(_req(tmp_path))
    assert out.final_text == "stdin:ship it"


def test_it_gets_the_per_rollout_endpoint_and_credential(tmp_path):
    """The whole premise: its model calls must reach our gateway."""
    tr = _T()
    out = _runner(tmp_path, tr,
                  ["sh", _script(tmp_path, 'echo "$OPENAI_BASE_URL|$OPENAI_API_KEY"\n')]
                  ).run_one(_req(tmp_path))
    base, key = out.final_text.split("|")
    assert base.endswith("/rollouts/rl_c")
    assert key == "perk_x"


def test_the_control_plane_credential_still_does_not_cross(tmp_path, monkeypatch):
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pehk_live_must_not_leak")
    tr = _T()
    out = _runner(tmp_path, tr,
                  ["sh", _script(tmp_path, 'echo "${PERCEPTEYE_API_KEY:-<absent>}"\n')]
                  ).run_one(_req(tmp_path))
    assert out.final_text == "<absent>"


def test_outcomes_come_from_the_trajectory_a_cli_can_append_to(tmp_path):
    """A non-Python agent reports tool calls by appending JSONL — no SDK, no
    import, just a file whose path it is given."""
    tr = _T()
    body = (
        'D="$PERCEPTEYE_TRAJECTORY_DIR"\n'
        'printf \'{"name":"deploy","arguments":{},"outcome":"ok","status_code":200}\\n\''
        ' >> "$D/tool_calls.jsonl"\n'
        'echo done\n'
    )
    out = _runner(tmp_path, tr, ["sh", _script(tmp_path, body)]).run_one(_req(tmp_path))
    assert out.tool_calls is not None and len(out.tool_calls) == 1
    assert (out.tool_calls[0].name, out.tool_calls[0].outcome) == ("deploy", "ok")


def test_a_nonzero_exit_is_excluded_not_scored(tmp_path):
    """A crashed run says nothing about the policy."""
    tr = _T()
    out = _runner(tmp_path, tr,
                  ["sh", _script(tmp_path, 'echo "partial"; echo boom >&2; exit 3\n')]
                  ).run_one(_req(tmp_path))
    # The agent RAN and exited non-zero — its own failure, not a
    # failure to start it. Popen raises for the latter.
    assert out.exclusion_reason == "agent_exit_nonzero"
    assert "exit 3" in (out.exclusion_detail or "")
    assert "boom" in (out.exclusion_detail or "")


def test_a_hung_command_is_actually_killed(tmp_path):
    tr = _T()
    started = time.time()
    out = RolloutRunner(["sh", _script(tmp_path, "sleep 120\n")], transport=tr,
                        input_shape="text", artifacts_dir=str(tmp_path),
                        rollout_timeout_s=1.5).run_one(_req(tmp_path, seconds=1.5))
    assert out.exclusion_reason == "deadline_exceeded"
    assert time.time() - started < 30, "the command was not killed"
