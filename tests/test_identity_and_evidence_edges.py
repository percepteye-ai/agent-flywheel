"""PY-M13, runner side: the execution identity covers the command output
adapter and the rollout ceiling; an artifact's digest and content come from
one read; one unreadable claimed rollout does not stop the loop."""
from __future__ import annotations

import hashlib
import time
import typing
import warnings

from agent_flywheel import RolloutRequest
from agent_flywheel import runner as _runner_mod
from agent_flywheel.contract import RolloutOutput
from agent_flywheel.errors import ContractViolation
from agent_flywheel.event_stream import CommandOutputProjection
from agent_flywheel.runner import RolloutRunner, _attempt_key
from tests.test_execution_identity import _command_identity


class _Adapter:
    def __init__(self, name):
        self.adapter_name = name

    def project(self, stdout):
        return CommandOutputProjection(recognized=False)


def test_the_output_adapter_and_the_ceiling_are_part_of_the_identity(tmp_path):
    argv = ["agent", "--mode", "json"]
    base = _command_identity(tmp_path, argv)
    adapter = _command_identity(tmp_path, argv, command_output_adapter=_Adapter("x"))
    other = _command_identity(tmp_path, argv, command_output_adapter=_Adapter("y"))
    ceiling = _command_identity(tmp_path, argv, rollout_timeout_s=120.0)
    assert len({base, adapter, other, ceiling}) == 4, (
        "two agents that read stdout differently, or run under different "
        "ceilings, shared one execution identity")


def test_an_artifacts_content_and_digest_are_the_same_bytes(tmp_path):
    """``read_text`` turned CRLF into LF, so the inlined content no longer
    matched the digest sent with it."""
    class _T:
        agent_id = "a"
    runner = RolloutRunner(["true"], transport=_T(), input_shape="text",
                           artifacts_dir=str(tmp_path))
    req = RolloutRequest(rollout_id="rl_c", turn_input="t", llm_url="u", api_key="k",
                         artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60)
    root = tmp_path / "_percepteye" / _attempt_key(req)
    root.mkdir(parents=True)
    body = b"line one\r\nline two\r\n"
    (root / "out.txt").write_bytes(body)
    out = RolloutOutput(final_text="")
    runner._collect_artifacts(req, out)
    entry = out.artifacts["out.txt"]
    assert entry["content"].encode("utf-8") == body
    assert entry["sha256"] == hashlib.sha256(body).hexdigest()


def test_an_unreadable_claimed_rollout_does_not_stop_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(_runner_mod, "sleep_ms", lambda ms: None)

    class _T:
        agent_id = "a"
        calls = 0
        reported: typing.ClassVar[list[str]] = []

        def claim(self, n):
            _T.calls += 1
            if _T.calls == 1:
                raise ContractViolation("RolloutRequest missing required field(s)")
            if _T.calls == 2:
                return [RolloutRequest(
                    rollout_id="rl_ok", turn_input="t", llm_url="u", api_key="k",
                    artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60)], 10
            return [], 10

        def heartbeat(self, r, *, progress=None): return None
        def report(self, r, o):
            _T.reported.append(r.rollout_id)
            return {}
        def abandon(self, r, *, reason, detail=""): return None

    runner = RolloutRunner(["true"], transport=_T(), input_shape="text",
                           artifacts_dir=str(tmp_path))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=1)
    assert _T.reported == ["rl_ok"]
