"""The subprocess path, end to end.

This is the load-bearing integration test. Per-rollout capture has to survive
a process boundary, because the customer's agent runs in a child and the
parent is the one that reports. A design that captured only in the parent
would record nothing while appearing to work -- and an empty tool-call list
does not read downstream as "capture failed", it reads as "this agent used no
tools", which is a much more expensive kind of wrong.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import pytest

from agent_flywheel import RolloutRequest
from agent_flywheel.errors import ConfigurationError
from agent_flywheel.runner import RolloutRunner

TESTS_DIR = str(pathlib.Path(__file__).resolve().parent)


class _NullTransport:
    """The runner needs a transport to construct; these tests drive run_one()
    directly and never claim or report."""

    def claim(self, n): return [], 1000
    def heartbeat(self, r, progress=None): return None
    def report(self, r, o): return {}
    def abandon(self, r, reason, detail=""): return None


def _rollout(tmp_path, task="do the thing", seconds=60.0) -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_test_1",
        turn_input=task,
        llm_url="https://gw.example.invalid/llm/v1/rollouts/rl_test_1",
        api_key="perk_single_use_token",
        artifacts_dir=str(tmp_path),
        deadline_ts=time.time() + seconds,
    )


def _runner(tmp_path, spec="fixture_agent:build", **kw):
    kwargs = dict(
        transport=_NullTransport(),
        input_shape="text",
        artifacts_dir=str(tmp_path),
        isolation="subprocess",
        concurrency=1,
        rollout_timeout_s=60.0,
    )
    kwargs.update(kw)
    return RolloutRunner(spec, **kwargs)


@pytest.fixture(autouse=True)
def _child_can_import(monkeypatch):
    """The child inherits PYTHONPATH, which is how it finds fixture_agent."""
    existing = os.environ.get("PYTHONPATH", "")
    monkeypatch.setenv(
        "PYTHONPATH", TESTS_DIR + (os.pathsep + existing if existing else "")
    )


def test_tool_calls_recorded_in_the_child_reach_the_parent(tmp_path):
    """THE integration. Three calls recorded inside a spawned process are
    merged back by the parent, with their outcomes intact."""
    out = _runner(tmp_path).run_one(_rollout(tmp_path))

    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.final_text == "handled:do the thing"
    assert out.tool_calls is not None, "capture did not survive the process boundary"
    assert len(out.tool_calls) == 3

    by_name = {c.name: c for c in out.tool_calls}
    assert (by_name["lookup"].outcome, by_name["lookup"].status_code) == ("ok", 200)
    # An app-level refusal returned as HTTP 200: both halves preserved.
    assert (by_name["charge"].outcome, by_name["charge"].status_code) == ("failed", 200)
    # The unknown call kept no status, in the field or the body.
    assert by_name["opaque"].outcome == "unknown"
    assert by_name["opaque"].status_code is None
    assert "status_code" not in str(by_name["opaque"].output)
    assert "done" in str(by_name["opaque"].output), "content must survive scrubbing"


def test_the_child_cannot_see_the_control_plane_credential(tmp_path, monkeypatch):
    """The security boundary, proved across a real process boundary rather
    than asserted about a dict."""
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pehk_live_must_not_leak")
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://api.example.invalid")
    monkeypatch.setenv("JIRA_API_TOKEN", "customer-token")

    out = _runner(tmp_path, spec="fixture_agent:build_leak_probe").run_one(
        _rollout(tmp_path)
    )
    assert out.exclusion_reason is None, out.exclusion_detail
    report = dict(part.split("=", 1) for part in out.final_text.split("|"))

    assert report["control_key"] == "<absent>"
    assert "pehk_live_must_not_leak" not in out.final_text
    assert set(report["percepteye_vars"].split(",")) == {
        "PERCEPTEYE_ROLLOUT_ID", "PERCEPTEYE_TRAJECTORY_DIR"
    }
    # Their own credentials still reach their agent -- that is the premise.
    assert report["customer_key"] == "customer-token"
    # And the completion wire is ours for this rollout.
    assert report["openai_base"].endswith("/rollouts/rl_test_1")
    assert report["openai_key"] == "perk_single_use_token"


def test_an_agent_that_raises_is_reported_not_propagated(tmp_path):
    """Fail open on the customer's application path. Their bug is a rollout
    result, not our crash."""
    out = _runner(tmp_path, spec="fixture_agent:build_boom").run_one(_rollout(tmp_path))
    assert out.exclusion_reason == "entrypoint_error"
    assert "agent exploded" in (out.exclusion_detail or "")


def test_a_hung_agent_is_actually_killed(tmp_path):
    """Only a child process gives a real timeout; in-process we could stop
    waiting but not stop running. The fixture blocks for two minutes, so a
    timeout that is not enforced hangs the suite rather than passing."""
    started = time.time()
    out = _runner(
        tmp_path, spec="fixture_agent:build_slow", rollout_timeout_s=2.0
    ).run_one(_rollout(tmp_path, seconds=2.0))
    assert out.exclusion_reason == "deadline_exceeded"
    assert time.time() - started < 60, "the child was not actually killed"


def test_the_deadline_comes_from_the_server_not_a_local_default(tmp_path):
    """The server's deadline wins when it is tighter than the local ceiling."""
    started = time.time()
    out = _runner(
        tmp_path, spec="fixture_agent:build_slow", rollout_timeout_s=900.0
    ).run_one(_rollout(tmp_path, seconds=2.0))
    assert out.exclusion_reason == "deadline_exceeded"
    assert time.time() - started < 60


def test_subprocess_isolation_refuses_a_live_object_at_registration(tmp_path):
    """A child cannot receive a Python object. Say so at startup, not at
    rollout time."""
    class Agent:
        def invoke(self, x): return x

    with pytest.raises(ConfigurationError, match="importable"):
        _runner(tmp_path, spec=Agent())


def test_inprocess_refuses_concurrency_above_one(tmp_path):
    class Agent:
        def invoke(self, x): return x

    with pytest.raises(ConfigurationError, match="concurrency"):
        _runner(tmp_path, spec=Agent(), isolation="inprocess", concurrency=4)


def test_inprocess_path_also_captures(tmp_path):
    out = _runner(tmp_path, isolation="inprocess").run_one(_rollout(tmp_path))
    assert out.tool_calls is not None and len(out.tool_calls) == 3


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX path assumptions")
def test_trajectory_is_written_under_the_rollouts_own_directory(tmp_path):
    _runner(tmp_path).run_one(_rollout(tmp_path))
    assert (tmp_path / "_percepteye" / "rl_test_1" / "tool_calls.jsonl").exists()
