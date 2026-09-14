"""Capture with no Percepteye code in the customer's agent.

This file exists because the adapter subsystem was BUILT and then wired to
nothing: `serve()` had no `adapters=` parameter, so the README's own example
raised TypeError and every framework integration was dead code. The tests below
fail if that happens again, because they go through the public surface rather
than calling the adapter directly.
"""
from __future__ import annotations

import inspect
import time

import pytest

import agent_flywheel as fw
from agent_flywheel.contract import RolloutRequest
from agent_flywheel.entrypoint import resolve_entrypoint
from agent_flywheel.runner import run_rollout_in_process

pytest.importorskip("langchain_core")


class Uninstrumented:
    """Not one Percepteye import. Not one record_tool_call."""

    def invoke(self, task: str) -> str:
        from langchain_core.tools import tool

        @tool
        def lookup(q: str) -> str:
            """Look something up."""
            return f"found:{q}"

        @tool
        def explode(q: str) -> str:
            """Always fails."""
            raise RuntimeError("upstream refused")

        out = lookup.invoke({"q": task})
        try:
            explode.invoke({"q": task})
        except Exception:
            pass
        return str(out)


def _req(tmp_path) -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_z", turn_input="widgets",
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_z", api_key="perk_x",
        artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60,
    )


def test_serve_accepts_the_adapters_the_readme_documents():
    """The README's framework example used a keyword serve() did not have, so
    following the documentation raised TypeError."""
    assert "adapters" in inspect.signature(fw.serve).parameters


def test_outcomes_are_captured_with_no_instrumentation(tmp_path):
    """THE claim. The agent above contains no Percepteye code; the outcomes
    come from the adapter binding the framework's own callbacks."""
    out = run_rollout_in_process(
        resolve_entrypoint(Uninstrumented(), input_shape="text"),
        _req(tmp_path), trajectory_dir=str(tmp_path), timeout_s=30.0, adapters="auto",
    )
    assert out.tool_calls is not None, "capture produced nothing"
    by_name = {c.name: c for c in out.tool_calls}
    assert set(by_name) == {"lookup", "explode"}

    # A tool that RAISED is a real verdict, and it is recorded as one.
    assert by_name["explode"].outcome == "failed"

    # A tool that returned normally is NOT. LangChain hands back raw content,
    # so the tool may equally have swallowed its own exception -- and calling
    # that "ok" is the single most consequential lie a capture can tell. This
    # asserts the honest answer, not a better-looking one.
    assert by_name["lookup"].outcome == "unknown"
    assert by_name["lookup"].status_code is None


def test_binding_off_captures_nothing_rather_than_guessing(tmp_path):
    """`adapters=[]` is a real choice, and it must be silent rather than
    half-instrumented."""
    out = run_rollout_in_process(
        resolve_entrypoint(Uninstrumented(), input_shape="text"),
        _req(tmp_path), trajectory_dir=str(tmp_path), timeout_s=30.0, adapters=[],
    )
    assert out.tool_calls is None


def test_auto_skips_a_framework_that_is_not_installed():
    """"auto" probes without committing: an absent framework is skipped, not
    an error. A NAMED adapter is a different assertion and does raise."""
    from agent_flywheel.binding import bind_selected

    report, unbinds = bind_selected("auto", llm_url="https://gw/x", api_key="k")
    for u in unbinds:
        u()
    assert "langchain" in report.bound
    assert report.capabilities.reports_outcomes is True
    # Every built-in whose framework is installed must appear. Asserting only
    # langchain would let a new adapter be added to the registry and never
    # reached by "auto" -- built, and wired to nothing, which is the defect
    # this repo keeps rediscovering.
    import importlib.util as _u

    if _u.find_spec("google.adk") is not None:
        assert "adk" in report.bound


def test_an_explicitly_named_missing_adapter_raises():
    from agent_flywheel.binding import bind_selected

    with pytest.raises(KeyError, match="nosuchframework"):
        bind_selected(["nosuchframework"], llm_url="https://gw/x", api_key="k")


# ── multi-agent attribution ────────────────────────────────────────────────


def test_a_call_is_attributed_to_the_agent_that_made_it(tmp_path):
    """Training a planner on a coder's tool calls is the multi-agent version
    of scoring an unobserved call as a success."""
    from langchain_core.tools import tool

    from agent_flywheel.binding import bind_selected
    from agent_flywheel.outcomes import Trajectory, _RolloutScope

    @tool
    def deploy(q: str) -> str:
        """Deploy."""
        return "ok"

    report, unbinds = bind_selected("auto", llm_url="https://gw/x", api_key="k")
    assert "langchain" in report.bound
    try:
        with _RolloutScope("rl_m", str(tmp_path)):
            deploy.invoke({"q": "x"}, config={"metadata": {"langgraph_node": "coder"}})
            deploy.invoke({"q": "y"})          # nothing names this one
    finally:
        for u in unbinds:
            u()

    calls = Trajectory(str(tmp_path)).read() or []
    assert len(calls) == 2, calls
    assert calls[0].agent_name == "coder"
    # Unattributed stays unattributed. Filling it in with the last-seen name
    # would be a fabrication in exactly the shape this contract refuses.
    assert calls[1].agent_name is None


def test_agent_name_survives_the_wire():
    from agent_flywheel import ToolCallOutcome

    wire = ToolCallOutcome(name="t", arguments={}, outcome="ok", status_code=200,
                           agent_name="planner").to_wire()
    assert wire["agent_name"] == "planner"
    # Absent rather than null when there is nothing to say.
    plain = ToolCallOutcome(name="t", arguments={}, outcome="ok",
                            status_code=200).to_wire()
    assert "agent_name" not in plain


# ── the same claim, for Google ADK ─────────────────────────────────────────


class UninstrumentedADK:
    """Not one Percepteye import, and no plugin passed to anything.

    Constructing a PluginManager is what every ADK Runner does internally, so
    this is the real install path rather than a stand-in for it.
    """

    def invoke(self, task: str) -> str:
        import asyncio

        from google.adk.plugins.plugin_manager import PluginManager

        class _Tool:
            def __init__(self, name):
                self.name = name

        class _Ctx:
            def __init__(self, cid, agent):
                self.function_call_id, self.agent_name = cid, agent

        pm = PluginManager()

        async def drive():
            ok, bad = _Tool("lookup"), _Tool("deploy")
            a, b = _Ctx("fc_1", "planner"), _Ctx("fc_2", "coder")
            await pm.run_before_tool_callback(
                tool=ok, tool_args={"q": task}, tool_context=a)
            await pm.run_after_tool_callback(
                tool=ok, tool_args={"q": task}, tool_context=a, result={"rows": 1})
            await pm.run_before_tool_callback(
                tool=bad, tool_args={"q": task}, tool_context=b)
            await pm.run_on_tool_error_callback(
                tool=bad, tool_args={"q": task}, tool_context=b,
                error=RuntimeError("upstream refused"))

        asyncio.run(drive())
        return "done"


def test_adk_outcomes_are_captured_with_no_instrumentation(tmp_path):
    """THE claim, through the public surface, for the second framework.

    Reaching the trajectory proves the whole chain: "auto" probed ADK, bound it by
    wrapping PluginManager.__init__, the customer's own runner picked the
    plugin up, and the outcomes were written where the parent reads them.
    """
    pytest.importorskip("google.adk")
    out = run_rollout_in_process(
        resolve_entrypoint(UninstrumentedADK(), input_shape="text"),
        _req(tmp_path), trajectory_dir=str(tmp_path), timeout_s=30.0, adapters="auto",
    )
    assert out.tool_calls is not None, "capture produced nothing"
    by_name = {c.name: c for c in out.tool_calls}
    assert set(by_name) == {"lookup", "deploy"}

    # A raise is a real verdict. This is the capability LangChain cannot give.
    assert by_name["deploy"].outcome == "failed"
    assert by_name["deploy"].agent_name == "coder"

    # A returned result is not evidence of success, in either framework.
    assert by_name["lookup"].outcome == "unknown"
    assert by_name["lookup"].agent_name == "planner"
