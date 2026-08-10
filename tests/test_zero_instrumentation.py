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

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel.contract import RolloutRequest
from percepteye_agent_flywheel.entrypoint import resolve_entrypoint
from percepteye_agent_flywheel.runner import run_rollout_in_process

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
        _req(tmp_path), ledger_dir=str(tmp_path), timeout_s=30.0, adapters="auto",
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
        _req(tmp_path), ledger_dir=str(tmp_path), timeout_s=30.0, adapters=[],
    )
    assert out.tool_calls is None


def test_auto_skips_a_framework_that_is_not_installed():
    """"auto" probes without committing: an absent framework is skipped, not
    an error. A NAMED adapter is a different assertion and does raise."""
    from percepteye_agent_flywheel.binding import bind_selected

    report, unbinds = bind_selected("auto", llm_url="https://gw/x", api_key="k")
    for u in unbinds:
        u()
    assert "langchain" in report.bound
    assert report.capabilities.reports_outcomes is True


def test_an_explicitly_named_missing_adapter_raises():
    from percepteye_agent_flywheel.binding import bind_selected

    with pytest.raises(KeyError, match="nosuchframework"):
        bind_selected(["nosuchframework"], llm_url="https://gw/x", api_key="k")
