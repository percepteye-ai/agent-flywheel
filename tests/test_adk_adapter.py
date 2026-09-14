"""The ADK adapter, tested through ADK's own dispatch.

Every test here drives a real ``PluginManager``, not a hand-rolled stand-in.
That is deliberate: the behaviours worth protecting are all properties of
ADK's ``_run_callbacks`` -- its early exit on a non-``None`` return, and its
``raise RuntimeError(...) from e`` around any plugin that throws. A fake
dispatcher would let all four traps pass while production hit every one.
"""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("google.adk")

from google.adk.plugins.plugin_manager import PluginManager

from agent_flywheel.adapters.adk import ADKAdapter
from agent_flywheel.outcomes import (
    Trajectory,
    _RolloutScope,
)


class _Tool:
    def __init__(self, name: str = "lookup") -> None:
        self.name = name


class _Ctx:
    """The fields the adapter reads off a real ToolContext."""

    def __init__(self, call_id: str = "fc_1", agent_name: str | None = "planner") -> None:
        self.function_call_id = call_id
        self.agent_name = agent_name


class _ExplodingCtx:
    """A context whose id cannot be read -- stands in for any capture bug."""

    agent_name = "planner"

    @property
    def function_call_id(self):
        raise ValueError("boom")


@pytest.fixture
def bound():
    """Bind, yield an unbind, and always restore -- the patch is process-global."""
    adapter = ADKAdapter()
    unbind = adapter.bind(llm_url="https://gw.invalid/v1", api_key="perk_x")
    try:
        yield adapter
    finally:
        unbind()


def _calls(tmp_path, rid="rl_adk"):
    return Trajectory(str(tmp_path)).read()


async def _one_call(pm, tool, ctx, *, result=None, error=None):
    """Drive one tool call through ADK's real dispatch. Returns what ADK saw."""
    before = await pm.run_before_tool_callback(
        tool=tool, tool_args={"q": "x"}, tool_context=ctx
    )
    if error is not None:
        handled = await pm.run_on_tool_error_callback(
            tool=tool, tool_args={"q": "x"}, tool_context=ctx, error=error
        )
        return before, handled
    after = await pm.run_after_tool_callback(
        tool=tool, tool_args={"q": "x"}, tool_context=ctx, result=result
    )
    return before, after


# ── the install ────────────────────────────────────────────────────────────


def test_the_plugin_installs_with_no_customer_code(bound):
    """Zero instrumentation, stated as a test. The customer constructs their
    own PluginManager -- as every ADK Runner does -- and ours is simply there.
    Binding at Runner instead would miss the app= path entirely."""
    pm = PluginManager()
    assert any(p.name == "agent_flywheel" for p in pm.plugins)


def test_a_customers_own_plugins_survive_the_install(bound):
    """We append, never replace. A customer plugin list that came back short
    would be us breaking their application path to observe it."""
    from google.adk.plugins.base_plugin import BasePlugin

    mine = BasePlugin(name="customer_own")
    pm = PluginManager(plugins=[mine])
    names = [p.name for p in pm.plugins]
    assert "customer_own" in names
    assert "agent_flywheel" in names


def test_unbind_restores_the_original_constructor():
    adapter = ADKAdapter()
    unbind = adapter.bind(llm_url="u", api_key="k")
    unbind()
    pm = PluginManager()
    assert not any(p.name == "agent_flywheel" for p in pm.plugins)


def test_installing_twice_does_not_double_record(bound, tmp_path):
    """ADK's duplicate-name guard is load-bearing: two runners in one process
    must not append two copies and record every call twice."""
    second = ADKAdapter().bind(llm_url="u", api_key="k")
    try:
        pm = PluginManager()
        assert sum(p.name == "agent_flywheel" for p in pm.plugins) == 1
        with _RolloutScope("rl_adk", str(tmp_path)):
            asyncio.run(_one_call(pm, _Tool(), _Ctx(), result={"ok": 1}))
        assert len(_calls(tmp_path)) == 1
    finally:
        second()


# ── the outcome, which is the whole point ──────────────────────────────────


def test_a_returned_result_is_unknown_and_never_ok(bound, tmp_path):
    """THE fabrication test. A tool that caught its own exception and returned
    prose is indistinguishable from one that worked, so a returned result is
    never evidence of success."""
    pm = PluginManager()
    with _RolloutScope("rl_adk", str(tmp_path)):
        asyncio.run(_one_call(pm, _Tool(), _Ctx(), result={"rows": []}))
    (call,) = _calls(tmp_path)
    assert call.outcome == "unknown"
    assert call.status_code is None, "ADK surfaces no HTTP status for a tool call"


def test_an_error_dict_is_a_real_failure(bound, tmp_path):
    """ADK's own tools return {"error": ...}; that IS a verdict."""
    pm = PluginManager()
    with _RolloutScope("rl_adk", str(tmp_path)):
        asyncio.run(_one_call(pm, _Tool(), _Ctx(), result={"error": "denied"}))
    (call,) = _calls(tmp_path)
    assert call.outcome == "failed"
    assert "denied" in (call.error or "")


def test_a_raised_tool_is_a_real_failure(bound, tmp_path):
    """The capability LangChain cannot give us: a raise is distinguishable
    from a return, so this adapter reports `failed` rather than `unknown`."""
    pm = PluginManager()
    with _RolloutScope("rl_adk", str(tmp_path)):
        asyncio.run(_one_call(pm, _Tool(), _Ctx(), error=RuntimeError("upstream")))
    (call,) = _calls(tmp_path)
    assert call.outcome == "failed"
    assert call.error_class == "RuntimeError"
    assert "upstream" in (call.error or "")


def test_the_agent_name_comes_from_adk_itself(bound, tmp_path):
    """ToolContext.agent_name is native here. Attributing every call to one
    agent would train a planner on a coder's tool calls."""
    pm = PluginManager()
    with _RolloutScope("rl_adk", str(tmp_path)):
        asyncio.run(_one_call(pm, _Tool(), _Ctx(agent_name="researcher"), result={}))
    assert _calls(tmp_path)[0].agent_name == "researcher"


def test_latency_is_measured_across_the_call(bound, tmp_path):
    pm = PluginManager()
    ctx, tool = _Ctx(), _Tool()

    async def drive():
        await pm.run_before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
        await asyncio.sleep(0.02)
        await pm.run_after_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx, result={}
        )

    with _RolloutScope("rl_adk", str(tmp_path)):
        asyncio.run(drive())
    assert _calls(tmp_path)[0].latency_ms >= 15.0


# ── the four traps ─────────────────────────────────────────────────────────


def test_before_returns_None_so_the_tool_actually_runs(bound):
    """A non-None return here means ADK SKIPS the tool and uses our value as
    its response. An observer that returns anything is not an observer."""
    pm = PluginManager()
    got = asyncio.run(
        pm.run_before_tool_callback(tool=_Tool(), tool_args={}, tool_context=_Ctx())
    )
    assert got is None


def test_after_returns_None_so_the_real_result_survives(bound, tmp_path):
    """A non-None return REPLACES the tool's result on its way to the model."""
    pm = PluginManager()
    with _RolloutScope("rl_adk", str(tmp_path)):
        _, after = asyncio.run(
            _one_call(pm, _Tool(), _Ctx(), result={"rows": [1, 2]})
        )
    assert after is None


def test_on_error_returns_None_so_the_exception_still_propagates(bound, tmp_path):
    """A non-None return marks the customer's error HANDLED and swallows it.
    Observing an exception must never suppress it: ADK re-raises only when
    every plugin returns None."""
    pm = PluginManager()
    with _RolloutScope("rl_adk", str(tmp_path)):
        _, handled = asyncio.run(
            _one_call(pm, _Tool(), _Ctx(), error=RuntimeError("upstream"))
        )
    assert handled is None, "returning a value here would swallow the customer's error"


def test_a_capture_bug_does_not_take_down_the_customers_run(bound, tmp_path):
    """The fourth trap, and the worst. ADK wraps any exception a plugin raises
    in RuntimeError and propagates it into the agent. A bug in our observer
    must cost us the observation and nothing else."""
    pm = PluginManager()
    ctx = _ExplodingCtx()
    with _RolloutScope("rl_adk", str(tmp_path)):
        # Every callback, because the guard has to be on every one of them.
        assert asyncio.run(
            pm.run_before_tool_callback(tool=_Tool(), tool_args={}, tool_context=ctx)
        ) is None
        assert asyncio.run(
            pm.run_after_tool_callback(
                tool=_Tool(), tool_args={}, tool_context=ctx, result={}
            )
        ) is None
        assert asyncio.run(
            pm.run_on_tool_error_callback(
                tool=_Tool(), tool_args={}, tool_context=ctx, error=ValueError("x")
            )
        ) is None
    assert _calls(tmp_path) is None, "nothing observable, and nothing invented"


# ── double firing ──────────────────────────────────────────────────────────


def test_an_error_handled_by_another_plugin_records_exactly_once(bound, tmp_path):
    """functions.py step 3: when some OTHER plugin handles the error, the call
    falls through to step 4 and after_tool_callback fires for the same call.
    State is popped, not read, so exactly one of them records it."""
    pm = PluginManager()
    tool, ctx = _Tool(), _Ctx()

    async def drive():
        await pm.run_before_tool_callback(tool=tool, tool_args={}, tool_context=ctx)
        await pm.run_on_tool_error_callback(
            tool=tool, tool_args={}, tool_context=ctx, error=RuntimeError("boom")
        )
        # ADK now continues with the handled response.
        await pm.run_after_tool_callback(
            tool=tool, tool_args={}, tool_context=ctx, result={"error": "boom"}
        )

    with _RolloutScope("rl_adk", str(tmp_path)):
        asyncio.run(drive())

    calls = _calls(tmp_path)
    assert len(calls) == 1, f"one tool call, one record; got {len(calls)}"
    assert calls[0].outcome == "failed"


def test_concurrent_tool_calls_do_not_overwrite_each_other(bound, tmp_path):
    """A parallel-function-calling model has several tools in flight at once.
    Keying on ADK's function_call_id is what keeps them apart; a single
    'current tool' field loses all but the last."""
    pm = PluginManager()
    a, b = _Ctx("fc_a", "planner"), _Ctx("fc_b", "coder")

    async def drive():
        await pm.run_before_tool_callback(
            tool=_Tool("alpha"), tool_args={}, tool_context=a
        )
        await pm.run_before_tool_callback(
            tool=_Tool("beta"), tool_args={}, tool_context=b
        )
        # Finish out of order, which is the normal case.
        await pm.run_after_tool_callback(
            tool=_Tool("beta"), tool_args={}, tool_context=b, result={"error": "no"}
        )
        await pm.run_after_tool_callback(
            tool=_Tool("alpha"), tool_args={}, tool_context=a, result={"ok": 1}
        )

    with _RolloutScope("rl_adk", str(tmp_path)):
        asyncio.run(drive())

    by_name = {c.name: c for c in _calls(tmp_path)}
    assert set(by_name) == {"alpha", "beta"}
    assert by_name["beta"].outcome == "failed"
    assert by_name["beta"].agent_name == "coder"
    assert by_name["alpha"].outcome == "unknown"
    assert by_name["alpha"].agent_name == "planner"


# ── the probe ──────────────────────────────────────────────────────────────


def test_the_probe_resolves_every_symbol_the_adapter_binds():
    """Detection and binding read the same list, so 'detected' cannot be true
    while binding is broken."""
    from agent_flywheel.binding import probe

    result = probe(ADKAdapter())
    assert result.ok, f"{result.status}: {result.reason} [{result.evidence}]"


def test_the_adapter_does_not_claim_to_route(bound):
    """ADK resolves its endpoint through the provider client the agent builds,
    so routing is the child environment's job. Claiming otherwise would assert
    something we never verified."""
    caps = ADKAdapter().capabilities()
    assert caps.reports_outcomes is True
    assert caps.routes_llm_calls is False
