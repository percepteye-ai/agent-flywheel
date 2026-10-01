"""The LangChain adapter, against the real langchain_core API.

Exercised against the installed library rather than a mock of it, because the
failure this adapter exists to prevent is precisely "the integration reported
success while binding to a symbol that no longer exists".
"""
from __future__ import annotations

import pytest

from agent_flywheel.adapters import BUILTIN_ADAPTERS, load_builtin
from agent_flywheel.adapters.langchain import (
    LangChainAdapter,
    _classify_output,
)
from agent_flywheel.binding import bind_all, probe
from agent_flywheel.errors import RoutingNotInstallable
from agent_flywheel.outcomes import ROLLOUT_ID_ENV, TRAJECTORY_DIR_ENV, Trajectory

lc = pytest.importorskip("langchain_core")


@pytest.fixture(autouse=True)
def _no_binding_leaks():
    """A binding must not survive its test.

    Not hygiene for its own sake: a bind() whose unbind() never runs leaves
    the next rollout attributing its tool calls to the previous one's context,
    which is exactly the cross-rollout leak the design is built to prevent. If
    a test leaks here, production can too.
    """
    from agent_flywheel.adapters.langchain import _ACTIVE

    assert _ACTIVE.get() is None, "a previous test left a binding active"
    yield
    leaked = _ACTIVE.get()
    _ACTIVE.set(None)
    assert leaked is None, f"this test left a binding active: {leaked!r}"


# ── it ships with the SDK ───────────────────────────────────────────────────


def test_the_adapter_is_a_builtin():
    assert "langchain" in BUILTIN_ADAPTERS
    assert isinstance(load_builtin("langchain"), LangChainAdapter)


def test_core_does_not_import_langchain_at_module_scope():
    """The adapter carries no dependency: an absent framework fails its probe
    rather than breaking the import."""
    import pathlib

    src = pathlib.Path(
        "src/agent_flywheel/adapters/langchain.py"
    ).read_text().split("class LangChainAdapter")[0]
    module_level = [
        ln for ln in src.splitlines()
        if ln.startswith(("import ", "from ")) and "langchain" in ln
    ]
    assert not module_level, f"module-scope langchain import: {module_level}"


# ── the probe ───────────────────────────────────────────────────────────────


def test_it_probes_clean_against_the_installed_library():
    """If langchain_core moves a symbol this adapter binds to, THIS fails —
    which is the whole point of the probe reading the same tuple bind does."""
    r = probe(LangChainAdapter())
    assert r.ok, f"{r.status}: {r.reason} [{r.evidence}]"
    assert r.version


def test_every_required_symbol_actually_resolves():
    from agent_flywheel.binding import _resolve_symbol

    for path in LangChainAdapter.required_symbols:
        assert _resolve_symbol(path) is not None


# ── outcome classification, the reason this adapter exists ──────────────────


def _tm(**kw):
    from langchain_core.messages.tool import ToolMessage

    kw.setdefault("content", "result")
    kw.setdefault("tool_call_id", "c1")
    return ToolMessage(**kw)


def test_an_explicit_error_status_is_failed():
    from langchain_core.messages.tool import ToolMessage

    outcome, err = _classify_output(_tm(status="error", content="boom"), ToolMessage)
    assert outcome == "failed"
    assert "boom" in (err or "")


def test_an_explicit_success_status_is_ok():
    from langchain_core.messages.tool import ToolMessage

    assert _classify_output(_tm(status="success"), ToolMessage)[0] == "ok"


def test_a_non_toolmessage_result_is_unknown_never_ok():
    """THE trap. `_format_output` returns raw content when there is no tool
    call id, and a tool with handle_tool_error=True returns a string that
    reads like a normal result. Both arrive at on_tool_end looking like
    success."""
    from langchain_core.messages.tool import ToolMessage

    assert _classify_output("looks like a normal result", ToolMessage)[0] == "unknown"
    assert _classify_output({"ok": True}, ToolMessage)[0] == "unknown"
    assert _classify_output(None, ToolMessage)[0] == "unknown"


def test_a_toolmessage_without_a_status_is_unknown():
    from langchain_core.messages.tool import ToolMessage

    msg = _tm()
    if getattr(msg, "status", None) in (None, ""):
        assert _classify_output(msg, ToolMessage)[0] == "unknown"
    else:
        # Newer versions default status to "success"; that IS an explicit
        # verdict from the library and is honoured.
        assert _classify_output(msg, ToolMessage)[0] == "ok"


# ── binding ─────────────────────────────────────────────────────────────────


def test_binding_without_a_model_reports_no_routing():
    """Honest capabilities: without the model object we cannot re-point
    anything, and routing is the child environment's job."""
    a = LangChainAdapter()
    caps = a.capabilities()
    assert caps.reports_outcomes is True
    assert caps.routes_llm_calls is False


def test_binding_is_idempotent_and_installs_one_handler():
    """Defining the handler inside bind() would mint a new class per call, and
    langchain's dedup is per-class — so every tool event would record twice."""
    from agent_flywheel.adapters.langchain import _build_handler_cls

    assert _build_handler_cls() is _build_handler_cls()


def test_the_handler_subclasses_the_real_base():
    """A previous SDK's handler subclassed a plain ABC; langchain reads
    handler.run_inline before dispatching, so it raised before delivering a
    single event."""
    from langchain_core.callbacks.base import BaseCallbackHandler

    from agent_flywheel.adapters.langchain import _build_handler_cls

    cls = _build_handler_cls()
    assert issubclass(cls, BaseCallbackHandler)
    h = cls()
    assert h.run_inline is False and h.raise_error is False


def test_unbind_clears_the_active_binding():
    from agent_flywheel.adapters.langchain import _ACTIVE

    a = LangChainAdapter()
    unbind = a.bind(llm_url="https://gw/x", api_key="perk_1")
    # The context variable carries the HANDLER, not the URL. langchain's
    # configure hook uses this value as a callback handler when no handler
    # class is registered alongside it -- a string here made every tool call
    # raise "'str' object has no attribute 'ignore_agent'" and captured
    # nothing at all.
    handler = _ACTIVE.get()
    assert handler is not None
    assert hasattr(handler, "on_tool_end"), f"not a callback handler: {handler!r}"
    unbind()
    assert _ACTIVE.get() is None


def test_a_model_that_cannot_be_repointed_is_a_refusal():
    """Setting an attribute a model ignores is worse than not setting it: the
    traffic goes to a live third party and the rollout still completes."""
    class Frozen:
        base_url = "https://api.openai.com/v1"

        def __setattr__(self, k, v):
            raise AttributeError("frozen")

    with pytest.raises(RoutingNotInstallable) as exc:
        LangChainAdapter(llm=Frozen()).bind(llm_url="https://gw/x", api_key="k")
    assert exc.value.reason == "HardcodedBaseUrl"
    assert exc.value.remedy


def test_a_repointable_model_is_rewritten_and_verified():
    class Model:
        base_url = "https://api.openai.com/v1"
        api_key = "sk-theirs"

    m = Model()
    a = LangChainAdapter(llm=m)
    assert a.capabilities().routes_llm_calls is True
    unbind = a.bind(llm_url="https://gw/rollouts/rl_1", api_key="perk_1")
    try:
        assert m.base_url == "https://gw/rollouts/rl_1"
        assert m.api_key == "perk_1"
    finally:
        unbind()


def test_bind_all_accepts_the_builtin():
    report, unbinds = bind_all([LangChainAdapter()], llm_url="https://gw/x",
                               api_key="perk_1")
    assert report.bound == ["langchain"]
    assert report.capabilities.reports_outcomes is True
    for u in unbinds:
        u()


# ── end to end through the trajectory ───────────────────────────────────────────


def test_tool_events_reach_the_trajectory_with_honest_outcomes(monkeypatch, tmp_path):
    from langchain_core.messages.tool import ToolMessage

    from agent_flywheel.adapters.langchain import _build_handler_cls

    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_lc")
    monkeypatch.setenv(TRAJECTORY_DIR_ENV, str(tmp_path))

    a = LangChainAdapter()
    unbind = a.bind(llm_url="https://gw/x", api_key="k")
    h = _build_handler_cls()()

    h.on_tool_start({"name": "search"}, "q", run_id="r1")
    h.on_tool_end(ToolMessage(content="ok", tool_call_id="c", status="success"),
                  run_id="r1")

    h.on_tool_start({"name": "charge"}, "x", run_id="r2")
    h.on_tool_end("silently swallowed an exception", run_id="r2")

    h.on_tool_start({"name": "boom"}, "x", run_id="r3")
    h.on_tool_error(RuntimeError("nope"), run_id="r3")
    unbind()

    calls = Trajectory(tmp_path).read() or []
    by = {c.name: c for c in calls}
    assert by["search"].outcome == "ok"
    assert by["charge"].outcome == "unknown", "a prose result is not a success"
    assert by["boom"].outcome == "failed"
    # No adapter invents a status; the framework surfaces none.
    assert all(c.status_code is None for c in calls)


def test_a_raised_traceback_keeps_its_exception_line(monkeypatch, tmp_path):
    """The raise was recorded whole while a returned error was cut to its
    first 500 chars. Both get the SDK's one body bound now."""
    from agent_flywheel._body_bound import bounded
    from agent_flywheel.adapters.langchain import _build_handler_cls

    from .test_error_evidence_survives_capture import _TRACEBACK_TAIL, _long_traceback

    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_lc")
    monkeypatch.setenv(TRAJECTORY_DIR_ENV, str(tmp_path))
    unbind = LangChainAdapter().bind(llm_url="https://gw/x", api_key="k")
    h = _build_handler_cls()()
    h.on_tool_start({"name": "boom"}, "x", run_id="r1")
    h.on_tool_error(RuntimeError(_long_traceback()), run_id="r1")
    unbind()

    (call,) = Trajectory(tmp_path).read() or []
    assert call.error.endswith(_TRACEBACK_TAIL)
    assert call.error == bounded(f"RuntimeError: {_long_traceback()}")


def test_events_outside_a_binding_are_ignored():
    """The handler is registered process-wide once; only a bound context
    records. Otherwise a sub-agent or worker thread would attribute its tools
    to whatever rollout ran last."""
    from agent_flywheel.adapters.langchain import _ACTIVE, _build_handler_cls

    assert _ACTIVE.get() is None
    h = _build_handler_cls()()
    h.on_tool_start({"name": "x"}, "in", run_id="r9")
    assert "r9" not in h._open
