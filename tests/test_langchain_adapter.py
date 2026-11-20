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
    from agent_flywheel.adapters import langchain as lcmod

    assert lcmod._active() is None, "a previous test left a binding active"
    yield
    leaked = lcmod._active()
    lcmod._ACTIVE.set(None)
    lcmod._fallback, lcmod._fallback_refs = None, 0
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
    """The hook is registered process-wide once and outlives every bind; with
    nothing bound -- no context binding and no process-wide one -- the
    handler records nothing."""
    from agent_flywheel.adapters.langchain import _ACTIVE, _build_handler_cls

    assert _ACTIVE.get() is None
    h = _build_handler_cls()()
    h.on_tool_start({"name": "x"}, "in", run_id="r9")
    assert "r9" not in h._open


# ── a bind is seen by threads that never saw the binding context ──────────────


def _calc_tool():
    from langchain_core.tools import tool

    @tool
    def calc(x: int) -> int:
        """Double x."""
        return 2 * x

    return calc


def test_a_thread_pool_worker_captures_under_a_startup_bind(monkeypatch, tmp_path):
    """attach() binds once, on the thread that starts the server; Flask,
    Django, gthread and ThreadPoolExecutor workers run with an EMPTY context.
    They used to see no binding and capture nothing at all."""
    import concurrent.futures as cf

    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_pool")
    monkeypatch.setenv(TRAJECTORY_DIR_ENV, str(tmp_path))
    calc = _calc_tool()
    unbind = LangChainAdapter().bind(llm_url="", api_key="")
    try:
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            assert list(pool.map(lambda x: calc.invoke({"x": x}), [1, 2])) == [2, 4]
    finally:
        unbind()
    calls = Trajectory(tmp_path).read() or []
    assert [c.name for c in calls] == ["calc", "calc"]


def test_a_raw_thread_captures_under_a_startup_bind(monkeypatch, tmp_path):
    import threading

    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_thread")
    monkeypatch.setenv(TRAJECTORY_DIR_ENV, str(tmp_path))
    calc = _calc_tool()
    unbind = LangChainAdapter().bind(llm_url="", api_key="")
    try:
        t = threading.Thread(target=lambda: calc.invoke({"x": 3}))
        t.start()
        t.join()
    finally:
        unbind()
    assert [c.name for c in Trajectory(tmp_path).read() or []] == ["calc"]


def test_an_attachment_captures_a_turn_served_on_a_worker_thread(tmp_path):
    """The production shape end to end: attach() on this thread, the turn and
    its tool call on a pool worker."""
    import concurrent.futures as cf

    import agent_flywheel as fw

    calc = _calc_tool()
    px = fw.attach(agent_id="pool-agent", trajectory_root=str(tmp_path),
                   register=False, adapters=["langchain"])
    try:
        def serve_one():
            with px.turn() as tid:
                calc.invoke({"x": 5})
            return tid

        with cf.ThreadPoolExecutor(max_workers=1) as pool:
            tid = pool.submit(serve_one).result()
    finally:
        px.close()
    assert [c.name for c in Trajectory(tmp_path / tid).read() or []] == ["calc"]


def test_unbind_from_another_thread_does_not_raise_and_retires_the_bind():
    """A server's shutdown hook runs on another thread. reset() refuses a
    token from another context; the unbind used to raise ValueError there."""
    import threading

    from agent_flywheel.adapters.langchain import _active

    unbind = LangChainAdapter().bind(llm_url="", api_key="")
    errors: list[BaseException] = []

    def _run():
        try:
            unbind()
        except BaseException as exc:  # pragma: no cover - the failure shown
            errors.append(exc)

    t = threading.Thread(target=_run)
    t.start()
    t.join()
    assert errors == []
    # Retired in the binding context too, not only in the unbinding one.
    assert _active() is None


def test_the_process_binding_lasts_while_any_bind_does():
    import threading

    from agent_flywheel.adapters.langchain import _active

    u1 = LangChainAdapter().bind(llm_url="", api_key="")
    u2 = LangChainAdapter().bind(llm_url="", api_key="")
    seen: list = []
    u1()
    u1()  # idempotent: a second call must not release u2's hold
    t = threading.Thread(target=lambda: seen.append(_active()))
    t.start()
    t.join()
    assert seen[0] is not None
    u2()
    assert _active() is None


# ── routing a real chat model ──────────────────────────────────────────────


def test_an_attach_bind_leaves_the_model_untouched():
    """attach() binds with llm_url="": a model handed to the adapter must keep
    its endpoint and key. Rewriting both to "" broke every production call."""
    class Model:
        base_url = "https://api.openai.com/v1"
        api_key = "sk-theirs"

    m = Model()
    unbind = LangChainAdapter(llm=m).bind(llm_url="", api_key="")
    try:
        # While bound -- the whole life of a production process.
        assert (m.base_url, m.api_key) == ("https://api.openai.com/v1", "sk-theirs")
    finally:
        unbind()


def test_unbind_restores_the_rewritten_fields():
    class Model:
        base_url = "https://api.openai.com/v1"
        api_key = "sk-theirs"

    m = Model()
    unbind = LangChainAdapter(llm=m).bind(llm_url="https://gw/r", api_key="perk_1")
    assert m.base_url == "https://gw/r"
    unbind()
    assert (m.base_url, m.api_key) == ("https://api.openai.com/v1", "sk-theirs")


class _Client:
    def __init__(self, base_url):
        self.base_url = base_url


def test_a_client_built_at_construction_is_rebuilt_not_just_relabelled():
    """A ChatOpenAI-shaped model: the field is what the client was built
    from, and the client is what sends. Rewriting the field alone left every
    request on the old URL while the check read the field back as routed."""
    class Model:
        def __init__(self):
            self.openai_api_base = "https://api.openai.com/v1"
            self.openai_api_key = "sk-theirs"
            self.client = None
            self.validate_environment()

        def validate_environment(self):
            if self.client is None:
                self.client = _Client(self.openai_api_base)
            return self

    m = Model()
    original = m.client
    unbind = LangChainAdapter(llm=m).bind(llm_url="https://gw/r", api_key="perk_1")
    assert m.client.base_url == "https://gw/r"
    unbind()
    assert m.client is original and m.openai_api_base == "https://api.openai.com/v1"


def test_a_client_that_cannot_be_rebuilt_is_a_refusal_and_is_restored():
    class Model:
        openai_api_base = "https://api.openai.com/v1"
        client = _Client("https://api.openai.com/v1")

    m = Model()
    with pytest.raises(RoutingNotInstallable) as exc:
        LangChainAdapter(llm=m).bind(llm_url="https://gw/r", api_key="k")
    assert "still sends to https://api.openai.com/v1" in exc.value.evidence
    assert m.openai_api_base == "https://api.openai.com/v1"


def test_a_real_chat_openai_sends_to_the_rollout_endpoint_and_back():
    """Against the installed langchain_openai, read off the provider clients
    the model actually sends through."""
    lco = pytest.importorskip("langchain_openai")
    m = lco.ChatOpenAI(model="gpt-4o", api_key="sk-theirs")
    clients = (m.root_client, m.client, m.async_client, m.root_async_client)
    unbind = LangChainAdapter(llm=m).bind(llm_url="https://gw.example/r1/v1",
                                          api_key="perk_1")
    try:
        for c in (m.root_client, m.root_async_client, m.client._client,
                  m.async_client._client):
            assert str(c.base_url).startswith("https://gw.example/r1/v1")
            assert c.api_key == "perk_1"
    finally:
        unbind()
    assert (m.root_client, m.client, m.async_client, m.root_async_client) == clients
    assert str(m.root_client.base_url).startswith("https://api.openai.com")
    assert m.openai_api_key.get_secret_value() == "sk-theirs"


def test_a_real_chat_anthropic_sends_to_the_rollout_endpoint_and_back():
    lca = pytest.importorskip("langchain_anthropic")
    m = lca.ChatAnthropic(model="claude-x", api_key="sk-theirs")
    _ = m._client
    unbind = LangChainAdapter(llm=m).bind(llm_url="https://gw.example/r1",
                                          api_key="perk_1")
    try:
        assert str(m._client.base_url).startswith("https://gw.example/r1")
        assert m._client.api_key == "perk_1"
    finally:
        unbind()
    assert str(m._client.base_url).startswith("https://api.anthropic.com")
    assert m._client.api_key == "sk-theirs"
