"""LangChain / LangGraph adapter.

**What this adapter is for, stated precisely.** Its job is per-tool-call
outcome capture. It routes LLM traffic only when you hand it the chat model
object, because LangChain's endpoint is a property of the client instance and
an already-constructed client cannot be reached from the outside. When you do
not pass one, routing is the child environment's job -- and that is the
default, working arrangement, not a degraded one.

The distinction matters because of the trap this adapter exists to surface:
``@langchain/openai`` and its Python sibling resolve a constructor ``base_url``
argument AHEAD of the environment. A customer whose code says
``ChatOpenAI(base_url=...)`` sends every completion to that URL no matter what
we put in the child's environment, the rollout completes, and the result is
indistinguishable from a clean one. Passing ``llm=`` lets us check and correct
it; not passing it means we cannot, and the agent registers at a grade that
says so.

**The outcome trap.** ``on_tool_end`` does NOT mean the tool succeeded. A tool
configured with ``handle_tool_error=True`` catches its own exception and
returns a string that reads like a normal result; and ``_format_output``
returns raw content rather than a ``ToolMessage`` whenever there is no tool
call id to attach. Both arrive at ``on_tool_end`` looking like success. This
adapter reports them as ``unknown``, never ``ok``. That is the whole reason it
is worth writing rather than telling people to call ``record_tool_call``.
"""
from __future__ import annotations

import threading
import time
from contextvars import ContextVar
from typing import Any

from ..binding import AdapterCapabilities
from ..errors import RoutingNotInstallable
from ..outcomes import record_tool_call

__all__ = ["LangChainAdapter"]

_DISTRIBUTION = "langchain-core"
#: Exactly the symbols this adapter touches. The probe resolves this tuple, so
#: detection cannot pass while binding is broken -- they read the same list.
_REQUIRED = (
    "langchain_core.callbacks.base:BaseCallbackHandler",
    "langchain_core.tracers.context:register_configure_hook",
    "langchain_core.messages.tool:ToolMessage",
)

#: The active binding. A context variable, not a module global: two rollouts in
#: one process must not see each other's.
#: Holds the HANDLER INSTANCE for the rollout on this context, not the URL.
#: langchain's configure hook reads this variable and, when no handler class is
#: registered alongside it, uses the value ITSELF as a callback handler -- so a
#: string here made every tool call raise
#: ``'str' object has no attribute 'ignore_agent'`` and captured nothing.
_ACTIVE: ContextVar[Any] = ContextVar("percepteye_lc_active", default=None)

_lock = threading.Lock()
_handler_cls: type | None = None
_hook_registered = False


def _build_handler_cls() -> type:
    """Create the callback handler class ONCE per process, and cache it.

    Defining it inside ``bind()`` would mint a new class object on every call.
    LangChain's de-duplication is ``isinstance(handler, handler_class)``, which
    is per-class, so a second bind would install a second handler and every
    tool event would be recorded twice.

    Built lazily rather than at import so this module carries no dependency on
    langchain being installed.
    """
    global _handler_cls
    if _handler_cls is not None:
        return _handler_cls

    from langchain_core.callbacks.base import BaseCallbackHandler
    from langchain_core.messages.tool import ToolMessage

    class _PercepteyeHandler(BaseCallbackHandler):
        """Records tool outcomes against the rollout bound in this context.

        Subclasses ``BaseCallbackHandler`` deliberately. A previous SDK's
        handler subclassed a plain ABC, and langchain_core reads
        ``handler.run_inline`` and ``handler.raise_error`` unconditionally
        before dispatching, so it raised an AttributeError before a single
        event was delivered.
        """

        raise_error = False
        run_inline = False

        def __init__(self) -> None:
            super().__init__()
            self._open: dict[Any, dict[str, Any]] = {}

        # State is keyed by run_id, which every tool callback receives. An
        # earlier implementation discarded it and kept a single mutable
        # "current tool" field, so concurrent tools overwrote each other.
        def on_tool_start(self, serialized, input_str, *, run_id=None, **kw):
            if _ACTIVE.get() is None:
                return
            name = (serialized or {}).get("name") or "tool"
            self._open[run_id] = {
                "name": name,
                "arguments": kw.get("inputs") or {"input": input_str},
                "t0": time.perf_counter(),
                # Which agent made this call, when the graph says so. LangGraph
                # puts the node name in metadata; tags are the older channel.
                # None when nothing names it -- unattributed is the honest
                # answer, and is not a claim that one agent made everything.
                "agent_name": _agent_name(kw),
            }

        def on_tool_end(self, output, *, run_id=None, **kw):
            state = self._open.pop(run_id, None)
            if state is None or _ACTIVE.get() is None:
                return
            outcome, error = _classify_output(output, ToolMessage)
            record_tool_call(
                state["name"], state["arguments"],
                outcome=outcome,
                # Always None: LangChain surfaces no HTTP status here, and
                # inventing one from the payload is how an unobserved call
                # becomes a landed one.
                status_code=None,
                output=_safe(output),
                error=error,
                latency_ms=(time.perf_counter() - state["t0"]) * 1000.0,
                agent_name=state.get("agent_name"),
            )

        def on_tool_error(self, error, *, run_id=None, **kw):
            state = self._open.pop(run_id, None)
            if state is None or _ACTIVE.get() is None:
                return
            record_tool_call(
                state["name"], state["arguments"],
                outcome="failed", status_code=None,
                agent_name=state.get("agent_name"),
                error=f"{type(error).__name__}: {error}",
                latency_ms=(time.perf_counter() - state["t0"]) * 1000.0,
            )

    _handler_cls = _PercepteyeHandler
    return _handler_cls


def _agent_name(kw: dict) -> str | None:
    """The enclosing agent's name, or None.

    Read from the framework's own run metadata rather than guessed. LangGraph
    stamps the node it is executing; older chains carry tags. Returning None
    when nothing names it is deliberate: attributing every call to one agent
    would train a planner on a coder's tool calls.
    """
    meta = kw.get("metadata") or {}
    for key in ("langgraph_node", "agent_name", "checkpoint_ns"):
        v = meta.get(key)
        if isinstance(v, str) and v:
            return v.split(":", 1)[0]
    for tag in kw.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("agent:"):
            return tag.split(":", 1)[1]
    return None


def _classify_output(output: Any, tool_message_cls: type) -> tuple[str, str | None]:
    """``(outcome, error)`` for one tool result.

    A ``ToolMessage`` carries an explicit ``status``, which is a real verdict.
    Anything else is genuinely unknown: the tool may have succeeded, or it may
    have swallowed its own exception and returned prose. Reporting that as
    ``ok`` is the single most consequential lie a capture can tell.
    """
    if isinstance(output, tool_message_cls):
        status = getattr(output, "status", None)
        if status == "error":
            return "failed", str(getattr(output, "content", ""))[:500] or None
        if status == "success":
            return "ok", None
    return "unknown", None


def _safe(value: Any) -> Any:
    try:
        content = getattr(value, "content", value)
        return content if isinstance(content, (str, int, float, bool, dict, list)) \
            else str(content)
    except Exception:
        return None


class LangChainAdapter:
    """Outcome capture for LangChain and LangGraph; routing when given a model."""

    name = "langchain"
    distribution = _DISTRIBUTION
    supported = ">=0.3,<2"
    required_symbols = _REQUIRED

    def __init__(self, llm: Any = None) -> None:
        #: The chat model to re-point. Optional: without it routing is the
        #: child environment's job, which is the ordinary arrangement.
        self.llm = llm

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            routes_llm_calls=self.llm is not None,
            reports_outcomes=True,
        )

    def bind(self, *, llm_url: str, api_key: str) -> Any:
        handler_cls = _build_handler_cls()
        self._route(llm_url, api_key)

        global _hook_registered
        with _lock:
            if not _hook_registered:
                from langchain_core.tracers.context import register_configure_hook

                # No ``env_var``: passing one makes langchain construct the
                # handler whenever that variable is set, EVEN IF the context
                # variable is unset -- which would attach an unbound handler
                # in forked children and worker threads.
                #
                # No handler CLASS either: langchain would then construct its
                # own instance and ignore ours, and our per-rollout state lives
                # on the instance. The context variable carries the instance.
                register_configure_hook(_ACTIVE, True)
                _hook_registered = True

        token = _ACTIVE.set(handler_cls())

        def _unbind() -> None:
            _ACTIVE.reset(token)

        return _unbind

    def _route(self, llm_url: str, api_key: str) -> None:
        """Re-point the supplied chat model, and verify it took.

        Setting an attribute that a model ignores is worse than not setting it,
        so this reads the value back. A client that cannot be re-pointed is a
        refusal: its traffic would go to a live third party and the rollout
        would still complete.
        """
        if self.llm is None:
            return
        applied = False
        for attr in ("openai_api_base", "base_url"):
            if hasattr(self.llm, attr):
                try:
                    setattr(self.llm, attr, llm_url)
                    applied = str(getattr(self.llm, attr, "")).startswith(llm_url)
                except Exception:
                    applied = False
                if applied:
                    break
        for attr in ("openai_api_key", "api_key"):
            if hasattr(self.llm, attr):
                try:
                    setattr(self.llm, attr, api_key)
                except Exception:
                    pass
        if not applied:
            raise RoutingNotInstallable(
                reason="HardcodedBaseUrl",
                evidence=(
                    f"{type(self.llm).__name__} did not accept a new base URL; "
                    f"its endpoint is fixed at construction"
                ),
                remedy=(
                    "construct the model inside your entrypoint factory so each "
                    "rollout builds its own, or drop llm= and let the child "
                    "environment route (only works if you did not pass an "
                    "explicit base_url)"
                ),
            )
