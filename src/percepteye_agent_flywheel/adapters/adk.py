"""Google ADK adapter.

The best-fitting framework we have bound so far, for one structural reason:
ADK has a first-class *plugin* interface that is global to a run, and it
separates a tool that RAISED from a tool that RETURNED. LangChain gives us
neither -- its callbacks are per-invocation and ``on_tool_end`` fires for
success and swallowed-failure alike. Here, ``on_tool_error_callback`` is a
real verdict, so this adapter reports genuine ``failed`` outcomes rather than
the defensive ``unknown`` the LangChain adapter is mostly limited to.

**Zero instrumentation.** The customer does not pass us a plugin, and does not
edit their ``Runner``. Binding wraps ``PluginManager.__init__``, which is the
chokepoint every runner goes through -- both the deprecated
``Runner(plugins=[...])`` argument and the current ``app=`` route construct one.
Binding above that, at ``Runner``, would miss whichever path we did not patch.

**Four traps, all of them the same trap.** ADK's ``_run_callbacks`` stops at
the first plugin returning non-``None`` and uses that value as the result:

* ``before_tool_callback`` -> the tool is NEVER CALLED and our value becomes
  its response;
* ``after_tool_callback`` -> our value REPLACES the tool's real result;
* ``on_tool_error_callback`` -> the error is marked HANDLED and swallowed.

An observer that returns anything is no longer an observer, so every callback
here returns ``None`` on every path, including the failure paths.

The fourth is the one that bites hardest. ``_run_callbacks`` wraps any
exception a plugin raises in ``raise RuntimeError(...) from e``, which
propagates into the customer's agent. A bug in our capture would therefore
take down their run -- the one thing this SDK must never do. Every callback
body is wrapped so that a capture failure costs us the observation and nothing
else.

**Double-firing.** In ``functions.py`` step 3, a tool that raises and whose
error is handled by *some other* plugin sets ``function_response`` and falls
through to step 4, so ``on_tool_error_callback`` AND ``after_tool_callback``
both fire for one call. State is therefore POPPED, not read: whichever fires
first records, and the second finds nothing and returns.
"""
from __future__ import annotations

import threading
import time
from typing import Any

from ..binding import AdapterCapabilities
from ..outcomes import record_tool_call

__all__ = ["ADKAdapter"]

_DISTRIBUTION = "google-adk"
#: Exactly the symbols this adapter touches, so detection cannot pass while
#: binding is broken. ``PluginManager.register_plugin`` is here because the
#: install wraps ``__init__`` and then calls it.
_REQUIRED = (
    "google.adk.plugins.base_plugin:BasePlugin",
    "google.adk.plugins.plugin_manager:PluginManager",
    "google.adk.plugins.plugin_manager:PluginManager.register_plugin",
    "google.adk.tools.tool_context:ToolContext",
)

#: One name, so ADK's own duplicate-name check in ``register_plugin`` becomes
#: our idempotence guarantee: a second runner in the same process cannot
#: install a second copy and double-record every call.
_PLUGIN_NAME = "percepteye_agent_flywheel"

_lock = threading.Lock()
_plugin_cls: type | None = None
_installed: Any = None  # the original PluginManager.__init__, while patched


def _build_plugin_cls() -> type:
    """Create the plugin class once per process, lazily.

    Lazily because this module must import with ADK absent -- an adapter whose
    framework is not installed fails its probe, it does not fail the import of
    the SDK.
    """
    global _plugin_cls
    if _plugin_cls is not None:
        return _plugin_cls

    from google.adk.plugins.base_plugin import BasePlugin

    class _PercepteyePlugin(BasePlugin):
        """Records tool outcomes. Returns ``None`` from everything, always."""

        def __init__(self) -> None:
            super().__init__(name=_PLUGIN_NAME)
            # Keyed by function_call_id: ADK's own per-call identifier. A
            # single "current tool" field would be overwritten whenever two
            # tool calls are in flight, which is the normal case for a
            # parallel function-calling model.
            self._open: dict[str, dict[str, Any]] = {}
            self._open_lock = threading.Lock()

        # -- start ------------------------------------------------------
        async def before_tool_callback(self, *, tool, tool_args, tool_context):
            try:
                key = _call_key(tool_context)
                if key is None:
                    return None
                with self._open_lock:
                    self._open[key] = {
                        "name": getattr(tool, "name", None) or "tool",
                        "arguments": dict(tool_args or {}),
                        "t0": time.perf_counter(),
                        # ADK's function_call_id doubles as the model-assigned
                        # call id: it reappears in the function_response part
                        # of the next request, which is what lets the control
                        # plane attribute this call to the completion that
                        # emitted it.
                        "tool_call_id": key,
                        # ADK names the calling agent natively. No metadata
                        # archaeology, and no attributing every call to one
                        # agent when a graph actually had several.
                        "agent_name": getattr(tool_context, "agent_name", None),
                    }
            except Exception:
                pass
            # MUST be None: a value here means ADK skips the tool entirely.
            return None

        # -- returned ---------------------------------------------------
        async def after_tool_callback(self, *, tool, tool_args, tool_context, result):
            try:
                state = self._take(tool_context)
                if state is None:
                    return None
                outcome, error = _classify(result)
                record_tool_call(
                    state["name"], state["arguments"],
                    outcome=outcome,
                    # ADK surfaces no HTTP status for a tool call. Inventing
                    # one from the payload is how an unobserved call becomes
                    # a landed one.
                    status_code=None,
                    output=_safe(result),
                    error=error,
                    latency_ms=(time.perf_counter() - state["t0"]) * 1000.0,
                    agent_name=state.get("agent_name"),
                    tool_call_id=state.get("tool_call_id"),
                )
            except Exception:
                pass
            # MUST be None: a value here REPLACES the tool's real result.
            return None

        # -- raised -----------------------------------------------------
        async def on_tool_error_callback(self, *, tool, tool_args, tool_context, error):
            try:
                state = self._take(tool_context)
                if state is None:
                    return None
                record_tool_call(
                    state["name"], state["arguments"],
                    outcome="failed", status_code=None,
                    error=f"{type(error).__name__}: {error}"[:500],
                    error_class=type(error).__name__,
                    latency_ms=(time.perf_counter() - state["t0"]) * 1000.0,
                    agent_name=state.get("agent_name"),
                    tool_call_id=state.get("tool_call_id"),
                )
            except Exception:
                pass
            # MUST be None: a value here marks the customer's error HANDLED
            # and swallows it. Observing an exception must not suppress it.
            return None

        def _take(self, tool_context) -> dict[str, Any] | None:
            """Pop the open state, so exactly one callback can record a call."""
            key = _call_key(tool_context)
            if key is None:
                return None
            with self._open_lock:
                return self._open.pop(key, None)

    _plugin_cls = _PercepteyePlugin
    return _plugin_cls


def _call_key(tool_context: Any) -> str | None:
    """ADK's own id for this tool call, or None if it did not supply one."""
    key = getattr(tool_context, "function_call_id", None)
    return str(key) if key else None


def _classify(result: Any) -> tuple[str, str | None]:
    """``(outcome, error)`` for a returned tool result.

    ADK's convention across its own tools is ``{"error": "..."}``, which is a
    real verdict and is reported as ``failed``. Everything else is genuinely
    ``unknown``: a tool that caught its own exception and returned prose is
    indistinguishable from one that worked, and calling that ``ok`` is the
    single most consequential lie a capture can tell -- it is the fabrication
    this SDK exists to prevent, one layer down.
    """
    if isinstance(result, dict):
        err = result.get("error")
        if err:
            return "failed", str(err)[:500]
    return "unknown", None


def _safe(value: Any) -> Any:
    try:
        if isinstance(value, (str, int, float, bool, dict, list)) or value is None:
            return value
        return str(value)
    except Exception:
        return None


class ADKAdapter:
    """Outcome capture for Google ADK, installed without touching the agent."""

    name = "adk"
    distribution = _DISTRIBUTION
    supported = ">=1,<3"
    required_symbols = _REQUIRED

    def capabilities(self) -> AdapterCapabilities:
        # routes_llm_calls is False on purpose. ADK resolves its model endpoint
        # through the provider client the agent constructs, so routing is the
        # child environment's job exactly as it is by default. Claiming to
        # route here would assert something we have not verified.
        return AdapterCapabilities(routes_llm_calls=False, reports_outcomes=True)

    def bind(self, *, llm_url: str, api_key: str) -> Any:
        """Wrap ``PluginManager.__init__`` so every runner gets the plugin."""
        from google.adk.plugins.plugin_manager import PluginManager

        plugin_cls = _build_plugin_cls()

        global _installed
        with _lock:
            if _installed is not None:
                # Already patched by an earlier bind in this process. Unbinding
                # would restore a stale __init__ over the live one, so this
                # bind owns nothing and its unbind is a no-op.
                return lambda: None

            original = PluginManager.__init__

            def _patched(self, *a: Any, **kw: Any) -> None:
                original(self, *a, **kw)
                try:
                    self.register_plugin(plugin_cls())
                except ValueError:
                    # ADK's duplicate-name guard. Already present: nothing to do.
                    pass
                except Exception:
                    # Capture is never worth breaking a runner over.
                    pass

            PluginManager.__init__ = _patched  # type: ignore[method-assign]
            _installed = original

        def _unbind() -> None:
            global _installed
            with _lock:
                if _installed is None:
                    return
                PluginManager.__init__ = _installed  # type: ignore[method-assign]
                _installed = None

        return _unbind
