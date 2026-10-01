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

from .._body_bound import bounded
from ..binding import AdapterCapabilities
from ..genai_otlp import new_span_id, now_unix_nano
from ..outcomes import model_capture_active, record_model_boundary, record_tool_call

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
_PLUGIN_NAME = "agent_flywheel"

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
            #: Model calls in flight, keyed by (invocation, agent): ADK makes
            #: one agent's model calls one at a time. Holds the LlmRequest
            #: OBJECT, not a copy -- see `before_model_callback`.
            self._open_models: dict[tuple[str, str], dict[str, Any]] = {}

        # -- the model boundary: OBSERVE ONLY -----------------------------
        #
        # Recorded only inside a production turn whose agent may upload model
        # content; everywhere else these return None before touching anything.
        # ALL THREE RETURN None ON EVERY PATH, and that is the same trap the
        # tool callbacks document: a value from before_model_callback is used
        # AS the model's response and the model is never called; a value from
        # after_model_callback REPLACES the response; a value from
        # on_model_error_callback swallows the error.
        async def before_model_callback(self, *, callback_context, llm_request):
            try:
                if model_capture_active():
                    key = _model_key(callback_context)
                    with self._open_lock:
                        # The REQUEST OBJECT, read when the response arrives:
                        # the agent's own before_model callbacks run after
                        # this plugin and may still edit it, and what matters
                        # is what the model was actually sent.
                        self._open_models[key] = {
                            "start": now_unix_nano(), "request": llm_request}
                        while len(self._open_models) > _MAX_OPEN_MODEL_CALLS:
                            self._open_models.pop(next(iter(self._open_models)))
            except Exception:
                pass
            return None

        async def after_model_callback(self, *, callback_context, llm_response):
            try:
                if model_capture_active() and not getattr(
                        llm_response, "partial", False):
                    with self._open_lock:
                        state = self._open_models.pop(
                            _model_key(callback_context), None)
                    if state is not None:
                        record_model_boundary(_chat_operation(
                            state, end=now_unix_nano(), response=llm_response))
            except Exception:
                pass
            return None

        async def on_model_error_callback(self, *, callback_context,
                                          llm_request, error):
            try:
                if model_capture_active():
                    with self._open_lock:
                        state = self._open_models.pop(
                            _model_key(callback_context), None)
                    if state is not None:
                        record_model_boundary(_chat_operation(
                            state, end=now_unix_nano(),
                            error_type=type(error).__name__))
            except Exception:
                pass
            return None

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
                        "start": now_unix_nano(),
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
                if model_capture_active():
                    # What the model is sent back: ADK wraps a result that is
                    # not a dict as {"result": value} before it builds the
                    # function_response, and so does this.
                    _record_execution(state, result=result if isinstance(
                        result, dict) else {"result": result})
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
                    error=bounded(f"{type(error).__name__}: {error}"),
                    error_class=type(error).__name__,
                    latency_ms=(time.perf_counter() - state["t0"]) * 1000.0,
                    agent_name=state.get("agent_name"),
                    tool_call_id=state.get("tool_call_id"),
                )
                if model_capture_active():
                    _record_execution(state, error_type=type(error).__name__)
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


#: Model calls whose response never came (an agent callback answered in the
#: model's place, so after_model_callback never fires) are forgotten oldest
#: first past this many, rather than held for the life of the process.
_MAX_OPEN_MODEL_CALLS = 1024

#: ADK's client-side function-call id prefix. ADK strips these ids from every
#: request before the model sees it, so the model's tool calls carry none.
_CLIENT_CALL_ID_PREFIX = "adk-"


def _model_key(callback_context: Any) -> tuple[str, str]:
    return (str(getattr(callback_context, "invocation_id", "") or ""),
            str(getattr(callback_context, "agent_name", "") or ""))


def _record_execution(state: dict[str, Any], *, result: Any = None,
                      error_type: str | None = None) -> None:
    """One tool execution on the turn's model boundary. Never raises."""
    try:
        op: dict[str, Any] = {
            "operation": "execute_tool",
            "span_id": new_span_id(),
            "start_time_unix_nano": state["start"],
            "end_time_unix_nano": now_unix_nano(),
            "tool_name": state["name"],
            "tool_type": "function",
        }
        call_id = state.get("tool_call_id")
        # An ADK client id is not an id the MODEL used: ADK removes it from
        # every request, so a tool span carrying it would join no tool call
        # the model made. The span is left to join by name instead.
        if call_id and not str(call_id).startswith(_CLIENT_CALL_ID_PREFIX):
            op["tool_call_id"] = str(call_id)
        op["arguments"] = state["arguments"]
        if error_type:
            op["error_type"] = error_type
        else:
            op["result"] = result
        record_model_boundary(op)
    except Exception:
        return


# -- ADK's request and response, as the GenAI semantic conventions spell them
#
# A TRANSLATION, never an addition: every part below is a part ADK was about
# to send, or one the model sent back.

#: A LiteLLM model name's provider prefix, as the conventions name it.
_PROVIDERS = {
    "openai": "openai", "azure": "azure.ai.openai", "anthropic": "anthropic",
    "bedrock": "aws.bedrock", "groq": "groq", "mistral": "mistral_ai",
    "deepseek": "deepseek", "xai": "x_ai", "cohere": "cohere",
    "vertex_ai": "gcp.vertex_ai", "gemini": "gcp.gemini",
}

#: Gemini's FinishReason names, as the output message's finish_reason spells
#: them.
_FINISH = {
    "STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter",
    "RECITATION": "content_filter", "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter", "SPII": "content_filter",
}


def _provider(model: Any) -> str:
    """Which provider served ``model``: a LiteLLM prefix when it has one, and
    otherwise the conventions' generic Google value -- ADK's native models are
    Gemini, served by either Google API, and which one is not in the request."""
    name = str(model or "")
    prefix, sep, _ = name.partition("/")
    if sep and prefix in _PROVIDERS:
        return _PROVIDERS[prefix]
    return "gcp.gen_ai"


def _jsonable(value: Any) -> Any:
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json", exclude_none=True)
    return value


def _part(part: Any) -> dict[str, Any] | None:
    text = getattr(part, "text", None)
    if isinstance(text, str):
        if getattr(part, "thought", None):
            return {"type": "reasoning", "content": text}
        return {"type": "text", "content": text}
    call = getattr(part, "function_call", None)
    if call is not None:
        call_id = getattr(call, "id", None)
        return {"type": "tool_call", "id": call_id, "name": call.name,
                "arguments": _jsonable(getattr(call, "args", None) or {})}
    response = getattr(part, "function_response", None)
    if response is not None:
        return {"type": "tool_call_response",
                "id": getattr(response, "id", None),
                "response": _jsonable(getattr(response, "response", None))}
    blob = getattr(part, "inline_data", None)
    if blob is not None:
        import base64

        mime = getattr(blob, "mime_type", None)
        data = getattr(blob, "data", None) or b""
        return {"type": "blob", "modality": (mime or "").partition("/")[0]
                or "unknown", "mime_type": mime,
                "content": base64.b64encode(data).decode("ascii")}
    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        mime = getattr(file_data, "mime_type", None)
        return {"type": "uri", "modality": (mime or "").partition("/")[0]
                or "unknown", "mime_type": mime,
                "uri": getattr(file_data, "file_uri", None)}
    dumped = _jsonable(part)
    if isinstance(dumped, dict) and dumped:
        kind = next(iter(dumped))
        return {"type": kind, kind: dumped[kind]}
    return None


def _message(content: Any) -> dict[str, Any]:
    parts = [p for p in (_part(x) for x in (getattr(content, "parts", None)
                                             or [])) if p is not None]
    role = getattr(content, "role", None)
    if role == "model":
        role = "assistant"
    elif parts and all(p["type"] == "tool_call_response" for p in parts):
        # ADK sends function responses in a "user" content; the conventions'
        # role for a tool's answer is "tool".
        role = "tool"
    else:
        role = role or "user"
    return {"role": role, "parts": parts}


def _system(config: Any) -> list[dict[str, Any]]:
    value = getattr(config, "system_instruction", None)
    if value is None:
        return []
    if isinstance(value, str):
        return [{"type": "text", "content": value}] if value else []
    items = getattr(value, "parts", None)
    if items is None:
        items = value if isinstance(value, list) else [value]
    out = []
    for item in items:
        if isinstance(item, str):
            out.append({"type": "text", "content": item})
        else:
            part = _part(item)
            if part is not None:
                out.append(part)
    return out


def _tools(config: Any) -> list[dict[str, Any]] | None:
    out: list[dict[str, Any]] = []
    for tool in getattr(config, "tools", None) or []:
        for decl in getattr(tool, "function_declarations", None) or []:
            flat: dict[str, Any] = {"type": "function", "name": decl.name}
            if getattr(decl, "description", None) is not None:
                flat["description"] = decl.description
            schema = getattr(decl, "parameters_json_schema", None)
            if schema is None and getattr(decl, "parameters", None) is not None:
                schema = _jsonable(decl.parameters)
            if schema is not None:
                flat["parameters"] = schema
            out.append(flat)
    return out or None


def _chat_operation(state: dict[str, Any], *, end: str, response: Any = None,
                    error_type: str | None = None) -> dict[str, Any]:
    request = state["request"]
    config = getattr(request, "config", None)
    op: dict[str, Any] = {
        "operation": "chat",
        "span_id": new_span_id(),
        "start_time_unix_nano": state["start"],
        "end_time_unix_nano": end,
        "provider_name": _provider(getattr(request, "model", None)),
    }
    if getattr(request, "model", None):
        op["request_model"] = str(request.model)
    system = _system(config)
    if system:
        op["system_instructions"] = system
    op["input_messages"] = [_message(c) for c in request.contents or []]
    tools = _tools(config)
    if tools:
        op["tool_definitions"] = tools
    if error_type is None and response is not None \
            and getattr(response, "error_code", None) \
            and getattr(response, "content", None) is None:
        error_type = str(response.error_code)
    if error_type:
        op["error_type"] = error_type
        return op
    if response is None or getattr(response, "content", None) is None:
        return op
    output = _message(response.content)
    output["role"] = "assistant"
    native = getattr(getattr(response, "finish_reason", None), "name", None)
    calls = any(p["type"] == "tool_call" for p in output["parts"])
    output["finish_reason"] = "tool_call" if calls else _FINISH.get(
        native, "stop")
    op["output_messages"] = [output]
    if native:
        op["finish_reasons"] = [native]
    if getattr(response, "model_version", None):
        op["response_model"] = str(response.model_version)
    return op


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
            return "failed", bounded(str(err))
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
