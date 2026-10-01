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

from .._body_bound import bounded
from ..binding import AdapterCapabilities
from ..errors import RoutingNotInstallable
from ..genai_otlp import new_span_id, now_unix_nano
from ..outcomes import model_capture_active, record_model_boundary, record_tool_call

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
            #: Model calls in flight, by run_id -- the request as it was SENT,
            #: snapshotted at start, waiting for what came back.
            self._open_models: dict[Any, dict[str, Any]] = {}

        # ── the model boundary: OBSERVE ONLY ───────────────────────────────
        #
        # Recorded only inside a production turn whose agent may upload model
        # content (`model_capture_active`); everywhere else these return
        # before touching anything. Every one returns None and swallows its
        # own failure: a callback's return value is ignored by LangChain, and
        # a capture bug must cost the observation, never the model call.
        def on_chat_model_start(self, serialized, messages, *, run_id=None,
                                **kw):
            if not model_capture_active():
                return None
            try:
                self._open_models[run_id] = {
                    "start": now_unix_nano(),
                    "request": _chat_request(messages[0] if messages else [],
                                             kw),
                }
            except Exception:
                pass
            return None

        def on_llm_end(self, response, *, run_id=None, **kw):
            state = self._open_models.pop(run_id, None)
            if state is None or not model_capture_active():
                return None
            try:
                record_model_boundary(_chat_operation(
                    state, end=now_unix_nano(), response=response))
            except Exception:
                pass
            return None

        def on_llm_error(self, error, *, run_id=None, **kw):
            state = self._open_models.pop(run_id, None)
            if state is None or not model_capture_active():
                return None
            try:
                record_model_boundary(_chat_operation(
                    state, end=now_unix_nano(),
                    error_type=type(error).__name__))
            except Exception:
                pass
            return None

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
                # Wall clock for the model boundary's execute_tool span, and
                # the model's own call id, which LangChain hands the tool.
                "start": now_unix_nano(),
                "tool_call_id": kw.get("tool_call_id"),
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
            # The model-assigned call id rides on the ToolMessage itself, and
            # the SAME id appears on the role="tool" message of the agent's
            # next request -- the join key that lets the control plane say
            # which completion this call followed. A tool that returns a plain
            # value has no id; None is the honest answer, not a reason to
            # invent one.
            _tcid = getattr(output, "tool_call_id", None)
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
                tool_call_id=str(_tcid) if _tcid else None,
            )
            if model_capture_active():
                # What the tool handed BACK to the model: the ToolMessage's
                # content, the same value the next model call is sent.
                _record_execution(state, _tcid, result=_safe(output))

        def on_tool_error(self, error, *, run_id=None, **kw):
            state = self._open.pop(run_id, None)
            if state is None or _ACTIVE.get() is None:
                return
            record_tool_call(
                state["name"], state["arguments"],
                outcome="failed", status_code=None,
                agent_name=state.get("agent_name"),
                error=bounded(f"{type(error).__name__}: {error}"),
                latency_ms=(time.perf_counter() - state["t0"]) * 1000.0,
            )
            if model_capture_active():
                _record_execution(state, None,
                                  error_type=type(error).__name__)

    _handler_cls = _PercepteyeHandler
    return _handler_cls


def _record_execution(state: dict[str, Any], tool_call_id: Any, *,
                      result: Any = None, error_type: str | None = None) -> None:
    """One tool execution on the turn's model boundary. Never raises."""
    try:
        call_id = tool_call_id or state.get("tool_call_id")
        op: dict[str, Any] = {
            "operation": "execute_tool",
            "span_id": new_span_id(),
            "start_time_unix_nano": state["start"],
            "end_time_unix_nano": now_unix_nano(),
            "tool_name": state["name"],
            "tool_type": "function",
        }
        if call_id:
            op["tool_call_id"] = str(call_id)
        op["arguments"] = state["arguments"]
        if error_type:
            op["error_type"] = error_type
        else:
            op["result"] = result
        record_model_boundary(op)
    except Exception:
        return


# ── LangChain's messages, as the GenAI semantic conventions spell them ─────
#
# A TRANSLATION, never an addition: every part below is a part LangChain was
# about to send. What LangChain holds that the provider would not see (ids,
# response metadata) is left out; what it would see and the conventions have
# no part for goes through as a generic part, typed as LangChain typed it.

#: LangChain's `ls_provider` spellings mapped to the conventions' well-known
#: `gen_ai.provider.name` values. Anything else is passed through verbatim:
#: the attribute is an open enum, and a guess would misname the provider.
_PROVIDERS = {
    "openai": "openai",
    "azure": "azure.ai.openai",
    "azure_openai": "azure.ai.openai",
    "azure_ai": "azure.ai.inference",
    "anthropic": "anthropic",
    "google_genai": "gcp.gen_ai",
    "google_vertexai": "gcp.vertex_ai",
    "google_anthropic_vertex": "gcp.vertex_ai",
    "amazon_bedrock": "aws.bedrock",
    "bedrock": "aws.bedrock",
    "bedrock_converse": "aws.bedrock",
    "groq": "groq",
    "mistralai": "mistral_ai",
    "deepseek": "deepseek",
    "xai": "x_ai",
    "cohere": "cohere",
    "perplexity": "perplexity",
    "ibm": "ibm.watsonx.ai",
}

#: Native finish reasons, as the output message's `finish_reason` spells them.
_FINISH = {
    "stop": "stop", "end_turn": "stop", "stop_sequence": "stop",
    "tool_calls": "tool_call", "tool_use": "tool_call",
    "function_call": "tool_call",
    "length": "length", "max_tokens": "length",
    "content_filter": "content_filter",
}


def _parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "content": content}] if content else []
    parts: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if isinstance(block, str):
            parts.append({"type": "text", "content": block})
            continue
        if not isinstance(block, dict):
            continue
        kind = block.get("type")
        if kind == "text":
            parts.append({"type": "text", "content": block.get("text", "")})
        elif kind in ("thinking", "reasoning"):
            text = block.get("thinking", block.get("reasoning"))
            if isinstance(text, str):
                parts.append({"type": "reasoning", "content": text})
        elif kind in ("tool_use", "tool_call", "function_call"):
            # Anthropic-style content blocks duplicate `message.tool_calls`,
            # which is what is read below; counting both would double a call.
            continue
        elif kind == "image_url":
            url = (block.get("image_url") or {}).get("url") \
                if isinstance(block.get("image_url"), dict) \
                else block.get("image_url")
            if isinstance(url, str) and url.startswith("data:") and ";base64," in url:
                head, _, data = url.partition(";base64,")
                parts.append({"type": "blob", "modality": "image",
                              "mime_type": head[len("data:"):] or None,
                              "content": data})
            elif isinstance(url, str):
                parts.append({"type": "uri", "modality": "image", "uri": url})
        elif isinstance(kind, str):
            parts.append(dict(block))
    return parts


def _message(m: Any) -> dict[str, Any]:
    role = getattr(m, "type", None)
    if role == "human":
        return {"role": "user", "parts": _parts(m.content)}
    if role == "ai":
        parts = _parts(m.content)
        for call in getattr(m, "tool_calls", None) or []:
            parts.append({"type": "tool_call", "id": call.get("id"),
                          "name": call.get("name"),
                          "arguments": call.get("args")})
        for call in getattr(m, "invalid_tool_calls", None) or []:
            parts.append({"type": "tool_call", "id": call.get("id"),
                          "name": call.get("name"),
                          "arguments": call.get("args")})
        return {"role": "assistant", "parts": parts}
    if role == "tool":
        return {"role": "tool", "parts": [{
            "type": "tool_call_response",
            "id": getattr(m, "tool_call_id", None),
            "response": m.content,
        }]}
    if role == "system":
        return {"role": "system", "parts": _parts(m.content)}
    return {"role": str(getattr(m, "role", role) or "user"),
            "parts": _parts(getattr(m, "content", ""))}


def _tools(tools: Any) -> list[dict[str, Any]] | None:
    """The tool definitions the call was made with, or None when it carried
    none that can be read -- never an empty list standing in for "unknown"."""
    if not isinstance(tools, list) or not tools:
        return None
    out: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if tool.get("type") == "function" else None
        if isinstance(fn, dict) and fn.get("name"):
            flat: dict[str, Any] = {"type": "function", "name": fn["name"]}
            if "description" in fn:
                flat["description"] = fn["description"]
            if "parameters" in fn:
                flat["parameters"] = fn["parameters"]
            out.append(flat)
        elif tool.get("name") and "input_schema" in tool:
            flat = {"type": "function", "name": tool["name"]}
            if "description" in tool:
                flat["description"] = tool["description"]
            flat["parameters"] = tool["input_schema"]
            out.append(flat)
        else:
            out.append(dict(tool))
    return out or None


def _chat_request(messages: list[Any], kw: dict[str, Any]) -> dict[str, Any]:
    """The request as it was sent: leading system messages as the
    conventions' system instructions, everything else as the input."""
    params = kw.get("invocation_params") or {}
    meta = kw.get("metadata") or {}
    system: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for m in messages:
        if getattr(m, "type", None) == "system" and not inputs:
            system.extend(_parts(m.content))
        else:
            inputs.append(_message(m))
    provider = meta.get("ls_provider")
    return {
        "provider_name": _PROVIDERS.get(provider, provider)
        if isinstance(provider, str) else None,
        "request_model": params.get("model") or params.get("model_name")
        or meta.get("ls_model_name"),
        "system_instructions": system,
        "input_messages": inputs,
        "tool_definitions": _tools(params.get("tools")),
    }


def _chat_operation(state: dict[str, Any], *, end: str, response: Any = None,
                    error_type: str | None = None) -> dict[str, Any]:
    request = state["request"]
    op: dict[str, Any] = {
        "operation": "chat",
        "span_id": new_span_id(),
        "start_time_unix_nano": state["start"],
        "end_time_unix_nano": end,
        "provider_name": request["provider_name"],
    }
    if request["request_model"]:
        op["request_model"] = str(request["request_model"])
    if request["system_instructions"]:
        op["system_instructions"] = request["system_instructions"]
    op["input_messages"] = request["input_messages"]
    if request["tool_definitions"]:
        op["tool_definitions"] = request["tool_definitions"]
    if error_type:
        op["error_type"] = error_type
        return op
    generation = response.generations[0][0]
    message = getattr(generation, "message", None)
    if message is None:
        return op
    meta = getattr(message, "response_metadata", None) or {}
    info = getattr(generation, "generation_info", None) or {}
    native = meta.get("finish_reason") or meta.get("stop_reason") \
        or info.get("finish_reason")
    output = _message(message)
    calls = any(p.get("type") == "tool_call" for p in output["parts"])
    output["finish_reason"] = "tool_call" if calls else _FINISH.get(
        native, native if isinstance(native, str) else "stop")
    op["output_messages"] = [output]
    if isinstance(native, str) and native:
        op["finish_reasons"] = [native]
    model = meta.get("model_name") or meta.get("model")
    if isinstance(model, str) and model:
        op["response_model"] = model
    return op


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
            return "failed", bounded(str(getattr(output, "content", ""))) or None
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
