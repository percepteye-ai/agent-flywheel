"""One-line model-client wrappers: ``wrap_openai``, ``wrap_anthropic``,
``wrap_gemini``.

For an agent that calls a provider's client directly, with no framework
adapter in between::

    from openai import OpenAI
    from agent_flywheel import attach, wrap_openai

    client = wrap_openai(OpenAI())
    fw = attach(agent_id="support-agent")
    with fw.turn(user_message):
        client.chat.completions.create(model=..., messages=...)

Inside a production turn whose agent may upload model content
(:func:`agent_flywheel.outcomes.model_capture_active`), every call records
ONE ``chat`` operation on the turn's model boundary: what it was sent (the
system instructions, the messages, the tool definitions, the output format)
and what it answered (text, reasoning, the tool calls it asked for), as the
framework adapters record theirs. Outside one, or without that consent, a
call pays one context-variable read and records nothing.

OBSERVE ONLY. The request is passed on as given and the response returned as
received; a streamed response is read on its way through to the caller, and
a capture bug costs the observation, never the call. The client is patched
in place (its ``create`` / ``generate_content`` methods) and returned, so
wrapping twice is wrapping once.

What is wrapped:
  OpenAI     ``chat.completions.create`` and ``responses.create``, sync and
             async, streamed or not.
  Anthropic  ``messages.create``, sync and async, streamed or not, and the
             ``messages.stream(...)`` helper.
  Gemini     ``models.generate_content`` and ``models.generate_content_stream``
             (google-genai), and their ``aio`` twins.

The Node SDK's ``wrapOpenAI`` / ``wrapAnthropic`` / ``wrapGemini``
(``src/wrappers.js``) are this module in JavaScript; the shared
``model_wrappers.json`` fixture holds both to the same operations.
"""
from __future__ import annotations

import base64
import enum
import functools
import inspect
import json
from collections.abc import Mapping
from typing import Any, Callable

from .adapters._media import data_url_part, media_part
from .genai_otlp import new_span_id, now_unix_nano, requested_output_format
from .outcomes import model_capture_active, record_model_boundary

__all__ = [
    "chat_operation",
    "request_of",
    "response_of",
    "stream_response_of",
    "wrap_anthropic",
    "wrap_gemini",
    "wrap_openai",
]

_MARK = "_percepteye_wrapped"

#: Native finish reasons as the output message's ``finish_reason`` (the table
#: both SDKs' adapters use), matched as given and then lower-cased.
_FINISH = {
    "stop": "stop", "end_turn": "stop", "stop_sequence": "stop",
    "tool_calls": "tool_call", "tool_use": "tool_call",
    "function_call": "tool_call",
    "length": "length", "max_tokens": "length",
    "content_filter": "content_filter", "safety": "content_filter",
}

_PROVIDERS = {"openai_chat": "openai", "openai_responses": "openai",
              "anthropic": "anthropic", "gemini": "gcp.gemini"}


# ── plain values ──────────────────────────────────────────────────────────

def _plain(value: Any) -> Any:
    """A provider SDK's object as plain JSON-shaped data: pydantic models
    dumped, enums by value, bytes as base64 text."""
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            value = value.model_dump(exclude_none=True)
        except Exception:
            return None
    if isinstance(value, enum.Enum):
        return _plain(value.value)
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items() if v is not None}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, (bytes, bytearray, memoryview)):
        return base64.b64encode(bytes(value)).decode("ascii")
    return value


def _get(obj: Any, *keys: str) -> Any:
    """The first of ``keys`` a mapping holds (snake_case and camelCase)."""
    if not isinstance(obj, Mapping):
        return None
    for key in keys:
        if obj.get(key) is not None:
            return obj[key]
    return None


def _args(value: Any) -> Any:
    """A tool call's arguments as the model sent them: parsed when JSON."""
    if not isinstance(value, str):
        return {} if value is None else value
    try:
        return json.loads(value)
    except ValueError:
        return value


def _text(content: str) -> list[dict[str, Any]]:
    return [{"type": "text", "content": content}] if content else []


def _flat_tool(tool: Any) -> dict[str, Any] | None:
    """A tool definition as the conventions flatten it."""
    if not isinstance(tool, Mapping):
        return None
    fn = tool.get("function") if tool.get("type") == "function" else None
    src = fn if isinstance(fn, Mapping) else tool
    if not isinstance(src.get("name"), str) or not src["name"]:
        return dict(tool)
    flat: dict[str, Any] = {"type": "function", "name": src["name"]}
    if "description" in src:
        flat["description"] = src["description"]
    schema = _get(src, "parameters", "input_schema", "inputSchema",
                  "parameters_json_schema", "parametersJsonSchema")
    if schema is not None:
        flat["parameters"] = schema
    return flat


def _tools(tools: Any) -> list[dict[str, Any]] | None:
    if not isinstance(tools, list) or not tools:
        return None
    out = [t for t in (_flat_tool(x) for x in tools) if t is not None]
    return out or None


# ── OpenAI Chat Completions ───────────────────────────────────────────────

_AUDIO = {"mp3": "audio/mpeg", "wav": "audio/wav"}


def _openai_parts(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return _text(content)
    parts: list[dict[str, Any]] = []
    for block in content if isinstance(content, list) else []:
        if not isinstance(block, Mapping):
            continue
        kind = block.get("type")
        if kind in ("text", "input_text", "output_text"):
            parts.append({"type": "text", "content": block.get("text", "")})
        elif kind == "refusal":
            parts.append({"type": "text", "content": block.get("refusal", "")})
        elif kind in ("image_url", "input_image"):
            url = block.get("image_url")
            url = url.get("url") if isinstance(url, Mapping) else url
            part = data_url_part(url, modality="image")
            if part is not None:
                parts.append(part)
        elif kind == "input_audio" and isinstance(block.get("input_audio"), Mapping):
            body = block["input_audio"]
            fmt = body.get("format")
            mime = _AUDIO.get(fmt, f"audio/{fmt}") if isinstance(fmt, str) and fmt else None
            parts.append(media_part(mime_type=mime, data=body.get("data"), modality="audio"))
        elif isinstance(kind, str):
            parts.append(dict(block))
    return parts


def _openai_message(m: Mapping[str, Any]) -> dict[str, Any]:
    role = m.get("role") or "user"
    if role == "tool":
        return {"role": "tool", "parts": [{
            "type": "tool_call_response", "id": m.get("tool_call_id"),
            "response": m.get("content")}]}
    parts = _openai_parts(m.get("content"))
    if role == "assistant":
        for call in m.get("tool_calls") or []:
            fn = call.get("function") or {}
            parts.append({"type": "tool_call", "id": call.get("id"),
                          "name": fn.get("name"),
                          "arguments": _args(fn.get("arguments"))})
    return {"role": role, "parts": parts}


def _openai_chat_request(req: Mapping[str, Any]) -> dict[str, Any]:
    system: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for m in req.get("messages") or []:
        if not isinstance(m, Mapping):
            continue
        if m.get("role") in ("system", "developer") and not inputs:
            system.extend(_openai_parts(m.get("content")))
        else:
            inputs.append(_openai_message(m))
    return {"system": system, "input": inputs}


def _openai_chat_response(resp: Mapping[str, Any]) -> dict[str, Any]:
    choice = (resp.get("choices") or [{}])[0]
    message = dict(choice.get("message") or {})
    message["role"] = "assistant"
    return {"output": _openai_message(message),
            "finish": choice.get("finish_reason"), "model": resp.get("model")}


def _openai_chat_stream(chunks: list[Any]) -> dict[str, Any]:
    text = ""
    calls: dict[int, dict[str, Any]] = {}
    finish = model = None
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        model = chunk.get("model") or model
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if isinstance(delta.get("content"), str):
                text += delta["content"]
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0),
                                        {"id": None, "name": None, "arguments": ""})
                fn = call.get("function") or {}
                slot["id"] = call.get("id") or slot["id"]
                slot["name"] = fn.get("name") or slot["name"]
                slot["arguments"] += fn.get("arguments") or ""
            finish = choice.get("finish_reason") or finish
    parts = _text(text)
    for index in sorted(calls):
        call = calls[index]
        parts.append({"type": "tool_call", "id": call["id"], "name": call["name"],
                      "arguments": _args(call["arguments"])})
    return {"output": {"role": "assistant", "parts": parts},
            "finish": finish, "model": model}


# ── OpenAI Responses ──────────────────────────────────────────────────────

def _openai_responses_request(req: Mapping[str, Any]) -> dict[str, Any]:
    system = _text(req["instructions"]) if isinstance(req.get("instructions"), str) else []
    inputs: list[dict[str, Any]] = []
    items = req.get("input")
    if isinstance(items, str):
        items = [{"role": "user", "content": items}]
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, Mapping):
            continue
        kind = item.get("type")
        if kind == "function_call":
            inputs.append({"role": "assistant", "parts": [{
                "type": "tool_call", "id": item.get("call_id"), "name": item.get("name"),
                "arguments": _args(item.get("arguments"))}]})
        elif kind == "function_call_output":
            inputs.append({"role": "tool", "parts": [{
                "type": "tool_call_response", "id": item.get("call_id"),
                "response": item.get("output")}]})
        elif item.get("role"):
            if item["role"] in ("system", "developer") and not inputs:
                system.extend(_openai_parts(item.get("content")))
            else:
                inputs.append({"role": item["role"],
                               "parts": _openai_parts(item.get("content"))})
    return {"system": system, "input": inputs}


def _openai_responses_response(resp: Mapping[str, Any]) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    for item in resp.get("output") or []:
        kind = item.get("type") if isinstance(item, Mapping) else None
        if kind == "message":
            parts.extend(_openai_parts(item.get("content")))
        elif kind == "function_call":
            parts.append({"type": "tool_call", "id": item.get("call_id"),
                          "name": item.get("name"),
                          "arguments": _args(item.get("arguments"))})
        elif kind == "reasoning":
            for s in item.get("summary") or []:
                if isinstance(s, Mapping) and isinstance(s.get("text"), str):
                    parts.append({"type": "reasoning", "content": s["text"]})
    details = resp.get("incomplete_details")
    finish = details.get("reason") if isinstance(details, Mapping) else None
    return {"output": {"role": "assistant", "parts": parts},
            "finish": finish, "model": resp.get("model")}


def _openai_responses_stream(events: list[Any]) -> dict[str, Any] | None:
    for event in reversed(events):
        if isinstance(event, Mapping) and event.get("type") in (
                "response.completed", "response.incomplete") \
                and isinstance(event.get("response"), Mapping):
            return _openai_responses_response(event["response"])
    return None


# ── Anthropic Messages ────────────────────────────────────────────────────

def _anthropic_block(block: Any) -> dict[str, Any] | None:
    if isinstance(block, str):
        return {"type": "text", "content": block}
    if not isinstance(block, Mapping):
        return None
    kind = block.get("type")
    if kind == "text":
        return {"type": "text", "content": block.get("text", "")}
    if kind == "thinking":
        return {"type": "reasoning", "content": block.get("thinking", "")}
    if kind == "redacted_thinking":
        return None
    if kind == "tool_use":
        return {"type": "tool_call", "id": block.get("id"), "name": block.get("name"),
                "arguments": block.get("input") if block.get("input") is not None else {}}
    if kind in ("image", "document") and isinstance(block.get("source"), Mapping):
        source = block["source"]
        modality = "image" if kind == "image" else None
        if source.get("type") == "base64":
            return media_part(mime_type=source.get("media_type"), data=source.get("data"),
                              modality=modality)
        if source.get("type") == "url":
            return data_url_part(source.get("url"), modality=modality,
                                 mime_type=source.get("media_type"))
    return dict(block) if isinstance(kind, str) else None


def _anthropic_request(req: Mapping[str, Any]) -> dict[str, Any]:
    raw = req.get("system")
    system = _text(raw) if isinstance(raw, str) else [
        p for p in (_anthropic_block(b) for b in raw or []) if p is not None]
    inputs: list[dict[str, Any]] = []
    for m in req.get("messages") or []:
        if not isinstance(m, Mapping):
            continue
        content = m.get("content")
        if isinstance(content, str):
            inputs.append({"role": m.get("role"), "parts": _text(content)})
            continue
        parts: list[dict[str, Any]] = []
        for block in content or []:
            if isinstance(block, Mapping) and block.get("type") == "tool_result":
                inputs.append({"role": "tool", "parts": [{
                    "type": "tool_call_response", "id": block.get("tool_use_id"),
                    "response": block.get("content")}]})
                continue
            part = _anthropic_block(block)
            if part is not None:
                parts.append(part)
        if parts or not content:
            inputs.append({"role": m.get("role"), "parts": parts})
    return {"system": system, "input": inputs}


def _anthropic_response(resp: Mapping[str, Any]) -> dict[str, Any]:
    parts = [p for p in (_anthropic_block(b) for b in resp.get("content") or [])
             if p is not None]
    return {"output": {"role": "assistant", "parts": parts},
            "finish": resp.get("stop_reason"), "model": resp.get("model")}


def _anthropic_stream(events: list[Any]) -> dict[str, Any]:
    blocks: dict[int, dict[str, Any]] = {}
    finish = model = None
    for event in events:
        if not isinstance(event, Mapping):
            continue
        kind = event.get("type")
        if kind == "message_start":
            model = (event.get("message") or {}).get("model") or model
        elif kind == "content_block_start":
            block = dict(event.get("content_block") or {})
            if block.get("type") == "tool_use":
                block["_json"] = ""
            blocks[event.get("index", len(blocks))] = block
        elif kind == "content_block_delta":
            block = blocks.get(event.get("index", 0))
            delta = event.get("delta") or {}
            if block is None:
                continue
            if delta.get("type") == "text_delta":
                block["text"] = block.get("text", "") + delta.get("text", "")
            elif delta.get("type") == "thinking_delta":
                block["thinking"] = block.get("thinking", "") + delta.get("thinking", "")
            elif delta.get("type") == "input_json_delta":
                block["_json"] = block.get("_json", "") + delta.get("partial_json", "")
        elif kind == "message_delta":
            finish = (event.get("delta") or {}).get("stop_reason") or finish
    content = []
    for index in sorted(blocks):
        block = blocks[index]
        if "_json" in block:
            raw = block.pop("_json")
            if raw:
                block["input"] = _args(raw)
        content.append(block)
    return _anthropic_response({"content": content, "stop_reason": finish, "model": model})


# ── Gemini (google-genai) ─────────────────────────────────────────────────

def _gemini_part(part: Any) -> tuple[str, dict[str, Any]] | None:
    """One Gemini part, with the role it belongs under (``tool`` for a
    function response)."""
    if isinstance(part, str):
        return "", {"type": "text", "content": part}
    if not isinstance(part, Mapping):
        return None
    if isinstance(part.get("text"), str):
        kind = "reasoning" if part.get("thought") else "text"
        return "", {"type": kind, "content": part["text"]}
    call = _get(part, "function_call", "functionCall")
    if isinstance(call, Mapping):
        return "", {"type": "tool_call", "id": call.get("id"), "name": call.get("name"),
                    "arguments": call.get("args") if call.get("args") is not None else {}}
    response = _get(part, "function_response", "functionResponse")
    if isinstance(response, Mapping):
        return "tool", {"type": "tool_call_response", "id": response.get("id"),
                        "response": response.get("response")}
    inline = _get(part, "inline_data", "inlineData")
    if isinstance(inline, Mapping):
        return "", media_part(mime_type=_get(inline, "mime_type", "mimeType"),
                              data=inline.get("data"))
    file = _get(part, "file_data", "fileData")
    if isinstance(file, Mapping):
        return "", media_part(mime_type=_get(file, "mime_type", "mimeType"),
                              uri=_get(file, "file_uri", "fileUri"))
    return None


def _gemini_content(content: Any) -> list[dict[str, Any]]:
    """One Gemini ``Content`` as messages: its function responses as a tool
    message, the rest under its role (``model`` is the assistant)."""
    if isinstance(content, str):
        return [{"role": "user", "parts": _text(content)}]
    if not isinstance(content, Mapping):
        return []
    role = "assistant" if content.get("role") == "model" else (content.get("role") or "user")
    own: list[dict[str, Any]] = []
    tool: list[dict[str, Any]] = []
    for part in content.get("parts") or []:
        seen = _gemini_part(part)
        if seen is not None:
            (tool if seen[0] == "tool" else own).append(seen[1])
    out = []
    if tool:
        out.append({"role": "tool", "parts": tool})
    if own or not tool:
        out.append({"role": role, "parts": own})
    return out


def _gemini_request(req: Mapping[str, Any]) -> dict[str, Any]:
    config = req.get("config") if isinstance(req.get("config"), Mapping) else {}
    raw = _get(config, "system_instruction", "systemInstruction")
    if isinstance(raw, str):
        system = _text(raw)
    elif isinstance(raw, Mapping):
        system = [p for _r, p in filter(None, map(_gemini_part, raw.get("parts") or []))]
    else:
        system = [p for _r, p in filter(None, map(_gemini_part, raw or []))]
    contents = req.get("contents")
    if isinstance(contents, (str, Mapping)):
        contents = [contents]
    inputs: list[dict[str, Any]] = []
    loose: list[dict[str, Any]] = []
    for item in contents if isinstance(contents, list) else []:
        if isinstance(item, Mapping) and "parts" in item:
            if loose:
                inputs.append({"role": "user", "parts": loose})
                loose = []
            inputs.extend(_gemini_content(item))
        else:
            seen = _gemini_part(item)
            if seen is not None:
                loose.append(seen[1])
    if loose:
        inputs.append({"role": "user", "parts": loose})
    tools: list[Any] = []
    for tool in _get(config, "tools") or []:
        decls = _get(tool, "function_declarations", "functionDeclarations") \
            if isinstance(tool, Mapping) else None
        if isinstance(decls, list):
            tools.extend(decls)
        else:
            tools.append(tool)
    return {"system": system, "input": inputs, "tools": tools, "format_source": config}


def _gemini_response(resp: Mapping[str, Any]) -> dict[str, Any]:
    candidate = (resp.get("candidates") or [{}])[0]
    content = candidate.get("content") if isinstance(candidate.get("content"), Mapping) else {}
    parts = [p for _r, p in filter(None, map(_gemini_part, content.get("parts") or []))]
    return {"output": {"role": "assistant", "parts": parts},
            "finish": _get(candidate, "finish_reason", "finishReason"),
            "model": _get(resp, "model_version", "modelVersion")}


def _gemini_stream(chunks: list[Any]) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    finish = model = None
    for chunk in chunks:
        if not isinstance(chunk, Mapping):
            continue
        one = _gemini_response(chunk)
        for part in one["output"]["parts"]:
            last = parts[-1] if parts else None
            if last and part["type"] in ("text", "reasoning") and last["type"] == part["type"]:
                last["content"] += part["content"]
            else:
                parts.append(dict(part))
        finish = one["finish"] or finish
        model = one["model"] or model
    return {"output": {"role": "assistant", "parts": parts}, "finish": finish, "model": model}


# ── the shared shape: request, response, operation ────────────────────────

_REQUESTS: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "openai_chat": _openai_chat_request,
    "openai_responses": _openai_responses_request,
    "anthropic": _anthropic_request,
    "gemini": _gemini_request,
}
_RESPONSES: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
    "openai_chat": _openai_chat_response,
    "openai_responses": _openai_responses_response,
    "anthropic": _anthropic_response,
    "gemini": _gemini_response,
}
_STREAMS: dict[str, Callable[[list[Any]], dict[str, Any] | None]] = {
    "openai_chat": _openai_chat_stream,
    "openai_responses": _openai_responses_stream,
    "anthropic": _anthropic_stream,
    "gemini": _gemini_stream,
}


def request_of(api: str, request: Any) -> dict[str, Any]:
    """The request as it was SENT: provider, requested model, system
    instructions, input messages, tool definitions and output format."""
    req = _plain(request) or {}
    seen = _REQUESTS[api](req)
    tools = seen.get("tools", req.get("tools"))
    return {
        "provider_name": _PROVIDERS[api],
        "request_model": req.get("model") if isinstance(req.get("model"), str) else None,
        "system_instructions": seen["system"],
        "input_messages": seen["input"],
        "tool_definitions": _tools(tools),
        "response_format": requested_output_format(seen.get("format_source", req)),
    }


def response_of(api: str, response: Any) -> dict[str, Any] | None:
    """What a call answered: ``{"output", "finish", "model"}``."""
    resp = _plain(response)
    return _RESPONSES[api](resp) if isinstance(resp, Mapping) else None


def stream_response_of(api: str, chunks: list[Any]) -> dict[str, Any] | None:
    """What a streamed call answered, from its chunks or events in order."""
    return _STREAMS[api]([_plain(c) for c in chunks])


def chat_operation(request: Mapping[str, Any], *, start: str, end: str,
                   answer: Mapping[str, Any] | None = None,
                   error_type: str | None = None) -> dict[str, Any]:
    """One model call as the model boundary's ``chat`` operation, keys in
    the order the adapters write them."""
    op: dict[str, Any] = {
        "operation": "chat",
        "span_id": new_span_id(),
        "start_time_unix_nano": start,
        "end_time_unix_nano": end,
        "provider_name": request["provider_name"],
    }
    if request.get("request_model"):
        op["request_model"] = request["request_model"]
    if request.get("system_instructions"):
        op["system_instructions"] = request["system_instructions"]
    op["input_messages"] = request["input_messages"]
    if request.get("tool_definitions"):
        op["tool_definitions"] = request["tool_definitions"]
    if request.get("response_format"):
        op["response_format"] = request["response_format"]
    if error_type:
        op["error_type"] = error_type
        return op
    if not answer:
        return op
    output = dict(answer["output"])
    native = answer.get("finish") if isinstance(answer.get("finish"), str) \
        and answer.get("finish") else None
    calls = any(p.get("type") == "tool_call" for p in output["parts"])
    output["finish_reason"] = "tool_call" if calls else (
        _FINISH.get(native) or _FINISH.get(native.lower()) or native
        if native else "stop")
    op["output_messages"] = [output]
    if native:
        op["finish_reasons"] = [native]
    if isinstance(answer.get("model"), str) and answer["model"]:
        op["response_model"] = answer["model"]
    return op


# ── the wrappers ──────────────────────────────────────────────────────────

def _record(api: str, request: dict[str, Any], start: str, *,
            answer: Any = None, error: BaseException | None = None) -> None:
    """Record one call. Never raises."""
    try:
        record_model_boundary(chat_operation(
            request, start=start, end=now_unix_nano(), answer=answer,
            error_type=type(error).__name__ if error is not None else None))
    except Exception:
        return


class _Tap:
    """A streamed response, read on its way through to the caller. Iterates
    exactly what the wrapped stream yields and delegates everything else to
    it; records once, when the stream ends or is closed."""

    def __init__(self, inner: Any, done: Callable[[list[Any], BaseException | None], None]):
        self._inner = inner
        self._done = done
        self._seen: list[Any] = []
        self._closed = False

    def _finish(self, error: BaseException | None = None) -> None:
        if not self._closed:
            self._closed = True
            self._done(self._seen, error)

    def __iter__(self):
        try:
            for item in self._inner:
                self._seen.append(item)
                yield item
        except GeneratorExit:
            # The caller stopped reading: what it read is what was answered.
            self._finish()
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        self._finish()

    async def __aiter__(self):
        try:
            async for item in self._inner:
                self._seen.append(item)
                yield item
        except GeneratorExit:
            # The caller stopped reading: what it read is what was answered.
            self._finish()
            raise
        except BaseException as exc:
            self._finish(exc)
            raise
        self._finish()

    def __enter__(self):
        if hasattr(self._inner, "__enter__"):
            self._inner.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._inner.__exit__(*exc) if hasattr(self._inner, "__exit__") else None
        finally:
            self._finish()

    async def __aenter__(self):
        if hasattr(self._inner, "__aenter__"):
            await self._inner.__aenter__()
        return self

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._inner.__aexit__(*exc) \
                if hasattr(self._inner, "__aexit__") else None
        finally:
            self._finish()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _streamed(api: str, kwargs: Mapping[str, Any], always: bool) -> bool:
    return always or kwargs.get("stream") is True


def _patch(owner: Any, name: str, api: str, *, always_stream: bool = False) -> None:
    """Replace ``owner.name`` with its observing twin, in place. Idempotent."""
    original = getattr(owner, name, None)
    if original is None or getattr(original, _MARK, False):
        return

    def begin(kwargs: Mapping[str, Any]) -> tuple[dict[str, Any], str] | None:
        if not model_capture_active():
            return None
        try:
            return request_of(api, kwargs), now_unix_nano()
        except Exception:
            return None

    def tap(result: Any, request: dict[str, Any], start: str) -> Any:
        def done(seen: list[Any], error: BaseException | None) -> None:
            if error is not None:
                _record(api, request, start, error=error)
                return
            try:
                answer = stream_response_of(api, seen)
            except Exception:
                answer = None
            _record(api, request, start, answer=answer)
        return _Tap(result, done)

    def answered(result: Any, request: dict[str, Any], start: str) -> None:
        try:
            answer = response_of(api, result)
        except Exception:
            answer = None
        _record(api, request, start, answer=answer)

    if inspect.iscoroutinefunction(original):
        @functools.wraps(original)
        async def observed_async(*args: Any, **kwargs: Any) -> Any:
            began = begin(kwargs)
            if began is None:
                return await original(*args, **kwargs)
            try:
                result = await original(*args, **kwargs)
            except BaseException as exc:
                _record(api, *began, error=exc)
                raise
            if _streamed(api, kwargs, always_stream):
                return tap(result, *began)
            answered(result, *began)
            return result
        wrapped: Any = observed_async
    else:
        @functools.wraps(original)
        def observed(*args: Any, **kwargs: Any) -> Any:
            began = begin(kwargs)
            if began is None:
                return original(*args, **kwargs)
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                _record(api, *began, error=exc)
                raise
            if _streamed(api, kwargs, always_stream):
                return tap(result, *began)
            answered(result, *began)
            return result
        wrapped = observed
    setattr(wrapped, _MARK, True)
    setattr(owner, name, wrapped)


def _resource(client: Any, *path: str) -> Any:
    for name in path:
        try:
            client = getattr(client, name)
        except Exception:
            return None
    return client


def wrap_openai(client: Any) -> Any:
    """Observe an ``openai.OpenAI`` / ``AsyncOpenAI`` client's Chat
    Completions and Responses calls. Returns the same client."""
    chat = _resource(client, "chat", "completions")
    if chat is not None:
        _patch(chat, "create", "openai_chat")
    responses = _resource(client, "responses")
    if responses is not None:
        _patch(responses, "create", "openai_responses")
    return client


class _ObservedStreamManager:
    """``messages.stream(...)``'s context manager, sync or async: the stream
    it opens is the caller's, untouched; on exit, the message it assembled
    (``current_message_snapshot``) is recorded. Everything else delegates."""

    def __init__(self, inner: Any, request: dict[str, Any], start: str):
        self._inner = inner
        self._request = request
        self._start = start
        self._stream: Any = None

    def _record(self, error: BaseException | None) -> None:
        if error is not None:
            _record("anthropic", self._request, self._start, error=error)
            return
        try:
            answer = response_of(
                "anthropic", getattr(self._stream, "current_message_snapshot", None))
        except Exception:
            answer = None
        _record("anthropic", self._request, self._start, answer=answer)

    def __enter__(self) -> Any:
        self._stream = self._inner.__enter__()
        return self._stream

    def __exit__(self, *exc: Any) -> Any:
        try:
            return self._inner.__exit__(*exc)
        finally:
            self._record(exc[1])

    async def __aenter__(self) -> Any:
        self._stream = await self._inner.__aenter__()
        return self._stream

    async def __aexit__(self, *exc: Any) -> Any:
        try:
            return await self._inner.__aexit__(*exc)
        finally:
            self._record(exc[1])

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def _patch_stream_helper(messages: Any) -> None:
    """Anthropic's ``messages.stream(...)`` helper posts on its own, not
    through ``create``: observed by its manager. Idempotent."""
    original = getattr(messages, "stream", None)
    if original is None or getattr(original, _MARK, False):
        return

    @functools.wraps(original)
    def observed(*args: Any, **kwargs: Any) -> Any:
        manager = original(*args, **kwargs)
        if not model_capture_active():
            return manager
        try:
            return _ObservedStreamManager(
                manager, request_of("anthropic", kwargs), now_unix_nano())
        except Exception:
            return manager

    setattr(observed, _MARK, True)
    setattr(messages, "stream", observed)


def wrap_anthropic(client: Any) -> Any:
    """Observe an ``anthropic.Anthropic`` / ``AsyncAnthropic`` client's
    Messages calls -- ``create``, streamed or not, and the
    ``messages.stream(...)`` helper. Returns the same client."""
    messages = _resource(client, "messages")
    if messages is not None:
        _patch(messages, "create", "anthropic")
        _patch_stream_helper(messages)
    return client


def wrap_gemini(client: Any) -> Any:
    """Observe a ``google.genai.Client``'s content generation, sync and
    ``aio``. Returns the same client."""
    for models in (_resource(client, "models"), _resource(client, "aio", "models")):
        if models is not None:
            _patch(models, "generate_content", "gemini")
            _patch(models, "generate_content_stream", "gemini", always_stream=True)
    return client
