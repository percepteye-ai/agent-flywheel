"""D: the one-line model-client wrappers (``wrap_openai``, ``wrap_anthropic``,
``wrap_gemini``).

The shared ``model_wrappers.json`` holds both SDKs to the same chat operation
for the same provider request and response. The rest pins OBSERVE ONLY: the
request goes on as given, the response comes back as received (a stream is
read on its way through, every chunk untouched), and nothing is recorded
outside a production turn whose agent may upload model content.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import agent_flywheel as fw
from agent_flywheel.wrappers import (
    chat_operation,
    request_of,
    response_of,
    stream_response_of,
)

FIXTURE = json.loads((Path(__file__).resolve().parent / "fixtures"
                      / "model_wrappers.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda c: c["name"])
def test_the_operation_matches_the_node_sdks(case):
    request = request_of(case["api"], case["request"])
    answer = (stream_response_of(case["api"], case["stream"]) if "stream" in case
              else response_of(case["api"], case["response"]))
    op = chat_operation(request, start="1", end="2", answer=answer)
    op.pop("span_id")
    assert op == case["expected"]
    # Byte for byte: key order is what both SDKs write.
    assert json.dumps(op) == json.dumps(case["expected"])


class _Transport:
    def __init__(self, *, model_content=True):
        self.model_content = model_content
        self.batches: list[list[dict]] = []

    def _capture(self):
        return {"capture": {"turns": True, "model_content": self.model_content}}

    def register(self, spec):
        return self._capture()

    def capture_current(self, agent_id):
        return self._capture()

    def policy_current(self):
        return {}

    def report_turns(self, agent_id, turns):
        self.batches.append(list(turns))
        return {"agent_id": agent_id,
                "turns": [{"turn_id": t["turn_id"], "accepted": True} for t in turns]}


@contextmanager
def _attached(monkeypatch, tmp_path, transport):
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_x")
    monkeypatch.setattr("agent_flywheel.transport.AttachTransport",
                        lambda client, agent_id: transport)
    monkeypatch.setattr("agent_flywheel._http.ControlPlaneClient.close",
                        lambda self: None)
    px = fw.attach(agent_id="acme-support", trajectory_root=str(tmp_path))
    try:
        yield px
    finally:
        px.close()


def _chats(transport):
    [turn] = [t for b in transport.batches for t in b]
    if "genai_trace" not in turn:
        return []
    spans = turn["genai_trace"]["resourceSpans"][0]["scopeSpans"][0]["spans"]
    attrs = [{a["key"]: a["value"].get("stringValue", a["value"])
              for a in s["attributes"]} for s in spans]
    return [a for a in attrs if a["gen_ai.operation.name"] == "chat"]


class _Ns:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _openai(create):
    return _Ns(chat=_Ns(completions=_Ns(create=create)), responses=None)


RESPONSE = {"model": "gpt-4o-mini-2024", "choices": [{"finish_reason": "stop",
            "message": {"role": "assistant", "content": "It shipped."}}]}


def test_an_openai_call_in_a_turn_is_recorded_and_returned_as_is(monkeypatch, tmp_path):
    sent = []

    def create(**kw):
        sent.append(kw)
        return RESPONSE

    client = _openai(create)
    assert fw.wrap_openai(client) is client
    assert fw.wrap_openai(client) is client  # wrapping twice is wrapping once
    tr = _Transport()
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "Where?"}]}
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="t1", input_text="Where?"):
            assert client.chat.completions.create(**body) is RESPONSE
        px.record_answer("t1", "It shipped.")
        px.flush()
    assert sent == [body]
    [chat] = _chats(tr)
    assert chat["gen_ai.provider.name"] == "openai"
    assert json.loads(chat["gen_ai.output.messages"])[0]["parts"] == [
        {"type": "text", "content": "It shipped."}]


def test_an_anthropic_stream_is_read_on_its_way_through(monkeypatch, tmp_path):
    events = [
        {"type": "message_start", "message": {"model": "claude-x"}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
    ]
    client = fw.wrap_anthropic(_Ns(messages=_Ns(create=lambda **kw: iter(events))))
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="t1", input_text="Hi"):
            seen = list(client.messages.create(model="claude", stream=True,
                                               messages=[{"role": "user", "content": "Hi"}]))
        px.record_answer("t1", "Hi")
        px.flush()
    assert seen == events
    [chat] = _chats(tr)
    assert chat["gen_ai.response.model"] == "claude-x"
    assert json.loads(chat["gen_ai.output.messages"])[0]["parts"] == [
        {"type": "text", "content": "Hi"}]


def test_async_gemini_stream_and_a_failed_call(monkeypatch, tmp_path):
    async def chunks():
        yield {"model_version": "g", "candidates": [
            {"content": {"parts": [{"text": "He"}]}}]}
        yield {"candidates": [{"finish_reason": "STOP",
                               "content": {"parts": [{"text": "llo"}]}}]}

    async def stream(**kw):
        return chunks()

    async def fail(**kw):
        raise RuntimeError("quota")

    models = _Ns(generate_content=fail, generate_content_stream=stream)
    client = fw.wrap_gemini(_Ns(models=models, aio=None))
    tr = _Transport()

    async def run():
        text = ""
        async for c in await client.models.generate_content_stream(model="g", contents="Hi"):
            text += c["candidates"][0]["content"]["parts"][0]["text"]
        with pytest.raises(RuntimeError):
            await client.models.generate_content(model="g", contents="Hi")
        return text

    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="t1", input_text="Hi"):
            assert asyncio.run(run()) == "Hello"
        px.record_answer("t1", "Hello")
        px.flush()
    streamed, failed = _chats(tr)
    assert json.loads(streamed["gen_ai.output.messages"])[0]["parts"] == [
        {"type": "text", "content": "Hello"}]
    assert failed["error.type"] == "RuntimeError"


def test_nothing_is_recorded_without_model_content_consent(monkeypatch, tmp_path):
    client = fw.wrap_openai(_openai(lambda **kw: RESPONSE))
    assert client.chat.completions.create(model="m", messages=[]) is RESPONSE
    tr = _Transport(model_content=False)
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="t1", input_text="x"):
            client.chat.completions.create(model="m", messages=[])
        px.record_answer("t1", "y")
        px.flush()
    assert _chats(tr) == []


def test_anthropics_stream_helper_is_recorded_from_its_final_message(
    monkeypatch, tmp_path,
):
    """``messages.stream()`` posts on its own, not through ``create``.
    Mutation: leave the helper unwrapped -- red."""
    final = {"model": "claude-x", "stop_reason": "end_turn",
             "content": [{"type": "text", "text": "Hi"}]}

    class _Stream:
        current_message_snapshot = final

        def __iter__(self):
            return iter(["e1", "e2"])

    class _Manager:
        def __enter__(self):
            return _Stream()

        def __exit__(self, *exc):
            return None

    client = fw.wrap_anthropic(_Ns(messages=_Ns(
        create=lambda **kw: None, stream=lambda **kw: _Manager())))
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="t1", input_text="Hi"):
            with client.messages.stream(
                    model="claude", messages=[{"role": "user", "content": "Hi"}]) as s:
                assert list(s) == ["e1", "e2"]
        px.record_answer("t1", "Hi")
        px.flush()
    [chat] = _chats(tr)
    assert chat["gen_ai.response.model"] == "claude-x"
    assert json.loads(chat["gen_ai.output.messages"])[0]["parts"] == [
        {"type": "text", "content": "Hi"}]
