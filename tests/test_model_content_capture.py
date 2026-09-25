"""Production MODEL-CONTENT capture, end to end through ``attach()``.

A production turn may carry what its model calls saw and said -- the system
prompt, the whole conversation history, the tool definitions and every tool
result -- as ONE OTel GenAI trace (``genai_trace`` on the turn upload). It is
the only turn content an OPSD row can be built on, so it has a consent tier of
its own, and the adapters record it by OBSERVING the model boundary: they
never change a call, a response or a tool result.

The happy paths come first and run a real agent loop through the real
adapters: LangChain (a chat model bound to tools, and LangGraph's own ReAct
agent) and Google ADK (through its PluginManager, the path every Runner
takes). Each asserts the turn that reaches the upload carries each model
call's WHOLE input and output. Everything after that pins a boundary:

* no consent, nothing RECORDED -- not merely nothing uploaded;
* a rollout records nothing, whatever it is passed (the SDK rollout lane's
  report is its own contract, and GRPO reads it byte for byte);
* a revocation the intake names stops recording;
* a grant made after startup is noticed;
* the same turn renders the same trace on every flush (the intake reads a
  different trace for the same turn as a conflict);
* Prime Agent's JSON stream is a boundary this SDK cannot observe, and it says
  so rather than render a prompt without its system instructions.
"""
from __future__ import annotations

import asyncio
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

import agent_flywheel as fw
from agent_flywheel.genai_otlp import OBSERVATION_FILENAME
from agent_flywheel.outcomes import _RolloutScope, model_capture_active

CONVERSATION = "7d0c2a8e-4b1f-4c3e-9a57-2f1e6b8d9c10"
SYSTEM = "You are Acme's order-support agent. Look orders up before answering."


class _Transport:
    """What the control plane says, and every batch it was handed."""

    def __init__(self, *, model_content=True, reason=None, refuse_traces=False):
        self.model_content = model_content
        self.reason = reason
        self.refuse_traces = refuse_traces
        self.batches: list[list[dict]] = []
        self.verdict_reads = 0

    def _capture(self):
        cap = {"turns": True, "model_content": self.model_content}
        if not self.model_content and self.reason:
            cap["model_content_reason"] = self.reason
        return {"capture": cap}

    def register(self, spec):
        return self._capture()

    def capture_current(self, agent_id):
        self.verdict_reads += 1
        return self._capture()

    def policy_current(self):
        return {}

    def report_turns(self, agent_id, turns):
        self.batches.append(list(turns))
        results = []
        for t in turns:
            r = {"turn_id": t["turn_id"], "accepted": True}
            if "genai_trace" in t and self.refuse_traces:
                # ``True`` is the consent refusal; a ``(code, detail)`` pair
                # is any other refusal the intake names.
                code, detail = (
                    self.refuse_traces if isinstance(self.refuse_traces, tuple)
                    else ("model_content_capture_not_enabled",
                          "an operator withdrew model-content capture"))
                r["genai_trace"] = {
                    "stored": False, "error": code, "detail": detail,
                }
            results.append(r)
        return {"agent_id": agent_id, "turns": results}

    @property
    def sent(self):
        return [t for b in self.batches for t in b]


@contextmanager
def _attached(monkeypatch, tmp_path, transport, **kw):
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_x")
    monkeypatch.setattr("agent_flywheel.transport.AttachTransport",
                        lambda client, agent_id: transport)
    monkeypatch.setattr("agent_flywheel._http.ControlPlaneClient.close",
                        lambda self: None)
    px = fw.attach(agent_id="acme-support", trajectory_root=str(tmp_path), **kw)
    try:
        yield px
    finally:
        px.close()


def _spans(trace):
    return trace["resourceSpans"][0]["scopeSpans"][0]["spans"]


def _attrs(span):
    return {a["key"]: a["value"].get("stringValue", a["value"])
            for a in span["attributes"]}


def _content(span, key):
    return json.loads(_attrs(span)[key])


# ── LangChain ───────────────────────────────────────────────────────────────


def _scripted_chat():
    pytest.importorskip("langchain_core")
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import Field

    class ScriptedChat(BaseChatModel):
        """A chat model that answers from a script, and says it is OpenAI's
        -- the provider LangChain names in the run's metadata."""

        script: list = Field(default_factory=list)

        def _generate(self, messages, stop=None, run_manager=None, **kw):
            return ChatResult(generations=[ChatGeneration(
                message=self.script.pop(0))])

        @property
        def _llm_type(self):
            return "scripted"

        def bind_tools(self, tools, **kw):
            return self.bind(tools=[convert_to_openai_tool(t) for t in tools],
                             **kw)

        def _get_ls_params(self, stop=None, **kw):
            return {"ls_provider": "openai", "ls_model_name": "gpt-4o-mini",
                    "ls_model_type": "chat"}

    return ScriptedChat


def _ai(content="", calls=(), finish="stop"):
    from langchain_core.messages import AIMessage

    return AIMessage(content=content, tool_calls=list(calls),
                     response_metadata={"finish_reason": finish,
                                        "model_name": "gpt-4o-mini-2024-07-18"})


def _lookup_tool():
    from langchain_core.tools import tool

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order's shipping status."""
        return json.dumps({"status": "shipped", "eta": "Friday"})

    return lookup_order


def _run_loop(model, tools, messages):
    """An ordinary tool loop, written the way a customer writes one."""
    by_name = {t.name: t for t in tools}
    bound = model.bind_tools(tools)
    while True:
        ai = bound.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:
            return ai.content
        for call in ai.tool_calls:
            messages.append(by_name[call["name"]].invoke(call))


def test_a_langchain_turn_uploads_its_model_calls_whole_input_and_output(
        monkeypatch, tmp_path):
    from langchain_core.messages import HumanMessage, SystemMessage

    tr = _Transport()
    ScriptedChat = _scripted_chat()
    lookup = _lookup_tool()
    call = {"name": "lookup_order", "args": {"order_id": "4417"},
            "id": "call_1", "type": "tool_call"}
    model = ScriptedChat(script=[
        _ai(calls=[call], finish="tool_calls"),
        _ai("Order 4417 has shipped and arrives Friday."),
    ])
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_capture_model_content is True
        assert "langchain" in px.adapters_bound
        messages = [SystemMessage(SYSTEM),
                    HumanMessage("Where is my order 4417?")]
        with px.turn(turn_id="t1", conversation_id=CONVERSATION,
                     input_text="Where is my order 4417?"):
            answer = _run_loop(model, [lookup], messages)
        px.record_answer("t1", answer)
        assert px.flush()["sent"] == 1

    [turn] = tr.sent
    # The published wire accepts it: `genai_trace` is part of ProductionTurn.
    from jsonschema import Draft202012Validator

    schema = json.loads((Path(__file__).resolve().parents[1] / "schema"
                         / "flywheel-1.json").read_text())
    Draft202012Validator({**schema, "$ref": "#/$defs/ProductionTurn"}
                         ).validate(turn)
    trace = turn["genai_trace"]
    root, chat1, tool_span, chat2 = _spans(trace)
    assert _attrs(root)["gen_ai.operation.name"] == "invoke_agent"
    assert _attrs(root)["gen_ai.provider.name"] == "openai"
    assert {_attrs(s)["gen_ai.conversation.id"] for s in _spans(trace)} == {
        CONVERSATION}
    assert not any(k.startswith("percepteye.")
                   for s in _spans(trace) for k in _attrs(s))

    # The system prompt is the conventions' system instructions, not a
    # message, and the first call was sent exactly the user's question.
    assert _content(chat1, "gen_ai.system_instructions") == [
        {"type": "text", "content": SYSTEM}]
    assert _content(chat1, "gen_ai.input.messages") == [
        {"role": "user", "parts": [
            {"type": "text", "content": "Where is my order 4417?"}]}]
    [definition] = _content(chat1, "gen_ai.tool.definitions")
    assert (definition["type"], definition["name"]) == ("function", "lookup_order")
    assert definition["parameters"]["properties"] == {
        "order_id": {"type": "string"}}
    assert _content(chat1, "gen_ai.output.messages") == [{
        "role": "assistant",
        "parts": [{"type": "tool_call", "id": "call_1",
                   "name": "lookup_order", "arguments": {"order_id": "4417"}}],
        "finish_reason": "tool_call"}]
    assert _attrs(chat1)["gen_ai.response.model"] == "gpt-4o-mini-2024-07-18"
    assert _attrs(chat1)["gen_ai.response.finish_reasons"] == {
        "arrayValue": {"values": [{"stringValue": "tool_calls"}]}}

    # The tool span holds what the tool handed BACK, the same value the next
    # call is sent -- and joins the model's call by its id.
    result = json.dumps({"status": "shipped", "eta": "Friday"})
    assert _attrs(tool_span)["gen_ai.tool.call.id"] == "call_1"
    assert _content(tool_span, "gen_ai.tool.call.result") == result
    assert _content(chat2, "gen_ai.input.messages") == [
        {"role": "user", "parts": [
            {"type": "text", "content": "Where is my order 4417?"}]},
        {"role": "assistant", "parts": [
            {"type": "tool_call", "id": "call_1", "name": "lookup_order",
             "arguments": {"order_id": "4417"}}]},
        {"role": "tool", "parts": [
            {"type": "tool_call_response", "id": "call_1", "response": result}]},
    ]
    assert _content(chat2, "gen_ai.output.messages")[0]["finish_reason"] == "stop"

    # Ordered, and inside the turn.
    starts = [int(s["startTimeUnixNano"]) for s in (chat1, tool_span, chat2)]
    ends = [int(s["endTimeUnixNano"]) for s in (chat1, tool_span, chat2)]
    assert starts == sorted(starts)
    assert int(root["startTimeUnixNano"]) <= starts[0]
    assert max(ends) <= int(root["endTimeUnixNano"])


def test_langgraphs_own_agent_is_captured_with_no_instrumentation(
        monkeypatch, tmp_path):
    """The README's zero-instrumentation claim, for model content: a
    LangGraph ReAct agent built the documented way, not one line changed."""
    pytest.importorskip("langgraph.prebuilt")
    from langgraph.prebuilt import create_react_agent

    tr = _Transport()
    ScriptedChat = _scripted_chat()
    call = {"name": "lookup_order", "args": {"order_id": "4417"},
            "id": "call_9", "type": "tool_call"}
    model = ScriptedChat(script=[
        _ai(calls=[call], finish="tool_calls"),
        _ai("Shipped; arrives Friday."),
    ])
    agent = create_react_agent(model, [_lookup_tool()], prompt=SYSTEM)
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="g1", conversation_id=CONVERSATION):
            out = agent.invoke({"messages": [("user", "Where is 4417?")]})
        px.record_answer("g1", out["messages"][-1].content)
        px.flush()
    trace = tr.sent[0]["genai_trace"]
    ops = [_attrs(s)["gen_ai.operation.name"] for s in _spans(trace)]
    assert ops == ["invoke_agent", "chat", "execute_tool", "chat"]
    chat1 = _spans(trace)[1]
    assert _content(chat1, "gen_ai.system_instructions") == [
        {"type": "text", "content": SYSTEM}]
    assert _content(chat1, "gen_ai.input.messages")[-1]["parts"] == [
        {"type": "text", "content": "Where is 4417?"}]


# ── Google ADK ──────────────────────────────────────────────────────────────


def _adk_turn(px, request, response, tool=None):
    """One model call and one tool call, through the PluginManager every ADK
    Runner constructs -- the install path, not a stand-in for it."""
    from google.adk.plugins.plugin_manager import PluginManager

    class _Ctx:
        invocation_id, agent_name = "inv-1", "support"
        function_call_id = "fc-live"

    class _Tool:
        name = "lookup_order"

    pm = PluginManager()

    async def drive():
        before = await pm.run_before_model_callback(
            callback_context=_Ctx(), llm_request=request)
        after = await pm.run_after_model_callback(
            callback_context=_Ctx(), llm_response=response)
        if tool is not None:
            await pm.run_before_tool_callback(
                tool=_Tool(), tool_args={"order_id": "4417"}, tool_context=_Ctx())
            await pm.run_after_tool_callback(
                tool=_Tool(), tool_args={"order_id": "4417"},
                tool_context=_Ctx(), result=tool)
        return before, after

    return asyncio.run(drive())


def _adk_request_response():
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    request = LlmRequest(
        model="gemini-2.5-flash",
        contents=[types.Content(role="user", parts=[
            types.Part(text="Where is my order 4417?")])],
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            tools=[types.Tool(function_declarations=[types.FunctionDeclaration(
                name="lookup_order",
                description="Look up an order's shipping status.",
                parameters_json_schema={
                    "type": "object",
                    "properties": {"order_id": {"type": "string"}},
                    "required": ["order_id"]})])]),
    )
    response = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(
            function_call=types.FunctionCall(
                name="lookup_order", args={"order_id": "4417"}))]),
        finish_reason=types.FinishReason.STOP,
        model_version="gemini-2.5-flash-001",
    )
    return request, response


def test_an_adk_turn_uploads_its_model_calls_whole_input_and_output(
        monkeypatch, tmp_path):
    pytest.importorskip("google.adk")
    request, response = _adk_request_response()
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert "adk" in px.adapters_bound
        with px.turn(turn_id="a1", conversation_id=CONVERSATION):
            before, after = _adk_turn(px, request, response,
                                      tool={"status": "shipped"})
        px.record_answer("a1", "shipped")
        px.flush()
    # OBSERVE ONLY: a value here would have replaced the model call and its
    # response.
    assert (before, after) == (None, None)
    _root, chat, tool_span = _spans(tr.sent[0]["genai_trace"])
    assert _attrs(chat)["gen_ai.provider.name"] == "gcp.gen_ai"
    assert _attrs(chat)["gen_ai.request.model"] == "gemini-2.5-flash"
    assert _attrs(chat)["gen_ai.response.model"] == "gemini-2.5-flash-001"
    assert _content(chat, "gen_ai.system_instructions") == [
        {"type": "text", "content": SYSTEM}]
    assert _content(chat, "gen_ai.input.messages") == [
        {"role": "user", "parts": [
            {"type": "text", "content": "Where is my order 4417?"}]}]
    assert _content(chat, "gen_ai.tool.definitions") == [{
        "type": "function", "name": "lookup_order",
        "description": "Look up an order's shipping status.",
        "parameters": {"type": "object",
                       "properties": {"order_id": {"type": "string"}},
                       "required": ["order_id"]}}]
    assert _content(chat, "gen_ai.output.messages") == [{
        "role": "assistant",
        "parts": [{"type": "tool_call", "id": None, "name": "lookup_order",
                   "arguments": {"order_id": "4417"}}],
        "finish_reason": "tool_call"}]
    # ADK sends a non-dict result as {"result": ...}; a dict as itself.
    assert _content(tool_span, "gen_ai.tool.call.result") == {"status": "shipped"}


def test_adk_reads_the_request_the_model_was_sent_not_an_early_copy(
        monkeypatch, tmp_path):
    """The agent's own before_model callbacks run AFTER this plugin and may
    edit the request. What is recorded is what the model was sent."""
    pytest.importorskip("google.adk")
    from google.genai import types

    request, response = _adk_request_response()
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="a2", conversation_id=CONVERSATION):
            from google.adk.plugins.plugin_manager import PluginManager

            class _Ctx:
                invocation_id, agent_name = "inv-2", "support"

            pm = PluginManager()

            async def drive():
                await pm.run_before_model_callback(
                    callback_context=_Ctx(), llm_request=request)
                request.contents.append(types.Content(
                    role="user", parts=[types.Part(text="(added by the agent)")]))
                await pm.run_after_model_callback(
                    callback_context=_Ctx(), llm_response=response)

            asyncio.run(drive())
        px.record_answer("a2", "ok")
        px.flush()
    chat = _spans(tr.sent[0]["genai_trace"])[1]
    assert _content(chat, "gen_ai.input.messages")[-1]["parts"] == [
        {"type": "text", "content": "(added by the agent)"}]


def test_an_adk_client_call_id_is_not_put_on_the_tool_span(monkeypatch, tmp_path):
    """ADK mints ``adk-`` ids for a model that returns none, and strips them
    from every request before the model sees it. A tool span carrying one
    would join no tool call the model made -- and the decoder refuses a
    decision after a tool execution no recorded message accounts for."""
    pytest.importorskip("google.adk")
    from google.adk.plugins.plugin_manager import PluginManager

    request, response = _adk_request_response()

    class _Ctx:
        invocation_id, agent_name = "inv-3", "support"
        function_call_id = "adk-6f0c"

    class _Tool:
        name = "lookup_order"

    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="a3", conversation_id=CONVERSATION):
            pm = PluginManager()

            async def drive():
                await pm.run_before_model_callback(
                    callback_context=_Ctx(), llm_request=request)
                await pm.run_after_model_callback(
                    callback_context=_Ctx(), llm_response=response)
                await pm.run_before_tool_callback(
                    tool=_Tool(), tool_args={"order_id": "4417"},
                    tool_context=_Ctx())
                await pm.run_after_tool_callback(
                    tool=_Tool(), tool_args={"order_id": "4417"},
                    tool_context=_Ctx(), result="shipped")

            asyncio.run(drive())
        px.record_answer("a3", "shipped")
        px.flush()
    tool_span = _spans(tr.sent[0]["genai_trace"])[-1]
    assert "gen_ai.tool.call.id" not in _attrs(tool_span)
    # ...and a result that is not a dict is what ADK sends the model.
    assert _content(tool_span, "gen_ai.tool.call.result") == {"result": "shipped"}


# ── consent ─────────────────────────────────────────────────────────────────


def _one_langchain_turn(px, tid="t1"):
    from langchain_core.messages import HumanMessage

    ScriptedChat = _scripted_chat()
    model = ScriptedChat(script=[_ai("hello")])
    with px.turn(turn_id=tid, conversation_id=CONVERSATION, input_text="hi"):
        model.invoke([HumanMessage("hi")])
    px.record_answer(tid, "hello")


def test_without_the_second_tier_nothing_is_recorded_at_all(monkeypatch, tmp_path):
    """Not merely not uploaded: the customer's prompt and tool results never
    reach disk, so a later grant cannot make this turn's content retroactive."""
    tr = _Transport(model_content=False, reason="an operator has not enabled it")
    with _attached(monkeypatch, tmp_path, tr) as px:
        assert px.may_upload is True
        assert px.may_capture_model_content is False
        assert px.model_content_refusal == "an operator has not enabled it"
        _one_langchain_turn(px)
        assert not (tmp_path / "t1" / OBSERVATION_FILENAME).exists()
        px.flush()
    assert tr.sent and "genai_trace" not in tr.sent[0]


def test_a_grant_made_after_startup_is_noticed(monkeypatch, tmp_path):
    tr = _Transport(model_content=False)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _one_langchain_turn(px, "before")
        tr.model_content = True
        px.flush()
        assert px.may_capture_model_content is True
        _one_langchain_turn(px, "after")
        px.flush()
    by_id = {t["turn_id"]: t for t in tr.sent}
    assert "genai_trace" not in by_id["before"]
    assert "genai_trace" in by_id["after"]


def test_a_recorded_trace_is_not_sent_once_the_tier_is_withdrawn(
        monkeypatch, tmp_path):
    """Recorded under a grant, flushed after a withdrawal: the turn goes up,
    its trace does not. The flush asks the verdict it holds NOW."""
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _one_langchain_turn(px)
        assert (tmp_path / "t1" / OBSERVATION_FILENAME).exists()
        tr.model_content = False
        px.may_capture_model_content = False
        px.flush()
    assert tr.sent and "genai_trace" not in tr.sent[0]


def test_a_named_refusal_from_the_intake_stops_recording(monkeypatch, tmp_path):
    tr = _Transport(refuse_traces=True)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _one_langchain_turn(px, "t1")
        with pytest.warns(RuntimeWarning, match="withdrew model-content"):
            px.flush()
        assert px.may_capture_model_content is False
        _one_langchain_turn(px, "t2")
        assert not (tmp_path / "t2" / OBSERVATION_FILENAME).exists()


def test_a_turn_attributed_to_no_product_stops_recording_until_bound(
        monkeypatch, tmp_path):
    """The agent became SHARED after the tier was granted: the intake keeps no
    trace of a turn attributed to no Product and refuses it by name. Read only
    as a consent refusal, every later turn's prompt and tool results were
    recorded and sent to be discarded. The SDK stops, and picks the tier back
    up when the verdict grants it again -- one Product declares the agent."""
    unattributed = ("genai_trace_turn_not_attributed_to_a_product",
                    "this agent's production turns are attributed to no "
                    "Product (ambiguous_shared_agent)")
    tr = _Transport(refuse_traces=unattributed)
    with _attached(monkeypatch, tmp_path, tr) as px:
        _one_langchain_turn(px, "t1")
        tr.model_content, tr.reason = False, unattributed[1]
        with pytest.warns(RuntimeWarning, match="attributed to no Product"):
            px.flush()
        assert px.may_capture_model_content is False
        assert px.model_content_refusal == unattributed[1]
        _one_langchain_turn(px, "t2")
        assert not (tmp_path / "t2" / OBSERVATION_FILENAME).exists()
        px.flush()
        assert px.may_capture_model_content is False
        # COMPLETES: the operator retires the second Product.
        tr.model_content, tr.refuse_traces = True, False
        px.flush()
        assert px.may_capture_model_content is True
        _one_langchain_turn(px, "t3")
        assert (tmp_path / "t3" / OBSERVATION_FILENAME).exists()


def test_a_refused_turn_upload_refuses_its_model_content_too(monkeypatch, tmp_path):
    """A 403 is the intake refusing this agent's TURNS; a trace rides on a
    turn, so recording model content for turns that will not be sent is
    recording the customer's prompt for nothing."""
    from agent_flywheel.errors import TransportError

    class _Revoked(_Transport):
        def report_turns(self, agent_id, turns):
            raise TransportError(
                "POST /agents/acme-support/turns -> HTTP 403: production "
                "capture is not enabled for this agent.")

    tr = _Revoked()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _one_langchain_turn(px)
        with pytest.warns(RuntimeWarning):
            px.flush()
        assert px.may_upload is False
        assert px.may_capture_model_content is False
        assert px.model_content_refusal == px.upload_refusal
        _one_langchain_turn(px, "t2")
        assert not (tmp_path / "t2" / OBSERVATION_FILENAME).exists()


def test_the_verdict_is_not_re_read_once_both_tiers_are_granted(
        monkeypatch, tmp_path):
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _one_langchain_turn(px)
        px.flush()
        px.flush()
    assert tr.verdict_reads == 0


# ── determinism and the unchanged paths ─────────────────────────────────────


def test_the_same_turn_renders_the_same_trace_on_every_flush(monkeypatch, tmp_path):
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        _one_langchain_turn(px)
        first = px._turn_wire(tmp_path / "t1")
        second = px._turn_wire(tmp_path / "t1")
    assert first == second and "genai_trace" in first


def test_a_turn_with_no_model_call_keeps_its_exact_wire(monkeypatch, tmp_path):
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="plain", conversation_id=CONVERSATION,
                     input_text="hi"):
            pass
        px.record_answer("plain", "hello")
        px.flush()
    assert tr.sent[0] == {
        "turn_id": "plain", "input_text": "hi", "final_text": "hello",
        "tool_calls": None, "tool_calls_omitted_count": 0,
        "conversation_id": CONVERSATION, "turn_index": 0,
    }


def test_a_rollout_records_no_model_boundary_whatever_it_is_passed(tmp_path):
    """The SDK rollout lane's report is its own contract (GRPO reads it byte
    for byte), so a rollout scope never records model content -- not even
    when a caller hands it the flag a turn uses."""
    from langchain_core.messages import HumanMessage

    from agent_flywheel.binding import bind_selected

    ScriptedChat = _scripted_chat()
    _, unbinds = bind_selected(["langchain"], llm_url="", api_key="")
    try:
        with _RolloutScope("rl_1", str(tmp_path), kind="rollout",
                           model_content=True):
            assert model_capture_active() is False
            ScriptedChat(script=[_ai("x")]).invoke([HumanMessage("q")])
    finally:
        for u in unbinds:
            u()
    assert list(Path(tmp_path).iterdir()) == []


def test_the_scope_is_the_one_answer_to_may_this_call_be_recorded(tmp_path):
    """Asked at TWO layers -- the adapters ask ``model_capture_active`` before
    converting anything, and the writer asks again before writing -- so each
    layer is pinned on its own: neither may be the only thing standing
    between a rollout, or an unconsented turn, and a recorded prompt."""
    from agent_flywheel.outcomes import record_model_boundary

    op = {"operation": "chat", "span_id": "1" * 16,
          "start_time_unix_nano": "1", "end_time_unix_nano": "2",
          "provider_name": "openai", "request_model": "m",
          "input_messages": []}
    cases = {
        "rollout, flag passed": ("rollout", True),
        "turn without consent": ("turn", False),
    }
    for name, (kind, flag) in cases.items():
        d = tmp_path / name.replace(" ", "_").replace(",", "")
        with _RolloutScope("s1", str(d), kind=kind, model_content=flag):
            assert model_capture_active() is False, name
            assert record_model_boundary(op) is False, name
        assert not (d / OBSERVATION_FILENAME).exists(), name
    assert model_capture_active() is False and record_model_boundary(op) is False
    granted = tmp_path / "granted"
    with _RolloutScope("t1", str(granted), kind="turn", model_content=True):
        assert model_capture_active() is True
        assert record_model_boundary(op) is True
    assert (granted / OBSERVATION_FILENAME).exists()


def test_a_rollout_pays_for_no_model_call_conversion(monkeypatch, tmp_path):
    """The LangChain callbacks return before converting a single message
    outside a consented turn: a rollout's model calls cost one context-variable
    read, not a copy of every prompt."""
    from langchain_core.messages import HumanMessage

    from agent_flywheel.adapters import langchain as lc
    from agent_flywheel.binding import bind_selected

    # COUNTED, not raised: the callbacks swallow their own failures (a
    # capture bug must never cost the model call), so a raise here would be
    # swallowed too and prove nothing.
    converted: list[int] = []
    real = lc._chat_request
    monkeypatch.setattr(lc, "_chat_request",
                        lambda *a, **k: converted.append(1) or real(*a, **k))
    ScriptedChat = _scripted_chat()
    _, unbinds = bind_selected(["langchain"], llm_url="", api_key="")
    try:
        with _RolloutScope("rl_2", str(tmp_path), kind="rollout"):
            ScriptedChat(script=[_ai("x")]).invoke([HumanMessage("q")])
        with _RolloutScope("t_2", str(tmp_path / "t"), kind="turn",
                           model_content=True):
            ScriptedChat(script=[_ai("y")]).invoke([HumanMessage("q")])
    finally:
        for u in unbinds:
            u()
    # One conversion: the consented turn's (the control), none for the rollout.
    assert converted == [1]


def test_adk_records_the_final_response_not_each_streamed_chunk(
        monkeypatch, tmp_path):
    """In streaming mode ADK calls after_model_callback for every partial
    chunk and once more for the final response. One model call is one chat
    span, carrying the final response."""
    pytest.importorskip("google.adk")
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins.plugin_manager import PluginManager
    from google.genai import types

    request, final = _adk_request_response()
    partial = LlmResponse(
        content=types.Content(role="model", parts=[types.Part(text="Look")]),
        partial=True)

    class _Ctx:
        invocation_id, agent_name = "inv-4", "support"

    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="a4", conversation_id=CONVERSATION):
            pm = PluginManager()

            async def drive():
                await pm.run_before_model_callback(
                    callback_context=_Ctx(), llm_request=request)
                await pm.run_after_model_callback(
                    callback_context=_Ctx(), llm_response=partial)
                await pm.run_after_model_callback(
                    callback_context=_Ctx(), llm_response=final)

            asyncio.run(drive())
        px.record_answer("a4", "ok")
        px.flush()
    spans = _spans(tr.sent[0]["genai_trace"])
    assert [_attrs(s)["gen_ai.operation.name"] for s in spans] == [
        "invoke_agent", "chat"]
    assert _content(spans[1], "gen_ai.output.messages")[0]["parts"][0][
        "type"] == "tool_call"


def test_prime_agents_stream_is_a_boundary_this_sdk_cannot_observe():
    """Prime Agent's JSON mode carries no system prompt and no tool
    definitions (its documented stream: a session header, then agent, turn,
    message and tool events), and it runs only as a serve() command -- a
    rollout, never an attach() turn. A trace rebuilt from its agent_end
    messages would present a prompt without the system instructions the model
    was given, so the adapter projects none: its projection has no field that
    could carry one."""
    from agent_flywheel.adapters.prime_agent import PrimeAgentOutputAdapter
    from agent_flywheel.event_stream import CommandOutputProjection

    stream = "\n".join(json.dumps(e) for e in [
        {"type": "session", "version": 3, "id": "s", "timestamp": "t",
         "cwd": "/w"},
        {"type": "agent_start"},
        {"type": "message_end", "message": {
            "role": "assistant", "content": [{"type": "text", "text": "ok"}],
            "stopReason": "stop"}},
        {"type": "agent_end", "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}]},
    ])
    projection = PrimeAgentOutputAdapter().project(stream)
    assert projection.final_text == "ok"
    assert set(CommandOutputProjection.__dataclass_fields__) == {
        "recognized", "final_text", "llm_call_count", "tool_calls",
        "tool_calls_omitted_count", "exclusion_reason", "exclusion_detail"}
    assert not any("system" in json.dumps(e)
                   for e in map(json.loads, stream.splitlines()))
