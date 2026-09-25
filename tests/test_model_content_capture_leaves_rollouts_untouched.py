"""Model-content capture leaves the SDK ROLLOUT lane byte-identical.

A ``serve()`` rollout is claimed work: its report is what Mission Control
grades and what GRPO reads, byte for byte. The observe-only model-boundary
callbacks now bound into the LangChain and ADK adapters run on that path too
-- the adapters are one binding for both lanes -- so this pins what a rollout
reports, through a LangChain tool loop AND an ADK model and tool call, to the
digest MEASURED ON THE PRE-CHANGE TREE (``d1ec738``, before the model-boundary
capture existed): the report's wire and every file the rollout's trajectory
directory holds, with the values that are clocks removed (each call's
``latency_ms``, and the report's ``timing``).

A rollout scope never records model content, whatever it is passed
(``_RolloutScope`` refuses the flag for anything but a turn); a mutation that
lets it do so writes ``genai_observation.jsonl`` beside the trajectory and
moves this digest.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time

import pytest

from agent_flywheel.contract import RolloutRequest
from agent_flywheel.entrypoint import resolve_entrypoint
from agent_flywheel.runner import run_rollout_in_process

pytest.importorskip("langchain_core")
pytest.importorskip("google.adk")

#: Measured with this file's scenario on the pre-change tree d1ec738 and on
#: the changed tree; identical.
PRE_CHANGE_DIGEST = (
    "98bce4694fe6975839b9ca53ce59bb6c92fdc30b1dcccb562f30a0c992eddb75")


def _agent():
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langchain_core.tools import tool
    from langchain_core.utils.function_calling import convert_to_openai_tool
    from pydantic import Field

    class ScriptedChat(BaseChatModel):
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

    @tool
    def lookup_order(order_id: str) -> str:
        """Look up an order's shipping status."""
        return json.dumps({"status": "shipped", "eta": "Friday"})

    class Agent:
        def invoke(self, task: str) -> str:
            meta = {"model_name": "gpt-4o-mini-2024-07-18"}
            model = ScriptedChat(script=[
                AIMessage(content="", tool_calls=[{
                    "name": "lookup_order", "args": {"order_id": task},
                    "id": "call_1", "type": "tool_call"}],
                    response_metadata={**meta, "finish_reason": "tool_calls"}),
                AIMessage(content="shipped", response_metadata={
                    **meta, "finish_reason": "stop"}),
            ])
            messages = [SystemMessage("be exact"), HumanMessage(task)]
            bound = model.bind_tools([lookup_order])
            while True:
                ai = bound.invoke(messages)
                messages.append(ai)
                if not ai.tool_calls:
                    break
                for call in ai.tool_calls:
                    messages.append(lookup_order.invoke(call))
            from google.adk.models.llm_request import LlmRequest
            from google.adk.models.llm_response import LlmResponse
            from google.adk.plugins.plugin_manager import PluginManager
            from google.genai import types

            class Ctx:
                invocation_id, agent_name, function_call_id = "inv", "a", "fc_1"

            class T:
                name = "lookup_order"

            pm = PluginManager()

            async def drive():
                await pm.run_before_model_callback(
                    callback_context=Ctx(), llm_request=LlmRequest(
                        model="gemini-2.5-flash", contents=[types.Content(
                            role="user", parts=[types.Part(text=task)])]))
                await pm.run_after_model_callback(
                    callback_context=Ctx(), llm_response=LlmResponse(
                        content=types.Content(role="model", parts=[
                            types.Part(text="ok")])))
                await pm.run_before_tool_callback(
                    tool=T(), tool_args={"order_id": task}, tool_context=Ctx())
                await pm.run_after_tool_callback(
                    tool=T(), tool_args={"order_id": task}, tool_context=Ctx(),
                    result={"status": "shipped"})

            asyncio.run(drive())
            return messages[-1].content

    return Agent()


def test_a_rollout_reports_exactly_what_it_reported_before(tmp_path):
    root = str(tmp_path)
    req = RolloutRequest(
        rollout_id="rl_grpo", turn_input="4417",
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_grpo", api_key="perk_x",
        artifacts_dir=root, deadline_ts=time.time() + 60,
    )
    out = run_rollout_in_process(
        resolve_entrypoint(_agent(), input_shape="text"), req,
        trajectory_dir=root, timeout_s=30.0, adapters=["langchain", "adk"],
    )
    wire = out.to_wire()
    wire.pop("timing", None)
    for call in wire.get("tool_calls") or []:
        call.pop("latency_ms", None)
    files = sorted(os.listdir(root))
    lines = []
    for name in files:
        path = os.path.join(root, name)
        if os.path.isfile(path):
            for line in open(path, encoding="utf-8").read().splitlines():
                rec = json.loads(line)
                if isinstance(rec, dict):
                    rec.pop("latency_ms", None)
                lines.append([name, rec])
    blob = json.dumps({"wire": wire, "files": files, "lines": lines},
                      sort_keys=True)
    assert files == ["tool_calls.jsonl"], files
    assert hashlib.sha256(blob.encode()).hexdigest() == PRE_CHANGE_DIGEST, blob
