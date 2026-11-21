"""B: the tool descriptions approved WITH a prompt reach the tools a turn sends.

Design 2026-11-21, contract B: ``/prompt/current`` serves ``tool_descriptions``
beside the approved text; :func:`agent_flywheel.prompt_decision` hands them out
only when the decision applies (``test_prompt_decision_parity.py`` holds that
to the Node SDK); a turn scoped with that decision applies them -- ADK by
itself, LangChain through :func:`agent_flywheel.apply_tool_descriptions` --
and outside one nothing changes.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import agent_flywheel as fw
from agent_flywheel.adapters.adk import _apply_tool_descriptions
from agent_flywheel.prompt import current_tool_descriptions

from .test_production_turn_upload import _attached, _Transport

TOOLS = {"lookup": "Look the order up first.", "refund": "Refund only shipped orders."}
SERVED = {"text": "You are careful.", "status": "approved", "tool_descriptions": TOOLS}


def test_a_turn_scoped_with_an_applying_decision_carries_its_overlay(
    monkeypatch, tmp_path,
):
    """Mutation: never set the turn's overlay -- red."""
    decision = fw.prompt_decision(SERVED)
    assert decision.apply and decision.tool_descriptions == TOOLS
    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        assert current_tool_descriptions() is None
        with px.turn(turn_id="t1", prompt=decision):
            assert current_tool_descriptions() == TOOLS
        assert current_tool_descriptions() is None
        refused = fw.prompt_decision({**SERVED, "status": "draft"})
        with px.turn(turn_id="t2", prompt=refused):
            assert current_tool_descriptions() is None
        with px.turn(turn_id="t3"):
            assert current_tool_descriptions() is None


def _adk_request(*names):
    decls = [SimpleNamespace(name=n, description="old") for n in names]
    return SimpleNamespace(config=SimpleNamespace(
        tools=[SimpleNamespace(function_declarations=decls)])), decls


def test_adk_applies_the_overlay_to_the_declarations_it_sends(monkeypatch, tmp_path):
    """Mutation: drop ``_apply_tool_descriptions`` from the ADK plugin's
    ``before_model_callback`` -- the callback test below goes red."""
    request, decls = _adk_request("lookup", "other")
    assert _apply_tool_descriptions(request) == 0
    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        with px.turn(turn_id="t1", prompt=fw.prompt_decision(SERVED)):
            assert _apply_tool_descriptions(request) == 1
    assert [d.description for d in decls] == [TOOLS["lookup"], "old"]


def test_adk_plugin_before_model_callback_applies_it_and_returns_none(
    monkeypatch, tmp_path,
):
    pytest.importorskip("google.adk")
    import asyncio

    from agent_flywheel.adapters import adk

    plugin_cls = adk._build_plugin_cls()
    monkeypatch.setattr(adk, "_binds", 1, raising=False)
    request, decls = _adk_request("refund")
    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        with px.turn(turn_id="t1", prompt=fw.prompt_decision(SERVED)):
            got = asyncio.run(plugin_cls().before_model_callback(
                callback_context=SimpleNamespace(invocation_id="i", agent_name="a"),
                llm_request=request))
    assert got is None
    assert decls[0].description == TOOLS["refund"]


def test_apply_tool_descriptions_copies_and_never_edits(monkeypatch, tmp_path):
    """LangChain's callbacks cannot change a request; the helper is the one
    line. Mutation: edit the caller's tool in place -- red."""
    openai_style = {"type": "function", "function": {"name": "lookup", "description": "old"}}
    flat = {"name": "refund", "description": "old"}
    other = {"name": "other", "description": "kept"}
    tools = [openai_style, flat, other]
    assert fw.apply_tool_descriptions(tools) is tools
    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        with px.turn(turn_id="t1", prompt=fw.prompt_decision(SERVED)):
            out = fw.apply_tool_descriptions(tools)
    assert out[0]["function"]["description"] == TOOLS["lookup"]
    assert out[1]["description"] == TOOLS["refund"]
    assert out[2] is other
    assert openai_style["function"]["description"] == "old"
    assert flat["description"] == "old"


def test_apply_tool_descriptions_to_a_langchain_tool(monkeypatch, tmp_path):
    pytest.importorskip("langchain_core")
    from langchain_core.tools import tool

    @tool
    def lookup(order_id: str) -> str:
        """old"""
        return order_id

    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        with px.turn(turn_id="t1", prompt=fw.prompt_decision(SERVED)):
            [copy] = fw.apply_tool_descriptions([lookup])
    assert copy.description == TOOLS["lookup"]
    assert lookup.description == "old"
    assert copy.invoke({"order_id": "7"}) == "7"
