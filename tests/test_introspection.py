"""Agent discovery from inside the process.

Two properties carry the weight here and both have a test that fails loudly if
they regress: an agent we could not read is reported as UNREAD rather than as
an agent with nothing in it, and a failure to introspect never reaches the
customer's agent.
"""
from __future__ import annotations

import json
from typing import ClassVar

import pytest

from agent_flywheel.introspection import (
    DiscoveredAgent,
    introspect,
)

# ── the absence contract ───────────────────────────────────────────────────


def test_an_unrecognised_agent_is_None_not_an_empty_agent():
    """THE test. Returning a DiscoveredAgent with no tools would assert that
    this agent HAS no tools, which we never observed. None is the honest
    answer and the caller reports it as such."""
    class SomeoneElsesAgent:
        pass

    assert introspect(SomeoneElsesAgent()) is None
    assert introspect(None) is None


def test_null_tool_definitions_and_empty_list_are_different_on_the_wire():
    """`null` = we did not look. `[]` = we looked, and there are none. The
    consumer grades on this difference, so it must survive serialisation."""
    unread = DiscoveredAgent(agent_name="a", framework="x").to_wire()
    empty = DiscoveredAgent(
        agent_name="a", framework="x", tool_definitions=[]
    ).to_wire()

    assert unread["tool_definitions"] is None
    assert empty["tool_definitions"] == []
    # And the key is always present — omitting it would collapse the two.
    assert "tool_definitions" in unread and "tool_definitions" in empty
    assert json.loads(json.dumps(unread))["tool_definitions"] is None


def test_absent_optional_fields_are_omitted_rather_than_nulled():
    """Everything EXCEPT tool_definitions is absent-when-unknown, so a reader
    cannot mistake a null for a measured value."""
    wire = DiscoveredAgent(agent_name="a", framework="x").to_wire()
    for k in ("system_prompt", "discovered_model_config", "capabilities",
              "sub_agents", "incomplete"):
        assert k not in wire, f"{k} should be absent, not null"


def test_source_type_is_always_sdk_runtime():
    """The one source type meaning 'the agent described itself from inside its
    own process' — which is what makes this vantage point different."""
    assert DiscoveredAgent(agent_name="a", framework="x").source_type == "sdk_runtime"


# ── never break the customer's agent ───────────────────────────────────────


def test_an_exploding_agent_costs_us_the_description_and_nothing_else():
    """Introspection runs at registration. A customer's agent must start
    whether or not we could describe it."""
    class Hostile:
        @property
        def tools(self):
            raise RuntimeError("boom")

        @property
        def system_prompt(self):
            raise RuntimeError("boom")

    assert introspect(Hostile()) is None    # no raise


def test_a_bound_method_resolves_to_the_agent_instance():
    """`build_agent()` returns an instance and the runner resolves one of its
    methods; the agent is the instance, not the method."""
    pytest.importorskip("langchain_core")
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return q

    class Agent:
        def __init__(self):
            self.tools = [lookup]

        def invoke(self, task: str) -> str:
            return task

    got = introspect(Agent().invoke)        # the METHOD, as the runner holds it
    assert got is not None, "should have followed __self__ to the instance"
    assert [t.name for t in got.tool_definitions] == ["lookup"]


# ── Google ADK: the near 1:1 mapping ───────────────────────────────────────


def _adk_agent(**kw):
    from google.adk.agents import LlmAgent
    return LlmAgent(**kw)


def test_adk_agent_yields_prompt_tools_and_model():
    pytest.importorskip("google.adk")

    def lookup(q: str) -> dict:
        """Look something up."""
        return {"q": q}

    agent = _adk_agent(
        name="researcher", model="gemini-2.0-flash",
        instruction="You are careful.", description="Finds things.",
        tools=[lookup],
    )
    got = introspect(agent)
    assert got is not None
    assert got.framework == "google_adk"
    assert got.agent_name == "researcher"
    assert got.system_prompt == "You are careful."
    assert got.discovered_model_config.model_name == "gemini-2.0-flash"
    assert [t.name for t in got.tool_definitions] == ["lookup"]
    # The tool's OWN declaration, not a guess at its signature.
    assert got.tool_definitions[0].original_format == "google_adk"
    assert "q" in json.dumps(got.tool_definitions[0].input_schema)


def test_adk_sub_agents_are_walked_and_the_parent_is_an_orchestrator():
    """The multi-agent hierarchy is the thing an external connector reading
    source cannot reliably reconstruct."""
    pytest.importorskip("google.adk")
    child = _adk_agent(name="coder", model="gemini-2.0-flash",
                       instruction="You write code.")
    parent = _adk_agent(name="planner", model="gemini-2.0-flash",
                        instruction="You plan.", sub_agents=[child])

    got = introspect(parent)
    assert got.role == "orchestrator"
    assert [s.agent_name for s in got.sub_agents] == ["coder"]
    assert got.sub_agents[0].role == "sub_agent"
    assert got.sub_agents[0].system_prompt == "You write code."


def test_a_callable_instruction_is_reported_unread_not_invoked():
    """ADK allows a callable instruction resolved per-invocation from state.
    Calling it here would execute customer code at registration with no
    invocation context, so we record that we could not read it."""
    pytest.importorskip("google.adk")
    called = []

    def dynamic(ctx):                      # pragma: no cover - must NOT run
        called.append(1)
        return "resolved"

    agent = _adk_agent(name="a", model="gemini-2.0-flash", instruction=dynamic)
    got = introspect(agent)
    assert got.system_prompt is None
    assert "system_prompt" in got.incomplete
    assert called == [], "a callable instruction must not be invoked"


def test_an_adk_agent_with_no_tools_reports_empty_not_unread():
    """The other side of the absence contract: we DID look here."""
    pytest.importorskip("google.adk")
    agent = _adk_agent(name="a", model="gemini-2.0-flash", instruction="hi")
    got = introspect(agent)
    assert got.tool_definitions == []
    assert got.to_wire()["tool_definitions"] == []


# ── LangChain ──────────────────────────────────────────────────────────────


def test_langchain_agent_yields_its_tool_catalogue():
    pytest.importorskip("langchain_core")
    from langchain_core.tools import tool

    @tool
    def deploy(service: str) -> str:
        """Deploy a service."""
        return service

    class Agent:
        tools: ClassVar[list] = [deploy]

    got = introspect(Agent())
    assert got is not None and got.framework == "langchain"
    assert [t.name for t in got.tool_definitions] == ["deploy"]
    assert got.tool_definitions[0].description == "Deploy a service."
    assert got.tool_definitions[0].input_schema["type"] == "object"
    assert "service" in got.tool_definitions[0].input_schema["properties"]
    # No prompt was reachable, and that is recorded rather than left blank.
    assert got.system_prompt is None and "system_prompt" in got.incomplete


def test_an_object_with_a_tools_attribute_that_is_not_langchain_does_not_match():
    """`tools` is a common attribute name. Matching on it alone would claim
    every object with a list called tools is a LangChain agent."""
    pytest.importorskip("langchain_core")

    class NotAnAgent:
        tools: ClassVar[list] = ["a string", 42]

    assert introspect(NotAnAgent()) is None
