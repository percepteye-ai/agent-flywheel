"""LangGraph discovery, against the real langgraph API.

A compiled graph has neither ``tools`` nor ``get_tools``, so the LangChain
probe matched nothing and a LangGraph customer was onboarded with NO
description at all. These tests run against the installed library rather than
a mock, because the failure being prevented is exactly "the probe reported
success while reading a shape the framework no longer has".

Two properties carry the weight and each has a test that fails loudly if it
regresses:

* a node we could not read reports ``None`` and says so in ``incomplete`` --
  never ``[]``, which would claim the node HAS no tools;
* the synthetic channel nodes never reach the operator, who would otherwise be
  offered ``__start__`` as something to optimise.
"""
from __future__ import annotations

import json
from typing import ClassVar, TypedDict

import pytest

from percepteye_agent_flywheel.introspection import introspect

pytest.importorskip("langgraph")
pytest.importorskip("langchain_core")

from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph


class _State(TypedDict):
    x: str


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return q


class _FakeLLM:
    model_name = "gpt-4o-mini"
    temperature = 0.3


class _Solver:
    """An object node: its prompt and tools are attributes, so they are
    reachable from outside."""

    system_prompt = "You are the solver."
    tools: ClassVar[list] = [lookup]
    llm = _FakeLLM()

    def __call__(self, state: _State) -> _State:
        return state


def _plain(state: _State) -> _State:
    """A function node. Nothing to read -- and that is not a failure."""
    return state


def _build(**nodes):
    """A compiled graph with ``nodes`` plus the object/function pair."""
    g = StateGraph(_State)
    g.add_node("solver", _Solver())
    g.add_node("plain", _plain)
    for name, impl in nodes.items():
        g.add_node(name, impl)
    g.add_edge(START, "solver")
    g.add_edge("solver", "plain")
    g.add_edge("plain", END)
    return g.compile()


@pytest.fixture
def graph():
    return _build()


def _by_name(agent):
    return {a.agent_name: a for a in agent.sub_agents}


# ── the graph is described at all ──────────────────────────────────────────


def test_a_compiled_graph_is_discovered_rather_than_returning_None():
    """THE regression. Before the langgraph probe existed this was None: the
    LangChain probe reads ``obj.tools`` / ``obj.get_tools()`` and a compiled
    graph has neither, so the customer got no description whatsoever."""
    app = _build()

    # Pin the reason the old probe failed, so this test still means something
    # if langgraph ever grows a `tools` attribute.
    assert not hasattr(app, "tools")
    assert not hasattr(app, "get_tools")

    got = introspect(app)
    assert got is not None, "a compiled LangGraph graph was not discovered"
    assert got.framework == "langgraph"


def test_every_user_node_is_a_sub_agent(graph):
    got = introspect(graph)
    assert set(_by_name(got)) == {"solver", "plain"}
    assert got.role == "orchestrator"
    assert all(a.role == "sub_agent" for a in got.sub_agents)


def test_synthetic_channel_nodes_never_reach_the_operator(graph):
    """``__start__``/``__end__`` are LangGraph's plumbing. Offering them as
    optimisable nodes would be offering something that does not exist."""
    # The graph really does carry one, so the filter is doing work.
    assert any(n.startswith("__") for n in graph.nodes)

    names = set(_by_name(introspect(graph)))
    assert not any(n.startswith("__") for n in names), names


def test_the_graph_probe_wins_over_the_langchain_probe():
    """Order is load-bearing, not decoration. The LangChain probe matches
    anything carrying a ``tools`` list of BaseTools, and it describes it as
    ONE flat agent -- so the moment a compiled graph carries such an attribute
    (a customer subclass today, langgraph itself tomorrow) the wrong probe
    would claim it and every node would be lost.
    """
    app = _build()
    app.tools = [lookup]  # the shape that makes the LangChain probe match

    got = introspect(app)
    assert got.framework == "langgraph", "the LangChain probe claimed the graph"
    assert set(_by_name(got)) == {"solver", "plain"}


def test_a_node_is_named_by_the_GRAPH_not_by_its_class(graph):
    """The node name is the operator-facing id, and the id the operator picks
    has to be the id the graph uses -- not ``_Solver``."""
    assert "solver" in _by_name(introspect(graph))
    assert "_Solver" not in _by_name(introspect(graph))


# ── what we can read, we read ──────────────────────────────────────────────


def test_an_object_nodes_prompt_tools_and_model_are_read(graph):
    solver = _by_name(introspect(graph))["solver"]

    assert solver.system_prompt == "You are the solver."
    assert [t.name for t in solver.tool_definitions] == ["lookup"]
    assert solver.tool_definitions[0].original_format == "langchain"
    assert solver.discovered_model_config.model_name == "gpt-4o-mini"
    assert solver.discovered_model_config.temperature == pytest.approx(0.3)
    assert solver.incomplete == []


def test_a_tool_node_is_read_through_tools_by_name():
    """A ToolNode is the tool surface of every prebuilt agent and keys its
    tools by name instead of listing them; reading only ``.tools`` reported
    the commonest LangGraph tool surface as unobserved."""
    ToolNode = pytest.importorskip("langgraph.prebuilt").ToolNode
    node = ToolNode([lookup])
    assert not hasattr(node, "tools"), "ToolNode grew .tools; simplify the read"

    tools_node = _by_name(introspect(_build(act=node)))["act"]
    assert [t.name for t in tools_node.tool_definitions] == ["lookup"]
    assert "tool_definitions" not in tools_node.incomplete


def test_a_node_registered_as_a_bound_method_is_read():
    """``add_node("x", obj.run)`` is ordinary in class-based builders; the
    agent is the instance, not the method."""
    class Builder:
        system_prompt = "You are the worker."
        tools: ClassVar[list] = [lookup]

        def run(self, state: _State) -> _State:
            return state

    worker = _by_name(introspect(_build(worker=Builder().run)))["worker"]
    assert worker.system_prompt == "You are the worker."
    assert [t.name for t in worker.tool_definitions] == ["lookup"]


# ── the absence contract ───────────────────────────────────────────────────


def test_an_unreadable_node_reports_None_and_names_what_it_missed(graph):
    """THE test. A function node holds its prompt in a closure cell, not an
    attribute, so there is nothing to read. Reporting ``tool_definitions=[]``
    would assert the node HAS no tools, which we never observed."""
    plain = _by_name(introspect(graph))["plain"]

    assert plain.tool_definitions is None
    assert plain.tool_definitions != []
    assert plain.system_prompt is None
    assert plain.discovered_model_config is None
    assert set(plain.incomplete) == {
        "system_prompt", "tool_definitions", "discovered_model_config"
    }


def test_a_node_that_really_has_no_tools_reports_an_empty_list():
    """The other half of the contract: we looked, and there are none. If this
    collapsed to None the operator could never tell a toolless node from an
    unread one."""
    class Toolless:
        system_prompt = "I have no tools."
        tools: ClassVar[list] = []

        def __call__(self, state: _State) -> _State:
            return state

    node = _by_name(introspect(_build(bare=Toolless())))["bare"]
    assert node.tool_definitions == []
    assert node.tool_definitions is not None
    assert "tool_definitions" not in node.incomplete


def test_the_graph_record_itself_claims_no_tools_of_its_own(graph):
    """A graph owns no tools -- its nodes do. ``[]`` on the root would be a
    fact we never measured about the customer's agent."""
    got = introspect(graph)
    assert got.tool_definitions is None
    assert got.system_prompt is None
    assert got.discovered_model_config is None


# ── it never raises ────────────────────────────────────────────────────────


def test_one_hostile_node_costs_its_own_description_and_nothing_else():
    """Introspection runs at registration, before any rollout. A customer's
    agent must start whether or not we could describe it."""
    class Hostile:
        @property
        def tools(self):
            raise RuntimeError("boom")

        def __call__(self, state: _State) -> _State:
            return state

    got = introspect(_build(hostile=Hostile()))
    nodes = _by_name(got)

    assert nodes["hostile"].tool_definitions is None
    assert "tool_definitions" in nodes["hostile"].incomplete
    # The readable node beside it is untouched.
    assert nodes["solver"].system_prompt == "You are the solver."


# ── nesting ────────────────────────────────────────────────────────────────


def test_a_subgraph_node_carries_its_own_nodes_beneath_it():
    """The supervisor pattern. The operator's id is the path through graph
    node names, so the inner node hangs under the outer node's name."""
    inner = StateGraph(_State)
    inner.add_node("summarizer", _Solver())
    inner.add_edge(START, "summarizer")
    inner.add_edge("summarizer", END)

    researcher = _by_name(introspect(_build(researcher=inner.compile())))["researcher"]

    assert [a.agent_name for a in researcher.sub_agents] == ["summarizer"]
    assert researcher.sub_agents[0].system_prompt == "You are the solver."


def test_the_depth_cap_says_it_cut_the_tree_rather_than_reporting_no_nodes():
    """A cycle must not hang registration, so the walk is bounded -- but a
    truncated graph reported as a graph with no nodes is a silent lie. The cut
    is named in ``incomplete``."""
    def wrap(inner, name):
        g = StateGraph(_State)
        g.add_node(name, inner)
        g.add_edge(START, name)
        g.add_edge(name, END)
        return g.compile()

    app = _Solver()
    for i in range(6):
        app = wrap(app, f"lvl{i}")

    node = introspect(app)
    seen = []
    while node.sub_agents:
        node = node.sub_agents[0]
        seen.append(node.agent_name)

    assert node.incomplete == ["sub_agents"], (seen, node.incomplete)
    assert node.sub_agents == []


# ── the wire ───────────────────────────────────────────────────────────────


def test_the_tree_round_trips_to_the_wire(graph):
    wire = json.loads(json.dumps(introspect(graph).to_wire()))

    assert wire["framework"] == "langgraph"
    subs = {a["agent_name"]: a for a in wire["sub_agents"]}
    assert set(subs) == {"solver", "plain"}

    assert subs["solver"]["system_prompt"] == "You are the solver."
    assert [t["name"] for t in subs["solver"]["tool_definitions"]] == ["lookup"]
    assert subs["solver"]["discovered_model_config"]["model_name"] == "gpt-4o-mini"

    # The load-bearing null survives serialisation, and the key stays present:
    # omitting it would collapse "not observed" into "no tools".
    assert "tool_definitions" in subs["plain"]
    assert subs["plain"]["tool_definitions"] is None
    assert "system_prompt" not in subs["plain"]
    assert set(subs["plain"]["incomplete"]) == {
        "system_prompt", "tool_definitions", "discovered_model_config"
    }
