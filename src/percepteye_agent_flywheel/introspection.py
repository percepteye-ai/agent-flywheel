"""Agent Discovery Protocol, run from inside the agent's own process.

Percepteye's discovery protocol onboards an agent by describing it: a
connector returns a ``DiscoveredAgent`` -- system prompt, tool definitions,
model config, and the sub-agent tree -- which a workflow generator turns into
the universe of tasks worth training on. Every connector shipped so far
reaches IN from the outside: it imports a module, reads a directory, or calls
a URL.

This module is the connector that reaches OUT from the inside, and that is a
strictly better vantage point. An external connector inspects an artifact --
source, a manifest, an endpoint's advertised schema. We are already in the
process holding the CONSTRUCTED agent, so we see what was actually built: the
prompt after its template was rendered, the tools after conditional
registration, the sub-agents after the graph was assembled.

**The absence contract applies here exactly as it does to tool outcomes.**
``tool_definitions=None`` means "we could not look"; ``[]`` means "we looked
and this agent has none". Those are different claims and this module never
collapses them -- an agent we failed to introspect must not be reported as an
agent with no tools, because the second is a fact and the first is a silence.
``incomplete`` names every part we could not read, so a thin result is legible
downstream rather than merely small.

**It never raises.** Introspection runs at registration, before any rollout,
and a customer's agent must start whether or not we could describe it. Every
probe is wrapped; a failure costs us the description and nothing else.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DiscoveredAgent",
    "ModelConfig",
    "ToolDefinition",
    "introspect",
]

#: How deep to walk a sub-agent tree. Deep enough for the multi-agent graphs
#: people actually build, shallow enough that a cycle cannot hang startup.
_MAX_DEPTH = 4

#: Cap on prompt bytes we ship. A system prompt is the customer's IP and can
#: also be enormous; the workflow generator needs its shape, not a novel.
_MAX_PROMPT_CHARS = 20_000


@dataclass(frozen=True)
class ToolDefinition:
    """One tool, in the discovery protocol's tool shape."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = field(default_factory=dict)
    original_format: str = "unknown"

    def to_wire(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "original_format": self.original_format,
        }


@dataclass(frozen=True)
class ModelConfig:
    """The discovery protocol's model config. Absent fields stay absent."""

    provider: str = "unknown"
    model_name: str = "unknown"
    temperature: float | None = None
    max_tokens: int | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"provider": self.provider, "model_name": self.model_name}
        if self.temperature is not None:
            out["temperature"] = self.temperature
        if self.max_tokens is not None:
            out["max_tokens"] = self.max_tokens
        return out


@dataclass(frozen=True)
class DiscoveredAgent:
    """What we could see of the constructed agent.

    Mirrors the discovery protocol's agent record field-for-field where we
    can populate it, and declines to invent the rest. ``source_type`` is always
    ``sdk_runtime`` -- the one source type that means "the agent described
    itself from inside its own process".
    """

    agent_name: str
    framework: str
    source_type: str = "sdk_runtime"
    system_prompt: str | None = None
    #: None means NOT OBSERVED. ``[]`` means observed, and empty.
    tool_definitions: list[ToolDefinition] | None = None
    discovered_model_config: ModelConfig | None = None
    capabilities: list[str] = field(default_factory=list)
    sub_agents: list[DiscoveredAgent] = field(default_factory=list)
    role: str = "standalone"
    #: Names of the parts we could not read, e.g. ``["system_prompt"]``.
    incomplete: list[str] = field(default_factory=list)

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "agent_name": self.agent_name,
            "framework": self.framework,
            "source_type": self.source_type,
            "role": self.role,
        }
        # Absent keys rather than null ones, EXCEPT tool_definitions, whose
        # null is load-bearing: it is the difference between "we did not look"
        # and "this agent has no tools", and the consumer grades on it.
        out["tool_definitions"] = (
            None if self.tool_definitions is None
            else [t.to_wire() for t in self.tool_definitions]
        )
        if self.system_prompt is not None:
            out["system_prompt"] = self.system_prompt
        if self.discovered_model_config is not None:
            out["discovered_model_config"] = self.discovered_model_config.to_wire()
        if self.capabilities:
            out["capabilities"] = list(self.capabilities)
        if self.sub_agents:
            out["sub_agents"] = [a.to_wire() for a in self.sub_agents]
        if self.incomplete:
            out["incomplete"] = list(self.incomplete)
        return out


# ── probes ──────────────────────────────────────────────────────────────────
#
# Each probe answers "is this my framework's agent?" and, if so, reads what it
# can. They are tried in order and the first match wins. A probe NEVER imports
# its framework at module scope -- an absent framework simply does not match.


def _safe(fn, *a: Any, **kw: Any) -> Any:
    """Run a probe step; a failure is an absence, never an exception."""
    try:
        return fn(*a, **kw)
    except Exception:
        return None


def _text(value: Any) -> str | None:
    """A prompt as a string, or None. ADK's ``instruction`` may be a CALLABLE
    (resolved per-invocation from state), and calling it here would run
    customer code at registration with no invocation context -- so a callable
    instruction is reported as not-observed rather than invoked."""
    if isinstance(value, str):
        return value[:_MAX_PROMPT_CHARS] if value else None
    return None


def _probe_adk(obj: Any, depth: int) -> DiscoveredAgent | None:
    """Google ADK. The closest thing to a 1:1 mapping we have: ``LlmAgent``
    declares ``instruction``, ``tools``, ``sub_agents``, ``model``, ``name``
    and ``description`` as model fields."""
    try:
        from google.adk.agents import BaseAgent
    except Exception:
        return None
    if not isinstance(obj, BaseAgent):
        return None

    incomplete: list[str] = []
    name = _safe(lambda: str(getattr(obj, "name", "") or "")) or "agent"

    prompt = _text(getattr(obj, "instruction", None))
    if prompt is None:
        prompt = _text(getattr(obj, "global_instruction", None))
    if prompt is None and hasattr(obj, "instruction"):
        # Present but not a plain string -- a callable resolved per turn.
        incomplete.append("system_prompt")

    tools = None
    raw_tools = getattr(obj, "tools", None)
    if raw_tools is not None:
        tools = []
        for t in raw_tools:
            d = _safe(_adk_tool, t)
            if d is not None:
                tools.append(d)
            else:
                incomplete.append("tool")

    model = _safe(_adk_model, obj)

    subs: list[DiscoveredAgent] = []
    if depth < _MAX_DEPTH:
        for child in (getattr(obj, "sub_agents", None) or []):
            got = _introspect_obj(child, depth + 1)
            if got is not None:
                subs.append(dataclasses.replace(got, role="sub_agent"))

    desc = _safe(lambda: str(getattr(obj, "description", "") or ""))
    return DiscoveredAgent(
        agent_name=name,
        framework="google_adk",
        system_prompt=prompt,
        tool_definitions=tools,
        discovered_model_config=model,
        capabilities=[desc] if desc else [],
        sub_agents=subs,
        role="orchestrator" if subs else "standalone",
        incomplete=incomplete,
    )


def _adk_tool(t: Any) -> ToolDefinition | None:
    """An ADK tool's OWN declaration, never a guess at its signature.

    ``LlmAgent.tools`` holds whatever was passed in, so a bare callable stays a
    bare function -- ADK wraps it lazily in ``canonical_tools``, which is async
    and wants a ``ReadonlyContext`` we do not have at registration. So we do
    the wrapping ourselves with ADK's own ``FunctionTool`` and read the
    declaration it derives. Deriving a JSON schema from the signature here
    instead would be a second implementation of ADK's rules, guaranteed to
    disagree with it eventually.
    """
    if not hasattr(t, "name") and callable(t):
        from google.adk.tools.function_tool import FunctionTool
        t = FunctionTool(t)

    name = getattr(t, "name", None)
    if not name:
        return None

    schema: dict[str, Any] = {}
    decl = _safe(lambda: t._get_declaration())  # ADK's own accessor
    if decl is not None:
        dumped = _safe(lambda: decl.model_dump(exclude_none=True))
        if isinstance(dumped, dict):
            # Newer ADK emits ``parameters_json_schema``; older emits a typed
            # ``parameters``. Take whichever is populated rather than pinning
            # to one and silently shipping an empty schema on the other.
            schema = (
                dumped.get("parameters_json_schema")
                or dumped.get("parameters")
                or {}
            )
    return ToolDefinition(
        name=str(name),
        description=str(getattr(t, "description", "") or ""),
        input_schema=schema if isinstance(schema, dict) else {},
        original_format="google_adk",
    )


def _adk_model(obj: Any) -> ModelConfig | None:
    m = getattr(obj, "model", None)
    if m is None:
        return None
    # ``model`` is either a string id or a BaseLlm carrying one.
    name = m if isinstance(m, str) else getattr(m, "model", None)
    if not name:
        return None
    return ModelConfig(provider="google_adk", model_name=str(name))


def _probe_langchain(obj: Any, depth: int) -> DiscoveredAgent | None:
    """LangChain. Far less uniform than ADK: there is no single agent base
    class, so we match on the tool surface instead and read what is reachable.

    NOT LangGraph -- this docstring used to claim it and the claim was false:
    a compiled graph has neither ``tools`` nor ``get_tools``, so this probe
    returned None for every one of them. ``_probe_langgraph`` handles those
    and runs first.
    """
    try:
        from langchain_core.tools import BaseTool
    except Exception:
        return None

    raw = getattr(obj, "tools", None)
    if raw is None:
        raw = _safe(lambda: obj.get_tools())
    if raw is None:
        return None
    if not any(isinstance(t, BaseTool) for t in raw):
        return None

    incomplete: list[str] = []
    tools: list[ToolDefinition] = []
    for t in raw:
        d = _safe(_langchain_tool, t)
        if d is not None:
            tools.append(d)
        else:
            incomplete.append("tool")

    prompt = _text(getattr(obj, "system_prompt", None)) or _text(
        getattr(obj, "system_message", None)
    )
    if prompt is None:
        incomplete.append("system_prompt")

    return DiscoveredAgent(
        agent_name=_safe(lambda: type(obj).__name__) or "agent",
        framework="langchain",
        system_prompt=prompt,
        tool_definitions=tools,
        discovered_model_config=_safe(_langchain_model, obj),
        incomplete=incomplete,
    )


def _langchain_tool(t: Any) -> ToolDefinition | None:
    name = getattr(t, "name", None)
    if not name:
        return None
    args = _safe(lambda: t.args)
    schema = (
        {"type": "object", "properties": args} if isinstance(args, dict) else {}
    )
    return ToolDefinition(
        name=str(name),
        description=str(getattr(t, "description", "") or ""),
        input_schema=schema,
        original_format="langchain",
    )


def _langchain_model(obj: Any) -> ModelConfig | None:
    llm = getattr(obj, "llm", None) or getattr(obj, "model", None)
    if llm is None:
        return None
    name = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if not name:
        return None
    return ModelConfig(
        provider=type(llm).__name__,
        model_name=str(name),
        temperature=_safe(lambda: float(llm.temperature)),
    )


# ── LangGraph ───────────────────────────────────────────────────────────────
#
# A compiled graph is NOT a LangChain agent, and probing it as one discovered
# nothing at all: `_probe_langchain` opens with ``obj.tools`` then
# ``obj.get_tools()``, and a ``CompiledStateGraph`` has neither, so the probe
# returned None and a LangGraph customer was onboarded with no description.
# The graph is the more specific shape, so it is tried FIRST.
#
# The unit of description here is the graph NODE: one record per user node,
# carried as ``sub_agents`` of a record describing the graph. A node's name is
# the operator-facing id, so it is taken from the graph rather than from the
# implementing class -- the id the operator picks must be the id the graph
# uses.

#: Attributes the unwrap hops through to reach a node's implementation.
#: LangGraph wraps every node (``PregelNode`` -> ``RunnableCallable`` -> the
#: callable it was given), and a class-based node is often registered as a
#: bound method, whose ``__self__`` is the object actually holding the prompt
#: and tools -- the same "the agent is the instance, not the method" unwrap
#: ``introspect`` already does for the entrypoint.
_NODE_UNWRAP_KEYS = (
    "func", "bound", "node", "steps", "runnable", "_func", "__self__",
)

#: What marks an object as a node's agent surface rather than one more
#: wrapper. Measured against langgraph 1.2: none of PregelNode,
#: RunnableCallable, RunnableSeq or ChannelWrite carries any of these, so the
#: walk cannot mistake plumbing for the customer's node.
_NODE_SURFACE_ATTRS = (
    "system_prompt", "system_message", "tools", "tools_by_name", "llm", "model",
)

#: Hops the unwrap takes before giving up. ``seen`` already stops cycles; this
#: stops a pathological chain from recursing into Python's own limit.
_MAX_UNWRAP_HOPS = 8


def _is_pregel(obj: Any) -> bool:
    """True for a compiled LangGraph graph.

    ``Pregel`` rather than ``CompiledStateGraph`` because it is the base every
    compiled form shares, so a graph built through the functional API matches
    too. An absent langgraph is simply not a match.
    """
    try:
        from langgraph.pregel import Pregel
    except Exception:
        return False
    return isinstance(obj, Pregel)


def _unwrap_node(node: Any, hops: int = 0, seen: set[int] | None = None) -> Any | None:
    """The object behind a graph node, or None when nothing readable is there.

    None is the correct answer for a plain function or a closure: a closure
    holds its prompt in a cell, not an attribute, and there is nothing to
    read. Reporting that as an agent with no tools would be a claim we never
    observed.
    """
    if seen is None:
        seen = set()
    if node is None or hops > _MAX_UNWRAP_HOPS or id(node) in seen:
        return None
    seen.add(id(node))
    if _is_pregel(node) or any(hasattr(node, a) for a in _NODE_SURFACE_ATTRS):
        return node
    for key in _NODE_UNWRAP_KEYS:
        child = getattr(node, key, None)
        if child is None:
            continue
        for candidate in (child if isinstance(child, (list, tuple)) else (child,)):
            got = _unwrap_node(candidate, hops + 1, seen)
            if got is not None:
                return got
    return None


def _unreadable_node(name: str) -> DiscoveredAgent:
    """A node whose implementation we could not reach.

    Every surface is None and named in ``incomplete``. Emitting ``[]`` for the
    tools here is the specific bug the absence contract exists to prevent: it
    would report "this node has no tools" when what happened is that we never
    got to look.
    """
    return DiscoveredAgent(
        agent_name=name,
        framework="langgraph",
        system_prompt=None,
        tool_definitions=None,
        discovered_model_config=None,
        role="sub_agent",
        incomplete=["system_prompt", "tool_definitions", "discovered_model_config"],
    )


def _langgraph_nodes(graph: Any, depth: int) -> list[DiscoveredAgent]:
    """One record per USER node of a compiled graph, at ``depth``.

    ``__start__`` and ``__end__`` are LangGraph's own channel plumbing, not
    customer agents; leaking them would put nodes in the operator's picker
    that cannot be optimised because they do not exist.
    """
    out: list[DiscoveredAgent] = []
    for raw_name, node in (_safe(lambda: dict(graph.nodes)) or {}).items():
        name = str(raw_name)
        if name.startswith("__"):
            continue
        # Per NODE, so one node with a property that raises costs that node's
        # description rather than the whole graph's.
        got = _safe(_langgraph_node, name, node, depth)
        out.append(got if got is not None else _unreadable_node(name))
    return out


def _langgraph_node(name: str, node: Any, depth: int) -> DiscoveredAgent:
    target = _unwrap_node(node)
    if target is None:
        return _unreadable_node(name)

    # A node can be an entire compiled graph -- the supervisor pattern. Its
    # own nodes hang below it, so the operator's id stays the path through
    # graph node names ("researcher/summarizer").
    if _is_pregel(target):
        if depth < _MAX_DEPTH:
            return DiscoveredAgent(
                agent_name=name,
                framework="langgraph",
                role="sub_agent",
                sub_agents=_langgraph_nodes(target, depth + 1),
            )
        # Say the tree was cut rather than reporting a graph with no nodes.
        return DiscoveredAgent(
            agent_name=name,
            framework="langgraph",
            role="sub_agent",
            incomplete=["sub_agents"],
        )

    incomplete: list[str] = []

    prompt = _text(getattr(target, "system_prompt", None)) or _text(
        getattr(target, "system_message", None)
    )
    if prompt is None:
        incomplete.append("system_prompt")

    raw_tools = getattr(target, "tools", None)
    if raw_tools is None:
        # A ToolNode -- the tool surface of every prebuilt agent -- keys its
        # tools by name rather than listing them.
        by_name = getattr(target, "tools_by_name", None)
        if isinstance(by_name, dict):
            raw_tools = list(by_name.values())

    tools: list[ToolDefinition] | None = None
    listed = None if raw_tools is None else _safe(list, raw_tools)
    if listed is None:
        # No tool surface, or one we could not read. Either way: not observed.
        incomplete.append("tool_definitions")
    else:
        tools = []
        for t in listed:
            d = _safe(_langchain_tool, t)
            if d is not None:
                tools.append(d)
            else:
                incomplete.append("tool")

    model = _safe(_langchain_model, target)
    if model is None:
        incomplete.append("discovered_model_config")

    return DiscoveredAgent(
        agent_name=name,
        framework="langgraph",
        system_prompt=prompt,
        tool_definitions=tools,
        discovered_model_config=model,
        role="sub_agent",
        incomplete=incomplete,
    )


def _probe_langgraph(obj: Any, depth: int) -> DiscoveredAgent | None:
    """A compiled LangGraph graph, described one user node at a time."""
    if not _is_pregel(obj):
        return None

    name = _safe(lambda: str(getattr(obj, "name", "") or "")) or "graph"
    if depth < _MAX_DEPTH:
        subs, incomplete = _langgraph_nodes(obj, depth + 1), []
    else:
        subs, incomplete = [], ["sub_agents"]

    return DiscoveredAgent(
        agent_name=name,
        framework="langgraph",
        # The graph owns no prompt, tools or model of its own -- its nodes do,
        # and they are the sub_agents below. None rather than ``[]`` because
        # "this agent has no tools" is not something we observed about it.
        system_prompt=None,
        tool_definitions=None,
        discovered_model_config=None,
        sub_agents=subs,
        role="orchestrator" if subs else "standalone",
        incomplete=incomplete,
    )


# Order matters: a compiled graph is the more specific shape, and probing it
# as a plain LangChain agent finds nothing.
_PROBES = (_probe_adk, _probe_langgraph, _probe_langchain)


def _introspect_obj(obj: Any, depth: int = 0) -> DiscoveredAgent | None:
    for probe in _PROBES:
        got = _safe(probe, obj, depth)
        if got is not None:
            return got
    return None


def introspect(entrypoint_obj: Any) -> DiscoveredAgent | None:
    """Describe the constructed agent, or return None having tried.

    ``None`` is a real answer and the honest one for a framework we do not
    recognise or a command entrypoint (a separate process has no object to
    read). The caller reports the absence rather than substituting an empty
    agent, because "we could not describe this" and "this agent has nothing"
    are different facts.
    """
    if entrypoint_obj is None:
        return None
    # A bound method is the usual shape: ``build_agent()`` returned an
    # instance and the runner resolved one of its methods. The agent is the
    # instance, not the method.
    obj = getattr(entrypoint_obj, "__self__", entrypoint_obj)
    return _introspect_obj(obj, 0)
