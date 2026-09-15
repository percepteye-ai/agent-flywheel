"""``describe=``: an agent describing itself when nothing can read it.

serve() learns an agent's prompt and tools by reflecting on the object an
entrypoint builds (``introspection.py``) or, for a command agent, by asking its
host (``host_description.py``). A custom agent -- a plain class, the shape the
Mission Control dashboard's own snippet shows -- is neither: reflection finds
no framework it knows, registration sends ``discovered_agent: null``, the
control plane records the agent as ``unreadable``, and no task is ever
generated for it. The customer knows the answer; this is where they say it::

    fw.serve("my_package.agent:build", agent_id="acme-bot", input_shape="text",
             describe={"system_prompt": SYSTEM_PROMPT, "tools": TOOLS})

THE CUSTOMER'S WORD WINS, AND ONLY WHERE THEY GAVE IT. Declared fields replace
what reflection found; anything left out still comes from reflection, so a
LangChain agent whose prompt reflection cannot reach declares just the prompt
and keeps its reflected tools. Leaving ``tools`` out never means "no tools" --
that is ``tools=[]``.

ONE PARSER. ``tools`` is the list the agent already sends its model -- OpenAI
function specs, Anthropic tools, or the protocol's own ``{name, description,
input_schema, read_only}`` -- read by the same code that reads a tool list off
a completion request, so a declared tool and an observed one cannot disagree
about shape. ``read_only`` is the one thing only a declaration carries: it goes
at the top level of an entry, whichever shape, and a request's is never read.

NOT EVIDENCE. A declaration is what the customer says their agent does, not an
observation of what ran, so a declared description never anchors an execution
identity: the callers decline one whenever this module contributed.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from .errors import ConfigurationError
from .introspection import (
    _MAX_PROMPT_CHARS,
    DiscoveredAgent,
    ModelConfig,
    ToolDefinition,
)

#: What ``describe=`` accepts: a mapping, or a zero-argument callable returning
#: one, called once the agent is built.
Describe = Mapping[str, Any] | Callable[[], Mapping[str, Any]]

#: The keys a declaration may carry. Anything else is a typo that would
#: otherwise declare nothing in silence -- ``prompt`` for ``system_prompt``,
#: ``tool_definitions`` for ``tools`` -- so it refuses at startup instead.
_KEYS = frozenset({"system_prompt", "tools", "model"})
#: The keys ``model`` may carry as a mapping, refused otherwise for the same
#: reason: ``{"model": "gpt-4.1"}`` inside it would register ``unknown``.
_MODEL_KEYS = frozenset({"provider", "model_name", "temperature", "max_tokens"})

#: ``source_type`` of a description a declaration contributed to. A claim about
#: where the description came from: the customer said it; nothing observed it.
SOURCE_TYPE = "sdk_declared"


@dataclass(frozen=True)
class Declaration:
    """A validated ``describe=``. ``None`` fields were not declared."""

    system_prompt: str | None = None
    tools: list[ToolDefinition] | None = None
    model: ModelConfig | None = None

    @property
    def complete(self) -> bool:
        """Every part reflection or a warm-up could observe is declared, so
        observing would add nothing. Anything less still observes the rest."""
        return all(p is not None for p in (self.system_prompt, self.tools, self.model))


def resolve(describe: Describe | None) -> Declaration | None:
    """``describe=`` validated and parsed, calling it if it is a callable.

    ``None`` when nothing was declared -- including a mapping whose every value
    is None. Loud and early, because a declaration nobody reads is an agent
    registered without its tools and a run that generates nothing hours later:
    the failure this argument exists to prevent. Every tool is parsed here, so
    a static declaration refuses before serve() connects to anything. A
    callable that raises is refused the same way, naming ``describe=``.
    """
    if describe is None:
        return None
    if callable(describe):
        try:
            spec = describe()
        except ConfigurationError:
            raise
        except Exception as exc:
            raise ConfigurationError(
                f"describe= raised {type(exc).__name__}: {exc}") from exc
    else:
        spec = describe
    if not isinstance(spec, Mapping):
        raise ConfigurationError(
            f"describe= must be a mapping (or a callable returning one), got "
            f"{type(spec).__name__}"
        )
    unknown = sorted(str(k) for k in spec if k not in _KEYS)
    if unknown:
        raise ConfigurationError(
            f"describe= got unknown key(s) {unknown}; it accepts "
            f"{sorted(_KEYS)}"
        )
    prompt, tools, model = (spec.get(k) for k in ("system_prompt", "tools", "model"))
    if prompt is not None and not isinstance(prompt, str):
        raise ConfigurationError("describe['system_prompt'] must be a string")
    if tools is not None and not isinstance(tools, list):
        raise ConfigurationError(
            "describe['tools'] must be a list -- the tools you send the model")
    if model is not None and not isinstance(model, (str, Mapping)):
        raise ConfigurationError(
            "describe['model'] must be a model name or a mapping of "
            "provider / model_name / temperature / max_tokens")
    if prompt is None and tools is None and model is None:
        return None
    return Declaration(
        system_prompt=prompt[:_MAX_PROMPT_CHARS] if prompt is not None else None,
        tools=_tools(tools) if tools is not None else None,
        model=_model(model) if model is not None else None,
    )


def _tools(declared: list[Any]) -> list[ToolDefinition]:
    """The declared tools as ``ToolDefinition``s, through the wire's parser.

    Unlike an observed request, a declaration that names a tool the parser
    cannot read is the customer's mistake, and it is said at startup rather
    than dropped: a silently shorter catalogue is a smaller task universe
    nobody chose. So is a schema that is not a mapping, or a ``read_only``
    that is not a bool, or one the parser would not see -- each would
    otherwise register as an empty schema or an undeclared tool.
    """
    from .host_description import _tool_from_wire, _tools_from_request

    out = []
    for i, entry in enumerate(declared):
        where = f"describe['tools'][{i}]"
        parsed = _tools_from_request({"tools": [entry]})
        tool = _tool_from_wire(parsed[0]) if parsed else None
        if tool is None:
            raise ConfigurationError(
                f"{where} is not a tool: expected an OpenAI function spec, an "
                f"Anthropic tool, or a mapping with a 'name', got {entry!r:.120}"
            )
        schema = parsed[0]["input_schema"]
        if schema is not None and not isinstance(schema, dict):
            raise ConfigurationError(
                f"{where}: the schema must be a JSON-schema mapping, got "
                f"{type(schema).__name__}")
        fn = entry.get("function")
        if isinstance(fn, dict) and "read_only" in fn:
            raise ConfigurationError(
                f"{where}: read_only goes beside 'function', not inside it")
        read_only = entry.get("read_only")
        if read_only is not None and not isinstance(read_only, bool):
            raise ConfigurationError(
                f"{where}['read_only'] must be true or false, got {read_only!r}")
        out.append(dataclasses.replace(tool, read_only=read_only))
    return out


def _model(declared: str | Mapping[str, Any]) -> ModelConfig:
    if isinstance(declared, str):
        if not declared.strip():
            raise ConfigurationError("describe['model'] must not be empty")
        return ModelConfig(model_name=declared)
    unknown = sorted(str(k) for k in declared if k not in _MODEL_KEYS)
    if unknown:
        raise ConfigurationError(
            f"describe['model'] got unknown key(s) {unknown}; it accepts "
            f"{sorted(_MODEL_KEYS)}")
    name, provider = declared.get("model_name"), declared.get("provider")
    temperature, max_tokens = declared.get("temperature"), declared.get("max_tokens")
    if not isinstance(name, str) or not name.strip():
        raise ConfigurationError("describe['model'] needs a 'model_name' string")
    if provider is not None and not isinstance(provider, str):
        raise ConfigurationError("describe['model']['provider'] must be a string")
    if temperature is not None and (
            isinstance(temperature, bool) or not isinstance(temperature, (int, float))):
        raise ConfigurationError("describe['model']['temperature'] must be a number")
    if max_tokens is not None and (
            isinstance(max_tokens, bool) or not isinstance(max_tokens, int)):
        raise ConfigurationError("describe['model']['max_tokens'] must be an integer")
    return ModelConfig(
        provider=provider or "unknown",
        model_name=name,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=max_tokens,
    )


def apply(
    declared: Declaration | None,
    *,
    reflected: DiscoveredAgent | None,
    agent_name: str,
) -> DiscoveredAgent | None:
    """``reflected`` with the declared fields in place, the declaration alone
    when nothing was reflected, or ``reflected`` unchanged when nothing was
    declared.

    A part the declaration supplies stops being listed in ``incomplete``: it is
    no longer unknown. The identity digest of what was reflected is dropped --
    it covered an observation this description no longer is.
    """
    if declared is None:
        return reflected
    base = reflected or DiscoveredAgent(agent_name=agent_name, framework="custom")
    fields: dict[str, Any] = {"source_type": SOURCE_TYPE,
                              "exact_observation_sha256": None}
    supplied: set[str] = set()
    if declared.system_prompt is not None:
        fields["system_prompt"] = declared.system_prompt
        supplied.add("system_prompt")
    if declared.tools is not None:
        fields["tool_definitions"] = list(declared.tools)
        supplied.update({"tool_definitions", "tool"})
    if declared.model is not None:
        fields["discovered_model_config"] = declared.model
        supplied.add("discovered_model_config")
    fields["incomplete"] = [i for i in base.incomplete if i not in supplied]
    return dataclasses.replace(base, **fields)
