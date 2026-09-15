"""``describe`` registers the same description from either SDK.

The Node plugin takes the declaration ``serve(describe=...)`` takes, as
``plugins.entries.agent-flywheel.config.describe``, through
``src/declaration.js`` -- a port of ``declaration.py`` rule for rule. This
runs both over one table. Every accepted declaration must produce the same
wire record, alone and merged over what the plugin observes of an OpenClaw
agent. Every refused one must refuse with the same sentence; only a value the
sentence quotes may differ, since each language renders it its own way.

The Node side answers the whole table in ONE child process, as
``test_prompt_decision_parity.py`` does.
"""
from __future__ import annotations

import json
import re

import pytest

from agent_flywheel import declaration
from agent_flywheel.errors import ConfigurationError
from agent_flywheel.introspection import DiscoveredAgent, ModelConfig
from tests.node_sdk import require_node_sdk, run_node

AGENT = "acme-bot"

#: What the Node plugin observes of an OpenClaw agent: a prompt and a model,
#: and no tools -- ``llm_input`` carries none. The case ``describe`` exists for.
OBSERVED_AGENT = DiscoveredAgent(
    agent_name=AGENT, framework="openclaw", system_prompt="observed",
    discovered_model_config=ModelConfig(provider="openai", model_name="gpt-4o"),
    incomplete=["tool_definitions"],
)
OBSERVED_WIRE = OBSERVED_AGENT.to_wire()


def _fn(**function):
    return {"type": "function", "function": {"name": "x", **function}}


ACCEPTED: dict[str, object] = {
    "every tool shape": {
        "system_prompt": "You resolve orders.",
        "model": "gpt-4.1",
        "tools": [
            {"type": "function", "function": {
                "name": "search_orders", "description": "Find orders.",
                "parameters": {"type": "object", "properties": {"email": {"type": "string"}}}}},
            {"name": "refund", "description": "Refund.",
             "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}}},
            {"name": "lookup", "input_schema": {}, "read_only": True},
            {"name": "legacy", "parameters": {"type": "object"}},
        ],
    },
    "read_only beside an OpenAI spec": {
        "tools": [{**_fn(parameters={}), "read_only": False}]},
    "an explicit null schema wins over parameters": {
        "tools": [{"name": "x", "input_schema": None, "parameters": {"type": "object"}}]},
    "names are stripped, descriptions coerced": {
        "tools": [{"name": "  padded  ", "description": 5},
                  {"name": "flag", "description": True}]},
    "a model mapping": {
        "model": {"provider": "openai", "model_name": "gpt-4.1",
                  "temperature": 0.2, "max_tokens": 512}},
    "an empty provider is unknown": {"model": {"model_name": "m", "provider": ""}},
    "no tools is an answer": {"tools": []},
    "a prompt alone": {"system_prompt": "Be exact."},
    "a key given as null declares nothing": {"system_prompt": None, "tools": None},
    "a long prompt is cut in code points": {"system_prompt": "\U0001F600" * 20_005},
}

REFUSED: dict[str, object] = {
    "a typo'd key": {"prompt": "typo"},
    "a tool with no name": {"tools": [{"description": "no name"}]},
    "an empty object is no name": {"tools": [{"name": {}}]},
    "a null tool": {"tools": [None]},
    "tools not a list": {"tools": "search"},
    "a list": ["not", "a", "mapping"],
    "a string": "a string",
    "a number": 5,
    "a non-string prompt": {"system_prompt": 5},
    "a numeric model": {"model": 5},
    "model inside model": {"model": {"provider": "openai", "model": "gpt-4.1"}},
    "no model_name": {"model": {"provider": "openai"}},
    "a string temperature": {"model": {"model_name": "m", "temperature": "0.2"}},
    "a boolean temperature": {"model": {"model_name": "m", "temperature": True}},
    "a fractional max_tokens": {"model": {"model_name": "m", "max_tokens": 1.5}},
    "a boolean max_tokens": {"model": {"model_name": "m", "max_tokens": True}},
    "a numeric provider": {"model": {"model_name": "m", "provider": 3}},
    "an empty model": {"model": " "},
    "a string schema": {"tools": [{"name": "x", "input_schema": "{}"}]},
    "a list of parameters": {"tools": [_fn(parameters=[])]},
    "a string read_only": {"tools": [{"name": "x", "read_only": "true"}]},
    "read_only inside function": {"tools": [_fn(read_only=True)]},
}


@pytest.fixture(scope="module")
def node() -> dict[str, dict]:
    sdk, binary = require_node_sdk()
    script = (
        "import {applyDeclaration, resolveDeclaration} from './src/declaration.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const {accepted, refused, observed, agent} = JSON.parse(raw);\n"
        "const out = {accepted: {}, refused: {}};\n"
        "for (const [name, spec] of Object.entries(accepted)) {\n"
        "  const d = resolveDeclaration(spec);\n"
        "  out.accepted[name] = {\n"
        "    alone: applyDeclaration(d, {reflected: null, agentName: agent}),\n"
        "    over: applyDeclaration(d, {reflected: observed, agentName: agent}),\n"
        "  };\n"
        "}\n"
        "for (const [name, spec] of Object.entries(refused)) {\n"
        "  try { resolveDeclaration(spec); out.refused[name] = null; }\n"
        "  catch (err) { out.refused[name] = `${err.constructor.name}: ${err.message}`; }\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    stdin = json.dumps({"accepted": ACCEPTED, "refused": REFUSED,
                        "observed": OBSERVED_WIRE, "agent": AGENT})
    return json.loads(run_node(sdk, binary, script, stdin=stdin))


def _python(spec: object) -> dict:
    declared = declaration.resolve(spec)  # type: ignore[arg-type]
    alone = declaration.apply(declared, reflected=None, agent_name=AGENT)
    over = declaration.apply(declared, reflected=OBSERVED_AGENT, agent_name=AGENT)
    return {"alone": alone.to_wire() if alone is not None else None,
            "over": over.to_wire() if over is not None else None}


@pytest.mark.parametrize("name", list(ACCEPTED))
def test_both_sdks_register_the_same_description(name, node):
    mine, theirs = _python(ACCEPTED[name]), node["accepted"][name]
    assert mine == theirs, (
        f"the two SDKs disagree about {name!r}:\n"
        f"  python: {json.dumps(mine, ensure_ascii=False)[:600]}\n"
        f"  node  : {json.dumps(theirs, ensure_ascii=False)[:600]}")


def _sentence(message: str) -> str:
    """The refusal with any quoted VALUE replaced: ``{'a': 1}`` in Python is
    ``{"a":1}`` in Node, and that difference is the renderer's, not the rule's.
    A type name (``got list``) or a key list is the rule's, and is kept."""
    return re.sub(
        r" got (?:[\[{'\"0-9-]|(?:None|null|True|true|False|false)\b).*$",
        " got <value>", message)


@pytest.mark.parametrize("name", list(REFUSED))
def test_both_sdks_refuse_with_the_same_sentence(name, node):
    with pytest.raises(ConfigurationError) as refused:
        declaration.resolve(REFUSED[name])  # type: ignore[arg-type]
    theirs = node["refused"][name]
    assert theirs is not None, f"Node ACCEPTED {name!r}, which Python refuses"
    assert theirs.startswith("ConfigurationError: "), theirs
    assert _sentence(theirs.removeprefix("ConfigurationError: ")) == _sentence(
        str(refused.value))
