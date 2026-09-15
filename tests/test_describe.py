"""``describe=``: an agent that nothing can read describes itself.

``tests.fixtures_agent:build`` is a plain class -- the custom agent the Mission
Control dashboard's own snippet shows. Reflection finds no framework it knows,
so without a declaration it registers ``discovered_agent: null``, the control
plane records it ``unreadable``, and no task is ever generated for it.
"""
from __future__ import annotations

import warnings

import pytest

import agent_flywheel as fw
from agent_flywheel.errors import ConfigurationError
from agent_flywheel.introspection import DiscoveredAgent, ToolDefinition
from agent_flywheel.runner import RolloutRunner
from tests.test_production_turn_upload import (
    _attached,
    _exact_description,
    _IdentityAgent,
    _Transport,
)
from tests.test_serve_registration import captured  # noqa: F401  (fixture)

PLAIN = "tests.fixtures_agent:build"

#: The three shapes an agent already sends its model, plus a declaration.
TOOLS = [
    {"type": "function", "function": {
        "name": "search_orders", "description": "Find orders.",
        "parameters": {"type": "object", "properties": {"email": {"type": "string"}}}}},
    {"name": "refund", "description": "Refund an order.",
     "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}}},
    {"name": "lookup", "description": "Read one.", "input_schema": {}, "read_only": True},
    # OpenAI's legacy flat ``functions`` shape: the schema is ``parameters``.
    {"name": "legacy", "parameters": {"type": "object", "properties": {"q": {"type": "string"}}}},
]


def _serve(captured, **kw):  # noqa: F811
    kw.setdefault("isolation", "inprocess")
    return captured(kw.pop("entrypoint", PLAIN), **kw)


# ── a custom agent becomes describable ─────────────────────────────────────


def test_a_custom_agent_registers_what_it_declared(captured):  # noqa: F811
    spec = _serve(captured, describe={
        "system_prompt": "You resolve orders.", "tools": TOOLS, "model": "gpt-4.1"})
    d = spec["discovered_agent"]
    assert d["system_prompt"] == "You resolve orders."
    assert [t["name"] for t in d["tool_definitions"]] == [
        "search_orders", "refund", "lookup", "legacy"]
    # Each shape through the one parser: schemas kept, the declaration carried.
    assert d["tool_definitions"][0]["input_schema"]["properties"] == {"email": {"type": "string"}}
    assert d["tool_definitions"][3]["input_schema"]["properties"] == {"q": {"type": "string"}}
    assert d["tool_definitions"][2]["read_only"] is True
    assert "read_only" not in d["tool_definitions"][1]        # undeclared stays undeclared
    assert d["discovered_model_config"]["model_name"] == "gpt-4.1"
    # Provenance says who said it; no identity is anchored on a declaration.
    assert d["source_type"] == "sdk_declared"
    assert d["framework"] == "custom"
    assert "agent_execution_sha256" not in spec


def test_without_a_declaration_the_same_agent_is_unreadable(captured):  # noqa: F811
    """The control: this is the gap describe= closes."""
    assert _serve(captured)["discovered_agent"] is None


def test_a_callable_is_called_once_at_registration(captured):  # noqa: F811
    calls: list[int] = []

    def describe():
        calls.append(1)
        return {"tools": TOOLS[:1]}

    spec = _serve(captured, describe=describe)
    assert calls == [1]
    assert [t["name"] for t in spec["discovered_agent"]["tool_definitions"]] == ["search_orders"]


# ── declared wins, and only where given ────────────────────────────────────


def _reflected(monkeypatch, **over):
    """Reflection that found tools and could not read the prompt."""
    got = DiscoveredAgent(agent_name="lc", framework="langchain",
                          tool_definitions=[ToolDefinition(name="search")],
                          incomplete=["system_prompt"], **over)
    monkeypatch.setattr(fw, "introspect_agent", lambda fn: got)


def test_a_declared_prompt_fills_what_reflection_could_not_read(captured, monkeypatch):  # noqa: F811
    _reflected(monkeypatch)
    d = _serve(captured, describe={"system_prompt": "Be exact."})["discovered_agent"]
    assert d["system_prompt"] == "Be exact."
    assert [t["name"] for t in d["tool_definitions"]] == ["search"]   # reflected, kept
    assert d["framework"] == "langchain"
    assert "incomplete" not in d                   # the prompt is no longer unknown


def test_declared_tools_replace_reflected_ones_and_empty_is_an_answer(captured, monkeypatch):  # noqa: F811
    _reflected(monkeypatch)
    d = _serve(captured, describe={"tools": []})["discovered_agent"]
    assert d["tool_definitions"] == []


def test_a_key_given_as_none_declares_nothing(captured, monkeypatch):  # noqa: F811
    _reflected(monkeypatch)
    d = _serve(captured, describe={"system_prompt": None, "tools": None})["discovered_agent"]
    assert d["source_type"] == "sdk_runtime"       # nothing declared; reflection stands
    assert d["incomplete"] == ["system_prompt"]


def test_a_declaration_never_anchors_an_execution_identity(captured, monkeypatch):  # noqa: F811
    asked: list[bool] = []
    real = RolloutRunner.configure_execution_identity

    def spy(self, described, **kw):
        asked.append(kw["discovery_enabled"])
        return real(self, described, **kw)

    monkeypatch.setattr(RolloutRunner, "configure_execution_identity", spy)
    _serve(captured)
    _serve(captured, describe={"tools": TOOLS})
    assert asked == [True, False]


# ── a command agent: the declaration replaces the warm-up ──────────────────


def test_a_declared_command_agent_is_not_warmed_up(captured, monkeypatch):  # noqa: F811
    def no_warm_up(*a, **kw):
        raise AssertionError("the warm-up ran for an agent that declared itself")

    monkeypatch.setattr(fw, "_describe_via_host", no_warm_up)
    spec = captured(["python", "my_agent.py"], describe={"tools": TOOLS[:1]})
    assert [t["name"] for t in spec["discovered_agent"]["tool_definitions"]] == ["search_orders"]


# ── refusals, before anything connects ─────────────────────────────────────


@pytest.mark.parametrize("describe, match", [
    ({"prompt": "typo"}, "unknown key"),
    ({"tools": [{"description": "no name"}]}, r"describe\['tools'\]\[0\]"),
    ({"tools": "search"}, "must be a list"),
    (["not", "a", "mapping"], "must be a mapping"),
])
def test_a_bad_declaration_refuses_at_startup(captured, monkeypatch, describe, match):  # noqa: F811
    # With no control plane configured, the only way to reach THIS error is to
    # check the declaration before anything connects.
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL")
    with pytest.raises(ConfigurationError, match=match):
        _serve(captured, describe=describe)


def test_describe_with_introspect_false_is_a_contradiction(captured):  # noqa: F811
    with pytest.raises(ConfigurationError, match="introspect=False"):
        _serve(captured, describe={"tools": TOOLS}, introspect=False)


def test_the_environment_opt_out_wins_with_a_warning(captured, monkeypatch):  # noqa: F811
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    with pytest.warns(RuntimeWarning, match="PERCEPTEYE_INTROSPECT is off"):
        spec = _serve(captured, describe={"tools": TOOLS})
    assert "discovered_agent" not in spec


# ── attach() ───────────────────────────────────────────────────────────────


def test_a_production_only_agent_can_describe_itself(monkeypatch, tmp_path):
    """No entrypoint: the declaration is the whole description it registers."""
    monkeypatch.delenv("PERCEPTEYE_INTROSPECT", raising=False)
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr, adapters=[],
                   describe={"system_prompt": "p", "tools": TOOLS[1:2]}):
        pass
    d = tr.registered["discovered_agent"]
    assert d["system_prompt"] == "p"
    assert [t["name"] for t in d["tool_definitions"]] == ["refund"]


def test_attach_with_an_entrypoint_merges_and_withholds_identity(monkeypatch, tmp_path):
    """The same setup yields an execution identity without a declaration."""
    monkeypatch.setattr("agent_flywheel.introspection.introspect",
                        lambda fn: _exact_description())
    kwargs = dict(adapters=[], execution_entrypoint=_IdentityAgent(),
                  execution_input_shape="messages",
                  attested_execution_components={"customer.release": "a" * 64})
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr, **kwargs) as px:
        assert px.agent_fingerprint is not None                     # the control
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr, describe={"system_prompt": "declared"},
                   **kwargs) as px:
        assert px.agent_fingerprint is None
    assert tr.registered["discovered_agent"]["system_prompt"] == "declared"
    assert tr.registered["discovered_agent"]["source_type"] == "sdk_declared"


def test_attach_honours_the_opt_out_over_a_declaration(monkeypatch, tmp_path):
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    tr = _Transport()
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        with _attached(monkeypatch, tmp_path, tr, adapters=[], describe={"tools": TOOLS}):
            pass
    assert "discovered_agent" not in tr.registered
    assert any("PERCEPTEYE_INTROSPECT is off" in str(w.message) for w in seen)
