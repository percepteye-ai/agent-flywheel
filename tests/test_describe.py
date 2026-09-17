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
from agent_flywheel.declaration import Declaration
from agent_flywheel.errors import ConfigurationError
from agent_flywheel.introspection import DiscoveredAgent, ModelConfig, ToolDefinition
from agent_flywheel.runner import RolloutRunner
from tests import fixtures_agent
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
    # Provenance says who said it. (Identity: see the spy test below.)
    assert d["source_type"] == "sdk_declared"
    assert d["framework"] == "custom"


def test_read_only_is_read_beside_any_shape(captured):  # noqa: F811
    """OpenAI's spec has no room for it inside ``function``; beside it, it is
    the customer's declaration, whichever shape the entry is."""
    spec = _serve(captured, describe={"tools": [
        {"type": "function", "function": {"name": "s", "parameters": {}}, "read_only": True}]})
    assert spec["discovered_agent"]["tool_definitions"][0]["read_only"] is True


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


def test_the_callable_runs_once_the_agent_is_built(captured, monkeypatch, tmp_path):  # noqa: F811
    """So it can read tools the build registers -- in serve() and attach()."""
    seen: list[list[str]] = []

    def describe():
        seen.append(list(fixtures_agent.BUILT))
        return {"tools": TOOLS[:1]}

    fixtures_agent.BUILT.clear()
    _serve(captured, entrypoint="tests.fixtures_agent:build_recorded", describe=describe)
    fixtures_agent.BUILT.clear()
    with _attached(monkeypatch, tmp_path, _Transport(), adapters=[], describe=describe,
                   execution_entrypoint="tests.fixtures_agent:build_recorded"):
        pass
    assert len(seen) == 2 and all(seen), f"a callable ran before the build: {seen}"


def test_a_callable_is_not_run_when_consent_is_off(captured, monkeypatch, tmp_path):  # noqa: F811
    """Its description will not be sent, so nothing of the customer's runs."""
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    calls: list[int] = []

    def describe():
        calls.append(1)
        return {"tools": TOOLS}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _serve(captured, describe=describe)
        with _attached(monkeypatch, tmp_path, _Transport(), adapters=[], describe=describe):
            pass
    assert calls == []


# ── declared wins, and only where given ────────────────────────────────────


def _reflected(monkeypatch, incomplete=("system_prompt",)):
    """Reflection that found tools and could not read the prompt."""
    got = DiscoveredAgent(agent_name="lc", framework="langchain",
                          tool_definitions=[ToolDefinition(name="search")],
                          incomplete=list(incomplete))
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


def test_a_whole_float_is_a_max_tokens(captured):  # noqa: F811
    """`1024.0` in a JSON config is 1024 to the Node plugin, so it is here."""
    d = _serve(captured, describe={"model": {"model_name": "m", "max_tokens": 1024.0}})
    assert d["discovered_agent"]["discovered_model_config"]["max_tokens"] == 1024
    assert isinstance(d["discovered_agent"]["discovered_model_config"]["max_tokens"], int)


def test_each_declared_part_stops_being_unknown(captured, monkeypatch):  # noqa: F811
    _reflected(monkeypatch, incomplete=[
        "system_prompt", "tool_definitions", "tool", "discovered_model_config"])
    d = _serve(captured, describe={"tools": TOOLS[:1], "model": "gpt-4.1"})["discovered_agent"]
    assert d["incomplete"] == ["system_prompt"]      # the one part not declared


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


# ── a command agent: a whole declaration replaces the warm-up ──────────────

COMPLETE = {"system_prompt": "p", "tools": TOOLS[:1], "model": "gpt-4.1"}


def test_a_fully_declared_command_agent_is_not_warmed_up(captured, monkeypatch):  # noqa: F811
    def no_warm_up(*a, **kw):
        raise AssertionError("the warm-up ran for an agent that declared all of itself")

    monkeypatch.setattr(fw, "_describe_via_host", no_warm_up)
    spec = captured(["python", "my_agent.py"], describe=COMPLETE)
    assert [t["name"] for t in spec["discovered_agent"]["tool_definitions"]] == ["search_orders"]


@pytest.mark.parametrize("declare", [{"model": "gpt-4.1-mini"}, {"system_prompt": "Be exact."}])
def test_a_partly_declared_command_agent_is_warmed_up_for_the_rest(captured, monkeypatch, declare):  # noqa: F811
    """Skipping the warm-up for any declaration registered a declared model
    with no tools and no prompt -- what the host would have observed, lost."""
    host = DiscoveredAgent(agent_name="a", framework="goose", source_type="host_wire",
                           system_prompt="observed", tool_definitions=[
                               ToolDefinition(name="read"), ToolDefinition(name="write")],
                           discovered_model_config=ModelConfig(model_name="gpt-4o"))
    handed: list = []

    def warm_up(*a, declared=None, **kw):
        handed.append(declared)
        return host

    monkeypatch.setattr(fw, "_describe_via_host", warm_up)
    d = captured(["python", "my_agent.py"], describe=declare)["discovered_agent"]
    assert [t["name"] for t in d["tool_definitions"]] == ["read", "write"]
    assert d["system_prompt"] == declare.get("system_prompt", "observed")
    assert d["discovered_model_config"]["model_name"] == declare.get("model", "gpt-4o")
    assert handed and handed[0] is not None     # the warning knows what was declared


def test_a_failed_warm_up_says_what_registers_instead(monkeypatch, tmp_path):
    """Its 'will register as unreadable' was false beside a declaration."""
    monkeypatch.setattr(fw, "warm_up_command", lambda *a, **kw: None)

    def said(declared):
        with pytest.warns(RuntimeWarning) as seen:
            fw._describe_via_host(["agent"], artifacts_dir=str(tmp_path), agent_id="a",
                                  declared=declared)
        return str(seen[0].message)

    assert "introspection_state=unreadable" in said(None)             # the control
    prompt_only = said(Declaration(system_prompt="p"))
    assert "unreadable" not in prompt_only
    assert "registers only what describe= declared; with no tools" in prompt_only
    assert "no workflows" not in said(Declaration(tools=[ToolDefinition(name="t")]))


# ── refusals, before anything connects ─────────────────────────────────────


def _fn(name="x", **fn):
    return {"type": "function", "function": {"name": name, **fn}}


@pytest.mark.parametrize("describe, match", [
    ({"prompt": "typo"}, "unknown key"),
    ({"tools": [{"description": "no name"}]}, r"describe\['tools'\]\[0\]"),
    ({"tools": "search"}, "must be a list"),
    (["not", "a", "mapping"], "must be a mapping"),
    ({"system_prompt": 5}, "must be a string"),
    ({"model": 5}, "must be a model name"),
    # Inside ``model``: each would otherwise register ``unknown`` or drop a
    # field, over whatever reflection found.
    ({"model": {"provider": "openai", "model": "gpt-4.1"}}, "unknown key"),
    ({"model": {"provider": "openai"}}, "needs a 'model_name'"),
    ({"model": {"model_name": "m", "temperature": "0.2"}}, "temperature'\\] must be a number"),
    # An infinity would go out as `Infinity`, which is not JSON; a huge int
    # overflowed float() with an OverflowError instead of this refusal.
    ({"model": {"model_name": "m", "temperature": float("inf")}}, "must be a finite number"),
    ({"model": {"model_name": "m", "temperature": 10**400}}, "must be a finite number"),
    ({"model": {"model_name": "m", "max_tokens": 1.5}}, "max_tokens'\\] must be a whole number"),
    ({"model": {"model_name": "m", "max_tokens": 0}}, "max_tokens'\\] must be a whole number"),
    # Beyond 2**53 the Node plugin, reading the same JSON, holds another number.
    ({"model": {"model_name": "m", "max_tokens": 2**53}}, "max_tokens'\\] must be a whole number"),
    ({"model": {"model_name": "m", "provider": 3}}, "provider'\\] must be a string"),
    ({"model": " "}, "must not be empty"),
    # Inside a tool: each would otherwise register an empty schema or an
    # undeclared tool.
    ({"tools": [{"name": "x", "input_schema": "{}"}]}, "must be a JSON-schema mapping"),
    ({"tools": [_fn(parameters=[])]}, "must be a JSON-schema mapping"),
    ({"tools": [{"name": "x", "read_only": "true"}]}, "must be true or false"),
    ({"tools": [_fn(read_only=True)]}, "beside 'function'"),
    # Coerced, a name or description became text no two languages agree on.
    ({"tools": [{"name": 5}]}, "'name' must be a string, got number"),
    ({"tools": [_fn(description=["a"])]}, "'description' must be a string, got array"),
])
def test_a_bad_declaration_refuses_at_startup(captured, monkeypatch, describe, match):  # noqa: F811
    # With no control plane configured, the only way to reach THIS error is to
    # check the declaration before anything connects.
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL")
    with pytest.raises(ConfigurationError, match=match):
        _serve(captured, describe=describe)


def test_a_callable_that_raises_is_refused_naming_describe(captured):  # noqa: F811
    def describe():
        return {"system_prompt": {}["SYSTEM_PROMPT"]}

    with pytest.raises(ConfigurationError, match="describe= raised KeyError"):
        _serve(captured, describe=describe)


def test_a_typo_refuses_whatever_the_opt_out_says(captured, monkeypatch, tmp_path):  # noqa: F811
    """serve() and attach() alike: lifting the opt-out later must not turn the
    same line into a refusal to start."""
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    with pytest.raises(ConfigurationError, match="unknown key"):
        _serve(captured, describe={"prompt": "typo"})
    with pytest.raises(ConfigurationError, match="unknown key"):
        fw.attach(agent_id="support", trajectory_root=str(tmp_path),
                  describe={"prompt": "typo"})


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


def test_that_warning_never_fails_attach_under_W_error(monkeypatch, tmp_path):
    """attach() must never fail a customer's startup over a warning."""
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    tr = _Transport()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with _attached(monkeypatch, tmp_path, tr, adapters=[], describe={"tools": TOOLS}):
            pass
    assert "discovered_agent" not in tr.registered


class _Bound:
    bound = ("fake",)


def _spy_binding(monkeypatch) -> tuple[list[int], list[int]]:
    bound, undone = [], []

    def bind_selected(*a, **kw):
        bound.append(1)
        return _Bound(), [lambda: undone.append(1)]

    monkeypatch.setattr("agent_flywheel.binding.bind_selected", bind_selected)
    return bound, undone


def test_attach_refuses_a_bad_declaration_before_binding_anything(monkeypatch, tmp_path):
    bound, _ = _spy_binding(monkeypatch)
    with pytest.raises(ConfigurationError, match="unknown key"):
        fw.attach(agent_id="support", trajectory_root=str(tmp_path),
                  describe={"prompt": "typo"})
    assert bound == []


def test_a_callable_that_raises_leaves_no_adapter_bound(monkeypatch, tmp_path):
    """The global binding is undone, as for a bad entrypoint."""
    monkeypatch.delenv("PERCEPTEYE_INTROSPECT", raising=False)
    bound, undone = _spy_binding(monkeypatch)

    def describe():
        raise KeyError("SYSTEM_PROMPT")

    with pytest.raises(ConfigurationError, match="describe= raised KeyError"):
        fw.attach(agent_id="support", trajectory_root=str(tmp_path), describe=describe)
    assert bound == [1] and undone == [1]
