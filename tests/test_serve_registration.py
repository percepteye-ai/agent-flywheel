"""What ``serve()`` actually sends at registration.

This file exists because there was no test of ``serve()`` at all. Every other
test builds a ``RolloutRunner`` directly, which is why a command entrypoint --
where ``runner.resolved`` is None by construction -- crashed on
``runner.resolved.name`` at registration and blocked every non-Python agent at
line one, with a green suite.
"""
from __future__ import annotations

import re

import pytest

import agent_flywheel as fw
from agent_flywheel.contract import CONTRACT_VERSION
from agent_flywheel.introspection import (
    DiscoveredAgent,
    ModelConfig,
    ToolDefinition,
)


class _FakeTransport:
    """Records the registration and then reports an empty queue, so serve()
    completes one full startup and returns without running anything."""

    def __init__(self, client, agent_id):
        self.agent_id = agent_id
        self.registered: dict | None = None

    def health(self):
        return {"contract_versions_supported": [CONTRACT_VERSION]}

    def register(self, spec):
        self.registered = spec
        return {"agent_id": self.agent_id, "grade": "advisory", "grade_reasons": []}

    def claim(self, max_rollouts):
        return [], 0


@pytest.fixture
def captured(monkeypatch, tmp_path):
    """Run serve() to completion against a fake control plane and hand back
    whatever it registered."""
    box: dict = {}

    def _factory(client, agent_id):
        tr = _FakeTransport(client, agent_id)
        box["tr"] = tr
        return tr

    monkeypatch.setattr(fw, "AttachTransport", _factory)
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid/v1")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "perk_test")

    def _run(entrypoint, **kw):
        kw.setdefault("agent_id", "a1")
        kw.setdefault("input_shape", "text")
        kw.setdefault("artifacts_dir", str(tmp_path))
        kw.setdefault("max_rollouts", 0)
        fw.serve(entrypoint, **kw)
        return box["tr"].registered

    return _run


# ── the command-entrypoint regression ──────────────────────────────────────


def test_serve_registers_a_command_entrypoint_without_crashing(captured):
    """THE regression. `resolved` is None for a command, and reading `.name`
    off it raised AttributeError before any rollout could be claimed."""
    spec = captured(["python", "my_agent.py", "--serve"])
    assert spec is not None, "registration never happened"
    assert spec["entrypoint"] == "python my_agent.py --serve"
    assert spec["isolation"] == "subprocess"


def test_serve_registers_a_python_entrypoint_by_name(captured):
    spec = captured("tests.fixtures_agent:build", isolation="inprocess")
    assert spec["entrypoint"]
    assert spec["contract_version"] == CONTRACT_VERSION


# ── introspection: on by default, and refusable two ways ───────────────────


def test_introspection_is_on_by_default(captured):
    """The owner's call: default ON, disclosed in the README. The key is
    present even when nothing could be read."""
    spec = captured("tests.fixtures_agent:build", isolation="inprocess")
    assert "discovered_agent" in spec


def test_a_command_entrypoint_reports_discovered_agent_null(captured):
    """A separate process has no object to read. `null` is the honest answer
    and it must still be SENT -- omitting the key would make 'we could not
    look' indistinguishable from 'we looked and found nothing'."""
    spec = captured(["python", "my_agent.py"])
    assert "discovered_agent" in spec
    assert spec["discovered_agent"] is None
    assert "agent_execution_sha256" not in spec


def test_introspect_false_omits_the_key_entirely(captured):
    """Opting out is not the same as introspecting and finding nothing, so it
    does not send a null -- it sends nothing."""
    spec = captured("tests.fixtures_agent:build", isolation="inprocess",
                    introspect=False)
    assert "discovered_agent" not in spec
    assert "agent_execution_sha256" not in spec


def test_the_env_kill_switch_disables_it_for_an_operator(captured, monkeypatch):
    """For someone who cannot edit the call site."""
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    spec = captured("tests.fixtures_agent:build", isolation="inprocess")
    assert "discovered_agent" not in spec


@pytest.mark.parametrize("value", ["1", "true", "yes", "", "banana"])
def test_only_a_deliberate_value_disables_capture(captured, monkeypatch, value):
    """A typo must not silently switch introspection off and leave the
    operator believing it is on."""
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", value)
    spec = captured("tests.fixtures_agent:build", isolation="inprocess")
    assert "discovered_agent" in spec, f"{value!r} should not have disabled it"


def test_a_real_agent_is_described_in_the_registration(captured):
    """End to end through the public surface: the tool catalogue reaches the
    payload without the customer writing anything."""
    pytest.importorskip("langchain_core")
    spec = captured("tests.fixtures_agent:build_langchain", isolation="inprocess")
    described = spec["discovered_agent"]
    assert described is not None
    assert described["framework"] == "langchain"
    assert described["source_type"] == "sdk_runtime"
    assert [t["name"] for t in described["tool_definitions"]] == ["lookup"]
    # This fixture does not expose its model configuration, so description is
    # useful for workflow generation but cannot claim exact execution identity.
    assert "agent_execution_sha256" not in spec


def test_exact_execution_identity_reaches_registration(captured, monkeypatch):
    described = DiscoveredAgent(
        agent_name="customer-agent",
        framework="customer-framework",
        system_prompt="Use the registered tools.",
        tool_definitions=[
            ToolDefinition(name="lookup", input_schema={"type": "object"}),
        ],
        discovered_model_config=ModelConfig(
            provider="gateway", model_name="policy",
        ),
    )
    monkeypatch.setattr(fw, "introspect_agent", lambda _agent: described)
    spec = captured(
        "tests.fixtures_agent:build",
        isolation="inprocess",
        attested_execution_components={"agent_code_sha256": "f" * 64},
    )
    # Computed from the same framework-neutral descriptor stamped on every
    # trajectory; it is never authored by the customer agent.
    assert re.fullmatch(r"[0-9a-f]{64}", spec["agent_execution_sha256"])


def test_description_alone_does_not_claim_exact_execution_identity(captured):
    """Prompt/tool discovery cannot see customer orchestration code."""
    pytest.importorskip("langchain_core")
    spec = captured(
        "tests.fixtures_agent:build_langchain", isolation="inprocess",
    )
    assert spec["discovered_agent"] is not None
    assert "agent_execution_sha256" not in spec


def test_serve_registers_as_a_rollout_and_attach_does_not(captured):
    """The two surfaces must be distinguishable by the field that names them.

    `attach()` sends mode="production" (see
    tests/test_attach_production_capture.py). This side keeps "rollout", which
    is the value intake has always received from a training donor. The point of
    the pair is that they DIFFER -- while both said "rollout" the field carried
    zero bits and could not answer the one question it exists for.

    "rollout" rather than "training" deliberately: this is the wire, where the
    field names the protocol, and renaming a value intake already accepts would
    be a breaking change for no gain. `PERCEPTEYE_AGENT_MODE=training` is the
    customer-facing spelling of the same thing.
    """
    spec = captured("tests.fixtures_agent:build", isolation="inprocess")
    assert spec["mode"] == "rollout"
    assert spec["entrypoint"]
