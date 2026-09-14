"""Generic execution identity must not depend on Prime Agent types."""
from __future__ import annotations

import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import pytest

from agent_flywheel.contract import RolloutRequest
from agent_flywheel.errors import ConfigurationError
from agent_flywheel.execution_identity import (
    canonical_observation_sha256,
    normalize_attested_components,
)
from agent_flywheel.introspection import (
    DiscoveredAgent,
    ModelConfig,
    ToolDefinition,
)
from agent_flywheel.runner import RolloutRunner


class _Transport:
    lease_ms = 1_800_000


@dataclass(frozen=True)
class _Child:
    environment: dict[str, str]
    label: str

    @property
    def discovery_environment(self) -> dict[str, str]:
        return dict(self.environment)


class _NonPrimeSnapshot:
    adapter_name = "customer-runtime"
    execution_components: ClassVar = {"customer.release": "a" * 64}
    execution_identity_complete = True
    diagnostic_fingerprint: ClassVar = {"customer_snapshot_sha256": "b" * 64}

    def __init__(self, environment: dict[str, str] | None = None) -> None:
        self.environment = dict(environment or {})
        self.events: list[str] = []

    def __bool__(self) -> bool:
        # Adapters are protocols, not truthy sentinels. Selection must be by
        # explicit None checks so a legitimate falsey implementation survives.
        return False

    def prepare_child(self, label: str = "rollout") -> _Child:
        self.events.append(f"prepare:{label}")
        return _Child(dict(self.environment), label)

    def attest_child(self, child: _Child) -> None:
        self.events.append(f"attest:{child.label}")

    def release_child(self, child: _Child) -> None:
        self.events.append(f"release:{child.label}")


def _described() -> DiscoveredAgent:
    return DiscoveredAgent(
        agent_name="customer-agent",
        framework="customer-framework",
        system_prompt="Use the registered tools to complete the task.",
        tool_definitions=[ToolDefinition(name="read", input_schema={"type": "object"})],
        discovered_model_config=ModelConfig(provider="gateway", model_name="policy"),
    )


def _request(rollout_id: str) -> RolloutRequest:
    return RolloutRequest(
        rollout_id=rollout_id,
        turn_input="run",
        llm_url=f"https://gateway.invalid/v1/rollouts/{rollout_id}",
        api_key="single-use",
        artifacts_dir=f"rollouts/{rollout_id}",
        deadline_ts=time.time() + 60,
    )


def _configure(runner: RolloutRunner) -> str | None:
    return runner.configure_execution_identity(
        _described(),
        discovery_enabled=True,
        sdk="agent-flywheel-python/0.1.0",
        contract_version="flywheel/1",
    )


def _script(tmp_path: Path) -> str:
    path = tmp_path / "agent.sh"
    path.write_text('printf "%s" "$FAKE_EXECUTION_MARKER"\n')
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _assert_identity_and_lifecycle(
    out, runner: RolloutRunner, snapshot: _NonPrimeSnapshot, rollout_id: str,
) -> None:
    assert out.exclusion_reason is None, out.exclusion_detail
    assert out.agent_fingerprint == {
        "execution_sha256": runner.execution_sha256,
        "customer_snapshot_sha256": "b" * 64,
    }
    assert snapshot.events == [
        f"prepare:{rollout_id}", f"attest:{rollout_id}", f"release:{rollout_id}",
    ]


def test_non_prime_command_snapshot_supplies_environment_and_lifecycle(tmp_path):
    snapshot = _NonPrimeSnapshot({"FAKE_EXECUTION_MARKER": "command-child"})
    runner = RolloutRunner(
        ["sh", _script(tmp_path)],
        transport=_Transport(),
        input_shape="text",
        artifacts_dir=str(tmp_path / "artifacts"),
        execution_snapshot=snapshot,
    )
    assert _configure(runner) is not None

    out = runner.run_one(_request("rl_command"))

    assert out.final_text == "command-child"
    _assert_identity_and_lifecycle(out, runner, snapshot, "rl_command")


def test_non_prime_import_snapshot_supplies_environment_and_lifecycle(tmp_path):
    snapshot = _NonPrimeSnapshot({"FAKE_EXECUTION_MARKER": "import-child"})
    runner = RolloutRunner(
        "tests.fixtures_agent:build_environment_probe",
        transport=_Transport(),
        input_shape="text",
        artifacts_dir=str(tmp_path / "artifacts"),
        execution_snapshot=snapshot,
    )
    assert _configure(runner) is not None

    out = runner.run_one(_request("rl_import"))

    assert out.final_text == "run|import-child"
    _assert_identity_and_lifecycle(out, runner, snapshot, "rl_import")


def test_non_prime_object_snapshot_attests_lifecycle_without_global_env(tmp_path):
    class _Agent:
        def invoke(self, task: str) -> str:
            return f"handled:{task}"

    snapshot = _NonPrimeSnapshot()
    runner = RolloutRunner(
        _Agent(),
        transport=_Transport(),
        input_shape="text",
        artifacts_dir=str(tmp_path / "artifacts"),
        isolation="inprocess",
        execution_snapshot=snapshot,
    )
    assert _configure(runner) is not None

    out = runner.run_one(_request("rl_object"))

    assert out.final_text == "handled:run"
    _assert_identity_and_lifecycle(out, runner, snapshot, "rl_object")

    with_environment = _NonPrimeSnapshot({"MUST_NOT_MUTATE_PARENT": "true"})
    refused = RolloutRunner(
        _Agent(),
        transport=_Transport(),
        input_shape="text",
        artifacts_dir=str(tmp_path / "refused"),
        isolation="inprocess",
        execution_snapshot=with_environment,
    )
    assert _configure(refused) is not None
    blocked = refused.run_one(_request("rl_object_env"))
    assert blocked.exclusion_reason == "cancelled"
    assert "process-global environment is not isolated" in (blocked.exclusion_detail or "")
    assert blocked.agent_fingerprint is None


def test_no_entrypoint_gets_an_exact_identity_without_complete_adapter_evidence(tmp_path):
    class _Agent:
        def invoke(self, task: str) -> str:
            return task

    runners = [
        RolloutRunner(
            ["agent"], transport=_Transport(), input_shape="text",
            artifacts_dir=str(tmp_path / "command"),
        ),
        RolloutRunner(
            "tests.fixtures_agent:build", transport=_Transport(), input_shape="text",
            artifacts_dir=str(tmp_path / "import"),
        ),
        RolloutRunner(
            _Agent(), transport=_Transport(), input_shape="text", isolation="inprocess",
            artifacts_dir=str(tmp_path / "object"),
        ),
    ]
    assert [_configure(runner) for runner in runners] == [None, None, None]

    incomplete = _NonPrimeSnapshot()
    incomplete.execution_identity_complete = False
    runner = RolloutRunner(
        ["agent"], transport=_Transport(), input_shape="text",
        artifacts_dir=str(tmp_path / "incomplete"),
        execution_snapshot=incomplete,
        attested_execution_components={"legacy.evidence": "c" * 64},
    )
    assert _configure(runner) is None

    empty = _NonPrimeSnapshot()
    empty.execution_components = {}
    runner = RolloutRunner(
        ["agent"], transport=_Transport(), input_shape="text",
        artifacts_dir=str(tmp_path / "empty"), execution_snapshot=empty,
    )
    assert _configure(runner) is None

    invalid = _NonPrimeSnapshot()
    invalid.execution_identity_complete = "yes"
    with pytest.raises(ConfigurationError, match="must be a bool"):
        RolloutRunner(
            ["agent"], transport=_Transport(), input_shape="text",
            artifacts_dir=str(tmp_path / "invalid"), execution_snapshot=invalid,
        )


def test_core_import_does_not_load_the_optional_prime_adapter():
    code = (
        "import sys; import agent_flywheel; "
        "assert 'agent_flywheel.adapters.prime_agent' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("failure", ["prepare", "attest", "release"])
def test_adapter_lifecycle_failure_abstains_without_losing_the_report(
    tmp_path, failure,
):
    class _Broken(_NonPrimeSnapshot):
        def prepare_child(self, label="rollout"):
            if failure == "prepare":
                raise RuntimeError("customer secret must not escape")
            return super().prepare_child(label)

        def attest_child(self, child):
            super().attest_child(child)
            if failure == "attest":
                raise RuntimeError("customer secret must not escape")

        def release_child(self, child):
            super().release_child(child)
            if failure == "release":
                raise RuntimeError("customer secret must not escape")

    snapshot = _Broken({"FAKE_EXECUTION_MARKER": "ran"})
    runner = RolloutRunner(
        ["sh", _script(tmp_path)],
        transport=_Transport(),
        input_shape="text",
        artifacts_dir=str(tmp_path / failure),
        execution_snapshot=snapshot,
    )
    assert _configure(runner) is not None
    out = runner.run_one(_request(f"rl_{failure}"))
    assert out.exclusion_reason == "cancelled"
    assert out.agent_fingerprint is None
    assert "customer secret" not in (out.exclusion_detail or "")
    if failure == "prepare":
        assert out.final_text == ""
    else:
        assert out.final_text == "ran"


def test_reserved_component_name_and_cross_language_canonical_vector():
    with pytest.raises(ConfigurationError, match="must not author"):
        normalize_attested_components({"execution_sha256": "a" * 64})
    with pytest.raises(ConfigurationError, match="invalid Unicode"):
        normalize_attested_components({"bad\ud800name": "a" * 64})

    value = {
        "unicode": "😀",
        "ordered": {"\ue000": 1, "😀": 2},
        "numbers": [1, 1.0, -0.0, 0.1, 9_007_199_254_740_991],
        "values": [None, True, False, "line\n"],
    }
    assert canonical_observation_sha256(value) == (
        "d7666c3413207b73217f8b97029984e2105d4b5a9128572af42609e60486cac5"
    )
    assert canonical_observation_sha256({"zero": -0.0}) == (
        canonical_observation_sha256({"zero": 0.0})
    )
    with pytest.raises(ConfigurationError, match="exact JSON range"):
        canonical_observation_sha256({"unsafe": 9_007_199_254_740_992})
    with pytest.raises(ConfigurationError, match="invalid Unicode"):
        canonical_observation_sha256({"bad": "\ud800"})
