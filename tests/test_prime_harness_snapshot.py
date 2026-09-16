"""Prime's global continual harness cannot vary inside an OPSD run."""
from __future__ import annotations

import concurrent.futures
import os
import re
import shlex
import stat
import time
from pathlib import Path

import pytest

import agent_flywheel as fw
import agent_flywheel.runner as runner_module
from agent_flywheel.adapters.prime_agent import PrimeHarnessSnapshot
from agent_flywheel.contract import RolloutOutput, RolloutRequest
from agent_flywheel.errors import ConfigurationError
from agent_flywheel.host_description import parse_discovered_agent
from agent_flywheel.introspection import (
    DiscoveredAgent,
    ModelConfig,
    ToolDefinition,
)
from agent_flywheel.isolation import build_child_env
from agent_flywheel.runner import RolloutRunner


class _Transport:
    lease_ms = 1_800_000


#: The settings.json every child must get, SPELLED OUT HERE rather than read
#: from the constant production writes.
#:
#: These assertions once pinned this literal; they were changed to compare
#: against ``prime_agent._FIXED_SETTINGS`` itself, which cannot fail -- the
#: writer and the expectation became the same object, so any wrong change to
#: that constant stays green. This file is the only thing standing between a
#: typo in those bytes and an OPSD run whose harness identity moved: the digest
#: of these exact bytes is attested into the agent's frozen execution identity,
#: and the provider ceiling inside them is what stops one wedged call eating an
#: episode. 306000 is ``harness_call_ceiling_s(894.0) * 1000`` -- if the
#: arithmetic behind it changes deliberately, change this literal deliberately
#: too, and know that the sealed identity moved with it.
_EXPECTED_SETTINGS = (
    b'{"autoRefine":{"enabled":false,"compact":false},'
    b'"telemetry":{"enabled":false,"noticeShown":true},'
    b'"retry":{"enabled":false,"maxRetries":0,"provider":'
    b'{"timeoutMs":306000,"maxRetries":0,"maxRetryDelayMs":60000}}}\n'
)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source-harness"
    (source / "skills").mkdir(parents=True, exist_ok=True)
    (source / "harness_state.json").write_text('{"schema":1,"entries":{}}')
    (source / "skills" / "opsd.md").write_text("fixed tactic\n")
    return source


def _snapshot(tmp_path: Path) -> PrimeHarnessSnapshot:
    kernel = tmp_path / "kernel-venv"
    kernel.mkdir(exist_ok=True)
    (kernel / "runtime.txt").write_text("fixed runtime\n")
    (kernel / "lib").mkdir(exist_ok=True)
    (kernel / "lib" / "package.py").write_text("VERSION = 1\n")
    (kernel / "lib64").symlink_to("lib", target_is_directory=True)
    runtime = tmp_path / "prime-runtime"
    runtime.mkdir(exist_ok=True)
    (runtime / "bundle.js").write_text("export const version = 1;\n")
    package_manifest = tmp_path / "package.json"
    package_manifest.write_text('{"name":"prime-agent","version":"test"}\n')
    route = tmp_path / "route.ts"
    route.write_text("export const route = 'gateway';\n")
    return PrimeHarnessSnapshot(
        _source(tmp_path),
        workspace_dir=tmp_path / "runs",
        kernel_venv=kernel,
        execution_artifacts={
            "prime_distribution": runtime,
            "prime_package_manifest": package_manifest,
            "route_extension": route,
        },
        route_configuration={
            "model_id": "policy",
            "context_window": "65536",
            "max_tokens": "4096",
        },
    )


def _described(
    prompt: str = "You are the fixed test agent.",
    *,
    sub_agents: list[DiscoveredAgent] | None = None,
) -> DiscoveredAgent:
    return DiscoveredAgent(
        agent_name="test-agent",
        framework="test",
        system_prompt=prompt,
        tool_definitions=[ToolDefinition(name="read", input_schema={"type": "object"})],
        discovered_model_config=ModelConfig(provider="gateway", model_name="policy"),
        sub_agents=sub_agents or [],
    )


def _fingerprint(
    snapshot: PrimeHarnessSnapshot, runner: RolloutRunner,
) -> dict[str, str]:
    assert snapshot.kernel_sha256 is not None
    assert runner.execution_sha256 is not None
    return {
        "execution_sha256": runner.execution_sha256,
        "harness_sha256": snapshot.harness_sha256,
        "kernel_sha256": snapshot.kernel_sha256,
    }


def _script(tmp_path: Path, body: str) -> str:
    path = tmp_path / "agent.sh"
    path.write_text("#!/bin/sh\nset -eu\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def _request(rollout_id: str = "rl_h", task: str = "run") -> RolloutRequest:
    return RolloutRequest(
        rollout_id=rollout_id,
        turn_input=task,
        llm_url=f"https://gateway.invalid/v1/rollouts/{rollout_id}",
        api_key="single-use",
        artifacts_dir=f"rollouts/{rollout_id}",
        deadline_ts=time.time() + 60,
    )


def _expired_request() -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_expired",
        turn_input="run",
        llm_url="https://gateway.invalid/v1/rollouts/rl_expired",
        api_key="single-use",
        artifacts_dir="rollouts/rl_expired",
        deadline_ts=time.time() - 1,
    )


def _runner(
    tmp_path: Path, snapshot: PrimeHarnessSnapshot, body: str,
) -> RolloutRunner:
    runner = RolloutRunner(
        ["sh", _script(tmp_path, body)],
        transport=_Transport(),
        input_shape="text",
        artifacts_dir=str(tmp_path / "artifacts"),
        execution_snapshot=snapshot,
    )
    runner.configure_execution_identity(
        _described(),
        discovery_enabled=True,
        sdk="agent-flywheel-python/0.1.0",
        contract_version="flywheel/1",
    )
    return runner


def test_snapshot_digest_is_canonical_and_children_are_private(tmp_path):
    snapshot = _snapshot(tmp_path)
    try:
        assert re.fullmatch(r"[0-9a-f]{64}", snapshot.harness_sha256)
        assert snapshot.kernel_sha256 is not None
        assert re.fullmatch(r"[0-9a-f]{64}", snapshot.kernel_sha256)
        assert all(
            re.fullmatch(r"[0-9a-f]{64}", digest)
            for digest in snapshot.execution_components.values()
        )
        first = snapshot.prepare_child("one")
        second = snapshot.prepare_child("two")
        try:
            assert first.home_dir != second.home_dir
            assert first.agent_dir != second.agent_dir
            assert first.environment["HOME"] == str(first.home_dir)
            assert first.environment["PRIME_AGENT_CODING_AGENT_DIR"] == str(
                first.agent_dir
            )
            assert first.environment["PRIME_AGENT_KERNEL_VENV"].endswith("kernel-venv")
            assert first.harness_sha256 == second.harness_sha256 == snapshot.harness_sha256
            assert (first.harness_dir / "skills" / "opsd.md").read_text() == "fixed tactic\n"
            assert first.settings_path.read_bytes() == _EXPECTED_SETTINGS
            assert not (first.settings_path.stat().st_mode & 0o222)
            assert not ((first.harness_dir / "harness_state.json").stat().st_mode & 0o222)
            snapshot.attest_child(first)
            snapshot.attest_child(second)
        finally:
            snapshot.release_child(first)
            snapshot.release_child(second)
    finally:
        snapshot.close()


def test_every_configured_result_carries_the_single_harness_digest(tmp_path):
    snapshot = _snapshot(tmp_path)
    try:
        runner = _runner(
            tmp_path,
            snapshot,
            'printf "%s|%s" "$HOME" "$PRIME_AGENT_CODING_AGENT_DIR"\n',
        )
        out = runner.run_one(_request())
        assert out.exclusion_reason is None, out.exclusion_detail
        assert out.agent_fingerprint == _fingerprint(snapshot, runner)
        home, agent_dir = out.final_text.split("|")
        assert home != str(Path.home())
        assert agent_dir.startswith(home + "/")
    finally:
        snapshot.close()


def test_child_env_builder_preserves_every_isolation_override(tmp_path):
    snapshot = _snapshot(tmp_path)
    child = snapshot.prepare_child("env")
    try:
        env = build_child_env(
            rollout_id="rl_env",
            llm_url="https://gateway.invalid/v1",
            api_key="single-use",
            trajectory_dir=str(tmp_path / "trajectory"),
            parent_env={
                "HOME": "/ambient/home",
                "XDG_CONFIG_HOME": "/ambient/config",
                "PRIME_AGENT_CODING_AGENT_DIR": "/ambient/prime",
            },
            extra=child.environment,
        )
        for name, value in child.environment.items():
            assert env[name] == value
        assert env["HOME"] != "/ambient/home"
        assert env["XDG_CONFIG_HOME"] != "/ambient/config"
        assert env["PRIME_AGENT_CODING_AGENT_DIR"] != "/ambient/prime"
    finally:
        snapshot.release_child(child)
        snapshot.close()


def test_expired_unstarted_result_does_not_claim_execution_identity(tmp_path):
    snapshot = _snapshot(tmp_path)
    try:
        runner = _runner(tmp_path, snapshot, "exit 99\n")
        out = runner.run_one(_expired_request())
        assert out.exclusion_reason == "deadline_exceeded"
        assert out.agent_fingerprint is None
    finally:
        snapshot.close()


def test_supervisor_environment_is_stable_and_contains_no_ambient_secrets(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("ARBITRARY_SECRET", "must-not-cross")
    monkeypatch.setenv("JIRA_TOKEN", "must-not-cross")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "must-not-cross")
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-cross")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "must-not-cross")
    snapshot = _snapshot(tmp_path)
    try:
        env = snapshot.daemon_environment()
        for name in (
            "ARBITRARY_SECRET",
            "JIRA_TOKEN",
            "AWS_ACCESS_KEY_ID",
            "OPENAI_API_KEY",
            "PERCEPTEYE_API_KEY",
        ):
            assert name not in env
        assert env["HOME"] != str(Path.home())
        agent_dir = Path(env["PRIME_AGENT_CODING_AGENT_DIR"])
        assert (agent_dir / "settings.json").read_bytes() == _EXPECTED_SETTINGS
        assert snapshot.daemon_working_directory.is_dir()
        assert list(snapshot.daemon_working_directory.iterdir()) == []
        assert snapshot.daemon_working_directory != tmp_path
        assert env["PRIME_AGENT_KERNEL_VENV"].endswith("kernel-venv")
        assert env["PRIME_AGENT_KERNEL_PYTHON"] == ""
        assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    finally:
        snapshot.close()


def test_concurrent_children_are_distinct_but_have_one_identity(tmp_path):
    snapshot = _snapshot(tmp_path)
    try:
        runner = _runner(
            tmp_path,
            snapshot,
            'cat "$PRIME_AGENT_CODING_AGENT_DIR/harness/harness_state.json" >/dev/null\n'
            'sleep 0.1\n'
            'printf "%s" "$HOME"\n',
        )
        requests = [_request(f"rl_{index}") for index in range(4)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(runner.run_one, requests))

        assert all(out.exclusion_reason is None for out in outputs)
        assert len({out.final_text for out in outputs}) == 4
        assert all(out.agent_fingerprint == _fingerprint(snapshot, runner) for out in outputs)

        signal = tmp_path / "kernel-mutated"
        kernel_runner = _runner(
            tmp_path,
            snapshot,
            'KERNEL="$PRIME_AGENT_KERNEL_VENV/runtime.txt"\n'
            'if [ "$PERCEPTEYE_TURN_INPUT" = mutate ]; then\n'
            '  ORIGINAL=$(cat "$KERNEL")\n'
            '  printf "changed runtime\\n" > "$KERNEL"\n'
            f"  touch {shlex.quote(str(signal))}\n"
            "  sleep 1\n"
            '  printf "%s\\n" "$ORIGINAL" > "$KERNEL"\n'
            "else\n"
            f"  while [ ! -e {shlex.quote(str(signal))} ]; do sleep 0.01; done\n"
            '  cat "$KERNEL"\n'
            "fi\n",
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            mutating = pool.submit(
                kernel_runner.run_one, _request("rl_kernel_mutate", "mutate")
            )
            observing = pool.submit(
                kernel_runner.run_one, _request("rl_kernel_observe", "observe")
            )
            mutation_out = mutating.result()
            observer_out = observing.result()

        assert mutation_out.exclusion_reason == "cancelled"
        assert observer_out.final_text == "changed runtime"
        assert observer_out.exclusion_reason == "cancelled"
        assert mutation_out.agent_fingerprint is None
        assert observer_out.agent_fingerprint is None
    finally:
        snapshot.close()


def test_refinement_mutation_is_excluded_and_cannot_reach_the_next_child(tmp_path):
    snapshot = _snapshot(tmp_path)
    try:
        runner = _runner(
            tmp_path,
            snapshot,
            'STATE="$PRIME_AGENT_CODING_AGENT_DIR/harness/harness_state.json"\n'
            'KERNEL="$PRIME_AGENT_KERNEL_VENV/runtime.txt"\n'
            'if [ "$PERCEPTEYE_TURN_INPUT" = refine ]; then\n'
            '  chmod u+w "$(dirname "$STATE")" "$STATE"\n'
            '  printf \'{"changed":true}\' > "$STATE"\n'
            '  printf \'{"type":"refine_complete","result":{"scope":"global"}}\\n\'\n'
            'elif [ "$PERCEPTEYE_TURN_INPUT" = restore ]; then\n'
            '  ORIGINAL=$(cat "$STATE")\n'
            '  chmod u+w "$(dirname "$STATE")" "$STATE"\n'
            '  printf \'{"changed":true}\' > "$STATE"\n'
            '  printf "%s" "$ORIGINAL" > "$STATE"\n'
            '  chmod 444 "$STATE"\n'
            '  chmod 555 "$(dirname "$STATE")"\n'
            '  echo restored\n'
            'elif [ "$PERCEPTEYE_TURN_INPUT" = kernel ]; then\n'
            '  ORIGINAL=$(cat "$KERNEL")\n'
            '  printf "changed runtime\\n" > "$KERNEL"\n'
            '  printf "%s\\n" "$ORIGINAL" > "$KERNEL"\n'
            '  echo restored-kernel\n'
            "else\n"
            '  cat "$STATE"\n'
            "fi\n",
        )

        refined = runner.run_one(_request("rl_refine", "refine"))
        assert refined.exclusion_reason == "cancelled"
        assert "harness" in (refined.exclusion_detail or "").lower()
        assert refined.agent_fingerprint is None

        restored = runner.run_one(_request("rl_restore", "restore"))
        assert restored.exclusion_reason == "cancelled"
        assert "harness" in (restored.exclusion_detail or "").lower()
        assert restored.agent_fingerprint is None

        untouched = runner.run_one(_request("rl_after", "read"))
        assert untouched.exclusion_reason is None, untouched.exclusion_detail
        assert untouched.final_text == '{"schema":1,"entries":{}}'
        assert untouched.agent_fingerprint == _fingerprint(snapshot, runner)
        assert (_source(tmp_path) / "harness_state.json").read_text() == (
            '{"schema":1,"entries":{}}'
        )

        kernel_changed = runner.run_one(_request("rl_kernel", "kernel"))
        assert kernel_changed.exclusion_reason == "cancelled"
        assert "execution state drifted" in (kernel_changed.exclusion_detail or "").lower()
        assert kernel_changed.agent_fingerprint is None

        # The child restored the original bytes, but the ctime/inode seal keeps
        # the shared venv poisoned. A later replica is refused before launch,
        # so it cannot silently train on a runtime another child touched.
        refused = runner.run_one(_request("rl_after_kernel", "read"))
        assert refused.exclusion_reason == "cancelled"
        assert refused.final_text == ""
        assert refused.agent_fingerprint is None
    finally:
        snapshot.close()


def test_mutating_auto_refine_settings_is_also_a_fail_closed_exclusion(tmp_path):
    snapshot = _snapshot(tmp_path)
    try:
        runner = _runner(
            tmp_path,
            snapshot,
            'SETTINGS="$PRIME_AGENT_CODING_AGENT_DIR/settings.json"\n'
            'ORIGINAL=$(cat "$SETTINGS")\n'
            'chmod u+w "$SETTINGS"\n'
            'printf \'{"autoRefine":{"enabled":true}}\\n\' > "$SETTINGS"\n'
            'printf "%s\\n" "$ORIGINAL" > "$SETTINGS"\n'
            'chmod 444 "$SETTINGS"\n'
            "echo done\n",
        )
        out = runner.run_one(_request("rl_settings"))
        assert out.exclusion_reason == "cancelled"
        assert "execution state drifted" in (out.exclusion_detail or "").lower()
        assert out.agent_fingerprint is None
    finally:
        snapshot.close()


def test_a_conflicting_agent_claim_is_replaced_and_excluded(tmp_path, monkeypatch):
    snapshot = _snapshot(tmp_path)
    try:
        monkeypatch.setattr(
            runner_module,
            "run_rollout_as_command",
            lambda *args, **kwargs: RolloutOutput(
                final_text="done",
                agent_fingerprint={
                    "sdk": "test",
                    "execution_sha256": "a" * 64,
                    "harness_sha256": "0" * 64,
                    "kernel_sha256": "f" * 64,
                },
            ),
        )
        runner = _runner(tmp_path, snapshot, "echo unused\n")
        out = runner.run_one(_request())
        assert out.exclusion_reason == "cancelled"
        assert out.agent_fingerprint == {
            "sdk": "test",
            "execution_sha256": runner.execution_sha256,
            "harness_sha256": snapshot.harness_sha256,
            "kernel_sha256": snapshot.kernel_sha256,
        }
        assert "different" in (out.exclusion_detail or "")

        def _launch_failure(*args, **kwargs):
            raise RuntimeError("original launcher failure")

        monkeypatch.setattr(runner_module, "run_rollout_as_command", _launch_failure)
        with pytest.raises(RuntimeError, match="original launcher failure"):
            _runner(tmp_path, snapshot, "echo unused\n").run_one(
                _request("rl_launch_failure")
            )
    finally:
        # The failed launch must have released its active private child rather
        # than making close() fail or leaking run state.
        snapshot.close()


def test_long_prompt_suffixes_do_not_collapse_after_wire_truncation(tmp_path):
    def _payload(suffix: str) -> dict:
        return {
            "agent_name": "prime",
            "framework": "prime",
            "system_prompt": "x" * 20_000 + suffix,
            "tool_definitions": [{"name": "read", "input_schema": {}}],
            "discovered_model_config": {"provider": "gateway", "model_name": "policy"},
        }

    first = parse_discovered_agent(_payload("first"))
    second = parse_discovered_agent(_payload("second"))
    assert first is not None and second is not None
    assert first.to_wire()["system_prompt"] == second.to_wire()["system_prompt"]
    assert first.exact_observation_sha256 != second.exact_observation_sha256

    runner_one = RolloutRunner(
        ["agent", fw.TASK], transport=_Transport(), input_shape="text",
        artifacts_dir=str(tmp_path / "one"),
        attested_execution_components={"agent.release": "1" * 64},
    )
    runner_two = RolloutRunner(
        ["agent", fw.TASK], transport=_Transport(), input_shape="text",
        artifacts_dir=str(tmp_path / "two"),
        attested_execution_components={"agent.release": "1" * 64},
    )
    for runner, described in ((runner_one, first), (runner_two, second)):
        runner.configure_execution_identity(
            described, discovery_enabled=True, sdk="sdk/1", contract_version="flywheel/1",
        )
    assert runner_one.execution_sha256 is not None
    assert runner_two.execution_sha256 is not None
    assert runner_one.execution_sha256 != runner_two.execution_sha256

    # A caller-created projected record has no proof of what lay beyond the
    # public cap, so it gets no affirmative OPSD identity at all.
    projected = _described("x" * 20_000)
    assert runner_one.configure_execution_identity(
        projected, discovery_enabled=True, sdk="sdk/1", contract_version="flywheel/1",
    ) is None


def test_constructed_object_and_nested_lossiness_withhold_exact_identity(tmp_path):
    class _Agent:
        def invoke(self, task):
            return task

    object_runner = RolloutRunner(
        _Agent(), transport=_Transport(), input_shape="text", isolation="inprocess",
        artifacts_dir=str(tmp_path / "object"),
    )
    assert object_runner.configure_execution_identity(
        _described(), discovery_enabled=True, sdk="sdk/1", contract_version="flywheel/1",
    ) is None

    attested_runner = RolloutRunner(
        _Agent(), transport=_Transport(), input_shape="text", isolation="inprocess",
        artifacts_dir=str(tmp_path / "attested-object"),
        attested_execution_components={"customer.object_state": "2" * 64},
    )
    assert attested_runner.configure_execution_identity(
        _described(), discovery_enabled=True, sdk="sdk/1", contract_version="flywheel/1",
    ) is not None

    incomplete_child = DiscoveredAgent(
        agent_name="child", framework="test", system_prompt="child prompt",
        tool_definitions=[ToolDefinition(name="read")],
        discovered_model_config=None,
        incomplete=["discovered_model_config"],
    )
    assert attested_runner.configure_execution_identity(
        _described(sub_agents=[incomplete_child]),
        discovery_enabled=True,
        sdk="sdk/1",
        contract_version="flywheel/1",
    ) is None

    # Host JSON has its own bounded recursive parser. A descendant beyond that
    # bound, or a malformed child that was dropped, must make the observation
    # incomplete instead of disappearing from an affirmative identity.
    leaf: dict = {
        "agent_name": "leaf", "framework": "test",
        "system_prompt": "leaf", "tool_definitions": [],
        "discovered_model_config": {"provider": "gateway", "model_name": "policy"},
    }
    nested = leaf
    for index in range(6):
        nested = {
            **leaf,
            "agent_name": f"parent-{index}",
            "sub_agents": [nested],
        }
    parsed = parse_discovered_agent(nested)
    assert parsed is not None
    assert attested_runner.configure_execution_identity(
        parsed, discovery_enabled=True, sdk="sdk/1", contract_version="flywheel/1",
    ) is None

    malformed = parse_discovered_agent({**leaf, "sub_agents": [{"bad": "child"}]})
    assert malformed is not None and "sub_agents" in malformed.incomplete


def test_execution_descriptor_binds_config_and_canonical_component_map(tmp_path):
    def _identity(*, gate: str, components: dict[str, str]) -> str:
        runner = RolloutRunner(
            ["agent", fw.TASK], transport=_Transport(), input_shape="text",
            artifacts_dir=str(tmp_path / gate.strip("-") or "none"),
            gate_flag=gate,
            reports_tool_calls=True,
            attested_execution_components=components,
        )
        digest = runner.configure_execution_identity(
            _described(), discovery_enabled=True,
            sdk="agent-flywheel-python/0.1.0",
            contract_version="flywheel/1",
        )
        assert digest is not None
        return digest

    first = _identity(gate="--gate", components={"z": "3" * 64, "a": "4" * 64})
    reordered = _identity(gate="--gate", components={"a": "4" * 64, "z": "3" * 64})
    changed_gate = _identity(gate="--other-gate", components={"a": "4" * 64, "z": "3" * 64})
    changed_component = _identity(
        gate="--gate", components={"a": "5" * 64, "z": "3" * 64},
    )
    assert first == reordered
    assert len({first, changed_gate, changed_component}) == 3

    with pytest.raises(ConfigurationError, match="lowercase 64-hex"):
        RolloutRunner(
            ["agent"], transport=_Transport(), input_shape="text",
            artifacts_dir=str(tmp_path / "invalid"),
            attested_execution_components={"release": "NOT-A-DIGEST"},
        )

def test_execution_artifact_mutate_then_restore_poison_is_fail_closed(tmp_path):
    snapshot = _snapshot(tmp_path)
    route = tmp_path / "route.ts"
    original = route.read_text()
    route.write_text("changed\n")
    route.write_text(original)
    try:
        runner = _runner(tmp_path, snapshot, "echo must-not-run\n")
        out = runner.run_one(_request("rl_artifact_drift"))
        assert out.exclusion_reason == "cancelled"
        assert out.final_text == ""
        assert out.agent_fingerprint is None
        assert "could not prepare" in (out.exclusion_detail or "")
    finally:
        snapshot.close()


def test_command_discovery_uses_and_attests_the_same_isolated_harness(
    tmp_path, monkeypatch,
):
    snapshot = _snapshot(tmp_path)
    seen: dict[str, str] = {}

    def _warm(*args, **kwargs):
        # The ambient parent process is passed separately so warm_up_command
        # can preserve only its tiny process baseline.  Adapter-owned paths
        # travel through the explicit, allowlisted isolation mapping.
        env = kwargs["isolation_environment"]
        seen["home"] = env["HOME"]
        seen["agent_dir"] = env["PRIME_AGENT_CODING_AGENT_DIR"]
        state = Path(seen["agent_dir"]) / "harness" / "harness_state.json"
        assert state.read_text() == '{"schema":1,"entries":{}}'
        assert (
            Path(seen["agent_dir"]) / "settings.json"
        ).read_bytes() == _EXPECTED_SETTINGS
        return DiscoveredAgent(agent_name="prime", framework="prime")

    monkeypatch.setattr(fw, "warm_up_command", _warm)
    try:
        described = fw._describe_via_host(  # type: ignore[attr-defined]
            ["prime-agent"],
            artifacts_dir=str(tmp_path / "artifacts"),
            agent_id="prime",
            execution_snapshot=snapshot,
        )
        assert described is not None
        assert seen["home"] != str(Path.home())
        assert seen["agent_dir"].startswith(seen["home"] + "/")
    finally:
        snapshot.close()


def test_source_links_fail_closed_and_generic_python_entrypoints_accept_adapter(tmp_path):
    source = _source(tmp_path)
    (source / "ambient").symlink_to(tmp_path)
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    with pytest.raises(ConfigurationError, match="link"):
        PrimeHarnessSnapshot(
            source, workspace_dir=tmp_path / "runs", kernel_venv=kernel,
        )
    (source / "ambient").unlink()
    os.mkfifo(source / "special")
    with pytest.raises(ConfigurationError, match="special"):
        PrimeHarnessSnapshot(
            source, workspace_dir=tmp_path / "runs-special", kernel_venv=kernel,
        )

    clean = tmp_path / "clean"
    clean.mkdir()
    snapshot = PrimeHarnessSnapshot(
        clean, workspace_dir=tmp_path / "runs-2", kernel_venv=kernel,
    )
    try:
        runner = RolloutRunner(
            "tests.fixtures_agent:build",
            transport=_Transport(),
            input_shape="text",
            artifacts_dir=str(tmp_path / "artifacts"),
            execution_snapshot=snapshot,
        )
        assert runner.execution_snapshot is snapshot
    finally:
        snapshot.close()


def test_kernel_external_directory_symlink_is_refused_but_lib64_alias_is_supported(
    tmp_path,
):
    source = _source(tmp_path)
    kernel = tmp_path / "kernel"
    kernel.mkdir()
    (kernel / "lib").mkdir()
    (kernel / "lib64").symlink_to("lib", target_is_directory=True)
    supported = PrimeHarnessSnapshot(
        source, workspace_dir=tmp_path / "supported", kernel_venv=kernel,
    )
    supported.close()

    outside = tmp_path / "outside"
    outside.mkdir()
    (kernel / "external").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ConfigurationError, match="resolve inside"):
        PrimeHarnessSnapshot(
            source, workspace_dir=tmp_path / "refused", kernel_venv=kernel,
        )
