"""The Prime example's on-policy identity and contamination controls."""
from __future__ import annotations

import importlib.util
import os
import pathlib
import secrets
import signal

from agent_flywheel.isolation import build_child_env

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_example(monkeypatch, tmp_path, **environment):
    monkeypatch.setenv("PRIME_WORKSPACE", str(tmp_path / "workspace"))
    defaults = {
        "PERCEPTEYE_MODEL_ID": "policy",
        "PERCEPTEYE_CONTEXT_WINDOW": "65536",
        "PERCEPTEYE_MAX_TOKENS": "4096",
    }
    # serve_prime deliberately makes these values explicit in os.environ for
    # its future children. Register every key with monkeypatch first so an
    # import in this test cannot leak those mutations into a later test.
    for name, default in defaults.items():
        monkeypatch.setenv(name, environment.get(name, os.environ.get(name, default)))
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    path = ROOT / "examples" / "prime_agent" / "serve_prime.py"
    spec = importlib.util.spec_from_file_location("_tested_serve_prime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parent_selection_and_child_route_configuration_have_identity_parity(
    monkeypatch, tmp_path
):
    module = _load_example(
        monkeypatch,
        tmp_path,
        PERCEPTEYE_MODEL_ID="tinker://opsd/run-9/final",
        PERCEPTEYE_CONTEXT_WINDOW="32768",
        PERCEPTEYE_MAX_TOKENS="3072",
    )
    selected = module.ARGV[module.ARGV.index("--model") + 1]
    core_child = build_child_env(
        rollout_id="rl_prime",
        llm_url="https://gateway.invalid/v1",
        api_key="single-use",
        trajectory_dir=str(tmp_path / "trajectory"),
    )
    assert "PERCEPTEYE_MODEL_ID" not in core_child
    assert "PERCEPTEYE_CONTEXT_WINDOW" not in core_child
    assert "PERCEPTEYE_MAX_TOKENS" not in core_child

    source = tmp_path / "harness"
    source.mkdir()
    snapshot = module.PrimeHarnessSnapshot(
        source,
        workspace_dir=tmp_path / "snapshot-runs",
        route_configuration={
            "model_id": selected,
            "context_window": "32768",
            "max_tokens": "3072",
        },
    )
    child = snapshot.prepare_child()
    try:
        assert child.environment["PRIME_AGENT_FLYWHEEL_MODEL_ID"] == selected
        assert child.environment["PRIME_AGENT_FLYWHEEL_CONTEXT_WINDOW"] == "32768"
        assert child.environment["PRIME_AGENT_FLYWHEEL_MAX_TOKENS"] == "3072"
        assert "prime.route_configuration" in snapshot.execution_components
    finally:
        snapshot.release_child(child)
        snapshot.close()


def test_prime_rollouts_disable_persistent_refinement_and_cancel_compaction(
    monkeypatch, tmp_path
):
    module = _load_example(monkeypatch, tmp_path)
    assert "--no-session" in module.ARGV
    assert "--no-extensions" in module.ARGV
    assert "--no-skills" in module.ARGV
    assert "--no-prompt-templates" in module.ARGV
    assert "--no-context-files" in module.ARGV
    route = (ROOT / "examples" / "prime_agent" / "route.ts").read_text()
    assert 'pi.on("session_before_compact", () => ({ cancel: true }))' in route


def test_invalid_route_limits_are_refused_before_any_rollout(monkeypatch, tmp_path):
    import pytest

    with pytest.raises(RuntimeError, match="MAX_TOKENS cannot exceed"):
        _load_example(
            monkeypatch,
            tmp_path,
            PERCEPTEYE_CONTEXT_WINDOW="1024",
            PERCEPTEYE_MAX_TOKENS="2048",
        )


def test_main_requires_explicit_harness_kernel_and_package_paths(monkeypatch, tmp_path):
    import pytest

    module = _load_example(monkeypatch, tmp_path)
    monkeypatch.delenv("PERCEPTEYE_PRIME_HARNESS_SOURCE", raising=False)
    monkeypatch.delenv("PRIME_AGENT_KERNEL_VENV", raising=False)
    # All THREE, not two of the three. Any machine that actually serves
    # prime-agent has this one exported -- and so does CI now, which installs
    # the package and names its root for the real-binary tests. Left ambient,
    # the third `raises` below falls straight through to `fw.serve` and the
    # test fails on a missing control plane, so the check that this example
    # demands an explicit package root was passing only where nobody had one.
    monkeypatch.delenv("PRIME_AGENT_PACKAGE_ROOT", raising=False)
    with pytest.raises(RuntimeError, match="PRIME_HARNESS_SOURCE"):
        module.main()

    source = tmp_path / "harness"
    source.mkdir()
    monkeypatch.setenv("PERCEPTEYE_PRIME_HARNESS_SOURCE", str(source))
    with pytest.raises(RuntimeError, match="PRIME_AGENT_KERNEL_VENV"):
        module.main()

    kernel = tmp_path / "kernel"
    kernel.mkdir()
    monkeypatch.setenv("PRIME_AGENT_KERNEL_VENV", str(kernel))
    with pytest.raises(RuntimeError, match="PRIME_AGENT_PACKAGE_ROOT"):
        module.main()


def test_main_refuses_a_launcher_outside_the_attested_package(monkeypatch, tmp_path):
    import pytest

    module = _load_example(monkeypatch, tmp_path)
    source = tmp_path / "empty-harness"
    kernel = tmp_path / "kernel"
    package = tmp_path / "prime-package"
    wrapper = tmp_path / "debug-wrapper" / "prime-agent"
    source.mkdir()
    kernel.mkdir()
    (package / "dist").mkdir(parents=True)
    (package / "package.json").write_text('{"name":"prime-agent"}\n')
    wrapper.parent.mkdir()
    wrapper.write_text("#!/bin/sh\n")
    monkeypatch.setenv("PERCEPTEYE_PRIME_HARNESS_SOURCE", str(source))
    monkeypatch.setenv("PRIME_AGENT_KERNEL_VENV", str(kernel))
    monkeypatch.setenv("PRIME_AGENT_PACKAGE_ROOT", str(package))
    monkeypatch.setattr(module.shutil, "which", lambda name: str(wrapper))

    with pytest.raises(RuntimeError, match="does not belong"):
        module.main()


def test_main_wires_and_closes_one_run_scoped_snapshot(monkeypatch, tmp_path):
    module = _load_example(monkeypatch, tmp_path)
    source = tmp_path / "harness"
    kernel = tmp_path / "kernel"
    source.mkdir()
    kernel.mkdir()
    package = tmp_path / "prime-package"
    (package / "dist").mkdir(parents=True)
    launcher = package / "dist" / "cli.js"
    launcher.write_text("#!/usr/bin/env node\n")
    (package / "package.json").write_text('{"name":"prime-agent"}\n')
    monkeypatch.setenv("PERCEPTEYE_PRIME_HARNESS_SOURCE", str(source))
    monkeypatch.setenv("PRIME_AGENT_KERNEL_VENV", str(kernel))
    monkeypatch.setenv("PRIME_AGENT_PACKAGE_ROOT", str(package))
    monkeypatch.setattr(module.shutil, "which", lambda name: str(launcher))
    seen: dict = {}
    order: list[str] = []

    class _Snapshot:
        def __init__(
            self, source_dir, *, workspace_dir, kernel_venv, execution_artifacts,
            route_configuration,
        ):
            seen["source"] = source_dir
            seen["workspace"] = workspace_dir
            seen["kernel"] = kernel_venv
            seen["execution_artifacts"] = execution_artifacts
            seen["route_configuration"] = route_configuration
            seen["closed"] = False

        def close(self):
            order.append("snapshot.close")
            seen["closed"] = True

        def daemon_environment(self):
            return {"PATH": os.environ["PATH"], "HOME": "/run-owned-daemon"}

        @property
        def daemon_working_directory(self):
            return "/run-owned-daemon-cwd"

    class _Daemon:
        def __init__(self, socket_path, env, *, cwd, executable):
            seen["daemon_socket"] = socket_path
            seen["daemon_env"] = env
            seen["daemon_cwd"] = cwd
            seen["daemon_executable"] = executable

        def start(self):
            order.append("daemon.start")

        def close(self):
            order.append("daemon.close")

    def _serve(argv, **kwargs):
        order.append("serve")
        seen["argv"] = argv
        seen["snapshot"] = kwargs["execution_snapshot"]
        seen["command_output_adapter"] = kwargs["command_output_adapter"]
        return 7

    monkeypatch.setattr(module, "PrimeHarnessSnapshot", _Snapshot)
    monkeypatch.setattr(module.fw, "serve", _serve)
    monkeypatch.setattr(module, "_RunOwnedPrimeDaemon", _Daemon)
    monkeypatch.setattr(
        module,
        "_preflight_prime_runtime",
        lambda executable, *, env, cwd: "0.test",
    )
    assert module.main() == 7
    assert seen["source"] == str(source)
    assert seen["kernel"] == str(kernel)
    assert seen["execution_artifacts"] == {
        "prime_distribution": package / "dist",
        "prime_package_manifest": package / "package.json",
        "route_extension": module.HERE / "route.ts",
    }
    assert seen["route_configuration"] == {
        "model_id": "policy",
        "context_window": "65536",
        "max_tokens": "4096",
    }
    assert seen["snapshot"].__class__ is _Snapshot
    assert seen["command_output_adapter"].adapter_name == "Prime Agent JSONL"
    assert seen["closed"] is True
    assert seen["daemon_env"]["HOME"] == "/run-owned-daemon"
    assert seen["daemon_cwd"] == "/run-owned-daemon-cwd"
    assert seen["daemon_executable"] == str(launcher)
    assert seen["argv"][0] == str(launcher)
    assert order == ["daemon.start", "serve", "daemon.close", "snapshot.close"]


def test_runtime_preflight_checks_launcher_and_required_flags(monkeypatch, tmp_path):
    module = _load_example(monkeypatch, tmp_path)
    seen: list[tuple[list[str], dict]] = []
    help_text = "\n".join(module._REQUIRED_PRIME_FLAGS).encode()

    def _run(argv, **kwargs):
        seen.append((argv, kwargs))
        stdout = b"0.7.2\n" if argv[-1] == "--version" else help_text
        return module.subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", _run)
    assert module._preflight_prime_runtime(
        "/attested/prime-agent",
        env={"PATH": "/pinned/node24/bin"},
        cwd=tmp_path,
    ) == "0.7.2"
    assert [call[0] for call in seen] == [
        ["/attested/prime-agent", "--version"],
        ["/attested/prime-agent", "--help"],
    ]
    assert all(call[1]["env"] == {"PATH": "/pinned/node24/bin"} for call in seen)
    assert all(call[1]["cwd"] == tmp_path for call in seen)


def test_runtime_preflight_names_a_broken_node_or_incompatible_prime(
    monkeypatch, tmp_path
):
    import pytest

    module = _load_example(monkeypatch, tmp_path)

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda argv, **kwargs: module.subprocess.CompletedProcess(
            argv, 134, stdout=b"", stderr=b"dyld: Library not loaded"
        ),
    )
    with pytest.raises(RuntimeError, match="dyld: Library not loaded"):
        module._preflight_prime_runtime("prime-agent", env={}, cwd=tmp_path)

    calls = 0

    def _missing_flag(argv, **kwargs):
        nonlocal calls
        calls += 1
        stdout = b"0.7.2\n" if calls == 1 else b"--mode\n--provider\n"
        return module.subprocess.CompletedProcess(argv, 0, stdout=stdout, stderr=b"")

    monkeypatch.setattr(module.subprocess, "run", _missing_flag)
    with pytest.raises(RuntimeError, match="required rollout flags"):
        module._preflight_prime_runtime("prime-agent", env={}, cwd=tmp_path)


def test_run_owned_daemon_is_started_and_explicitly_shut_down(monkeypatch, tmp_path):
    module = _load_example(monkeypatch, tmp_path)
    calls: list[tuple[str, object]] = []

    class _Process:
        returncode = None

        def poll(self):
            return None

        def wait(self, timeout):
            calls.append(("wait", timeout))
            self.returncode = 0
            return 0

        def terminate(self):
            calls.append(("terminate", None))

        def kill(self):
            calls.append(("kill", None))

    process = _Process()

    def _popen(argv, **kwargs):
        calls.append(("popen", (argv, kwargs)))
        return process

    def _run(argv, **kwargs):
        calls.append(("run", (argv, kwargs)))
        return None

    monkeypatch.setattr(module.subprocess, "Popen", _popen)
    monkeypatch.setattr(module.subprocess, "run", _run)
    # macOS limits AF_UNIX paths to 103 bytes; pytest's tmp_path is longer.
    daemon_socket = f"/tmp/percepteye-test-{secrets.token_hex(8)}.sock"
    daemon = module._RunOwnedPrimeDaemon(
        daemon_socket,
        {"PATH": os.environ["PATH"], "HOME": "/isolated-daemon-home"},
        cwd=tmp_path / "isolated-daemon-cwd",
    )
    monkeypatch.setattr(daemon, "_accepting_connections", lambda: True)

    daemon.start()
    # Represent the daemon-created socket without requiring the unit-test
    # sandbox to permit AF_UNIX binds; lstat/unlink remain real.
    pathlib.Path(daemon.socket_path).write_text("socket placeholder")
    monkeypatch.setattr(module.stat, "S_ISSOCK", lambda mode: True)
    daemon.close()

    launch = next(value for kind, value in calls if kind == "popen")
    assert launch[0] == [
        "prime-agent",
        "--mode",
        "daemon",
        "--daemon-socket",
        daemon_socket,
    ]
    assert launch[1]["env"]["HOME"] == "/isolated-daemon-home"
    assert launch[1]["cwd"] == str(tmp_path / "isolated-daemon-cwd")
    shutdown = next(value for kind, value in calls if kind == "run")
    assert shutdown[0] == [
        "prime-agent",
        "shutdown",
        "--force",
        "--daemon-socket",
        daemon_socket,
    ]
    assert shutdown[1]["cwd"] == str(tmp_path / "isolated-daemon-cwd")
    assert calls[-1] == ("wait", 5)
    assert not pathlib.Path(daemon.socket_path).exists()


def test_service_stop_signals_unwind_through_the_cleanup_path(monkeypatch, tmp_path):
    module = _load_example(monkeypatch, tmp_path)
    changes: dict[int, list[object]] = {}
    originals = {
        signal.SIGTERM: object(),
        signal.SIGINT: object(),
        **({signal.SIGHUP: object()} if hasattr(signal, "SIGHUP") else {}),
    }

    monkeypatch.setattr(module.signal, "getsignal", lambda signum: originals[signum])

    def _signal(signum, handler):
        changes.setdefault(signum, []).append(handler)

    monkeypatch.setattr(module.signal, "signal", _signal)
    previous = module._install_cleanup_signal_handlers()

    assert previous == originals
    import pytest

    first_signum = next(iter(changes))
    with pytest.raises(SystemExit) as stopped:
        changes[first_signum][0](first_signum, None)
    assert stopped.value.code == 128 + first_signum
    # Service managers commonly send a second TERM while graceful shutdown
    # drains. No later stop signal can interrupt child teardown.
    for signum, handlers in changes.items():
        assert handlers[0](signum, None) is None

    module._ignore_cleanup_signals(previous)
    module._restore_signal_handlers(previous)
    assert changes == {
        signum: [changes[signum][0], signal.SIG_IGN, original]
        for signum, original in originals.items()
    }


def test_the_example_keeps_one_execution_identity_across_restarts(
    monkeypatch, tmp_path
):
    """The registered command carries a PID; the identity must not.

    Driven through the example's own ARGV and its own declaration rather than
    a hand-written command, so a change to how serve_prime spells either one
    is caught here.
    """
    from agent_flywheel.introspection import (
        DiscoveredAgent,
        ModelConfig,
        ToolDefinition,
    )
    from agent_flywheel.runner import RolloutRunner

    class _Transport:
        lease_ms = 1_800_000

    def _identity(pid: int, *, declare: bool) -> tuple[list[str], str]:
        monkeypatch.setattr(os, "getpid", lambda: pid)
        module = _load_example(monkeypatch, tmp_path / f"pid-{pid}")
        argv = list(module.ARGV)
        runner = RolloutRunner(
            argv,
            transport=_Transport(),
            input_shape="text",
            artifacts_dir=str(tmp_path / f"artifacts-{pid}"),
            attested_execution_components={"prime.continual_harness": "a" * 64},
            volatile_command_values=(
                module.VOLATILE_COMMAND_VALUES if declare else None
            ),
        )
        digest = runner.configure_execution_identity(
            DiscoveredAgent(
                agent_name="prime-agent",
                framework="prime",
                system_prompt="Complete the task.",
                tool_definitions=[ToolDefinition(name="shell", input_schema={})],
                discovered_model_config=ModelConfig(
                    provider="percepteye", model_name="policy",
                ),
            ),
            discovery_enabled=True,
            sdk="agent-flywheel-python/0.1.1",
            contract_version="flywheel/1",
        )
        assert digest is not None
        return argv, digest

    first_argv, first = _identity(4242, declare=True)
    second_argv, second = _identity(9999, declare=True)

    # The PID really does reach the registered command -- otherwise this test
    # would pass against an example that never had the defect.
    socket_index = first_argv.index("--daemon-socket") + 1
    assert first_argv[socket_index] != second_argv[socket_index]
    assert "4242" in first_argv[socket_index]
    assert first == second

    # And the declaration is what does it, not an accident of the argv.
    assert _identity(4242, declare=False)[1] != _identity(9999, declare=False)[1]


def test_the_example_hands_its_volatile_declaration_to_serve(monkeypatch, tmp_path):
    """A declaration nothing passes on is a constant, not a mechanism."""
    package_root = tmp_path / "prime-package"
    (package_root / "dist").mkdir(parents=True)
    (package_root / "package.json").write_text("{}")
    launcher = package_root / "bin" / "prime-agent"
    launcher.parent.mkdir()
    launcher.write_text("#!/bin/sh\n")
    harness_source = tmp_path / "harness-source"
    harness_source.mkdir()

    module = _load_example(
        monkeypatch,
        tmp_path / "serve",
        PERCEPTEYE_PRIME_HARNESS_SOURCE=str(harness_source),
        PRIME_AGENT_KERNEL_VENV=str(tmp_path / "kernel-venv"),
        PRIME_AGENT_PACKAGE_ROOT=str(package_root),
    )

    class _Harness:
        daemon_working_directory = str(tmp_path)

        def __init__(self, *args, **kwargs) -> None:
            pass

        def daemon_environment(self) -> dict[str, str]:
            return {}

        def close(self) -> None:
            pass

    class _Daemon:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def start(self) -> None:
            pass

        def close(self) -> None:
            pass

    captured: dict = {}

    def _serve(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return 0

    monkeypatch.setattr(module.shutil, "which", lambda name: str(launcher))
    monkeypatch.setattr(module, "PrimeHarnessSnapshot", _Harness)
    monkeypatch.setattr(module, "_RunOwnedPrimeDaemon", _Daemon)
    monkeypatch.setattr(module, "_preflight_prime_runtime", lambda *a, **k: "1.0.0")
    monkeypatch.setattr(module, "_install_cleanup_signal_handlers", dict)
    monkeypatch.setattr(module, "_ignore_cleanup_signals", lambda previous: None)
    monkeypatch.setattr(module, "_restore_signal_handlers", lambda previous: None)
    monkeypatch.setattr(module.fw, "serve", _serve)

    assert module.main() == 0
    assert captured["kwargs"]["volatile_command_values"] == module.VOLATILE_COMMAND_VALUES
    # And the declared option is in the command that was actually registered.
    for option in module.VOLATILE_COMMAND_VALUES:
        assert option in captured["command"]
