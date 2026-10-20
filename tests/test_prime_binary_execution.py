"""Run the real prime-agent harness, locally, and require it to PRODUCE.

The failure these guard is "claims work, produces nothing, expires". Every
existing test in ``test_prime_agent_example.py`` mocks ``fw.serve`` and checks
the argv, so all of them stayed green through a week of rollouts that claimed
within half a second and then returned nothing at all. Checking the command is
built correctly proves the command is built correctly and no more.

So these start the run-owned daemon, snapshot the harness, and execute the
example's own ARGV against a local endpoint, asserting that a completion came
back -- and, against an endpoint that accepts the connection and never
answers, that the child hands control back long before its budget instead of
being killed on it. Each rollout first passes the example's runtime
preflight, Prime's own kernel check included, and is then attested the way the
runner attests it: the kernel is a real one, provisioned once per module by
Prime's own bootstrap, as the example requires, and handed to Prime as the
interpreter it uses as it is -- so a rollout during which Prime deleted or
rebuilt the sealed venv fails here, as production excludes it.

THE SKIP RULE is the one already written down in ``tests/node_sdk.py``, and
the two cases are not the same case. AUTO-DISCOVERY MAY SKIP: a developer who
has not installed prime-agent has nothing to execute, and turning that into a
red suite teaches everyone to ignore the failure. AN EXPLICIT
``PRIME_AGENT_PACKAGE_ROOT`` IS A PROMISE: somebody said the harness is there,
so a launcher that cannot start is a broken environment and FAILS. ``test-full``
fails the build on any skip, so a deployment that serves prime-agent -- which
sets that variable to serve at all -- cannot turn these into a green no-op.

``test_a_wedged_provider_call_cannot_eat_the_episode.py`` holds the same
claims in a form that always runs, so the fix cannot be deleted silently on a
machine where this file skips.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_flywheel import RolloutRequest
from agent_flywheel.adapters import prime_agent
from agent_flywheel.adapters.prime_agent import (
    PrimeAgentOutputAdapter,
    PrimeHarnessSnapshot,
)
from agent_flywheel.isolation import build_child_env
from agent_flywheel.runner import run_rollout_as_command
from tests.test_prime_agent_example import _load_example

ROOT = pathlib.Path(__file__).resolve().parents[1]
#: Ceiling used for the wedged-gateway case only. The SHIPPED value has to be
#: generous enough for a real generation (see ``_FIXED_SETTINGS``), which would
#: make this a five-minute test; the value proves the mechanism, and the
#: shipped one is asserted to fit the episode in the always-running file.
WEDGED_CEILING_MS = 8_000


def _package_root() -> tuple[pathlib.Path | None, bool]:
    """``(root, was_named)``. ``root`` is None only when nobody named one."""
    raw = os.environ.get("PRIME_AGENT_PACKAGE_ROOT", "").strip()
    if raw:
        return pathlib.Path(raw).expanduser(), True
    launcher = shutil.which("prime-agent")
    if launcher is None:
        return None, False
    resolved = pathlib.Path(launcher).resolve()
    for parent in resolved.parents:
        if (parent / "package.json").is_file() and (parent / "dist").is_dir():
            return parent, False
    return None, False


def _starts(argv: list[str]) -> str | None:
    """``--version`` output, or None if that command cannot run at all."""
    try:
        probe = subprocess.run(
            [*argv, "--version"], stdin=subprocess.DEVNULL,
            capture_output=True, timeout=60, check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    return (probe.stdout + probe.stderr).decode("utf-8", "replace").strip()


def _runnable_launcher() -> list[str]:
    """The launcher, invoked EXACTLY as serve_prime invokes it.

    Through its own ``#!/usr/bin/env node`` shebang, deliberately: running it
    under some other interpreter the test picked would prove a launch
    production never performs. A box whose ``node`` cannot start the launcher
    cannot serve prime-agent either, which is why naming
    ``PRIME_AGENT_PACKAGE_ROOT`` turns that into a failure rather than a skip.
    """
    package_root, named = _package_root()
    if package_root is None:
        pytest.skip(
            "prime-agent is not installed here, so the harness was not "
            "EXECUTED: the launch command, the run-owned daemon and the "
            "attested provider ceiling go unproven against the real binary "
            "on this machine. Set PRIME_AGENT_PACKAGE_ROOT to require them."
        )
    launcher = package_root / "dist" / "bundle" / "cli.js"
    if not launcher.is_file():
        message = f"prime-agent at {package_root} has no dist/bundle/cli.js"
        pytest.fail(message) if named else pytest.skip(message)
    if _starts([str(launcher)]) is not None:
        return [str(launcher)]
    message = (
        f"{launcher} could not start, so the harness was not EXECUTED. A "
        f"broken `node` on PATH does exactly this -- the launcher is a "
        f"#!/usr/bin/env node script, and it is why the example preflights "
        f"the runtime before serving."
    )
    pytest.fail(message) if named else pytest.skip(message)
    raise AssertionError("unreachable")  # pragma: no cover -- both of those raise


class _Gateway:
    """An OpenAI-compatible endpoint that either answers or never does.

    With ``ipython`` code, its first answer is a call to Prime's ``ipython``
    tool running that code in the kernel, streamed as Prime requests it, and
    the answers after that are text; ``tool_results`` is what came back.
    """

    def __init__(self, *, answer: bool, ipython: str | None = None):
        self.requests: list[dict] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _send(self, payload):
                raw = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self):
                self._send({"data": [{"id": "policy", "object": "model"}]})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                outer.requests.append(json.loads(self.rfile.read(n) or b"{}"))
                if not answer:
                    # Accepted, read, and never replied: the shape a gateway
                    # waiting on a dead upstream presents to its client.
                    outer._wedged.wait()
                    return
                if ipython is not None and len(outer.requests) == 1:
                    self._stream_tool_call(ipython)
                    return
                self._send({
                    "id": "chatcmpl-test", "object": "chat.completion",
                    "created": int(time.time()), "model": "policy",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": "hello from the policy"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                              "total_tokens": 2},
                })

            def _stream_tool_call(self, code):
                chunk = {"id": "chatcmpl-tool", "object": "chat.completion.chunk",
                         "created": int(time.time()), "model": "policy"}
                call = {"index": 0, "id": "call_ipython", "type": "function",
                        "function": {"name": "ipython",
                                     "arguments": json.dumps({"code": code})}}
                events = [
                    {**chunk, "choices": [{"index": 0, "finish_reason": None,
                                           "delta": {"role": "assistant",
                                                     "tool_calls": [call]}}]},
                    {**chunk, "choices": [{"index": 0, "delta": {},
                                           "finish_reason": "tool_calls"}]},
                    {**chunk, "choices": [], "usage": {
                        "prompt_tokens": 1, "completion_tokens": 1,
                        "total_tokens": 2}},
                ]
                raw = "".join(f"data: {json.dumps(e)}\n\n" for e in events)
                raw = (raw + "data: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._wedged = threading.Event()
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def tool_results(self) -> list[str]:
        return [str(message.get("content")) for request in self.requests
                for message in request.get("messages", [])
                if message.get("role") == "tool"]

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._wedged.set()
        self._server.shutdown()
        self._server.server_close()


def _example(tmp_path: pathlib.Path):
    """The example, imported by the one loader its other tests use.

    That loader gives it a PRIME_WORKSPACE under ``tmp_path``. A loader of
    this file's own imported it with none, and the example then created its
    default workspace, ``~/.percepteye/prime-agent-workspace``, in the
    developer's real home on every run that had the binary. What the loader
    sets is undone once the import is done: the module keeps what it read.
    """
    with pytest.MonkeyPatch.context() as patch:
        return _load_example(patch, tmp_path)


@pytest.fixture(scope="module")
def provisioned_kernel(tmp_path_factory):
    """A kernel venv Prime will use, made by Prime's own bootstrap.

    The example requires a provisioned PRIME_AGENT_KERNEL_VENV, and refuses
    one whose interpreter Prime would refuse. Built here by the installed
    prime-agent itself rather than imitated: its bootstrap, run with
    PRIME_AGENT_KERNEL_PYTHON unset, provisions the venv that the same file
    later checks, with that variable set, in the example's preflight. Prime
    needs ``uv`` to build it, and the network on a cold uv cache; the skip
    rule is the one ``_runnable_launcher`` applies.
    """
    launcher = pathlib.Path(_runnable_launcher()[-1])
    _, named = _package_root()
    bootstrap = _example(tmp_path_factory.mktemp("prime-example"))._prime_kernel_check(
        launcher.parents[2])
    node = shutil.which("node")
    kernel = tmp_path_factory.mktemp("prime-kernel") / "kernel-venv"
    env = {k: v for k, v in os.environ.items() if k != "PRIME_AGENT_KERNEL_PYTHON"}
    env["PRIME_AGENT_KERNEL_VENV"] = str(kernel)
    failure = None
    if node is None or not bootstrap.is_file():
        failure = f"no node on PATH or no {bootstrap} to provision a kernel with"
    else:
        try:
            built = subprocess.run(
                [node, str(bootstrap)], stdin=subprocess.DEVNULL, env=env,
                capture_output=True, timeout=600, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            failure = f"Prime's kernel bootstrap did not finish: {exc}"
        else:
            if built.returncode != 0:
                failure = "Prime's kernel bootstrap failed: " + (
                    built.stdout + built.stderr).decode("utf-8", "replace")[-800:]
    if failure is not None:
        shutil.rmtree(kernel, ignore_errors=True)
        message = (
            f"{failure}. Without a provisioned kernel the example cannot "
            f"serve, so the harness was not EXECUTED."
        )
        pytest.fail(message) if named else pytest.skip(message)
    yield kernel
    shutil.rmtree(kernel, ignore_errors=True)


def _path_without_uv(tmp_path) -> str:
    """PATH with a ``uv`` in front that refuses, as if none were installed.

    Prime is handed the kernel as PRIME_AGENT_KERNEL_PYTHON, which it never
    rebuilds, so it should never reach for ``uv``. If the hand-off ever goes
    back to PRIME_AGENT_KERNEL_VENV alone, Prime ``rm -rf``s a venv it judges
    stale FIRST and only then looks for ``uv`` -- the refusal cannot save the
    venv, and is not meant to. What it stops is the rebuild: without it,
    ``uv python install`` and ``uv pip install`` run on in the background into
    the child's private HOME after the session ends, still writing while the
    child is removed (``Directory not empty``), and orphaned. With it, the
    venv is simply gone and the attestation after the rollout says so on
    every machine.
    """
    bin_dir = tmp_path / "no-uv"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "uv"
    stub.write_text("#!/bin/sh\necho 'no Prime kernel is provisioned here' >&2\nexit 1\n")
    stub.chmod(0o755)
    return f"{bin_dir}{os.pathsep}{os.environ['PATH']}"


def _run_one_rollout(
    tmp_path, launch, gateway, kernel, *, budget_s: float, preflight: bool = True,
):
    """One rollout through the example's own ARGV and run-owned daemon.

    Preflighted as ``main`` preflights before serving, and attested before
    release, as ``RolloutRunner`` attests every rollout: a kernel Prime would
    not use, or a drifted kernel, harness or settings, fails here instead of
    passing. ``preflight=False`` serves a kernel the preflight would refuse,
    as a kernel the preflight accepted and Prime then refused in a rollout.
    """
    module = _example(tmp_path)
    launcher = pathlib.Path(launch[-1])
    package_root = launcher.parents[2]
    source = tmp_path / "empty-harness"
    source.mkdir()
    harness = PrimeHarnessSnapshot(
        source,
        workspace_dir=tmp_path / "_harness_runs",
        kernel_venv=kernel,
        execution_artifacts={
            "prime_distribution": package_root / "dist",
            "prime_package_manifest": package_root / "package.json",
            "route_extension": ROOT / "examples" / "prime_agent" / "route.ts",
        },
        route_configuration={"model_id": "policy", "context_window": "65536",
                             "max_tokens": "4096"},
    )
    # Placed by the example's own rule, not a second copy of it: a pytest
    # tmp_path is past the AF_UNIX path limit, and so is the temp directory
    # itself under a long TMPDIR -- where a copy that joined onto
    # tempfile.gettempdir() made the real daemon exit on `listen EINVAL`.
    socket_path = module._daemon_socket_path(
        f"pe-test-{os.getpid()}-{time.monotonic_ns()}.sock")
    daemon = module._RunOwnedPrimeDaemon(
        socket_path, harness.daemon_environment(),
        cwd=harness.daemon_working_directory, executable=str(launcher),
    )
    child = None
    try:
        if preflight:
            module._preflight_prime_runtime(
                str(launcher), env=harness.daemon_environment(),
                cwd=harness.daemon_working_directory,
                kernel_check=module._prime_kernel_check(package_root),
            )
        daemon.start()
        child = harness.prepare_child("test")
        trajectory = tmp_path / "trajectory"
        trajectory.mkdir()
        rollout = RolloutRequest(
            rollout_id="rl_executed",
            turn_input="Reply with the word hello. Do not use any tools.",
            llm_url=gateway.base_url, api_key="perk_test",
            artifacts_dir=str(trajectory),
            deadline_ts=time.time() + budget_s,
            metadata={"completion_gate": "true"},
        )
        argv = list(launch) + [
            socket_path if arg == module._DAEMON_SOCKET else arg
            for arg in module.ARGV[1:]
        ]
        env = build_child_env(
            rollout_id=rollout.rollout_id, llm_url=gateway.base_url,
            api_key="perk_test", trajectory_dir=str(trajectory),
            extra={"PERCEPTEYE_TURN_INPUT": rollout.turn_input,
                   "PATH": _path_without_uv(tmp_path), **child.environment},
        )
        started = time.monotonic()
        out = run_rollout_as_command(
            argv, rollout, trajectory_dir=str(trajectory), timeout_s=budget_s,
            gate_flag="--autonomous-gate", env=env,
            output_adapter=PrimeAgentOutputAdapter(), cwd=str(trajectory),
        )
        elapsed = time.monotonic() - started
        assert kernel.is_dir(), (
            "Prime deleted the kernel venv the snapshot sealed, as it does "
            "one it judges not ready, to rebuild it; production excludes that "
            "rollout as drift")
        harness.attest_child(child)
        return out, elapsed
    finally:
        # Each step runs whatever the one before it raised. A failed release
        # used to skip daemon.close(), and every such run left its run-owned
        # supervisor alive and orphaned -- dozens of them, days old, on one
        # developer machine.
        try:
            if child is not None:
                harness.release_child(child)
        finally:
            try:
                daemon.close()
            finally:
                harness.close()


def _ci_pinned_prime_version() -> str:
    """The version CI installs, read out of the workflow that installs it.

    READ, not restated. The pin has to live in ``ci.yml`` -- it is an argument
    to ``npm install`` -- so a constant here would be a second declaration of
    it, and the two would drift the first time somebody bumped one. This is the
    consumer of that single declaration.
    """
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    # Two spellings of one pin. `prime-agent` was unpublished from the npm
    # registry in October 2026, so CI installs the identical npm tarball from
    # the upstream GitHub release instead (its own dependencies are pinned
    # tarball URLs). A release URL whose tag and file name disagree is two
    # versions, not one, and is refused below.
    registry = re.findall(r"npm install -g prime-agent@(\S+)", workflow)
    release = re.findall(
        r"npm install -g https://github\.com/PrimeIntellect-ai/prime-agent/"
        r"releases/download/v(\S+?)/prime-agent-(\S+?)\.tgz", workflow)
    for tag, filename in release:
        assert tag == filename, (
            f"ci.yml installs prime-agent from release v{tag} but its file "
            f"names version {filename}")
    found = registry + [tag for tag, _ in release]
    assert len(found) == 1, (
        f"ci.yml must install prime-agent at exactly one pinned version, found "
        f"{found}"
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", found[0]), (
        f"the prime-agent install is not pinned to an exact version: {found[0]!r}. "
        f"The installed dist/ tree is hashed into agent_execution_sha256, so a "
        f"floating install executes a binary the deployment does not attest."
    )
    return found[0]


def test_the_harness_on_this_machine_is_the_version_ci_pins(tmp_path):
    """The pin is a claim about the binary; this is what makes it one.

    ``.github/workflows/ci.yml`` installs a pinned ``prime-agent`` so the
    real-binary tests below execute SOMETHING rather than skipping -- and the
    pin matters beyond reproducibility: the installed ``dist/`` tree is hashed
    into ``agent_execution_sha256``, which is the identity a sealed OPSD
    rollout is claimed under. A comment saying "pinned to what the deployment
    attests" is not a mechanism and cannot be checked from this repo; what CAN
    be checked, and is, is that the harness actually on PATH is the one the
    workflow said it would install. An `npm install` that resolved to something
    else fails here instead of silently changing the attested identity.
    """
    launcher = _runnable_launcher()
    reported = _starts(launcher)
    assert reported is not None, f"{launcher} stopped being runnable"
    pinned = _ci_pinned_prime_version()
    assert reported.splitlines()[-1].strip() == pinned, (
        f"ci.yml pins prime-agent@{pinned} but the harness being executed here "
        f"reports {reported!r}. Bump the pin, or install the pinned version: "
        f"the dist/ tree is hashed into the sealed execution identity."
    )


def test_the_example_argv_actually_produces_a_completion(
    tmp_path, provisioned_kernel,
):
    """The whole point of claiming a rollout: something comes back.

    Executes the example's real launch command against the real binary. An
    ARGV that parses but cannot reach a provider, a daemon that never yields a
    session, an extension that throws on load -- every one of those fails here
    and none of them fails a test that inspects the argv.
    """
    launch = _runnable_launcher()
    with _Gateway(answer=True) as gateway:
        out, elapsed = _run_one_rollout(
            tmp_path, launch, gateway, provisioned_kernel, budget_s=120.0)

    assert gateway.requests, (
        "the harness claimed the rollout and never reached the gateway"
    )
    assert out.exclusion_reason != "deadline_exceeded", out.exclusion_detail
    assert out.llm_call_count and out.llm_call_count >= 1, (
        f"no assistant message observed on the event stream: {out.exclusion_detail}"
    )
    assert elapsed < 120.0


def test_a_gateway_that_never_answers_does_not_consume_the_whole_budget(
    tmp_path, provisioned_kernel,
):
    """The 2026-09-15 failure, against the real binary.

    Prime's provider client has no argv flag for this; the ceiling is
    ``retry.provider.timeoutMs`` in the settings the adapter writes into every
    child. Without it the child sits in one request for Prime's own 600000ms
    default and the runner kills it with nothing to report.
    """
    launch = _runnable_launcher()
    shipped = json.loads(prime_agent._FIXED_SETTINGS)
    assert shipped.get("retry", {}).get("provider", {}).get("timeoutMs"), (
        "the attested settings carry no provider request ceiling, so nothing "
        "bounds a wedged completion"
    )
    shipped["retry"]["provider"]["timeoutMs"] = WEDGED_CEILING_MS
    body = (json.dumps(shipped, separators=(",", ":")) + "\n").encode()

    import hashlib
    original, original_sha = (
        prime_agent._FIXED_SETTINGS, prime_agent._FIXED_SETTINGS_SHA256)
    prime_agent._FIXED_SETTINGS = body
    prime_agent._FIXED_SETTINGS_SHA256 = hashlib.sha256(
        prime_agent._SETTINGS_DIGEST_DOMAIN + body).hexdigest()
    budget_s = 150.0
    try:
        with _Gateway(answer=False) as gateway:
            out, elapsed = _run_one_rollout(
                tmp_path, launch, gateway, provisioned_kernel, budget_s=budget_s)
    finally:
        prime_agent._FIXED_SETTINGS = original
        prime_agent._FIXED_SETTINGS_SHA256 = original_sha

    assert gateway.requests, "the child never reached the wedged gateway"
    # It came back on the CEILING, not on the runner's kill.
    assert elapsed < budget_s / 2, (
        f"the child spent {elapsed:.0f}s of a {budget_s:.0f}s budget inside a "
        f"request the gateway never answered"
    )
    assert out.exclusion_reason != "deadline_exceeded", (
        f"killed on the deadline rather than returning on the ceiling: "
        f"{out.exclusion_detail}"
    )


def _kernel_copy(provisioned, destination: pathlib.Path) -> pathlib.Path:
    """A private copy of the module's kernel, for a test that alters one."""
    shutil.copytree(provisioned, destination, symlinks=True)
    return destination


#: What the policy runs in Prime's kernel: which interpreter, from which venv,
#: and anything the serving shell's Python configuration put there.
_WHICH_KERNEL = (
    "import importlib.util, os, sys\n"
    "import jupyter\n"
    "print('kernel-prefix=' + sys.prefix)\n"
    "print('ambient=' + repr(\n"
    "    [n for n in ('PYTHONPATH', 'PYTHONHOME', 'PYTHONSTARTUP', 'IPYTHONDIR',\n"
    "                 'PYTHONPYCACHEPREFIX', 'PYTHONEXECUTABLE',\n"
    "                 '__PYVENV_LAUNCHER__')\n"
    "     if n in os.environ]\n"
    "    + [m for m in ('ambient_marker', 'ambient_venv_marker')\n"
    "       if importlib.util.find_spec(m)]\n"
    "    + [g for g in ('AMBIENT_STARTUP', 'AMBIENT_PROFILE') if g in globals()]\n"
    "    + [a for a in ('AMBIENT_BYTECODE',) if hasattr(jupyter, a)]))\n"
)

#: A module of the kernel's own that nothing imports when the kernel starts:
#: jupyter_core's ``python -m jupyter`` shim in its site-packages, which
#: ``_WHICH_KERNEL`` imports.
_BYTECODE_SHADOWED_MODULE = "jupyter"

#: Run by the kernel's own interpreter under the ambient PYTHONPYCACHEPREFIX:
#: compiles the module's source plus a marker and writes it where that
#: interpreter looks for the module's bytecode, behind the PEP 552 header it
#: checks -- its magic number, then the source's modification time and size
#: -- so it loads this in place of the source, though no byte of the venv
#: changed.
_PLANT_BYTECODE = (
    "import importlib.util, marshal, os, sys\n"
    "origin = importlib.util.find_spec(sys.argv[1]).origin\n"
    "with open(origin, 'rb') as f:\n"
    "    source = f.read()\n"
    "code = compile(source + b'\\nAMBIENT_BYTECODE = 1\\n', origin, 'exec')\n"
    "cached = importlib.util.cache_from_source(origin)\n"
    "os.makedirs(os.path.dirname(cached), exist_ok=True)\n"
    "st = os.stat(origin)\n"
    "header = importlib.util.MAGIC_NUMBER + b''.join(\n"
    "    (n & 0xFFFFFFFF).to_bytes(4, 'little')\n"
    "    for n in (0, int(st.st_mtime), st.st_size))\n"
    "with open(cached, 'wb') as f:\n"
    "    f.write(header + marshal.dumps(code))\n"
    "print(cached)\n"
)


def _ambient_bytecode(kernel: pathlib.Path, prefix: pathlib.Path) -> None:
    """Compiled code for one of ``kernel``'s modules, in a cache tree outside it.

    Written by the kernel's own interpreter, so it carries that interpreter's
    magic number; a PYTHONPYCACHEPREFIX naming ``prefix`` makes the kernel run
    it in place of the module's source in the venv ``kernel_sha256`` names.
    """
    planted = subprocess.run(
        [str(kernel / "bin" / "python"), "-c", _PLANT_BYTECODE,
         _BYTECODE_SHADOWED_MODULE],
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=60,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPYCACHEPREFIX": str(prefix),
             "PYTHONDONTWRITEBYTECODE": "1"},
        cwd=str(prefix.parent), check=False,
    )
    assert planted.returncode == 0, planted.stderr[-800:]
    assert pathlib.Path(planted.stdout.strip()).is_relative_to(prefix), planted.stdout


def _ambient_python_configuration(tmp_path, monkeypatch, provisioned) -> None:
    """The serving shell's Python configuration, each piece able to run code.

    A module on PYTHONPATH, a file IPython runs in the kernel at startup
    (PYTHONSTARTUP), an IPython profile whose startup script it runs there too
    (IPYTHONDIR), a PYTHONHOME no interpreter can start under, a bytecode
    cache tree (PYTHONPYCACHEPREFIX) that ``_ambient_bytecode`` fills for the
    kernel the test serves, and, on macOS, the interpreter path of another
    venv (PYTHONEXECUTABLE and __PYVENV_LAUNCHER__), whose site-packages the
    kernel would then run on -- a copy of the provisioned kernel, so it still
    starts, plus a marker module. None of it is in the venv ``kernel_sha256``
    names.
    """
    path = tmp_path / "ambient-path"
    path.mkdir()
    (path / "ambient_marker.py").write_text("")
    startup = tmp_path / "ambient-startup.py"
    startup.write_text("AMBIENT_STARTUP = 1\n")
    profile_startup = tmp_path / "ambient-ipython" / "profile_default" / "startup"
    profile_startup.mkdir(parents=True)
    (profile_startup / "00-ambient.py").write_text("AMBIENT_PROFILE = 1\n")
    monkeypatch.setenv("PYTHONPATH", str(path))
    monkeypatch.setenv("PYTHONSTARTUP", str(startup))
    monkeypatch.setenv("IPYTHONDIR", str(tmp_path / "ambient-ipython"))
    monkeypatch.setenv("PYTHONHOME", str(tmp_path / "no-python-home"))
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", str(tmp_path / "ambient-pycache"))
    elsewhere = _kernel_copy(provisioned, tmp_path / "ambient-venv")
    site = next((elsewhere / "lib").glob("python*/site-packages"))
    (site / "ambient_venv_marker.py").write_text("")
    monkeypatch.setenv("PYTHONEXECUTABLE", str(elsewhere / "bin" / "python"))
    monkeypatch.setenv("__PYVENV_LAUNCHER__", str(elsewhere / "bin" / "python"))


def _kernel_without_a_default_package(provisioned, destination: pathlib.Path):
    """A copy of the kernel with ``ipykernel`` and ``rlm`` and without
    ``tyro``, one of the default packages this prime-agent installs and then
    requires -- a kernel Prime refuses, and only the installed release can
    say why."""
    kernel = _kernel_copy(provisioned, destination)
    required = json.loads((kernel / ".bootstrap-version").read_text())["extraUvArgs"]
    assert "tyro" in required, (
        f"this prime-agent no longer installs tyro by default ({required}); "
        f"remove another package it does")
    site = next((kernel / "lib").glob("python*/site-packages"))
    shutil.rmtree(site / "tyro")
    return kernel


@pytest.mark.parametrize("provisioned_by", ["this prime-agent", "another prime-agent"])
def test_the_sealed_kernel_runs_the_policy_code_as_it_is(
    tmp_path, monkeypatch, provisioned_kernel, provisioned_by,
):
    """Prime runs ipython in the sealed venv and never rebuilds it.

    A venv another prime-agent version provisioned carries that version's
    readiness marker. Handed over as PRIME_AGENT_KERNEL_VENV, Prime judged it
    stale at the first kernel start, ``rm -rf``ed it and began rebuilding it
    in place; the rollout completed anyway and only the attestation after it
    saw the sealed kernel gone. Handed over as PRIME_AGENT_KERNEL_PYTHON --
    Prime's documented way to use an existing environment without
    auto-bootstrap -- the interpreter is checked and used as it is: the tool
    call runs in the sealed venv, which is still there, unchanged, afterwards.

    And it runs only the sealed venv. Prime's kernel inherits its session's
    environment, so the serving shell's Python configuration used to reach it
    -- a PYTHONPATH module importable, a startup file run, a venv module's
    bytecode loaded from a cache outside the venv, on macOS another venv's
    site-packages -- as code the seal never covered; none of it does now.
    """
    _ambient_python_configuration(tmp_path, monkeypatch, provisioned_kernel)
    kernel = provisioned_kernel
    if provisioned_by == "another prime-agent":
        kernel = _kernel_copy(provisioned_kernel, tmp_path / "kernel-venv")
        marker = kernel / ".bootstrap-version"
        recorded = json.loads(marker.read_text())
        assert recorded.get("runtime"), (
            "Prime's bootstrap no longer records the runtime it installed; "
            "this test no longer models a kernel another version provisioned")
        # The runtime identity is what differs between prime-agent versions:
        # a hash of the bundled prime-agent-runtime source.
        recorded["runtime"] = "sha256:" + "0" * 64
        marker.write_text(json.dumps(recorded, separators=(",", ":")) + "\n")
    _ambient_bytecode(kernel, tmp_path / "ambient-pycache")
    launch = _runnable_launcher()
    with _Gateway(answer=True, ipython=_WHICH_KERNEL) as gateway:
        out, _ = _run_one_rollout(
            tmp_path, launch, gateway, kernel, budget_s=120.0)

    assert out.exclusion_reason is None, out.exclusion_detail
    assert gateway.tool_results, "the policy's ipython call never came back"
    assert f"kernel-prefix={kernel.resolve()}" in gateway.tool_results[0], (
        f"the ipython call did not run in the sealed kernel venv: "
        f"{gateway.tool_results[0][:400]}")
    assert "ambient=[]" in gateway.tool_results[0], (
        f"the serving shell's Python configuration reached the kernel: "
        f"{gateway.tool_results[0][:400]}")


def test_a_kernel_prime_would_refuse_is_refused_before_serving(
    tmp_path, provisioned_kernel,
):
    """Prime's own verdict, before the first rollout, in Prime's words.

    PRIME_AGENT_KERNEL_PYTHON is used as it is or refused -- and a refused
    kernel does not stop a session: it completes, with every ``ipython`` call
    answered by the refusal, a rollout that would only be excluded one at a
    time (see the next test). The snapshot probes only the two packages
    Prime's docs name for it, ``ipykernel`` and ``rlm``; this kernel has both
    and lacks one of the default packages THIS release requires, which only
    the installed prime-agent can say, so the example's preflight asks it.
    """
    kernel = _kernel_without_a_default_package(
        provisioned_kernel, tmp_path / "kernel-venv")

    module = _example(tmp_path)
    launcher = pathlib.Path(_runnable_launcher()[-1])
    source = tmp_path / "empty-harness"
    source.mkdir()
    harness = PrimeHarnessSnapshot(  # ipykernel and rlm import: accepted
        source, workspace_dir=tmp_path / "_harness_runs", kernel_venv=kernel)
    try:
        with pytest.raises(RuntimeError) as refused:
            module._preflight_prime_runtime(
                str(launcher), env=harness.daemon_environment(),
                cwd=harness.daemon_working_directory,
                kernel_check=module._prime_kernel_check(launcher.parents[2]),
            )
    finally:
        harness.close()
    assert "prime-agent refuses the kernel PRIME_AGENT_KERNEL_VENV names" in str(
        refused.value)
    assert "tyro" in str(refused.value)


def test_a_rollout_whose_kernel_prime_refuses_is_excluded(
    tmp_path, provisioned_kernel,
):
    """The refusal the startup preflight cannot rule out, excluded where it lands.

    Prime checks the kernel each time a session starts it, one interpreter
    per package, and counts a package that merely failed to spawn as missing,
    so a kernel the preflight accepted can still be refused in a rollout --
    and the session completes regardless, the policy's ``ipython`` call
    answered by the refusal instead of running. Served here with the
    preflight skipped, a kernel without one of Prime's default packages gives
    exactly that rollout: attested, completed, its call refused. It is
    excluded as an environment fault, not graded as the policy's attempt.
    """
    kernel = _kernel_without_a_default_package(
        provisioned_kernel, tmp_path / "kernel-venv")
    launch = _runnable_launcher()
    with _Gateway(answer=True, ipython=_WHICH_KERNEL) as gateway:
        out, _ = _run_one_rollout(
            tmp_path, launch, gateway, kernel, budget_s=120.0, preflight=False)

    refusal = "PRIME_AGENT_KERNEL_PYTHON points to a Python missing"
    assert gateway.tool_results, "the policy's ipython call never came back"
    assert gateway.tool_results[0].startswith(refusal), (
        f"Prime did not refuse this kernel, so this rollout shows nothing: "
        f"{gateway.tool_results[0][:400]}")
    assert out.exclusion_reason == "substrate_unavailable", out.exclusion_detail
    assert refusal in (out.exclusion_detail or "")
    assert str(kernel) not in (out.exclusion_detail or ""), (
        "the exclusion carried the kernel's path")
