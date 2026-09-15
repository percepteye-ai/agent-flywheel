"""Serve prime-agent rollouts. No Python integration — it is a TypeScript binary.

    PERCEPTEYE_CONTROL_PLANE_URL=... PERCEPTEYE_API_KEY=... \
      python examples/prime_agent/serve_prime.py

prime-agent is served through the COMMAND entrypoint, which is the path that
exists for exactly this: an agent that is a process rather than an importable
Python object. It needs no Percepteye code, and the only thing added to it is
one extension that points its completions at the per-rollout endpoint.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import signal
import socket
import stat
import subprocess
import tempfile
import time
from collections.abc import Callable
from types import FrameType
from typing import Any

import agent_flywheel as fw
from agent_flywheel.adapters.prime_agent import (
    PrimeAgentOutputAdapter,
    PrimeHarnessSnapshot,
)

HERE = pathlib.Path(__file__).resolve().parent


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer; got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be a positive integer; got {raw!r}")
    return value


# One explicit configuration object feeds both sides of Prime's model lookup:
# ARGV selects the model in the parent, and route.ts registers that exact id and
# limits in the rollout child. The optional Prime adapter translates these
# parent settings into adapter-owned child variables and attests their digest;
# core keeps its whole PERCEPTEYE_ namespace scrubbed.
MODEL_ID = os.environ.get("PERCEPTEYE_MODEL_ID", "policy").strip()
if not MODEL_ID:
    raise RuntimeError("PERCEPTEYE_MODEL_ID must not be blank")
CONTEXT_WINDOW = _positive_int_env("PERCEPTEYE_CONTEXT_WINDOW", 65536)
MAX_TOKENS = _positive_int_env("PERCEPTEYE_MAX_TOKENS", 4096)
if MAX_TOKENS > CONTEXT_WINDOW:
    raise RuntimeError("PERCEPTEYE_MAX_TOKENS cannot exceed PERCEPTEYE_CONTEXT_WINDOW")

#: WHERE ROLLOUTS RUN. Each rollout's child process is started with its cwd
#: inside this directory, and the agent has a real shell — so this must never
#: be a source tree. Left to the default (``$PWD/.percepteye-artifacts``) and
#: launched from a checkout, rollouts write their task artifacts into the
#: repository and can reach the rest of it with ordinary relative paths: on
#: 2026-08-16 a sweep launched that way left shell scripts beside the package
#: and removed tracked files from the working tree, including the licence and
#: the build config.
#:
#: A dedicated directory outside any checkout is the cheap part of the fix.
#: The complete one is an OS-level sandbox — this SDK's ``isolation`` is a
#: process boundary and an environment scrub, and says nothing about the
#: filesystem. Run untrusted agents accordingly.
WORKSPACE = pathlib.Path(
    os.environ.get("PRIME_WORKSPACE")
    or (pathlib.Path.home() / ".percepteye" / "prime-agent-workspace"),
).expanduser()
WORKSPACE.mkdir(parents=True, exist_ok=True)

#: `-p` is print mode: one task, non-interactive, answer on stdout. The task
#: also arrives on stdin, which prime-agent merges into the prompt — the SDK
#: sends it both ways because CLIs differ and guessing wrongly is silent.
#:
#: The autonomous limits are bounded deliberately. A rollout that runs until it
#: decides to stop is a rollout whose cost we do not control, and the episode
#: budget is the only clock the control plane can enforce from outside.
#:
#: Continuations raised 2 -> 6 on 2026-08-12. At 2, run 08e40b36's rollout
#: rl_dz73eyv died `entrypoint_error: exit 1: Autonomous run stopped before
#: terminal evidence; maxContinuations reached (2/2)` — the cap cut the agent
#: off BEFORE it produced the evidence the reward reads, so the rollout scored
#: nothing and told us nothing. A bound that stops the work short of any
#: observable result is not cost control, it is a wasted rollout. The episode
#: budget still bounds the real cost from outside.
#: ONE daemon socket per harness process, at a path nothing else uses.
#:
#: prime-agent runs a daemon supervisor that owns a registry entry. When a
#: second generation takes that entry over, every agent still bound to the old
#: one dies instantly with "Daemon supervisor generation ... no longer owns its
#: registry entry" — which the SDK reports as ``entrypoint_error`` and the
#: liveness canary reports as "no successful real-provider LLM call". Run
#: 32403f96 lost 16 of 40 rollouts that way, mid-run, after a manual daemon
#: kill had "fixed" it an hour earlier.
#:
#: The default socket is shared by every prime-agent on the box: a previous
#: harness's orphaned daemons, and the operator's own interactive sessions.
#: A path keyed to THIS process shares with none of them, so a stale
#: generation cannot evict us and we cannot evict anyone. Per-ROLLOUT sockets
#: would isolate too, but daemons outlive the agent that started them, so that
#: leaks a daemon per rollout; one per harness run is the same isolation
#: without the accumulation.
#:
#: ``prime-agent --version`` works regardless — it never touches the daemon —
#: so it is not a liveness check for this.
#:
#: The path carries this process's PID, so it is different on every restart.
#: THAT is transport, not identity: the same build of this agent, restarted,
#: is the same agent. It is declared as such in ``VOLATILE_COMMAND_VALUES``
#: below and reaches ``fw.serve``, which drops the value — and only the value —
#: from ``execution_sha256``. Without that declaration each restart minted a
#: new execution identity and invalidated any OPSD run already prepared
#: against the previous one (observed: prime-agent registered
#: 3502c7cd… on 2026-09-12 and 409c43ee… on 2026-09-15).
_DAEMON_SOCKET = str(
    pathlib.Path(tempfile.gettempdir()) / f"percepteye-prime-{os.getpid()}.sock"
)

#: The one argument in ARGV whose VALUE varies per process. ``--daemon-socket``
#: itself stays in the identity, so a build that runs without a private daemon
#: socket remains a different execution from this one.
VOLATILE_COMMAND_VALUES = ("--daemon-socket",)


class _RunOwnedPrimeDaemon:
    """Keep Prime's supervisor outside every disposable rollout HOME.

    Prime sends each client-owned session's full launch environment to its
    worker, so the workers still use the per-rollout agent directories. The
    supervisor itself persists descriptor files beside the agent directory it
    started with, though. If the first rollout starts it, deleting that
    rollout's HOME deletes live daemon state. Start it once under the fixed
    daemon environment owned by ``PrimeHarnessSnapshot`` and stop it before
    the snapshot is removed.
    """

    def __init__(
        self,
        socket_path: str,
        env: dict[str, str],
        *,
        cwd: str | os.PathLike[str],
        executable: str = "prime-agent",
    ) -> None:
        self.socket_path = socket_path
        self.env = dict(env)
        self.cwd = str(cwd)
        self.executable = executable
        self.process: subprocess.Popen[bytes] | None = None
        self._owns_socket_path = False

    def start(self, timeout_s: float = 20.0) -> None:
        if self.process is not None:
            raise RuntimeError("Prime daemon already started")
        socket_entry = pathlib.Path(self.socket_path)
        if os.name != "nt" and (socket_entry.exists() or socket_entry.is_symlink()):
            raise RuntimeError(
                f"Prime daemon socket already exists; refusing ambiguous owner: "
                f"{self.socket_path}"
            )
        self.process = subprocess.Popen(
            [self.executable, "--mode", "daemon", "--daemon-socket", self.socket_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=self.cwd,
            env=self.env,
            start_new_session=(os.name != "nt"),
        )
        self._owns_socket_path = True
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"Prime daemon exited during startup with code {self.process.returncode}"
                )
            if self._accepting_connections():
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"Prime daemon did not become ready within {timeout_s:g} seconds"
        )

    def _accepting_connections(self) -> bool:
        if os.name == "nt" or not hasattr(socket, "AF_UNIX"):
            return pathlib.Path(self.socket_path).exists()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(0.2)
        try:
            client.connect(self.socket_path)
            return True
        except OSError:
            return False
        finally:
            client.close()

    def close(self) -> None:
        process = self.process
        if process is None:
            return
        try:
            subprocess.run(
                [
                    self.executable,
                    "shutdown",
                    "--force",
                    "--daemon-socket",
                    self.socket_path,
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=self.cwd,
                env=self.env,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        finally:
            self.process = None
            self._remove_owned_socket()

    def _remove_owned_socket(self) -> None:
        """Remove only the Unix socket this instance started with."""
        if not self._owns_socket_path or os.name == "nt":
            return
        path = pathlib.Path(self.socket_path)
        try:
            entry = path.lstat()
            if stat.S_ISSOCK(entry.st_mode):
                path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._owns_socket_path = False


SignalHandler = signal.Handlers | int | Callable[[int, FrameType | None], Any] | None


def _install_cleanup_signal_handlers() -> dict[int, SignalHandler]:
    """Turn service-manager termination into the normal cleanup path.

    ``fw.serve`` deliberately owns no process-wide signal policy. This
    standalone example does own a run-scoped Prime supervisor, though, and a
    default SIGTERM/SIGHUP would bypass its ``finally`` block and leave that
    supervisor behind. Raising ``SystemExit`` on the main thread preserves the
    conventional exit status while still unwinding through ``daemon.close``.

    SIGKILL and machine failure remain uncatchable. The unique socket keeps an
    orphan from colliding with the next run; this handler prevents the ordinary
    service-stop cases from creating one in the first place.
    """
    previous: dict[int, SignalHandler] = {}
    terminating = False

    def _exit(signum: int, _frame: FrameType | None) -> None:
        nonlocal terminating
        # ThreadPoolExecutor drains its running rollouts while ``fw.serve``
        # unwinds. A service manager may repeat SIGTERM during that interval;
        # a second exception would interrupt the drain and race snapshot
        # teardown against live children. The first signal owns shutdown.
        if terminating:
            return
        terminating = True
        raise SystemExit(128 + signum)

    cleanup_signals = [signal.SIGTERM, signal.SIGINT]
    if hasattr(signal, "SIGHUP"):
        cleanup_signals.append(signal.SIGHUP)
    try:
        for signum in cleanup_signals:
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, _exit)
    except BaseException:
        # ``signal.signal`` rejects a caller outside the main thread. Do not
        # leave the first handler installed if a later one could not be.
        for restore_signum, handler in previous.items():
            signal.signal(restore_signum, handler)
        raise
    return previous


def _ignore_cleanup_signals(previous: dict[int, SignalHandler]) -> None:
    """Keep a repeated stop from interrupting daemon/snapshot teardown."""
    for signum in previous:
        signal.signal(signum, signal.SIG_IGN)


def _restore_signal_handlers(previous: dict[int, SignalHandler]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


_REQUIRED_PRIME_FLAGS = (
    "--daemon-socket",
    "--mode",
    "--provider",
    "--model",
    "--no-session",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-context-files",
    "--autonomous",
    "--autonomous-max-turns",
    "--autonomous-max-continuations",
    "--autonomous-gate-timeout-ms",
    "--autonomous-gate",
)


def _preflight_prime_runtime(
    executable: str,
    *,
    env: dict[str, str],
    cwd: str | os.PathLike[str],
) -> str:
    """Prove the pinned launcher can start and implements this adapter contract."""

    def _run(option: str) -> subprocess.CompletedProcess[bytes]:
        try:
            return subprocess.run(
                [executable, option],
                stdin=subprocess.DEVNULL,
                capture_output=True,
                cwd=cwd,
                env=env,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError(
                f"prime-agent {option} preflight could not execute: {exc}"
            ) from exc

    version_probe = _run("--version")
    version = (version_probe.stdout or version_probe.stderr).decode(
        "utf-8", "replace"
    ).strip()
    if version_probe.returncode != 0:
        detail = version[:500] or f"exit {version_probe.returncode} without output"
        raise RuntimeError(f"prime-agent runtime preflight failed: {detail}")

    help_probe = _run("--help")
    help_text = (help_probe.stdout + help_probe.stderr).decode("utf-8", "replace")
    if help_probe.returncode != 0:
        detail = help_text.strip()[:500] or (
            f"exit {help_probe.returncode} without output"
        )
        raise RuntimeError(f"prime-agent flag preflight failed: {detail}")
    missing = [flag for flag in _REQUIRED_PRIME_FLAGS if flag not in help_text]
    if missing:
        raise RuntimeError(
            "prime-agent does not implement the required rollout flags: "
            + ", ".join(missing)
        )
    return version.splitlines()[0] if version else "version not reported"


ARGV = [
    "prime-agent",
    "--daemon-socket", _DAEMON_SOCKET,
    "-p",
    "--mode", "json",              # structured events, not prose
    # Discovery outside the attested continual-harness snapshot is another
    # scaffold input.  HOME and the agent directory are isolated below, while
    # these flags also stop upward cwd searches (notably AGENTS.md).  Explicit
    # resources remain supported: route.ts is named immediately afterward.
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-context-files",
    "-e", str(HERE / "route.ts"),  # the routing extension
    # SELECTING the provider the extension registers. Registering one does not
    # use it: without these two flags prime-agent kept its own built-in
    # provider and its own credentials, every completion went somewhere we
    # never saw, and `inference_calls` stayed empty platform-wide — so no
    # trajectory carried an llm_call and the substrate-liveness canary
    # correctly reported "no successful real-provider LLM call" on every run
    # for a week. The rollouts looked fine; we simply owned none of the
    # completion wire, which is the exact failure route.ts warns about in its
    # own header and could not itself prevent.
    "--provider", "percepteye",
    "--model", MODEL_ID,
    # A persisted session exposes Prime's continual-harness refine machinery
    # and lets one rollout rewrite what later replicas see.  Rollouts need an
    # immutable harness identity, so there is deliberately no session to
    # refine or resume. route.ts independently cancels compaction, whose
    # summarizer call would also contaminate the captured policy token stream.
    "--no-session",
    "--autonomous",
    # 12 -> 15 on 2026-08-12. NOT because the agent needs more room to find the
    # answer: on run 516b7a55 it wrote a CORRECT top_ips.sh and kept going,
    # because the grading test is hidden by design and it has no way to know it
    # is finished. Raising this to 30 did not buy completions — it converted
    # `maxTurns` exclusions into `deadline_exceeded` ones against the 300s
    # episode budget, and a second attempt at the same task drifted from a
    # correct solution to a wrong one. 15 is a modest allowance over the point
    # where the work is typically done; the real fix is that the artifact is now
    # graded whether or not the agent terminates tidily.
    "--autonomous-max-turns", os.environ.get("PRIME_MAX_TURNS", "15"),
    "--autonomous-max-continuations", "6",
    # Bounded because the DEFAULT is 300000ms — exactly the 300s episode
    # budget, so one hanging gate consumes the entire rollout and the agent
    # never gets to report. A gate is an existence-and-smoke check; the
    # derived `test -s <file>` form returns instantly, and a model-authored
    # one that runs the deliverable should too. 30s is generous for both and
    # leaves the budget to the actual work.
    "--autonomous-gate-timeout-ms", "30000",
]


def main() -> int:
    source = os.environ.get("PERCEPTEYE_PRIME_HARNESS_SOURCE", "").strip()
    if not source:
        raise RuntimeError(
            "PERCEPTEYE_PRIME_HARNESS_SOURCE is required and must name the "
            "Prime harness directory to snapshot (an empty directory is valid)"
        )
    kernel_venv = os.environ.get("PRIME_AGENT_KERNEL_VENV", "").strip()
    if not kernel_venv:
        raise RuntimeError(
            "PRIME_AGENT_KERNEL_VENV is required and must name a pre-provisioned "
            "Prime kernel runtime; per-child HOME isolation disables its ambient default"
        )
    package_root_raw = os.environ.get("PRIME_AGENT_PACKAGE_ROOT", "").strip()
    if not package_root_raw:
        raise RuntimeError(
            "PRIME_AGENT_PACKAGE_ROOT is required and must name the installed "
            "Prime package containing package.json and dist/"
        )
    package_root = pathlib.Path(package_root_raw).expanduser().resolve()
    package_manifest = package_root / "package.json"
    package_dist = package_root / "dist"
    if not package_manifest.is_file() or not package_dist.is_dir():
        raise RuntimeError(
            "PRIME_AGENT_PACKAGE_ROOT must contain package.json and a dist/ directory"
        )
    prime_launcher = shutil.which("prime-agent")
    if prime_launcher is None:
        raise RuntimeError("prime-agent is not present on PATH")
    try:
        resolved_launcher = pathlib.Path(prime_launcher).resolve()
        resolved_launcher.relative_to(package_root)
    except ValueError as exc:
        raise RuntimeError(
            "prime-agent on PATH does not belong to PRIME_AGENT_PACKAGE_ROOT"
        ) from exc

    harness = PrimeHarnessSnapshot(
        source,
        workspace_dir=WORKSPACE / "_harness_runs",
        kernel_venv=kernel_venv,
        execution_artifacts={
            "prime_distribution": package_dist,
            "prime_package_manifest": package_manifest,
            "route_extension": HERE / "route.ts",
        },
        route_configuration={
            "model_id": MODEL_ID,
            "context_window": str(CONTEXT_WINDOW),
            "max_tokens": str(MAX_TOKENS),
        },
    )
    daemon: _RunOwnedPrimeDaemon | None = None
    previous_signal_handlers: dict[int, SignalHandler] = {}
    try:
        daemon_env = harness.daemon_environment()
        version = _preflight_prime_runtime(
            str(resolved_launcher),
            env=daemon_env,
            cwd=harness.daemon_working_directory,
        )
        print(f"Prime runtime preflight passed: {version}", flush=True)
        daemon = _RunOwnedPrimeDaemon(
            _DAEMON_SOCKET,
            daemon_env,
            cwd=harness.daemon_working_directory,
            executable=str(resolved_launcher),
        )
        previous_signal_handlers = _install_cleanup_signal_handlers()
        daemon.start()
        return fw.serve(
            [str(resolved_launcher), *ARGV[1:]],
            agent_id=os.environ.get("PERCEPTEYE_AGENT_ID", "prime-agent"),
            input_shape="text",
        # TRUE as of 2026-08-12: the event parser landed. prime-agent's
        # `--mode json` stream emits tool_execution_start / tool_execution_end
        # with toolCallId, toolName, args and isError, and the SDK reads them
        # off stdout (event_stream.parse_tool_calls) with no instrumentation in
        # the agent at all. So this is now an assertion we can actually back.
        #
        # It stays honest at the edges rather than by promise: a call that ends
        # without an isError is recorded `unknown`, never rounded up to ok, and
        # a stream carrying no tool events leaves tool_calls None rather than
        # claiming zero calls were made.
            reports_tool_calls=True,
        # Raised 1 -> 4 on 2026-08-12. A GRPO group mints its rollouts at once,
        # each stamped with an ABSOLUTE deadline_at; at concurrency 1 the rest
        # sat in the queue burning that clock while this process worked one of
        # them. Run 08e40b36 lost 5 of 6 rollouts that way — two never claimed
        # at all ("rollout budget elapsed before any agent claimed it") and one
        # was claimed with 55s of its 300s left and killed mid-flight, because
        # the SDK takes min(rollout_timeout_s, seconds_remaining) rather than a
        # fresh clock. Serving concurrency has to match the group width or the
        # group's own deadline is what kills it.
            concurrency=int(os.environ.get("PRIME_CONCURRENCY", "4")),
        # How prime-agent spells its completion gate. `--autonomous` is
        # documented as "continue until gates pass or a limit is reached", so
        # with no gate registered a LIMIT is the only exit that exists — which
        # is why every rollout on runs b6b4388d and b60067db ended on maxTurns
        # or maxTokens and was excluded from training even when its artifact
        # was correct. The gate COMMAND is per task and rides in the rollout
        # metadata; this only says how to spell the option.
            gate_flag="--autonomous-gate",
        # Anchor every rollout's working directory outside any checkout. See
        # WORKSPACE above — without this the cwd defaults to the directory the
        # server happened to be started from.
            artifacts_dir=str(WORKSPACE),
            # Every command child (including the discovery warm-up) receives a
            # private HOME/agent directory cloned from this one immutable
            # run-scoped snapshot.  Post-run drift is an exclusion, and the
            # generic execution identity is emitted as
            # agent_fingerprint.execution_sha256; the Prime-only component
            # hashes remain diagnostic evidence.
            execution_snapshot=harness,
            command_output_adapter=PrimeAgentOutputAdapter(),
            # The socket path is per-process; see VOLATILE_COMMAND_VALUES.
            volatile_command_values=VOLATILE_COMMAND_VALUES,
        )
    finally:
        # Ignore repeated stop requests while the daemon and its attested
        # snapshot are being dismantled. The installed handler independently
        # suppresses repeats during fw.serve's in-flight-rollout drain.
        try:
            _ignore_cleanup_signals(previous_signal_handlers)
        finally:
            try:
                if daemon is not None:
                    daemon.close()
            finally:
                try:
                    harness.close()
                finally:
                    _restore_signal_handlers(previous_signal_handlers)


if __name__ == "__main__":
    raise SystemExit(main())
