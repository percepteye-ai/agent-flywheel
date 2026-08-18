"""Running one rollout, and the loop that runs many.

Isolation, honestly stated
--------------------------
``subprocess`` is the default and the only mode with a real timeout. Python
cannot interrupt a synchronous call in another thread; in-process, a hung
agent means we stop *waiting* while it keeps running and holding its
resources. A child process can actually be killed. It is also the only mode
where two concurrent rollouts cannot see each other's endpoint or credential.

The cost is a constraint the caller must satisfy: a child cannot receive a
live Python object, so subprocess isolation requires an entrypoint the child
can IMPORT -- ``"my_package.agent:build"``. Passing a constructed object is
supported, and then isolation must be ``inprocess`` with ``concurrency=1``.
That is checked at registration, not discovered at rollout time.
"""
from __future__ import annotations

import concurrent.futures as _cf
import contextvars
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .contract import RolloutOutput, RolloutRequest, ToolCallOutcome
from .entrypoint import InputShape, ResolvedEntrypoint, resolve_entrypoint
from .errors import ConfigurationError, LeaseLost, TransportError
from .isolation import build_child_env
from .outcomes import LEDGER_FILENAME, OutcomeLedger, _RolloutScope
from .transport import AttachTransport, sleep_ms

__all__ = ["RolloutRunner", "run_rollout_in_process"]

_SPEC_SEP = ":"

#: Registration is a digest and a size, so the ceiling is about what is
#: plausibly a deliverable rather than what fits in memory. Above it the
#: artifact is recorded as skipped, with its size, rather than dropped.
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
#: Per-file inline cap: a small UTF-8 artifact travels with the report so a
#: grader can READ it, not just verify a digest of it. 64KB covers scripts,
#: configs, and test outputs — the deliverables of a coding/REPL agent.
MAX_INLINE_ARTIFACT_BYTES = 64 * 1024
#: Total inline budget per report: two hundred small files must not become a
#: 12MB report. Files past the budget keep their digest-string value.
MAX_INLINE_TOTAL_BYTES = 256 * 1024


def _is_import_spec(obj: Any) -> bool:
    return isinstance(obj, str) and _SPEC_SEP in obj


def _load_spec(spec: str) -> Any:
    """Import ``module:attr`` and return the attribute (calling a factory)."""
    import importlib

    mod_name, _, attr = spec.partition(_SPEC_SEP)
    mod = importlib.import_module(mod_name)
    try:
        target = getattr(mod, attr)
    except AttributeError as exc:
        raise ConfigurationError(f"{spec!r}: module {mod_name} has no attribute {attr!r}") from exc
    return target


def build_agent(spec: str) -> Any:
    """``"module:attr"`` -> the agent object. The ONE place this is decided.

    The parent resolves the entrypoint at startup and the child re-resolves it
    per rollout. When those two arrived at the agent by different code the
    child could run something the parent never validated, so both call this.

    A CLASS is instantiated. Probing for agent-shaped methods first would find
    ``SomeAgent.invoke`` on the class itself -- an unbound function -- and call
    it with the turn input as ``self``, which either raises deep inside the
    agent or, for a ``staticmethod``, quietly runs with no instance at all.
    """
    import inspect

    target = _load_spec(spec)
    if inspect.isclass(target):
        try:
            return target()
        except TypeError as exc:
            raise ConfigurationError(
                f"{spec!r} names a class that cannot be constructed with no "
                f"arguments ({exc}). Point the entrypoint at a zero-argument "
                f"factory that builds a configured instance instead."
            ) from exc
    if callable(target) and not _looks_like_agent(target):
        return target()
    return target


def run_rollout_in_process(
    resolved: ResolvedEntrypoint,
    rollout: RolloutRequest,
    *,
    ledger_dir: str,
    timeout_s: float,
    adapters: list[str] | str | None = None,
) -> RolloutOutput:
    """Drive the agent once, in this process, under a rollout-scoped context.

    ``adapters`` binds framework integrations for the duration of the rollout,
    which is what makes outcome capture require no instrumentation in the
    customer's code. Bound HERE rather than in the parent because this is the
    process the agent runs in -- a callback registered in the parent would
    never see a child's tool calls.
    """
    started = time.time()
    unbinds: list[Any] = []
    if adapters is not None:
        from .binding import bind_selected

        report, unbinds = bind_selected(
            adapters, llm_url=rollout.llm_url, api_key=rollout.api_key
        )
        _ = report
    with _RolloutScope(rollout.rollout_id, ledger_dir):
        arg = resolved.build_arg(rollout.turn_input)
        try:
            if resolved.is_async:
                import asyncio

                raw = asyncio.run(_await_with_timeout(resolved.fn(arg), timeout_s))
            else:
                # A worker thread starts with an EMPTY context, so the rollout
                # binding set just above would be invisible inside the agent and
                # every recorded outcome would be silently discarded. Copy the
                # context in explicitly and run the agent inside it.
                ctx = contextvars.copy_context()
                # Deliberately NOT a `with` block. Executor.__exit__ calls
                # shutdown(wait=True), which on the timeout path BLOCKS during
                # unwinding until the hung agent returns -- so the TimeoutError
                # could not escape and the budget was cosmetic. Measured: a 1s
                # budget against an 8s agent returned after 8s, carrying the
                # right reason, which is why tests asserting the reason passed.
                # The subprocess path below carries the same warning; this
                # was the one place that had the bug the comment describes.
                pool = _cf.ThreadPoolExecutor(max_workers=1)
                fut = pool.submit(ctx.run, resolved.fn, arg)
                try:
                    raw = fut.result(timeout=timeout_s)
                except BaseException:
                    # wait=False: stop waiting on the wedged worker. Python
                    # cannot kill a thread, which is exactly why subprocess is
                    # the default and the only isolation with a real timeout.
                    pool.shutdown(wait=False, cancel_futures=True)
                    raise
                pool.shutdown(wait=True)
        except _cf.TimeoutError:
            return RolloutOutput(
                final_text="",
                exclusion_reason="deadline_exceeded",
                exclusion_detail=f"entrypoint exceeded {timeout_s:.0f}s",
                timing={"elapsed_s": time.time() - started},
            )
        except Exception as exc:
            # The customer's agent raising is a rollout result, not our error.
            return RolloutOutput(
                final_text="",
                success=False,
                exclusion_reason="entrypoint_error",
                exclusion_detail=f"{type(exc).__name__}: {exc}",
                timing={"elapsed_s": time.time() - started},
            )

    for u in unbinds:
        # Best effort: a failed unbind must not lose a completed rollout, and
        # the process is short-lived anyway under the default isolation.
        try:
            if callable(u):
                u()
        except Exception:
            pass

    out = _coerce_output(raw)
    out.timing["elapsed_s"] = time.time() - started
    _attach_ledger(out, ledger_dir, rollout.rollout_id)
    return out


async def _await_with_timeout(coro: Any, timeout_s: float) -> Any:
    import asyncio

    return await asyncio.wait_for(coro, timeout=timeout_s)


def _coerce_output(raw: Any) -> RolloutOutput:
    """Turn whatever the agent returned into a RolloutOutput.

    Deliberately shallow. Guessing at rich structure is how an earlier SDK
    produced confident nonsense; anything beyond text belongs in an explicit
    ``RolloutOutput`` the agent returns, or in ``record_tool_call``.
    """
    if isinstance(raw, RolloutOutput):
        return raw
    if isinstance(raw, str):
        return RolloutOutput(final_text=raw)
    if isinstance(raw, dict):
        for key in ("final_text", "output", "result", "content", "text", "answer"):
            v = raw.get(key)
            if isinstance(v, str):
                return RolloutOutput(final_text=v)
        return RolloutOutput(final_text=json.dumps(raw, default=str))
    return RolloutOutput(final_text="" if raw is None else str(raw))


def _attach_ledger(out: RolloutOutput, ledger_dir: str, rollout_id: str) -> None:
    """Merge recorded tool calls, preserving the None/[] distinction."""
    if out.tool_calls is not None:
        return
    recorded = OutcomeLedger(ledger_dir, rollout_id).read()
    if recorded is not None:
        out.tool_calls = recorded


#: Wall-clock held back from the agent so its RESULT can still be delivered.
#: Collecting artifacts and posting the report are network calls against a
#: lease, and a rollout run right up to its deadline has no time left for
#: either — the deliverable is produced and then dropped at the final step.
#: Small enough not to matter against a real episode budget, large enough to
#: cover a slow round-trip.
REPORT_MARGIN_S = 10.0

#: How often the heartbeat thread wakes to see what is due. Deliberately small
#: and fixed: the cadence that matters is PER ROLLOUT, because concurrent
#: rollouts can hold leases of different lengths, and a single global interval
#: computed once is wrong for all but one of them -- computed BEFORE anything
#: is in flight, it is wrong for all of them.
_BEAT_TICK_S = 1.0


def _kill_tree(proc: subprocess.Popen[str]) -> None:
    """Kill the command AND everything it spawned.

    Whatever the agent started is in the same process group, so the group is
    the unit that has to die. Falls back to killing the one process where
    process groups do not exist.
    """
    if os.name != "nt":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            return
        except (ProcessLookupError, PermissionError, OSError):
            pass
    proc.kill()


def _is_command(entrypoint: Any) -> bool:
    """A list/tuple argv means the agent is a PROCESS, not a Python object.

    The importable-entrypoint requirement is the single biggest limit on who
    can be served: an agent that is a compiled binary, a Node CLI, or anything
    not written in Python simply had no way in. A command has the same
    contract as everything else -- it is handed the rollout through the
    environment, it writes artifacts, and its outcomes come from the ledger.
    """
    return isinstance(entrypoint, (list, tuple)) and bool(entrypoint)


def _argv_with_gate(
    argv: list[str], rollout: RolloutRequest, gate_flag: str | None
) -> list[str]:
    """``argv`` plus this rollout's completion gate, when both sides declare one.

    Two independent facts, deliberately kept apart. ``gate_flag`` is how THIS
    binary spells the option and belongs to the serve configuration — it cannot
    be guessed, and passing an unknown flag to an arbitrary agent would break
    it outright. The gate COMMAND is a property of the TASK and arrives per
    rollout, in ``metadata``.

    Either side missing returns the argv untouched, which is exactly the
    pre-gate behaviour: an agent whose CLI has no gate concept keeps running
    until it self-terminates or hits a budget.
    """
    if not gate_flag:
        return argv
    gate = (rollout.metadata or {}).get("completion_gate")
    if not isinstance(gate, str) or not gate.strip():
        return argv
    return [*argv, gate_flag, gate.strip()]


def run_rollout_as_command(
    argv: list[str],
    rollout: RolloutRequest,
    *,
    ledger_dir: str,
    timeout_s: float,
    env: dict[str, str],
    cwd: str | None = None,
    gate_flag: str | None = None,
) -> RolloutOutput:
    """Run the customer's agent as a process and read what it left behind.

    ``turn_input`` reaches it two ways, because CLIs differ and guessing
    wrongly is silent: on stdin, and as PERCEPTEYE_TURN_INPUT. The final text
    is the process's stdout, trimmed -- there is no protocol to implement, so
    an existing CLI needs no changes to be served.
    """
    started = time.time()
    proc = subprocess.Popen(
        _argv_with_gate(list(argv), rollout, gate_flag),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, text=True, cwd=cwd,
        # Its own process GROUP. A real CLI agent spawns freely -- a shell, a
        # compiler, a test runner -- and killing only the process we launched
        # leaves those children alive holding the stdout pipe open, so
        # communicate() blocks and the timeout never returns. Measured: a
        # 1.5s budget against `sh -c 'sleep 120'` took the full 120s.
        start_new_session=(os.name != "nt"),
    )
    # EVERY exit path is finalised the same way. Capture attached only to the
    # clean-exit path is capture that is missing precisely when it matters — a
    # rollout killed on its budget is the one whose tool calls you most want to
    # see. This module learned that once with artifacts, and the stream capture
    # was then added to the success path only and repeated it: on run b60067db
    # every rollout exited non-zero, so tool_calls came back null even though
    # the parser worked. One finaliser, and no path that can forget it.
    def _finalise(out: RolloutOutput, stdout_text: str) -> RolloutOutput:
        out.timing.setdefault("elapsed_s", time.time() - started)
        _attach_ledger(out, ledger_dir, rollout.rollout_id)
        _attach_stream_tool_calls(out, stdout_text or "")
        return out

    try:
        stdout, stderr = proc.communicate(input=rollout.turn_input,
                                          timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        stdout, stderr = proc.communicate()
        # Killed on the clock, but it ran — whatever it did before the kill is
        # evidence and is captured like any other outcome.
        return _finalise(RolloutOutput(
            final_text=(stdout or "").strip(),
            exclusion_reason="deadline_exceeded",
            exclusion_detail=f"command exceeded {timeout_s:.0f}s and was killed",
        ), stdout or "")
    if proc.returncode != 0:
        # The agent's own failure, not ours — and the process RAN, which is
        # what separates this from ``entrypoint_error``. Popen raises when a
        # binary cannot be started at all, so reaching here means the agent
        # executed and chose to exit non-zero: a crash after doing work, or a
        # completion gate it could not satisfy. Both are the policy's own
        # behaviour and both are evidence.
        return _finalise(RolloutOutput(
            final_text=(stdout or "").strip(),
            exclusion_reason="agent_exit_nonzero",
            exclusion_detail=f"exit {proc.returncode}: {(stderr or '')[-1500:]}",
        ), stdout or "")
    return _finalise(RolloutOutput(final_text=(stdout or "").strip()), stdout or "")


def _attach_stream_tool_calls(out: RolloutOutput, stdout: str) -> None:
    """Recover tool outcomes from a command agent's own event stream.

    Only when the ledger recorded NOTHING. An agent that called
    ``record_tool_call`` made a deliberate assertion about its own behaviour
    and a stream parse must never overwrite it — the ledger is the record the
    control plane grades.

    This is what lets a process agent report tool calls at all. A framework
    adapter binds to a Python object; a Node CLI or a compiled binary has no
    object to bind to, so its calls were unobservable and the honest setting
    was ``reports_tool_calls=False``. Without per-call outcomes there is no
    verdict to grade, no retry to detect, and nothing a customer can debug a
    rollout from.

    ``None`` stays ``None``: a stream carrying no tool-execution events leaves
    the field unset rather than asserting the agent made zero calls.
    """
    if out.tool_calls is not None:
        return
    try:
        from .event_stream import parse_tool_calls

        parsed = parse_tool_calls(stdout)
    except Exception:  # noqa: BLE001 - never fail a rollout over telemetry
        return
    if parsed:
        out.tool_calls = parsed


class _InFlight:
    """One rollout being served right now, as the heartbeat thread sees it."""

    __slots__ = ("lease_lost", "next_beat_at", "proc", "rollout")

    def __init__(self, rollout: RolloutRequest, next_beat_at: float) -> None:
        self.rollout = rollout
        self.proc: subprocess.Popen[str] | None = None
        self.lease_lost = False
        self.next_beat_at = next_beat_at


def _mark_substrate_unavailable(out: RolloutOutput) -> None:
    """Every tool call FAILED → the substrate is broken, not the agent.

    A rollout whose every tool call errored is not a low-scoring attempt; it is
    an attempt that never got to happen. Reporting it as a legitimate zero
    teaches the policy from an environment fault and hides the fault behind
    what looks like poor performance — on 2026-08-18 an agent overwrote
    IPython's own ``interactiveshell.py``, every ``ipython`` call died at
    import, and a full day of runs reported 0% grounded while reading as "the
    model cannot use tools".

    STRICT, and deliberately so:

    * every call must be an OBSERVED failure. ``unknown`` is not failure — it
      is the honest answer when the framework exposed no verdict, and an agent
      that reports none must behave exactly as it did before.
    * at least one call must have been made. A rollout that never called a
      tool has told us nothing about the substrate.
    * partial failure is left alone. Some calls failing is ordinary agent
      behaviour and is exactly what we want to train on.

    Never overwrites an exclusion already set: an earlier, more specific
    reason (a crash, a dead credential) knows more than this does.
    """
    if out.exclusion_reason is not None:
        return
    calls = out.tool_calls
    if not calls:                      # None (unreported) or [] (none made)
        return
    if any(getattr(c, "outcome", None) != "failed" for c in calls):
        return
    out.exclusion_reason = "substrate_unavailable"
    out.exclusion_detail = (
        f"all {len(calls)} tool call(s) failed — "
        f"{', '.join(sorted({str(getattr(c, 'name', '?')) for c in calls}))}. "
        f"The agent could not act on its environment, so this is a substrate "
        f"fault rather than a low score: do not train on it."
    )


class RolloutRunner:
    """Claim work, run it, report it. One rollout at a time per slot."""

    def __init__(
        self,
        entrypoint: Any,
        *,
        transport: AttachTransport,
        input_shape: InputShape,
        artifacts_dir: str,
        isolation: str = "subprocess",
        concurrency: int = 1,
        rollout_timeout_s: float = 900.0,
        method: str | None = None,
        adapters: list[str] | str | None = "auto",
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        #: How THIS binary spells its completion-gate option, e.g.
        #: ``"--autonomous-gate"``. Serve configuration, because it is a fact
        #: about the agent's CLI and cannot be guessed — passing an unknown
        #: flag to an arbitrary binary would break it. The gate COMMAND is a
        #: property of the task and arrives per rollout in metadata; both are
        #: required before anything is appended.
        gate_flag: str | None = None,
    ) -> None:
        if isolation not in ("subprocess", "inprocess"):
            raise ConfigurationError(
                f"isolation must be 'subprocess' or 'inprocess', got {isolation!r}"
            )
        if isolation == "inprocess" and concurrency != 1:
            raise ConfigurationError(
                "isolation='inprocess' supports concurrency=1 only. Concurrent "
                "in-process rollouts share one set of environment variables, so "
                "they would race for the per-rollout endpoint and credential and "
                "silently cross-attribute each other's completions. Use "
                "isolation='subprocess' for concurrency > 1."
            )
        self._spec: str | None = None
        self._command: list[str] | None = None
        if _is_command(entrypoint):
            # A process, not an object. Nothing to resolve and nothing to
            # import; isolation is inherent because it is already its own
            # process.
            self._command = [str(a) for a in entrypoint]
            self.resolved = None  # type: ignore[assignment]
            self.transport = transport
            self.artifacts_dir = str(artifacts_dir)
            self.isolation = "subprocess"
            self.concurrency = max(1, concurrency)
            self.rollout_timeout_s = rollout_timeout_s
            self.gate_flag = gate_flag
            self._method = method
            self._adapters = adapters
            self._on_event = on_event
            self._stop = False
            self._inflight = {}
            self._inflight_lock = threading.Lock()
            self._beat_stop = threading.Event()
            return
        if _is_import_spec(entrypoint):
            self._spec = entrypoint
            obj = build_agent(entrypoint)
        else:
            obj = entrypoint
            if isolation == "subprocess":
                raise ConfigurationError(
                    "isolation='subprocess' needs an importable entrypoint so the "
                    "child can rebuild the agent -- pass "
                    "'my_package.agent:build' instead of a constructed object, "
                    "or set isolation='inprocess' (concurrency=1)."
                )
        # Resolve NOW. A bad entrypoint must stop startup, in front of the
        # person who wrote it, not surface hours later as a missing member of
        # a comparison group.
        self.resolved = resolve_entrypoint(obj, input_shape=input_shape, method=method)
        self.transport = transport
        self.artifacts_dir = str(artifacts_dir)
        self.isolation = isolation
        self.concurrency = max(1, concurrency)
        self.rollout_timeout_s = rollout_timeout_s
        self.gate_flag = gate_flag
        # Carried so the CHILD resolves the same callable the parent validated.
        # This was hardcoded to None on the subprocess payload, so a customer
        # who named an entrypoint with method= had it honoured at startup and
        # silently ignored on every rollout: the child re-probed and could run
        # a different method entirely.
        self._method = method
        #: Framework adapters, by NAME, bound inside whichever process runs the
        #: agent. "auto" binds every built-in whose framework is installed,
        #: which is what makes outcome capture need no instrumentation.
        self._adapters = adapters
        self._on_event = on_event
        self._stop = False
        #: Rollouts currently being served, by id. Read by the heartbeat thread.
        self._inflight: dict[str, _InFlight] = {}
        self._inflight_lock = threading.Lock()
        self._beat_stop = threading.Event()

    # ── one rollout ────────────────────────────────────────────────────────
    def run_one(self, rollout: RolloutRequest) -> RolloutOutput:
        budget = rollout.seconds_remaining(time.time())
        if budget <= 0.0:
            # Already past its deadline when we got it -- a queue backlog, a
            # slow claim, a clock that moved. Running it anyway spent the full
            # local timeout on work the server had already given up on, and
            # then reported into a lease that no longer existed. Hand it back
            # immediately instead.
            return RolloutOutput(
                final_text="",
                exclusion_reason="deadline_exceeded",
                exclusion_detail="deadline had already passed when claimed",
                timing={"elapsed_s": 0.0},
            )
        # Stop the agent SHORT of the deadline, so there is time left to
        # collect its artifacts and get the report in while the lease is still
        # good. Running to the deadline exactly meant the kill and the expiry
        # landed together and the report chased a rollout the server had
        # already swept — the work was done and then thrown away at the last
        # step. The control plane reserves the same kind of margin on its own
        # waiting loop, for the same reason; this side had none.
        timeout_s = min(self.rollout_timeout_s, max(1.0, budget - REPORT_MARGIN_S))
        ledger_dir = self._ledger_dir()
        Path(ledger_dir, rollout.rollout_id).mkdir(parents=True, exist_ok=True)

        if self._command is not None:
            return run_rollout_as_command(
                self._command, rollout, ledger_dir=ledger_dir,
                timeout_s=timeout_s, gate_flag=self.gate_flag,
                env=build_child_env(
                    rollout_id=rollout.rollout_id, llm_url=rollout.llm_url,
                    api_key=rollout.api_key, ledger_dir=ledger_dir,
                    extra={"PERCEPTEYE_TURN_INPUT": rollout.turn_input},
                ),
                # The per-rollout directory is the child's working directory,
                # so the files a CLI agent leaves behind ARE its artifacts —
                # _collect_artifacts scans exactly this directory. Without
                # this, a coding agent's work landed wherever the serving
                # process happened to be started from, collection found
                # nothing, and the artifact grader scored an absence the
                # agent never caused.
                cwd=str(Path(ledger_dir) / rollout.rollout_id),
            )
        if self.isolation == "inprocess":
            return run_rollout_in_process(
                self.resolved, rollout, ledger_dir=ledger_dir, timeout_s=timeout_s,
                adapters=self._adapters,
            )
        return self._run_in_subprocess(rollout, ledger_dir=ledger_dir, timeout_s=timeout_s)

    def _ledger_dir(self) -> str:
        return str(Path(self.artifacts_dir) / "_percepteye")

    def _run_in_subprocess(
        self, rollout: RolloutRequest, *, ledger_dir: str, timeout_s: float
    ) -> RolloutOutput:
        assert self._spec is not None
        payload = {
            "spec": self._spec,
            "method": self._method,
            "input_shape": self.resolved.input_shape,
            "adapters": self._adapters,
            "rollout": {
                "rollout_id": rollout.rollout_id,
                "turn_input": rollout.turn_input,
                "llm_url": rollout.llm_url,
                "api_key": rollout.api_key,
                "artifacts_dir": rollout.artifacts_dir,
                "deadline_ts": rollout.deadline_ts,
                "metadata": rollout.metadata,
            },
            "ledger_dir": ledger_dir,
            "timeout_s": timeout_s,
        }
        env = build_child_env(
            rollout_id=rollout.rollout_id,
            llm_url=rollout.llm_url,
            api_key=rollout.api_key,
            ledger_dir=ledger_dir,
        )
        started = time.time()
        # Popen rather than subprocess.run: the heartbeat thread must be able
        # to KILL this child when the lease is lost. Under run() the child kept
        # working for the rest of its timeout against a credential the server
        # had already revoked -- burning the customer's compute to produce a
        # result nothing would accept.
        proc = subprocess.Popen(
            [sys.executable, "-m", "percepteye_agent_flywheel.child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
        )
        self._attach_proc(rollout.rollout_id, proc)
        try:
            stdout, stderr = proc.communicate(
                input=json.dumps(payload), timeout=timeout_s + 15.0
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
            return RolloutOutput(
                final_text="",
                exclusion_reason="deadline_exceeded",
                exclusion_detail=f"child exceeded {timeout_s:.0f}s and was killed",
                timing={"elapsed_s": time.time() - started},
            )
        finally:
            self._detach_proc(rollout.rollout_id)

        if self._lease_lost(rollout.rollout_id):
            # Killed by the heartbeat thread. Not the agent's failure, and not
            # ours to hand back: the lease is already gone, so there is nothing
            # left to abandon.
            return RolloutOutput(
                final_text="",
                exclusion_reason="credential_expired",
                exclusion_detail="lease lost while the rollout was running",
                timing={"elapsed_s": time.time() - started},
            )
        if proc.returncode != 0:
            return RolloutOutput(
                final_text="",
                exclusion_reason="flywheel_crash",
                exclusion_detail=(stderr or "")[-2000:],
                timing={"elapsed_s": time.time() - started},
            )
        try:
            out = _output_from_wire(json.loads(stdout.strip().splitlines()[-1]))
        except Exception as exc:
            return RolloutOutput(
                final_text="",
                exclusion_reason="flywheel_crash",
                exclusion_detail=f"unparseable child result: {exc}",
                timing={"elapsed_s": time.time() - started},
            )
        out.timing.setdefault("elapsed_s", time.time() - started)
        _attach_ledger(out, ledger_dir, rollout.rollout_id)
        return out

    # ── leases ─────────────────────────────────────────────────────────────
    def _attach_proc(self, rollout_id: str, proc: subprocess.Popen[str]) -> None:
        with self._inflight_lock:
            entry = self._inflight.get(rollout_id)
            if entry is not None:
                entry.proc = proc

    def _detach_proc(self, rollout_id: str) -> None:
        with self._inflight_lock:
            entry = self._inflight.get(rollout_id)
            if entry is not None:
                entry.proc = None

    def _lease_lost(self, rollout_id: str) -> bool:
        with self._inflight_lock:
            entry = self._inflight.get(rollout_id)
            return bool(entry and entry.lease_lost)

    def _beat_interval_for(self, rollout: RolloutRequest) -> float:
        """A third of THIS rollout's lease, so two beats may be lost first.

        Measured from the lease the server actually GRANTED, not the one we
        asked for. The server clamps our request, so a deployment with a lower
        ceiling than our default would reap every long rollout underneath a
        heartbeat pacing itself against a lease it never got. Falls back to the
        requested value for a server that does not report what it granted.
        """
        granted = rollout.lease_expires_ts
        if granted is not None:
            return max(1.0, (granted - time.time()) / 3.0)
        lease_ms = getattr(self.transport, "lease_ms", 1_800_000)
        return max(1.0, (lease_ms / 1000.0) / 3.0)

    def _heartbeat_loop(self) -> None:
        """Keep every in-flight lease alive until its rollout finishes.

        Without this the server's reaper requeued any rollout that outlived its
        lease WHILE THE AGENT WAS STILL RUNNING IT: the per-rollout credential
        was revoked mid-flight, the work was handed to someone else, and the
        report arrived against a lease that no longer existed. Long rollouts --
        the ones this product exists for -- hit it every time.
        """
        while not self._beat_stop.wait(_BEAT_TICK_S):
            now = time.time()
            with self._inflight_lock:
                entries = [
                    e for e in self._inflight.values()
                    if not e.lease_lost and now >= e.next_beat_at
                ]
            for entry in entries:
                entry.next_beat_at = now + self._beat_interval_for(entry.rollout)
                try:
                    self.transport.heartbeat(entry.rollout)
                except LeaseLost:
                    # The server has already taken it back. Stop the work now
                    # rather than let it run to completion for nothing.
                    entry.lease_lost = True
                    self._emit("lease_lost", {"rollout_id": entry.rollout.rollout_id})
                    proc = entry.proc
                    if proc is not None and proc.poll() is None:
                        proc.kill()
                except TransportError as exc:
                    # Transient. The next beat may well succeed, and there are
                    # two more before the lease lapses.
                    self._emit("heartbeat_failed", {
                        "rollout_id": entry.rollout.rollout_id, "error": str(exc),
                    })

    # ── the loop ───────────────────────────────────────────────────────────
    def serve_forever(self, *, max_rollouts: int | None = None) -> int:
        """Claim → run → report, until stopped. Returns rollouts completed."""
        done = 0
        beat = threading.Thread(
            target=self._heartbeat_loop, name="percepteye-heartbeat", daemon=True
        )
        beat.start()
        try:
            with _cf.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                while not self._stop:
                    if max_rollouts is not None and done >= max_rollouts:
                        break
                    # Never claim more than we still intend to run. Claiming a
                    # full batch when one rollout remained left the rest leased
                    # to a runner that was about to exit.
                    want = self.concurrency
                    if max_rollouts is not None:
                        want = min(want, max_rollouts - done)
                    try:
                        rollouts, poll_after_ms = self.transport.claim(want)
                    except TransportError as exc:
                        self._emit("claim_failed", {"error": str(exc)})
                        sleep_ms(5_000)
                        continue
                    if not rollouts:
                        sleep_ms(poll_after_ms)
                        continue
                    # Concurrently, which is what claiming N meant all along.
                    # Serving them one after another held N-1 leases on work
                    # nobody was doing, and with no heartbeat those were the
                    # rollouts most likely to lapse.
                    now = time.time()
                    with self._inflight_lock:
                        for r in rollouts:
                            self._inflight[r.rollout_id] = _InFlight(
                                r, now + self._beat_interval_for(r)
                            )
                    futures = [pool.submit(self._serve_one, r) for r in rollouts]
                    for fut in _cf.as_completed(futures):
                        fut.result()  # _serve_one never raises; surface it if it does
                        done += 1
        finally:
            self._beat_stop.set()
            beat.join(timeout=5.0)
        return done

    def _serve_one(self, rollout: RolloutRequest) -> None:
        self._emit("rollout_start", {"rollout_id": rollout.rollout_id})
        try:
            self._serve_one_inner(rollout)
        finally:
            with self._inflight_lock:
                self._inflight.pop(rollout.rollout_id, None)
            self._purge_ledger(rollout.rollout_id)

    def _serve_one_inner(self, rollout: RolloutRequest) -> None:
        try:
            out = self.run_one(rollout)
        except Exception as exc:
            try:
                self.transport.abandon(
                    rollout, reason="flywheel_crash", detail=f"{type(exc).__name__}: {exc}"
                )
            except (TransportError, LeaseLost):
                pass
            self._emit("rollout_crash", {"rollout_id": rollout.rollout_id, "error": str(exc)})
            return
        if self._lease_lost(rollout.rollout_id):
            # Already requeued server-side. Reporting or abandoning would only
            # earn a 410; the heartbeat thread has emitted the event.
            return
        try:
            # Evidence FIRST, whatever the outcome. An agent that exhausted its
            # turn or time budget still wrote files, and for a coding/REPL agent
            # those files ARE what the reward reads. Collecting only on the
            # clean-exit path meant a rollout could write a CORRECT solution and
            # have it discarded because the process did not terminate tidily —
            # measured on run 516b7a55, where a correct top_ips.sh sat in the
            # ledger while the row carried artifacts={}.
            self._collect_artifacts(rollout, out)
            _mark_substrate_unavailable(out)
            # EVERY rollout is reported, whatever happened to it.
            #
            # This used to abandon when an exclusion carried no work product —
            # "nothing to grade, do not train on this". That is a decision
            # about TRAINABILITY, and it is not the SDK's to make: abandon
            # suppresses the record entirely, so the control plane could not
            # see rollouts it might have wanted to count, and the reason it
            # would have judged them by never arrived. An operator asking "what
            # did my agent do" got silence for exactly the rollouts that went
            # worst.
            #
            # The SDK's job is to observe faithfully and say so. Whether a
            # typed reason means retry, exclude, or train is the control
            # plane's call, and it has the reason to make it with.
            # Reported WITH its exclusion when it has one: to_wire carries the
            # exclusion block, so the server records the work product and still
            # knows not to train on it. The agent's inability to declare itself
            # finished must not decide the score — the hidden test does.
            ack = self.transport.report(rollout, out)
            self._emit("rollout_reported", {
                "rollout_id": rollout.rollout_id,
                "grade": ack.get("grade"),
                "duplicate": ack.get("duplicate", False),
                "exclusion": out.exclusion_reason,
            })
        except LeaseLost:
            self._emit("lease_lost", {"rollout_id": rollout.rollout_id})
        except TransportError as exc:
            # The rollout ran; only the report failed. The lease will lapse and
            # the server will hand the work to someone else, so this costs a
            # repeat rather than a hole -- but it is the expensive half we are
            # throwing away, so it is emitted loudly rather than swallowed.
            self._emit("report_failed", {"rollout_id": rollout.rollout_id, "error": str(exc)})

    def _collect_artifacts(
        self, rollout: RolloutRequest, out: RolloutOutput
    ) -> None:
        """Register the files the agent produced, before the report goes out.

        For a coding or REPL agent the ARTIFACT is the reward -- a diff, a
        built binary, a screenshot -- and a scorer that only sees final text is
        grading the commentary rather than the work. ``register_artifact`` has
        existed on the transport since the beginning with no caller, so this
        capability was fully implemented and entirely unreachable.

        Registration carries a digest and a size, never the bytes. The REPORT
        additionally inlines the content of SMALL TEXT artifacts — for a
        coding or REPL agent the files are the deliverable, and a grader that
        can verify a digest but never read a file can prove integrity while
        grading nothing. Two caps bound it: per-file (utf-8, ≤64KB) and a
        total per report, so a rollout that produced a 200MB core dump — or
        two hundred small ones — cannot bloat the report. Anything over a cap
        keeps today's digest-string value: registered, named, not inlined.
        """
        # The LOCAL per-rollout directory, which is the one the child is told
        # about through PERCEPTEYE_LEDGER_DIR and can therefore actually write
        # to. ``rollout.artifacts_dir`` is the SERVER's name for this rollout's
        # storage ("rollouts/rl_x") and is not a path on this machine.
        root = Path(self._ledger_dir()) / rollout.rollout_id
        if not root.is_dir():
            return
        # Registration is OPTIONAL. It is a server-side index; the report body
        # is what the grader actually reads, so a transport without it (or one
        # whose registration call fails) must still deliver the bytes.
        registrar = getattr(self.transport, "register_artifact", None)
        if not callable(registrar):
            registrar = None
        inline_budget = MAX_INLINE_TOTAL_BYTES
        # Resolve the collection root ONCE so every candidate can be proven to
        # live under it. ``strict=False`` because the dir is ours and exists.
        root_real = root.resolve()
        for path in sorted(root.rglob("*")):
            # SYMLINKS ARE NEVER COLLECTED. ``Path.is_file()`` and
            # ``read_text()`` both FOLLOW links, so a link the agent drops in
            # its own directory (``ln -s ~/.aws/credentials creds.txt``) used
            # to be read at its TARGET and inlined verbatim into the report
            # posted to the control plane — up to MAX_INLINE_ARTIFACT_BYTES of
            # a host secret, exfiltrated through the artifact path, defeating
            # the boundary the isolation module exists to draw. The customer's
            # agent is untrusted code by construction; it authors the contents
            # of this directory, so the collector must not follow anything it
            # can point somewhere else.
            #
            # Named, not silently dropped — a link IS a fact about the rollout,
            # and silence here would hide an exfiltration attempt.
            if path.is_symlink():
                out.artifacts[str(path.relative_to(root))] = "skipped:symlink"
                continue
            if not path.is_file() or path.name == LEDGER_FILENAME:
                continue
            # Defence in depth against a link ABOVE this path (a symlinked
            # parent dir would make every child look ordinary): prove the
            # resolved path is still inside the resolved root.
            try:
                path.resolve().relative_to(root_real)
            except (ValueError, OSError):
                out.artifacts[str(path.relative_to(root))] = "skipped:escapes_root"
                continue
            name = str(path.relative_to(root))
            try:
                digest, size = sha256_file(path)
            except OSError:
                continue
            if size > MAX_ARTIFACT_BYTES:
                # Named, not silently dropped: an artifact too large to
                # register is a fact about the rollout, and a scorer expecting
                # it deserves to know it existed.
                out.artifacts[name] = f"skipped:too_large:{size}"
                continue
            # Small text files travel WITH the report, because the grader
            # needs bytes, not proofs about bytes. Binary or over-cap files
            # keep the digest-string value — same wire shape as before.
            content: str | None = None
            if size <= MAX_INLINE_ARTIFACT_BYTES and size <= inline_budget:
                try:
                    content = path.read_text(encoding="utf-8", errors="strict")
                except (OSError, UnicodeDecodeError):
                    content = None
            # Recorded BEFORE registration is attempted. Registration is a
            # network call to the same server that may have just expired this
            # rollout, and it used to gate the assignment below — so on a
            # deadline_exceeded rollout, the one case where the evidence
            # matters most, registration 4xx'd and the deliverable vanished
            # from the report. Losing an index entry must not lose the bytes.
            if content is not None:
                inline_budget -= size
                out.artifacts[name] = {
                    "sha256": digest, "size_bytes": size, "content": content,
                }
            else:
                out.artifacts[name] = digest
            if registrar is None:
                continue
            try:
                registrar(rollout, name=name, size_bytes=size, sha256=digest)
            except (TransportError, LeaseLost) as exc:
                # The rollout itself is fine. Losing an artifact registration
                # must not lose the result it belongs to.
                self._emit("artifact_failed", {
                    "rollout_id": rollout.rollout_id, "name": name,
                    "error": str(exc),
                })

    def _purge_ledger(self, rollout_id: str) -> None:
        """Drop this rollout's ledger directory once it is no longer needed.

        One directory per rollout accumulated for the life of the process, so a
        flywheel serving continuously grew its artifacts tree without bound.
        """
        try:
            shutil.rmtree(Path(self._ledger_dir()) / rollout_id, ignore_errors=True)
        except OSError:  # pragma: no cover - best effort, never fails a rollout
            pass

    def stop(self) -> None:
        self._stop = True

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        if self._on_event:
            self._on_event(kind, payload)


def _looks_like_agent(target: Any) -> bool:
    """A factory is called; an already-built agent is used as-is."""
    from .entrypoint import ENTRYPOINT_ORDER

    return any(callable(getattr(target, n, None)) for n in ENTRYPOINT_ORDER[:-1])


def _output_from_wire(d: dict[str, Any]) -> RolloutOutput:
    tcs = d.get("tool_calls")
    calls = (
        None if tcs is None
        else [
            ToolCallOutcome(
                name=t["name"], arguments=t.get("arguments") or {},
                outcome=t["outcome"], status_code=t.get("status_code"),
                output=t.get("output"), error=t.get("error"),
                error_class=t.get("error_class"), latency_ms=t.get("latency_ms"),
                agent_name=t.get("agent_name"),
                tool_call_id=t.get("tool_call_id"),
            )
            for t in tcs
        ]
    )
    exc = d.get("exclusion") or {}
    return RolloutOutput(
        final_text=d.get("final_text", ""),
        tool_calls=calls,
        tool_calls_omitted_count=int(d.get("tool_calls_omitted_count") or 0),
        llm_call_count=d.get("llm_call_count"),
        artifacts=dict(d.get("artifacts") or {}),
        reward=d.get("reward"),
        success=d.get("success"),
        exclusion_reason=exc.get("reason"),
        exclusion_detail=exc.get("detail"),
        events=list(d.get("events") or []),
        timing=dict(d.get("timing") or {}),
        agent_fingerprint=d.get("agent_fingerprint"),
    )


def sha256_file(path: str | os.PathLike[str]) -> tuple[str, int]:
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            size += len(chunk)
    return h.hexdigest(), size


_ = LEDGER_FILENAME  # re-exported for adapters that write their own ledger
