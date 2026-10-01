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
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contract import RolloutOutput, RolloutRequest, ToolCallOutcome
from .entrypoint import TASK, InputShape, ResolvedEntrypoint, resolve_entrypoint
from .errors import ConfigurationError, LeaseLost, TransportError
from .event_stream import CommandOutputAdapter, CommandOutputProjection
from .execution_identity import (
    ExecutionChild,
    ExecutionSnapshot,
    canonical_command_argv,
    canonical_observation_sha256,
    execution_descriptor_sha256,
    is_lower_sha256,
    normalize_attested_components,
    normalize_volatile_command_values,
)
from .introspection import _MAX_PROMPT_CHARS, DiscoveredAgent
from .isolation import build_child_env
from .outcomes import TOOL_CALLS_FILENAME, Trajectory, _RolloutScope
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
    trajectory_dir: str,
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
    with _RolloutScope(rollout.rollout_id, trajectory_dir):
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
    _attach_trajectory(out, trajectory_dir)
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


def _attach_trajectory(out: RolloutOutput, trajectory_dir: str) -> None:
    """Merge recorded tool calls, preserving the None/[] distinction."""
    if out.tool_calls is not None:
        return
    traj = Trajectory(trajectory_dir)
    recorded = traj.read()
    if recorded is not None:
        out.tool_calls = recorded
        # A line we could not parse is a call we KNOW executed and cannot
        # report, which is exactly what this field means on the wire --
        # "executed calls deliberately not reported; non-zero forbids training
        # grade". Reporting the survivors while claiming the list is complete
        # would be the truncated-list failure the contract already forbids
        # agents from committing by hand.
        out.tool_calls_omitted_count += traj.malformed


#: FLOOR on the wall-clock held back from the agent so its RESULT can still be
#: delivered. Collecting artifacts and posting the report are network calls
#: against a lease, and a rollout run right up to its deadline has no time left
#: for either — the deliverable is produced and then dropped at the final step.
#:
#: It used to be the WHOLE reserve, and as a bare constant it was a fiction:
#: the client that spends this time defaults to 30s per attempt and 4 attempts
#: (``_http.DEFAULT_TIMEOUT_S`` / ``DEFAULT_MAX_ATTEMPTS``), so one report's own
#: worst case is ~123.5s against a ten-second reserve — and `_collect_artifacts`
#: spends one such envelope PER FILE before the report is even attempted. The
#: reserve is now derived from the transport that will spend it (see
#: :func:`episode_plan`); this survives as the floor, for a transport that
#: declares no envelope at all.
REPORT_MARGIN_S = 10.0

#: The reserve may never take more than this share of the episode. Without it,
#: a client envelope larger than the budget would leave the agent no time at
#: all: this deployment has minted 300s episodes, where the full 123.5s
#: envelope is 41% of everything there is.
MAX_RESERVE_FRACTION = 0.25

#: Reserved for killing the child and reading what it left on its pipes.
#: MEASURED, not guessed: a benign in-group process tree drains in ~0.001s.
#: The only case that exceeds this is a descendant that escaped the process
#: group and still holds the pipes, and waiting longer for that buys nothing
#: while costing the delivery window — see :func:`_kill_and_drain`.
TEARDOWN_DRAIN_S = 5.0

#: Default agent budget, named so the value that bounds a rollout and the value
#: an identity-sealed adapter setting is sized against are the same object.
DEFAULT_ROLLOUT_TIMEOUT_S = 900.0

#: The share of the child's window ONE blocking harness call may take. Half
#: would mean a single wedged provider call can eat the episode and still look
#: like an agent doing slow work; below half there is room to come back and say
#: what happened. Consumed by :func:`harness_call_ceiling_s`, which is how an
#: adapter sizes its own per-call ceiling without copying this arithmetic.
CALL_CEILING_FRACTION = 0.4


@dataclass(frozen=True)
class EpisodePlan:
    """How one episode's budget is divided. Produced in exactly one place."""

    #: What the agent gets. The child is killed here.
    child_s: float
    #: Killing the child and draining its pipes.
    teardown_s: float
    #: Collecting artifacts and delivering the report.
    delivery_s: float


def episode_plan(
    budget_s: float,
    *,
    rollout_timeout_s: float = DEFAULT_ROLLOUT_TIMEOUT_S,
    envelope_s: float = REPORT_MARGIN_S,
) -> EpisodePlan:
    """Turn an episode budget into a child deadline. THE one place.

    ``envelope_s`` is what the transport says one call costs it, worst case.
    Reserving it and spending it become the same number, which is the whole
    point: the same value is handed to the HTTP client as its ``deadline``, so
    the loop cannot spend more than was held back for it.

    Every adapter inherits this by construction — it is upstream of the
    command/python split and knows nothing about any harness.
    """
    budget = max(0.0, float(budget_s))
    delivery_s = min(max(REPORT_MARGIN_S, float(envelope_s)), budget * MAX_RESERVE_FRACTION)
    child_s = max(1.0, min(float(rollout_timeout_s), budget - delivery_s - TEARDOWN_DRAIN_S))
    return EpisodePlan(child_s=child_s, teardown_s=TEARDOWN_DRAIN_S, delivery_s=delivery_s)


def harness_call_ceiling_s(
    budget_s: float,
    *,
    rollout_timeout_s: float = DEFAULT_ROLLOUT_TIMEOUT_S,
    envelope_s: float | None = None,
) -> float:
    """The longest ONE blocking harness call may run, for an episode this long.

    The translation layer's input. An adapter whose harness has a per-call
    timeout setting sizes it from HERE rather than from a number typed into the
    adapter, so raising the delivery reserve automatically tightens every
    harness ceiling derived from it instead of silently invalidating them.

    ``envelope_s`` defaults to the envelope of a client built with this SDK's
    own defaults, because a setting frozen into an agent's identity cannot read
    a live transport.
    """
    if envelope_s is None:
        from ._http import default_delivery_envelope_s

        envelope_s = default_delivery_envelope_s()
    plan = episode_plan(
        budget_s, rollout_timeout_s=rollout_timeout_s, envelope_s=envelope_s
    )
    return plan.child_s * CALL_CEILING_FRACTION

#: Version tag on a spooled outcome. A file written by a future SDK with a
#: different shape must be left alone rather than half-read.
_OUTBOX_SCHEMA = "percepteye.outbox/v1"

#: Spooled outcomes older than this stop being retried and are moved aside,
#: named, for an operator. Long, because the thing being protected is a
#: rollout's entire work product and the usual cause of a week-long refusal is
#: a credential an operator has to reissue by hand.
_OUTBOX_MAX_AGE_S = 7 * 24 * 3600.0

#: Consecutive PERMANENT refusals before the loop stops claiming. Counts only
#: the terminal 4xx branch -- never 408/429/5xx/network, which recover on their
#: own -- and resets on the first success. Three, because one is a bad payload
#: and three in a row is a configuration that will refuse the next thousand
#: too: "every rollout since 2026-09-10" becomes one rollout and a named error.
_PERMANENT_REFUSAL_LIMIT = 3

#: Refusals redelivering the SAME BYTES cannot change. 404 (no such rollout),
#: 409 (a different result already landed), 410 (the lease/identity fence), 400
#: and 422 (the payload itself). A result refused with one of these is RETIRED:
#: renamed out of the retry set, kept on disk, and named out loud. Everything
#: else -- 401 and 403 above all -- stays IN the retry set, because a reissued
#: key delivers it.
#:
#: The difference is which set the file is in, and nothing here decides whether
#: the work survives: it always does. It used to be the predicate for unlinking
#: the file, and 410 is the refusal an identity-sealed rollout answers forever
#: once its lease is reaped -- so the one status that guaranteed the row was
#: unrecoverable also guaranteed the bytes were destroyed.
_REDELIVERY_CANNOT_FIX = frozenset({400, 404, 409, 410, 422})


def _is_permanent_refusal(exc: TransportError) -> bool:
    """A 4xx that will refuse the identical request forever.

    ``status_code`` is None when no response was ever produced, which is the
    transient case and must NOT count: a network blip that stopped the fleet
    claiming would be a worse outage than the one this exists to end.
    """
    code = getattr(exc, "status_code", None)
    return isinstance(code, int) and 400 <= code < 500 and code not in (408, 429)


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


def _kill_and_drain(
    proc: subprocess.Popen[str], grace_s: float
) -> tuple[str, str, bool]:
    """Kill the tree and read what it left, WITHOUT the chance of never returning.

    Returns ``(stdout, stderr, drained)``. ``drained`` false means the pipes
    were still held when the grace period ran out and nothing could be read.

    REPRODUCED, 2026-09-16: ``_kill_tree`` kills one process group, so a
    descendant that called ``setsid`` itself survives, and if it inherited the
    write end of stdout the follow-up ``communicate()`` — which had no timeout
    on either of its two call sites — blocks forever. Measured: >15s and in
    principle unbounded with such a descendant, against 0.001s for an ordinary
    in-group tree. That hang is inside the window reserved for delivering the
    result, so an escaped grandchild silently cost the whole rollout.

    Q10, stated rather than hidden: when the grace period expires the child's
    output IS LOST, and the caller must say so in the detail it reports rather
    than leave an empty stderr reading as "the agent said nothing".
    """
    _kill_tree(proc)
    try:
        stdout, stderr = proc.communicate(timeout=max(0.1, grace_s))
        return stdout or "", stderr or "", True
    except subprocess.TimeoutExpired:
        for pipe in (proc.stdout, proc.stderr, proc.stdin):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
        return "", "", False


def _is_command(entrypoint: Any) -> bool:
    """A list/tuple argv means the agent is a PROCESS, not a Python object.

    The importable-entrypoint requirement is the single biggest limit on who
    can be served: an agent that is a compiled binary, a Node CLI, or anything
    not written in Python simply had no way in. A command has the same
    contract as everything else -- it is handed the rollout through the
    environment, it writes artifacts, and its outcomes come from the trajectory.
    """
    return isinstance(entrypoint, (list, tuple)) and bool(entrypoint)


def _argv_with_task(argv: list[Any], turn_input: str) -> tuple[list[str], bool]:
    """Substitute this rollout's task wherever ``TASK`` appears.

    Returns the argv and whether anything was substituted. The caller uses the
    flag to decide stdin: an agent told the task through argv must NOT also be
    told it on stdin, or it acts on the same instruction twice. WHICH of the
    two channels is used is DECLARED by whether the sentinel appears; it is
    never inferred from the binary.
    """
    if not any(a is TASK for a in argv):
        return [str(a) for a in argv], False
    return [turn_input if a is TASK else str(a) for a in argv], True


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
    trajectory_dir: str,
    timeout_s: float,
    env: dict[str, str],
    cwd: str | None = None,
    gate_flag: str | None = None,
    output_adapter: CommandOutputAdapter | None = None,
    teardown_s: float = TEARDOWN_DRAIN_S,
    #: Handed the child's ``Popen`` the instant it exists. The heartbeat
    #: thread's lease-lost kill reads ``_InFlight.proc``, and that was attached
    #: on the PYTHON entrypoint path only — so for prime-agent, goose and every
    #: other BYOH command agent (the entire class this product serves) the kill
    #: at ``_heartbeat_loop`` was unreachable code and a child kept burning the
    #: customer's compute against a revoked credential.
    proc_sink: Callable[[subprocess.Popen[str]], None] | None = None,
) -> RolloutOutput:
    """Run the customer's agent as a process and read what it left behind.

    ``turn_input`` reaches it two ways by default, because CLIs differ and
    guessing wrongly is silent: on stdin, and as PERCEPTEYE_TURN_INPUT. A CLI
    that takes the task as a positional argument gets a third: put ``fw.TASK``
    in the argv and it is substituted there, and stdin is then left empty so
    the task is not delivered twice. For an ordinary command, final text is
    stdout trimmed. Structured stdout is interpreted only when the caller
    supplies an explicit adapter; core has no customer-agent vocabulary.
    """
    started = time.time()
    resolved_argv, task_in_argv = _argv_with_task(list(argv), rollout.turn_input)
    proc = subprocess.Popen(
        _argv_with_gate(resolved_argv, rollout, gate_flag),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, text=True, cwd=cwd,
        # Its own process GROUP. A real CLI agent spawns freely -- a shell, a
        # compiler, a test runner -- and killing only the process we launched
        # leaves those children alive holding the stdout pipe open, so
        # communicate() blocks and the timeout never returns. Measured: a
        # 1.5s budget against `sh -c 'sleep 120'` took the full 120s.
        start_new_session=(os.name != "nt"),
    )
    if proc_sink is not None:
        proc_sink(proc)
    # EVERY exit path is finalised the same way. Capture attached only to the
    # clean-exit path is capture that is missing precisely when it matters — a
    # rollout killed on its budget is the one whose tool calls you most want to
    # see. This module learned that once with artifacts, and the stream capture
    # was then added to the success path only and repeated it: on run b60067db
    # every rollout exited non-zero, so tool_calls came back null even though
    # the parser worked. One finaliser, and no path that can forget it.
    def _finalise(out: RolloutOutput, stdout_text: str) -> RolloutOutput:
        out.timing.setdefault("elapsed_s", time.time() - started)
        _attach_trajectory(out, trajectory_dir)
        _attach_command_projection(out, stdout_text or "", output_adapter)
        return out

    try:
        # Empty stdin when the task went in through argv. Sending it both
        # ways would have an agent that reads both act on the task twice.
        stdout, stderr = proc.communicate(
            input="" if task_in_argv else rollout.turn_input, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        stdout, stderr, drained = _kill_and_drain(proc, teardown_s)
        # Killed on the clock, but it ran — whatever it did before the kill is
        # evidence and is captured like any other outcome, AND THAT INCLUDES
        # ITS STDERR. It was read on the line above and then dropped, so the
        # one exit path with no other explanation was also the only one that
        # carried none: the agent_exit_nonzero branch below has always kept
        # `stderr[-1500:]`, and this branch reported nothing but the clock.
        #
        # Measured 2026-09-15, LOCALLY, against a loopback endpoint that
        # accepted the connection and never answered: the prime-agent child
        # consumed its entire budget inside that one request and was killed
        # here, and everything it had said about the blocked call was sitting
        # unread in this variable. A timeout is not a cause; whatever the child
        # managed to say about why it stopped moving is.
        #
        # This branch is NOT what the three production rollouts of the same day
        # hit, and nothing here should be read as claiming it was. Those rows
        # carry `status=abandoned`, which only the control plane's reaper
        # writes, with its own `exclusion_detail`; a report of any kind — this
        # one included — sets `status=reported` instead. So on those three the
        # SDK's report never landed at all. WHICH step dropped it was never
        # determined, and this module no longer depends on knowing: every
        # outcome now goes through `_deliver_outcome`, which spools it before
        # the first network call and redelivers it until the control plane
        # answers or names a refusal it cannot answer differently.
        return _finalise(RolloutOutput(
            final_text=(stdout or "").strip(),
            exclusion_reason="deadline_exceeded",
            exclusion_detail=(
                f"command exceeded {timeout_s:.0f}s and was killed"
                + (f": {stderr[-1500:]}" if stderr and stderr.strip() else "")
                # NOT an empty stderr. The pipes were still held when the kill
                # grace expired, so what the child said is unread rather than
                # unsaid, and an investigation must not read one as the other.
                + ("" if drained else
                   f"; its output could not be read within {teardown_s:.0f}s "
                   f"of the kill — a descendant outside the killed process "
                   f"group still held the pipes")
            ),
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


def _attach_command_projection(
    out: RolloutOutput,
    stdout: str,
    adapter: CommandOutputAdapter | None,
) -> None:
    """Merge one explicit adapter projection without trusting malformed claims."""
    if adapter is None:
        return
    try:
        projected = adapter.project(stdout)
        if not isinstance(projected, CommandOutputProjection):
            raise TypeError("adapter returned the wrong projection type")
    except Exception:
        # The process ran, but the configured interpretation boundary failed.
        # Raw lifecycle JSON is not an assistant answer and must not be graded
        # as one. Preserve excluded evidence without letting an observer
        # failure interrupt the customer process.
        out.final_text = ""
        detail = "command output adapter failed; structured evidence was withheld"
        if out.exclusion_reason is None:
            out.exclusion_reason = "cancelled"
            out.exclusion_detail = detail
        else:
            out.exclusion_detail = "; ".join(
                filter(None, (out.exclusion_detail, detail))
            )
        return
    if not projected.recognized:
        return

    if projected.final_text is not None:
        out.final_text = projected.final_text
    if projected.llm_call_count is not None and out.llm_call_count is None:
        count = projected.llm_call_count
        if type(count) is int and count >= 0:
            out.llm_call_count = count
        else:
            detail = "command output adapter returned an invalid model-call count"
            if out.exclusion_reason is None:
                out.exclusion_reason = "cancelled"
                out.exclusion_detail = detail
            else:
                out.exclusion_detail = "; ".join(
                    filter(None, (out.exclusion_detail, detail))
                )
    omitted = projected.tool_calls_omitted_count
    if type(omitted) is int and omitted >= 0:
        out.tool_calls_omitted_count += omitted
    else:
        out.tool_calls_omitted_count += 1
    # A trajectory is an explicit assertion and wins over stream projection.
    if out.tool_calls is None and projected.tool_calls is not None:
        out.tool_calls = projected.tool_calls

    if projected.exclusion_reason is not None:
        detail = projected.exclusion_detail or "command output adapter excluded trace"
        if out.exclusion_reason is None:
            out.exclusion_reason = projected.exclusion_reason
            out.exclusion_detail = detail
        else:
            out.exclusion_detail = "; ".join(
                filter(None, (out.exclusion_detail, detail))
            )

class _InFlight:
    """One rollout being served right now, as the heartbeat thread sees it."""

    __slots__ = ("beat_interval", "lease_lost", "next_beat_at", "plan", "proc", "rollout")

    def __init__(self, rollout: RolloutRequest, beat_interval: float,
                 next_beat_at: float) -> None:
        self.rollout = rollout
        self.proc: subprocess.Popen[str] | None = None
        self.lease_lost = False
        #: Fixed for the life of the rollout. See RolloutRunner._beat_interval_for.
        self.beat_interval = beat_interval
        self.next_beat_at = next_beat_at
        #: Written by ``run_one`` (the producer), read by ``_serve_one_inner``
        #: (the consumer) so the time RESERVED for delivery is the same number
        #: delivery is then allowed to spend. ``None`` until the rollout starts,
        #: and for a ``run_one`` called outside the serving loop.
        self.plan: EpisodePlan | None = None


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


def _exclude_for_execution_identity(out: RolloutOutput, detail: str) -> None:
    """Make identity failure untrainable without hiding prior evidence."""
    if out.exclusion_reason is None:
        out.exclusion_reason = "cancelled"
        out.exclusion_detail = detail
        return
    out.exclusion_detail = "; ".join(filter(None, (out.exclusion_detail, detail)))


def _adapter_config_identity(adapters: Any) -> Any:
    """Stable adapter registration facts, without address-bearing reprs."""
    if adapters is None or isinstance(adapters, str):
        return adapters
    values = adapters if isinstance(adapters, (list, tuple)) else [adapters]
    return [
        item if isinstance(item, str) else {
            "adapter_type": f"{type(item).__module__}.{type(item).__qualname__}",
            "configuration_unobserved": True,
        }
        for item in values
    ]


def _description_is_exact(described: DiscoveredAgent) -> bool:
    """Every behavior-bearing node must survive discovery without ambiguity."""
    if (
        described.system_prompt is None
        or described.tool_definitions is None
        or described.discovered_model_config is None
    ):
        return False
    if any(item != "tool_annotations" for item in described.incomplete):
        return False
    if (
        len(described.system_prompt) >= _MAX_PROMPT_CHARS
        and described.exact_observation_sha256 is None
    ):
        return False
    if (
        described.exact_observation_sha256 is not None
        and not is_lower_sha256(described.exact_observation_sha256)
    ):
        return False
    return all(_description_is_exact(child) for child in described.sub_agents)


def _bind_execution_fingerprint(
    out: RolloutOutput,
    execution_expected: str | None,
    diagnostics_expected: Mapping[str, str] | None = None,
) -> None:
    """Stamp SDK-reconciled identity, refusing agent-authored claims."""
    fingerprint = out.agent_fingerprint
    if fingerprint is None:
        merged: dict[str, Any] = {}
    elif isinstance(fingerprint, dict):
        merged = dict(fingerprint)
    else:
        merged = {}
        _exclude_for_execution_identity(out, "agent_fingerprint was not an object")

    execution_claimed = merged.get("execution_sha256")
    if execution_expected is None:
        if execution_claimed is not None:
            _exclude_for_execution_identity(
                out,
                "agent claimed execution_sha256 without an exact SDK-reconciled identity",
            )
            merged.pop("execution_sha256", None)
    else:
        if execution_claimed is not None and execution_claimed != execution_expected:
            _exclude_for_execution_identity(
                out,
                "agent claimed an execution_sha256 different from the SDK-reconciled identity",
            )
        merged["execution_sha256"] = execution_expected

    diagnostics = dict(diagnostics_expected or {})
    for name, expected in diagnostics.items():
        claimed = merged.get(name)
        if claimed is not None and claimed != expected:
            _exclude_for_execution_identity(
                out,
                f"agent claimed {name} different from the SDK-attested evidence",
            )
        merged[name] = expected
    out.agent_fingerprint = merged or None


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
        #: THE CONSTANT, not a copy of its value — see
        #: :data:`DEFAULT_ROLLOUT_TIMEOUT_S`. This signature is what every
        #: production rollout is actually bounded by, and it spelled the number
        #: out while the adapter that freezes a harness ceiling into an agent's
        #: sealed identity derived that ceiling from the constant. Two
        #: declarations of one budget, with nothing comparing them.
        rollout_timeout_s: float = DEFAULT_ROLLOUT_TIMEOUT_S,
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
        endpoint_env: tuple[str, str] | None = None,
        command_output_adapter: CommandOutputAdapter | None = None,
        execution_snapshot: ExecutionSnapshot | None = None,
        reports_tool_calls: bool = False,
        attested_execution_components: Mapping[str, str] | None = None,
        #: Command options whose VALUE is transport rather than identity, by
        #: name. Declared, never detected — see execution_identity.
        volatile_command_values: Sequence[str] | None = None,
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
        #: The command's canonical identity argv, computed once at startup so a
        #: bad declaration stops the process in front of whoever wrote it
        #: rather than at registration.
        self._identity_argv: list[dict[str, Any]] | None = None
        self._volatile_command_values = normalize_volatile_command_values(
            volatile_command_values
        )
        self._entrypoint_object_type: str | None = None
        self._execution_sha256: str | None = None
        self._execution_descriptor: dict[str, Any] | None = None
        self._reports_tool_calls = bool(reports_tool_calls)
        self._attested_execution_components = normalize_attested_components(
            attested_execution_components
        )
        if "execution_sha256" in self._attested_execution_components:
            raise ConfigurationError(
                "attested execution components must not author execution_sha256"
            )
        self._configured_input_shape = input_shape
        self._configured_method = method
        self._configured_adapters = _adapter_config_identity(adapters)
        self._configured_gate_flag = gate_flag
        self._configured_endpoint_env = list(endpoint_env) if endpoint_env is not None else None
        self._command_output_adapter = command_output_adapter
        if command_output_adapter is not None:
            adapter_name = getattr(command_output_adapter, "adapter_name", None)
            if not isinstance(adapter_name, str) or not adapter_name.strip():
                raise ConfigurationError(
                    "command_output_adapter.adapter_name must be a non-empty string"
                )
            if not callable(getattr(command_output_adapter, "project", None)):
                raise ConfigurationError(
                    "command_output_adapter must provide project(stdout)"
                )
        self.execution_snapshot = execution_snapshot
        self._execution_snapshot_name: str | None = None
        self._snapshot_components: dict[str, str] = {}
        self._snapshot_diagnostics: dict[str, str] = {}
        self._snapshot_complete = False
        if self.execution_snapshot is not None:
            raw_name = self.execution_snapshot.adapter_name
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise ConfigurationError("execution snapshot adapter_name must be non-empty")
            self._execution_snapshot_name = raw_name.strip()
            self._snapshot_components = normalize_attested_components(
                self.execution_snapshot.execution_components
            )
            if "execution_sha256" in self._snapshot_components:
                raise ConfigurationError(
                    "execution snapshot components must not author execution_sha256"
                )
            self._snapshot_diagnostics = normalize_attested_components(
                self.execution_snapshot.diagnostic_fingerprint
            )
            if "execution_sha256" in self._snapshot_diagnostics:
                raise ConfigurationError(
                    "execution snapshot diagnostics must not author execution_sha256"
                )
            raw_complete = self.execution_snapshot.execution_identity_complete
            if not isinstance(raw_complete, bool):
                raise ConfigurationError(
                    "execution snapshot execution_identity_complete must be a bool"
                )
            self._snapshot_complete = raw_complete
        if _is_command(entrypoint):
            # A process, not an object. Nothing to resolve and nothing to
            # import; isolation is inherent because it is already its own
            # process.
            # str() everything EXCEPT the sentinel, which must survive to
            # substitution. Stringifying it here is how it would silently
            # become the literal '<task>' in someone's argv.
            self._command = [a if a is TASK else str(a) for a in entrypoint]
            self._identity_argv = canonical_command_argv(
                self._command,
                volatile_value_options=self._volatile_command_values,
                task_sentinel=TASK,
            )
            self.resolved = None  # type: ignore[assignment]
            self.transport = transport
            self.artifacts_dir = str(artifacts_dir)
            self.isolation = "subprocess"
            self.concurrency = max(1, concurrency)
            self.rollout_timeout_s = rollout_timeout_s
            self.gate_flag = gate_flag
            self.endpoint_env = endpoint_env
            self._method = method
            self._adapters = adapters
            self._on_event = on_event
            self._stop = False
            self._inflight = {}
            self._inflight_lock = threading.Lock()
            self._beat_stop = threading.Event()
            self._ceiling_warned = False
            self._permanent_refusals = 0
            self._outbox_lock = threading.Lock()
            self._lease_tokens = {}
            return
        if command_output_adapter is not None:
            raise ConfigurationError(
                "command_output_adapter is supported only for command entrypoints"
            )
        if self._volatile_command_values:
            raise ConfigurationError(
                "volatile_command_values names command options, and this "
                "entrypoint is an importable agent with no argv. An in-process "
                "agent's identity carries no per-process launch arguments; "
                "remove the declaration."
            )
        if _is_import_spec(entrypoint):
            self._spec = entrypoint
            obj = build_agent(entrypoint)
        else:
            obj = entrypoint
            self._entrypoint_object_type = (
                f"{type(obj).__module__}.{type(obj).__qualname__}"
            )
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
        self.endpoint_env = endpoint_env
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
        #: Once per process: an inert harness ceiling is a property of the
        #: configuration, not of the rollout, so saying it every rollout would
        #: bury it.
        self._ceiling_warned = False
        #: Consecutive deliveries the control plane REFUSED permanently. See
        #: serve_forever: past a small number, claiming more work only produces
        #: more refusals. Deliberately unlocked: worker threads increment it,
        #: so a simultaneous pair can be counted as one. A brake that trips
        #: after four refusals instead of three is not a defect worth a lock on
        #: the delivery path; a brake that never trips would be.
        self._permanent_refusals = 0
        #: Serialises outbox writes against the drain, which runs on the serve
        #: loop's thread while worker threads are spooling their own results.
        self._outbox_lock = threading.Lock()
        #: Lease tokens for rollouts whose result this process is still holding,
        #: by rollout id. IN MEMORY ON PURPOSE: the token is a bearer credential
        #: and the outbox is a directory the customer's untrusted agent can read
        #: (see ``_spool_write``). Dropped the moment the outcome settles.
        self._lease_tokens: dict[str, str] = {}

    @property
    def execution_sha256(self) -> str | None:
        """SDK-authored opaque identity, or ``None`` when discovery was lossy."""
        return self._execution_sha256

    def configure_execution_identity(
        self,
        described: DiscoveredAgent | None,
        *,
        discovery_enabled: bool,
        sdk: str,
        contract_version: str,
    ) -> str | None:
        """Reconcile generic registration facts into one exact execution digest.

        No digest is emitted when discovery is disabled, unreadable, loses a
        behavior-bearing field, or lacks adapter-attested implementation/state
        evidence. In particular, a prompt projected at the public 20,000-character
        cap needs a pre-projection observation digest; hashing the truncated prefix
        would let distinct agents collide.
        """
        self._execution_sha256 = None
        self._execution_descriptor = None
        if not discovery_enabled or described is None:
            return None
        if not _description_is_exact(described):
            return None
        exact_observation = described.exact_observation_sha256
        projected = described.to_wire()
        try:
            observation_sha256 = exact_observation or canonical_observation_sha256(projected)
        except ConfigurationError:
            return None

        components = dict(self._attested_execution_components)
        for name, digest in self._snapshot_components.items():
            prior = components.get(name)
            if prior is not None and prior != digest:
                raise ConfigurationError(
                    f"attested execution component {name!r} conflicts with "
                    f"{self._execution_snapshot_name} adapter evidence"
                )
            components[name] = digest
        components = dict(sorted(components.items()))
        if self.execution_snapshot is not None and not self._snapshot_complete:
            return None
        # A completeness flag does not itself identify a single byte of
        # implementation/state.  At least one canonical component digest must
        # exist; otherwise every empty adapter would collapse to the same
        # affirmative execution identity.
        adapter_evidence_complete = bool(components)
        if not adapter_evidence_complete:
            # Prompt/tool discovery does not observe the orchestration code or
            # mutable object state. A caller that wants an OPSD-eligible exact
            # identity must use an execution adapter to attest those bytes.
            return None

        if self._command is not None:
            assert self._identity_argv is not None  # set with self._command
            entrypoint: dict[str, Any] = {
                "kind": "command",
                # Already canonical, and already validated at construction. The
                # declared-volatile values are absent from these entries, so the
                # digest survives a restart that moves a socket path; every
                # other token is verbatim, so it does not survive a real change.
                "argv": list(self._identity_argv),
            }
        elif self._spec is not None:
            entrypoint = {
                "kind": "import_spec",
                "spec": self._spec,
                "resolved_method": self.resolved.name,
            }
        else:
            entrypoint = {
                "kind": "constructed_object",
                "object_type": self._entrypoint_object_type,
                "resolved_method": self.resolved.name,
            }

        descriptor: dict[str, Any] = {
            "schema": "percepteye.agent_execution/v1",
            "sdk": sdk,
            "contract_version": contract_version,
            "entrypoint": entrypoint,
            "configuration": {
                "input_shape": self._configured_input_shape,
                "isolation": self.isolation,
                "concurrency": self.concurrency,
                "method": self._configured_method,
                "adapters": self._configured_adapters,
                "gate_flag": self._configured_gate_flag,
                "endpoint_env": self._configured_endpoint_env,
                "reports_tool_calls": self._reports_tool_calls,
            },
            "discovered_agent": projected,
            "discovery_observation_sha256": observation_sha256,
            "attested_components": components,
        }
        digest = execution_descriptor_sha256(descriptor)
        self._execution_descriptor = descriptor
        self._execution_sha256 = digest
        return digest

    # ── one rollout ────────────────────────────────────────────────────────
    def run_one(self, rollout: RolloutRequest) -> RolloutOutput:
        budget = rollout.seconds_remaining(time.time())
        if budget <= 0.0:
            # Already past its deadline when we got it -- a queue backlog, a
            # slow claim, a clock that moved. Running it anyway spent the full
            # local timeout on work the server had already given up on, and
            # then reported into a lease that no longer existed. Hand it back
            # immediately instead.
            out = RolloutOutput(
                final_text="",
                exclusion_reason="deadline_exceeded",
                exclusion_detail="deadline had already passed when claimed",
                timing={"elapsed_s": 0.0},
            )
            _bind_execution_fingerprint(out, None)
            return out
        # Stop the agent SHORT of the deadline, so there is time left to kill
        # it, collect its artifacts and get the report in while the lease is
        # still good. Running to the deadline exactly meant the kill and the
        # expiry landed together and the report chased a rollout the server had
        # already swept — the work was done and then thrown away at the last
        # step.
        #
        # THE ONLY CALL SITE of episode_plan, and the only line in this SDK
        # that turns an episode budget into a child deadline. Both child paths
        # and therefore every adapter — prime-agent, goose, a Python
        # entrypoint, a harness that does not exist yet — are downstream of it.
        plan = episode_plan(
            budget,
            rollout_timeout_s=self.rollout_timeout_s,
            envelope_s=self._delivery_envelope_s(),
        )
        self._record_plan(rollout.rollout_id, plan)
        self._emit("episode_plan", {
            "rollout_id": rollout.rollout_id,
            "budget_s": round(budget, 3),
            "child_s": round(plan.child_s, 3),
            "teardown_s": round(plan.teardown_s, 3),
            "delivery_s": round(plan.delivery_s, 3),
        })
        self._warn_if_call_ceiling_is_inert(plan)
        timeout_s = plan.child_s
        trajectory_dir = self._trajectory_dir(rollout.rollout_id)
        Path(trajectory_dir).mkdir(parents=True, exist_ok=True)

        if self._command is not None:
            return self._run_command_with_execution_snapshot(
                rollout, trajectory_dir=trajectory_dir, timeout_s=timeout_s,
                teardown_s=plan.teardown_s,
            )
        return self._run_python_with_execution_snapshot(
            rollout, trajectory_dir=trajectory_dir, timeout_s=timeout_s,
            teardown_s=plan.teardown_s,
        )

    def _delivery_envelope_s(self) -> float:
        """What the transport says one call costs it, worst case.

        Read with a default, because ``FlywheelTransport`` is a Protocol: a
        transport that never heard of this keeps the floor margin and the old
        behaviour instead of raising at the last step of a rollout.
        """
        try:
            declared = getattr(self.transport, "delivery_envelope_s", None)
        except Exception:  # a property that raises must not stop a rollout
            declared = None
        if isinstance(declared, (int, float)) and declared > 0:
            return float(declared)
        return REPORT_MARGIN_S

    def _record_plan(self, rollout_id: str, plan: EpisodePlan) -> None:
        with self._inflight_lock:
            entry = self._inflight.get(rollout_id)
            if entry is not None:
                entry.plan = plan

    def _delivery_budget_for(self, rollout_id: str) -> float:
        """What ``run_one`` held back for delivering THIS rollout's result."""
        with self._inflight_lock:
            entry = self._inflight.get(rollout_id)
            plan = entry.plan if entry is not None else None
        return plan.delivery_s if plan is not None else self._delivery_envelope_s()

    def _warn_if_call_ceiling_is_inert(self, plan: EpisodePlan) -> None:
        """Say when an adapter's per-call ceiling cannot fire inside the episode.

        OPTIONAL and read with a ``getattr``: an adapter that declares
        ``worst_case_call_s`` on its execution snapshot gets this check, and one
        that does not is served exactly as before. It exists because the
        prime-agent ceiling's own inertness floor was a COMMENT and a test over
        constants — so at the 300s budget this deployment has actually minted,
        one wedged provider call could still outlast the whole episode and
        nothing at runtime said so. Rollout #1 now does.
        """
        if self._ceiling_warned:
            return
        declared = getattr(self.execution_snapshot, "worst_case_call_s", None)
        if not isinstance(declared, (int, float)) or declared <= 0:
            return
        if float(declared) <= plan.child_s * CALL_CEILING_FRACTION:
            return
        self._ceiling_warned = True
        message = (
            f"{self._execution_snapshot_name or 'this harness'} bounds one "
            f"blocking call at {float(declared):.0f}s, but this episode leaves "
            f"the agent only {plan.child_s:.0f}s: a single wedged call can "
            f"consume the rollout and the ceiling cannot fire inside it. "
            f"Mint a longer episode, or lower the harness ceiling."
        )
        self._emit("harness_call_ceiling_inert", {
            "worst_case_call_s": float(declared),
            "child_s": round(plan.child_s, 3),
        })
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def _snapshot_prepare_failure(self) -> RolloutOutput:
        return RolloutOutput(
            final_text="",
            exclusion_reason="cancelled",
            exclusion_detail=(
                f"{self._execution_snapshot_name} execution snapshot could not "
                "prepare an attested child; the rollout was not run"
            ),
        )

    def _prepared_child_environment(self, child: ExecutionChild) -> dict[str, str]:
        try:
            environment = dict(child.environment)
        except Exception as exc:
            raise ConfigurationError(
                f"{self._execution_snapshot_name} execution child environment "
                "must be a string mapping"
            ) from exc
        if any(
            not isinstance(name, str) or not isinstance(value, str)
            for name, value in environment.items()
        ):
            raise ConfigurationError(
                f"{self._execution_snapshot_name} execution child environment "
                "must contain only string names and values"
            )
        return environment

    def _release_unrun_snapshot_child(self, child: ExecutionChild) -> None:
        snapshot = self.execution_snapshot
        assert snapshot is not None
        try:
            snapshot.attest_child(child)
        except Exception:
            pass
        finally:
            try:
                snapshot.release_child(child)
            except Exception:
                pass

    def _attest_and_release_snapshot_child(
        self, out: RolloutOutput, child: ExecutionChild,
    ) -> bool:
        snapshot = self.execution_snapshot
        assert snapshot is not None
        try:
            snapshot.attest_child(child)
        except Exception as exc:
            _exclude_for_execution_identity(
                out,
                f"{self._execution_snapshot_name} execution state drifted during "
                f"the rollout: {type(exc).__name__}",
            )
            attested = False
        else:
            attested = True
        finally:
            try:
                snapshot.release_child(child)
            except Exception as exc:
                _exclude_for_execution_identity(
                    out,
                    f"{self._execution_snapshot_name} execution child release "
                    f"failed: {type(exc).__name__}",
                )
                attested = False
        return attested

    def _run_command_with_execution_snapshot(
        self, rollout: RolloutRequest, *, trajectory_dir: str, timeout_s: float,
        teardown_s: float = TEARDOWN_DRAIN_S,
    ) -> RolloutOutput:
        """Run one command child, binding and then attesting adapter state."""
        assert self._command is not None
        snapshot = self.execution_snapshot
        child: ExecutionChild | None = None
        extra = {"PERCEPTEYE_TURN_INPUT": rollout.turn_input}
        if snapshot is not None:
            try:
                child = snapshot.prepare_child(rollout.rollout_id)
                extra.update(self._prepared_child_environment(child))
            except Exception:
                if child is not None:
                    self._release_unrun_snapshot_child(child)
                out = self._snapshot_prepare_failure()
                _bind_execution_fingerprint(out, None)
                return out

        try:
            out = run_rollout_as_command(
                self._command, rollout, trajectory_dir=trajectory_dir,
                timeout_s=timeout_s, gate_flag=self.gate_flag,
                teardown_s=teardown_s,
                output_adapter=self._command_output_adapter,
                # The command path's Popen, registered for the FIRST time. Every
                # BYOH agent this product serves is a command, and the heartbeat
                # thread's lease-lost kill only ever reached Python entrypoints.
                proc_sink=lambda p: self._attach_proc(rollout.rollout_id, p),
                env=build_child_env(
                    rollout_id=rollout.rollout_id, llm_url=rollout.llm_url,
                    api_key=rollout.api_key, trajectory_dir=trajectory_dir,
                    endpoint_env=self.endpoint_env, extra=extra,
                ),
                # The per-rollout directory is the child's working directory,
                # so the files a CLI agent leaves behind ARE its artifacts —
                # _collect_artifacts scans exactly this directory. Without
                # this, a coding agent's work landed wherever the serving
                # process happened to be started from, collection found
                # nothing, and the artifact grader scored an absence the
                # agent never caused.
                cwd=str(trajectory_dir),
            )
        except BaseException:
            self._detach_proc(rollout.rollout_id)
            # Popen and pre-result failures retain their existing exception
            # semantics, but must not strand an active private copy. There is
            # no result to train, so a failed post-attestation cannot create a
            # false success and must not mask the original launcher failure.
            if snapshot is not None and child is not None:
                try:
                    snapshot.attest_child(child)
                except Exception:
                    pass
                finally:
                    try:
                        snapshot.release_child(child)
                    except Exception:
                        pass
            raise
        self._detach_proc(rollout.rollout_id)

        if snapshot is None:
            _bind_execution_fingerprint(out, self.execution_sha256)
            return out
        assert child is not None

        attested = self._attest_and_release_snapshot_child(out, child)
        _bind_execution_fingerprint(
            out,
            self.execution_sha256 if attested else None,
            self._snapshot_diagnostics if attested else None,
        )
        return out

    def _run_python_with_execution_snapshot(
        self, rollout: RolloutRequest, *, trajectory_dir: str, timeout_s: float,
        teardown_s: float = TEARDOWN_DRAIN_S,
    ) -> RolloutOutput:
        snapshot = self.execution_snapshot
        child: ExecutionChild | None = None
        child_environment: dict[str, str] = {}
        ran = False
        if snapshot is not None:
            try:
                child = snapshot.prepare_child(rollout.rollout_id)
                child_environment = self._prepared_child_environment(child)
            except Exception:
                if child is not None:
                    self._release_unrun_snapshot_child(child)
                out = self._snapshot_prepare_failure()
                _bind_execution_fingerprint(out, None)
                return out

        try:
            if self.isolation == "inprocess":
                if child_environment:
                    out = RolloutOutput(
                        final_text="",
                        exclusion_reason="cancelled",
                        exclusion_detail=(
                            f"{self._execution_snapshot_name} execution snapshot prepared "
                            "environment overrides for an in-process entrypoint; the rollout "
                            "was not run because process-global environment is not isolated"
                        ),
                    )
                else:
                    ran = True
                    out = run_rollout_in_process(
                        self.resolved, rollout, trajectory_dir=trajectory_dir,
                        timeout_s=timeout_s, adapters=self._adapters,
                    )
            else:
                ran = True
                out = self._run_in_subprocess(
                    rollout, trajectory_dir=trajectory_dir, timeout_s=timeout_s,
                    extra_env=child_environment, teardown_s=teardown_s,
                )
        except BaseException:
            if snapshot is not None and child is not None:
                try:
                    snapshot.attest_child(child)
                except Exception:
                    pass
                finally:
                    try:
                        snapshot.release_child(child)
                    except Exception:
                        pass
            raise

        if snapshot is not None:
            assert child is not None
            attested = self._attest_and_release_snapshot_child(out, child)
        else:
            attested = True
        _bind_execution_fingerprint(
            out,
            self.execution_sha256 if ran and attested else None,
            self._snapshot_diagnostics if ran and attested else None,
        )
        return out

    def _trajectory_dir(self, rollout_id: str) -> str:
        return str(Path(self.artifacts_dir) / "_percepteye" / rollout_id)

    def _run_in_subprocess(
        self, rollout: RolloutRequest, *, trajectory_dir: str, timeout_s: float,
        extra_env: Mapping[str, str] | None = None,
        teardown_s: float = TEARDOWN_DRAIN_S,
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
            "trajectory_dir": trajectory_dir,
            "timeout_s": timeout_s,
        }
        env = build_child_env(
            rollout_id=rollout.rollout_id,
            llm_url=rollout.llm_url,
            api_key=rollout.api_key,
            trajectory_dir=trajectory_dir,
            endpoint_env=self.endpoint_env,
            extra=dict(extra_env or {}),
        )
        started = time.time()
        # Popen rather than subprocess.run: the heartbeat thread must be able
        # to KILL this child when the lease is lost. Under run() the child kept
        # working for the rest of its timeout against a credential the server
        # had already revoked -- burning the customer's compute to produce a
        # result nothing would accept.
        proc = subprocess.Popen(
            [sys.executable, "-m", "agent_flywheel.child"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            # Its own process GROUP, exactly as the command path does and for
            # exactly the reason stated there. This path never had it: a plain
            # in-group grandchild held the pipes open, the bare `proc.kill()`
            # below left it running, and the untimed `communicate()` that
            # followed blocked (reproduced: >15s). The command path learned this
            # and the lesson was never carried across.
            start_new_session=(os.name != "nt"),
        )
        self._attach_proc(rollout.rollout_id, proc)
        try:
            stdout, stderr = proc.communicate(
                input=json.dumps(payload), timeout=timeout_s + 15.0
            )
        except subprocess.TimeoutExpired:
            stdout, stderr, drained = _kill_and_drain(proc, teardown_s)
            return RolloutOutput(
                final_text="",
                exclusion_reason="deadline_exceeded",
                exclusion_detail=(
                    f"child exceeded {timeout_s:.0f}s and was killed"
                    + (f": {stderr[-1500:]}" if stderr and stderr.strip() else "")
                    + ("" if drained else
                       f"; its output could not be read within {teardown_s:.0f}s "
                       f"of the kill — a descendant outside the killed process "
                       f"group still held the pipes")
                ),
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
        _attach_trajectory(out, trajectory_dir)
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
        """A third of the lease the server GRANTED AT CLAIM.

        Measured from the lease the server actually granted, not the one we
        asked for. The server clamps our request, so a deployment with a lower
        ceiling than our default would reap every long rollout underneath a
        heartbeat pacing itself against a lease it never got. Falls back to the
        requested value for a server that does not report what it granted.

        HONEST ABOUT WHAT THIS DOES NOT GIVE YOU. This used to claim "so two
        beats may be lost first", and that is true only of the FIRST beat. Each
        beat asks the control plane to extend by ``min(lease_ms, 600_000)``,
        and the control plane SETS the expiry to now plus that -- so once the
        claim-time lease has been replaced, the interval and the extension are
        the same length and every later beat is scheduled at the expiry it
        exists to pre-empt. The margin after beat #1 is one POST's latency, not
        two intervals.

        Left as it is deliberately, and reported rather than fixed here: the
        arithmetic says it cannot fire inside the episode lengths this
        deployment mints (the first beat is not due until ~600s and pushes the
        expiry to ~1200s, against episodes of 300-894s), and bundling an
        unrelated change into this one would make the mutation test for the
        actual fix ambiguous. It is a real defect for any rollout past ~1200s.
        The seam it needs is already there on the wire: the control plane
        returns the new ``lease_expires_ts`` from every heartbeat and
        ``AttachTransport.heartbeat`` discards it.
        """
        granted = rollout.lease_expires_ts
        if granted is not None:
            # A DURATION, measured once. `lease_expires_ts` is an absolute
            # timestamp frozen at claim time, so recomputing this against
            # `now` on every beat paced against the shrinking REMAINDER rather
            # than the lease: the interval halved, then quartered, then pinned
            # itself to the 1.0 floor, turning a 300s lease on a 900s rollout
            # into roughly 600 heartbeat POSTs where the design intends nine.
            # Nothing was reaped -- the server extends on every beat -- so it
            # was pure load, invisible from here and paid for by the control
            # plane. Callers must compute this once, at claim, and reuse it.
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
                entry.next_beat_at = now + entry.beat_interval
                try:
                    self.transport.heartbeat(entry.rollout)
                except LeaseLost:
                    # The server has already taken it back. Stop the work now
                    # rather than let it run to completion for nothing.
                    entry.lease_lost = True
                    self._emit("lease_lost", {"rollout_id": entry.rollout.rollout_id})
                    proc = entry.proc
                    if proc is not None and proc.poll() is None:
                        # The GROUP, not the one process. A command agent is a
                        # shell that spawns freely, so killing only the process
                        # we launched left its children alive holding the
                        # pipes: MEASURED, the runner then sat in the child's
                        # own `communicate()` for the full 120s timeout after
                        # the lease was already gone. `proc.kill()` here was
                        # written when this could only ever be a Python child.
                        _kill_tree(proc)
                except TransportError as exc:
                    # Transient. The next beat may well succeed, and there are
                    # two more before the lease lapses.
                    self._emit("heartbeat_failed", {
                        "rollout_id": entry.rollout.rollout_id, "error": str(exc),
                    })
                except Exception as exc:
                    # ANYTHING ELSE, and this is the whole reason the loop is
                    # worth guarding: ``FlywheelTransport`` is a Protocol, so a
                    # customer implementation can raise whatever it likes, and
                    # an exception raised here does not fail ONE rollout -- it
                    # unwinds the single thread that renews EVERY lease this
                    # process holds. Every in-flight rollout then runs to
                    # completion against a lease nobody is renewing, the
                    # control plane's reaper writes abandoned/deadline_exceeded
                    # over all of them, and no report is ever accepted. That is
                    # the incident this module was rewritten for, reproduced
                    # from one unexpected exception type.
                    #
                    # Treated exactly like a transient failure: the next beat
                    # is already scheduled, and a beat that raises is not
                    # evidence the lease is gone (only a LeaseLost is).
                    self._emit("heartbeat_failed", {
                        "rollout_id": entry.rollout.rollout_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    })

    # ── the loop ───────────────────────────────────────────────────────────
    def serve_forever(self, *, max_rollouts: int | None = None) -> int:
        """Claim → run → deliver, until stopped.

        Returns how many outcomes were DELIVERED — acknowledged by the control
        plane, whether in the episode or later from the outbox. It used to
        count rollouts that finished running, which counted a result nobody
        ever received as a success. ``max_rollouts`` still bounds how many
        rollouts are SERVED, because a loop that re-claimed until N landed
        would spin forever against a control plane refusing every report.
        """
        served = 0
        delivered = 0
        beat = threading.Thread(
            target=self._heartbeat_loop, name="percepteye-heartbeat", daemon=True
        )
        beat.start()
        try:
            with _cf.ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                while not self._stop:
                    if max_rollouts is not None and served >= max_rollouts:
                        break
                    # DRAIN BEFORE CLAIMING. A result the previous process (or
                    # the previous minute) could not deliver is worth more than
                    # the next rollout, and doing it here is what makes a
                    # restart recover rather than forget.
                    #
                    # GUARDED, like the drain in the `finally` below. Delivering
                    # old results is housekeeping; claiming new work is the job.
                    # Unguarded, this was the FIRST statement of the loop, so
                    # anything it raised stopped the flywheel before it claimed
                    # anything at all -- a fleet that registers, claims nothing,
                    # and leaves every rollout to be reaped as
                    # abandoned/deadline_exceeded. That is the incident this
                    # whole mechanism exists to end, and it must not be
                    # reachable FROM the mechanism.
                    try:
                        delivered += self._drain_outbox()
                    except Exception as exc:
                        self._emit("drain_failed", {
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        warnings.warn(
                            f"the outbox drain failed ({type(exc).__name__}: "
                            f"{exc}); held results stay on disk and claiming "
                            f"continues.",
                            RuntimeWarning, stacklevel=2,
                        )
                    if self._permanent_refusals >= _PERMANENT_REFUSAL_LIMIT:
                        message = (
                            f"the control plane refused {self._permanent_refusals} "
                            f"deliveries in a row with an error that resending "
                            f"cannot change; this flywheel has stopped claiming. "
                            f"Results already produced are in "
                            f"{self._outbox_dir()} and will be delivered on the "
                            f"next start once the cause is fixed."
                        )
                        self._emit("delivery_brake", {
                            "consecutive_permanent_refusals": self._permanent_refusals,
                            "outbox": str(self._outbox_dir()),
                        })
                        warnings.warn(message, RuntimeWarning, stacklevel=2)
                        break
                    # Never claim more than we still intend to run. Claiming a
                    # full batch when one rollout remained left the rest leased
                    # to a runner that was about to exit.
                    want = self.concurrency
                    if max_rollouts is not None:
                        want = min(want, max_rollouts - served)
                    try:
                        rollouts, poll_after_ms = self.transport.claim(want)
                    except TransportError as exc:
                        self._emit("claim_failed", {"error": str(exc)})
                        sleep_ms(5_000)
                        continue
                    if not rollouts:
                        # A successful claim that returned no work. Emitted so
                        # an operator can tell "the queue is empty" from "the
                        # claim route is broken" -- both were previously a
                        # silent sleep, and the only other signal, claim_failed,
                        # fires only on the failing one.
                        self._emit("claim_empty", {"poll_after_ms": poll_after_ms})
                        sleep_ms(poll_after_ms)
                        continue
                    # Concurrently, which is what claiming N meant all along.
                    # Serving them one after another held N-1 leases on work
                    # nobody was doing, and with no heartbeat those were the
                    # rollouts most likely to lapse.
                    now = time.time()
                    with self._inflight_lock:
                        for r in rollouts:
                            _interval = self._beat_interval_for(r)
                            self._inflight[r.rollout_id] = _InFlight(
                                r, _interval, now + _interval
                            )
                    futures = [pool.submit(self._serve_one, r) for r in rollouts]
                    for fut in _cf.as_completed(futures):
                        # _serve_one really does not raise now — it has the
                        # `except Exception` this line's comment used to CLAIM
                        # it had. Before, a ContractViolation from `to_wire`
                        # (not a TransportError) came out here, stopped the
                        # loop, and took the heartbeat thread down with it, so
                        # ONE malformed payload ended lease renewal for every
                        # concurrent rollout.
                        if fut.result():
                            delivered += 1
                        served += 1
        finally:
            # One last drain while the process is still alive: anything still
            # undelivered has had every chance this process can give it, and
            # what remains on disk is what the next start will pick up.
            try:
                delivered += self._drain_outbox()
            except Exception:  # never let housekeeping mask the real exit
                pass
            self._beat_stop.set()
            beat.join(timeout=5.0)
        return delivered

    def _serve_one(self, rollout: RolloutRequest) -> bool:
        """Serve one rollout. Returns whether its outcome was DELIVERED.

        This never raises. The call site in ``serve_forever`` carried a comment
        saying so for a long time while only ``LeaseLost`` and
        ``TransportError`` were caught below it — so a ``ContractViolation``
        from ``to_wire``, which is not a ``TransportError``, escaped through
        ``fut.result()`` and stopped the heartbeat thread for every concurrent
        lease. A comment is not a mechanism; this is.
        """
        self._emit("rollout_start", {"rollout_id": rollout.rollout_id})
        delivered = False
        try:
            delivered = self._serve_one_inner(rollout)
            return delivered
        except Exception as exc:
            self._emit("serve_failed", {
                "rollout_id": rollout.rollout_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            try:
                # Even a bug in this SDK leaves a record. Silence here is what
                # the control plane's reaper turns into abandoned/deadline
                # _exceeded twenty minutes later, with no trace of a cause.
                delivered = self._deliver_outcome(
                    rollout,
                    RolloutOutput(
                        final_text="",
                        exclusion_reason="flywheel_crash",
                        exclusion_detail=f"{type(exc).__name__}: {exc}",
                    ),
                    lease_lost=self._lease_lost(rollout.rollout_id),
                )
                return delivered
            except Exception:
                warnings.warn(
                    f"rollout {rollout.rollout_id} failed inside the flywheel "
                    f"and its outcome could not be recorded: "
                    f"{type(exc).__name__}: {exc}",
                    RuntimeWarning, stacklevel=2,
                )
                return False
        finally:
            with self._inflight_lock:
                self._inflight.pop(rollout.rollout_id, None)
            # DELIVERED, not "no longer in the retry set". The trajectory is
            # the only local copy of this rollout's evidence -- the artifacts
            # too big or too binary to travel inside the report exist nowhere
            # else -- so it is kept until the control plane actually has the
            # result. The predicate used to be "no spool file at this path",
            # which a RETIRED record satisfies the instant it is renamed: a
            # permanently refused rollout kept its result and lost the evidence
            # behind it, in the one case an operator has to reconstruct by hand.
            if delivered:
                self._purge_trajectory(rollout.rollout_id)

    def _serve_one_inner(self, rollout: RolloutRequest) -> bool:
        try:
            out = self.run_one(rollout)
        except Exception as exc:
            # The flywheel could not RUN this rollout: nothing was produced, so
            # handing it back is right and lets another worker try it. The
            # boundary is deliberate — a crash before the agent produced
            # anything is an abandon (requeued), a crash after it did is a
            # report carrying an exclusion (recorded, not repeated).
            detail = f"{type(exc).__name__}: {exc}"
            self._emit("rollout_crash", {"rollout_id": rollout.rollout_id, "error": str(exc)})
            try:
                self.transport.abandon(rollout, reason="flywheel_crash", detail=detail)
                return False
            except (TransportError, LeaseLost) as abandon_exc:
                # Used to be a bare `pass`, which is how a crashed rollout
                # became a row nobody could explain: the control plane never
                # learned of the crash and the reaper wrote its own reason over
                # it. Fall through and DELIVER the crash instead.
                self._emit("abandon_failed", {
                    "rollout_id": rollout.rollout_id,
                    "error": f"{type(abandon_exc).__name__}: {abandon_exc}",
                })
                out = RolloutOutput(
                    final_text="",
                    exclusion_reason="flywheel_crash",
                    exclusion_detail=detail,
                )
        lease_lost = self._lease_lost(rollout.rollout_id)
        deadline = time.time() + self._delivery_budget_for(rollout.rollout_id)
        try:
            # Evidence FIRST, whatever the outcome. An agent that exhausted its
            # turn or time budget still wrote files, and for a coding/REPL agent
            # those files ARE what the reward reads. Collecting only on the
            # clean-exit path meant a rollout could write a CORRECT solution and
            # have it discarded because the process did not terminate tidily —
            # measured on run 516b7a55, where a correct top_ips.sh sat in the
            # trajectory while the row carried artifacts={}.
            #
            # Collected even when the lease is lost: the bytes travel INSIDE the
            # report, which is the only place the control plane reads them.
            self._collect_artifacts(rollout, out)
            _mark_substrate_unavailable(out)
        except Exception as exc:
            # Evidence collection must never decide whether a result is
            # delivered. It used to sit inside the same try as the report, so
            # an OSError from a vanished directory took the whole outcome with
            # it — and out through `fut.result()`.
            _exclude_for_execution_identity(
                out, f"artifact collection failed: {type(exc).__name__}: {exc}"
            )
            self._emit("artifact_failed", {
                "rollout_id": rollout.rollout_id, "name": "*",
                "error": f"{type(exc).__name__}: {exc}",
            })
        # EVERY rollout is delivered, whatever happened to it.
        #
        # This used to abandon when an exclusion carried no work product —
        # "nothing to grade, do not train on this". That is a decision about
        # TRAINABILITY, and it is not the SDK's to make: abandon suppresses the
        # record entirely, so the control plane could not see rollouts it might
        # have wanted to count, and the reason it would have judged them by
        # never arrived. An operator asking "what did my agent do" got silence
        # for exactly the rollouts that went worst.
        #
        # Reported WITH its exclusion when it has one: to_wire carries the
        # exclusion block, so the server records the work product and still
        # knows not to train on it.
        return self._deliver_outcome(
            rollout, out, lease_lost=lease_lost, deadline=deadline
        )

    # ── delivery: the one exit every outcome leaves through ────────────────
    def _deliver_outcome(
        self,
        rollout: RolloutRequest,
        out: RolloutOutput,
        *,
        lease_lost: bool,
        deadline: float | None = None,
    ) -> bool:
        """Deliver this outcome, or record it durably and say so OUT LOUD.

        THE ONLY CALLER of ``transport.report`` in this SDK. Before it there
        were four terminal paths out of ``_serve_one_inner`` and only one of
        them reported; the other three returned, each emitting an event into a
        hook production had never wired, and a ``finally`` then deleted the
        trajectory. A rollout whose report was refused or lost left NOTHING —
        not on the row, not on disk, not in the log — and the control plane's
        reaper wrote ``abandoned`` / ``deadline_exceeded`` over it twenty
        minutes later with a detail that says no agent ever claimed it.

        The result is written to the outbox BEFORE the first network call, so
        the process can die between the two and still deliver. What is spooled
        is the exact ``to_wire()`` payload — the flywheel/1 contract shape that
        every adapter produces — which is why the drain needs to know nothing
        about the harness that produced it.

        Redelivery is safe by construction: the control plane keys the report
        on the rollout id and answers an identical resubmission as a duplicate,
        and for an ORDINARY rollout it accepts a late report and overwrites the
        reaper's verdict, so a drain minutes later REPAIRS the row rather than
        merely recording the loss.

        WHAT REPAIR DOES NOT REACH, because selling it would be worse than not
        having it:

          * An IDENTITY-SEALED rollout. Its report route additionally requires
            a live, matching lease, so a late report -- the only kind the hold
            below can make once a lease is lost -- is refused. The result is
            kept on disk under a name that says it was refused; the ROW stays
            lost. Sealed rollouts are repaired by re-running them, not here.
          * Training credit. The control plane overwrites a row's
            training-exclusion only when the incoming report carries one of its
            own, so a repaired row can read ``reported`` and still be excluded
            from training. This restores THE RECORD, not necessarily the data.

        AND THE SDK CANNOT TELL WHICH KIND IT IS HOLDING. The requirement that
        makes a rollout sealed lives in the control plane's rollout metadata
        under ``_mc_required_agent_execution_sha256``, and the claim route
        NARROWS the metadata it returns to an allowlist -- ``task_id``,
        ``rollout_id``, ``completion_gate`` -- deliberately, so an agent cannot
        condition on what the platform knows about it. There is therefore no
        predicate here that separates the two, and none is invented: every
        undeliverable result is treated the way a sealed one has to be, which
        is the safe direction. That costs an ordinary rollout nothing, because
        a result that DOES land is removed from the outbox on the spot.
        """
        if deadline is None:
            deadline = time.time() + self._delivery_budget_for(rollout.rollout_id)
        # IN MEMORY, NEVER ON DISK. The lease token is a bearer credential --
        # it authorises reporting and abandoning this rollout -- and the outbox
        # lives under ``artifacts_dir``, one level up from the working
        # directory the customer's UNTRUSTED agent is handed as its cwd. It ran
        # as this uid, so no file mode could keep it out, and any sweep of that
        # tree into an artifact bundle carried the credential with it.
        #
        # Redelivery inside THIS process still has the token, which is the only
        # redelivery that could ever have used it: a token survives on disk
        # only to outlive the process, and a lease that outlives its process is
        # one the control plane's reaper has already cleared. Keeping it in
        # memory loses nothing that was real and ends the disclosure.
        if rollout.lease_token:
            with self._outbox_lock:
                self._lease_tokens[rollout.rollout_id] = rollout.lease_token
        record = {
            "schema": _OUTBOX_SCHEMA,
            "rollout_id": rollout.rollout_id,
            "deadline_ts": float(rollout.deadline_ts),
            "lease_lost": bool(lease_lost),
            "first_seen_ts": time.time(),
            "attempts": 0,
            "last_error": None,
            "result": _wire_payload(out),
        }
        path = self._spool_path(rollout.rollout_id)
        try:
            self._spool_write(path, record)
        except OSError as exc:
            # No durable copy. Say so before trying the network, because if
            # the send also fails this outcome is genuinely gone.
            warnings.warn(
                f"could not spool rollout {rollout.rollout_id}'s result to "
                f"{path}: {exc}. If delivery now fails the result is lost.",
                RuntimeWarning, stacklevel=2,
            )
            self._emit("outcome_spool_failed", {
                "rollout_id": rollout.rollout_id, "error": str(exc),
            })
        return self._deliver_record(path, record, deadline=deadline)

    def _deliver_record(
        self, path: Path, record: dict[str, Any], *, deadline: float
    ) -> bool:
        """Send ONE spooled outcome. The single place report failures are read.

        Used by the in-episode path and by the drain alike, so a redelivery
        cannot take a route the first attempt did not.
        """
        rollout_id = str(record.get("rollout_id") or "")
        try:
            # REBUILT BEFORE ANYTHING ELSE, and inside the try. ``_read_spool``
            # checks the envelope -- schema, rollout id, result-is-a-dict -- and
            # cannot check that the result's CONTENTS still rebuild: a tool call
            # missing "name" raises KeyError out of ``_output_from_wire``. That
            # used to escape this method, escape the drain, and escape the first
            # statement of the claim loop, so ONE unrebuildable file stopped the
            # flywheel before it claimed anything -- and the file survived to do
            # it again on the next start. A record this SDK cannot rebuild is a
            # settled record, not a fatal one.
            deadline_ts = float(record.get("deadline_ts") or 0.0)
            rollout = _spooled_rollout(
                record, lease_token=self._remembered_lease_token(rollout_id)
            )
            out = _output_from_wire(dict(record.get("result") or {}))
        except Exception as exc:
            record["last_error"] = f"{type(exc).__name__}: {exc}"
            kept = self._settle(path, record, delivered=False, retire="unreadable")
            self._emit("outcome_unrebuildable", {
                "rollout_id": rollout_id,
                "error": f"{type(exc).__name__}: {exc}",
                "path": str(kept),
            })
            warnings.warn(
                f"rollout {rollout_id}'s spooled result could not be rebuilt "
                f"({type(exc).__name__}: {exc}) and will not be retried. The "
                f"file is kept at {kept}.",
                RuntimeWarning, stacklevel=2,
            )
            return False
        if record.get("lease_lost") and time.time() < deadline_ts:
            # NEVER RACE A LIVE HOLDER. The lease is gone, so the control plane
            # may already have handed this rollout to someone else, and for an
            # ordinary rollout its report route performs no ownership check —
            # our stale result would simply overwrite theirs. Past the
            # deadline no new claim is possible (the control plane refuses to
            # claim a rollout whose budget has elapsed), so the drain delivers
            # it then. DEPENDS ON THAT SERVER-SIDE PREDICATE: if claiming past
            # the deadline ever becomes possible, this becomes a race.
            self._emit("outcome_spooled", {
                "rollout_id": rollout_id,
                "reason": "lease_lost_before_deadline",
                "path": str(path),
            })
            warnings.warn(
                f"rollout {rollout_id} lost its lease; its result is held in "
                f"{path} and will be attempted once the rollout's deadline has "
                f"passed and no other worker can hold it. AN IDENTITY-SEALED "
                f"ROLLOUT CANNOT BE REPAIRED THIS WAY: its report route also "
                f"requires the lease that was just lost, so a deferred "
                f"delivery will be refused and the result kept for inspection "
                f"rather than recorded.",
                RuntimeWarning, stacklevel=2,
            )
            return False

        record["attempts"] = int(record.get("attempts") or 0) + 1
        try:
            ack = self._report(rollout, out, deadline=deadline)
        except LeaseLost as exc:
            # THE LEASE OR THE IDENTITY, NOT THE CONFIGURATION. Only an
            # identity-sealed rollout's report route can answer 410 at all --
            # an ordinary rollout's performs no ownership check -- and what it
            # refuses is THIS ROLLOUT's lease, which lapsed or was reassigned.
            #
            # So two things that used to happen here are wrong:
            #
            #  * the file was UNLINKED. Resending cannot fix a lost lease, but
            #    that is not a reason to destroy the only copy of work a
            #    customer's agent actually did: for a sealed rollout the
            #    deferred delivery below is refused BY CONSTRUCTION (the hold
            #    waits for the deadline, by which time the lease it needs is
            #    certainly gone), so unlinking made lease loss a guaranteed
            #    shredder. Moved aside NAMED instead: the row is still lost,
            #    the bytes are not, and an operator can see what was produced.
            #  * it counted toward the permanent-refusal brake. That brake
            #    exists for a CONFIGURATION fault -- a key missing
            #    ``rollout.report`` scope, an expired key -- which every
            #    delivery will hit until a human fixes it. A lapsed lease is
            #    per-rollout and says nothing about the next one, so three
            #    unlucky rollouts must not stop the fleet claiming.
            record["last_error"] = str(exc)
            # Written back so the file kept for inspection SAYS why it was
            # refused, then renamed out of the retry set in one step.
            kept = self._settle(
                path, record, delivered=False, retire="refused-lease-lost"
            )
            self._emit("outcome_undeliverable", {
                "rollout_id": rollout_id, "status": 410, "error": str(exc),
                "path": str(kept),
            })
            warnings.warn(
                f"the control plane refused rollout {rollout_id}'s result: "
                f"{exc}. The row is lost and will be swept, but the result "
                f"itself is kept at {kept} and this rollout's evidence at "
                f"{self._trajectory_dir(rollout_id)}.",
                RuntimeWarning, stacklevel=2,
            )
            return False
        except TransportError as exc:
            permanent = _is_permanent_refusal(exc)
            status = getattr(exc, "status_code", None)
            # RETIRED, NEVER DESTROYED. This used to unlink the file for every
            # status in ``_REDELIVERY_CANNOT_FIX``, and for a sealed rollout
            # that is the whole failure mode: the control plane answers 410
            # forever once the reaper has cleared the lease, the SDK correctly
            # stops resending -- and then deleted the only copy of work the
            # customer's agent actually did. "Cannot be delivered" is a reason
            # to stop TRYING, not a reason to shred. The file leaves the retry
            # set under a name that says which refusal ended it.
            #
            # The SDK cannot tell a sealed rollout from an ordinary one (see
            # ``_deliver_outcome``), so this is not conditioned on guessing:
            # every permanent refusal keeps its bytes.
            retire = f"refused-{status}" if (
                permanent and status in _REDELIVERY_CANNOT_FIX
            ) else None
            record["last_error"] = str(exc)
            kept = self._settle(path, record, delivered=False, retire=retire)
            if permanent:
                self._permanent_refusals += 1
                self._emit("outcome_undeliverable", {
                    "rollout_id": rollout_id, "status": status, "error": str(exc),
                    "path": str(kept),
                })
                warnings.warn(
                    f"the control plane refused rollout {rollout_id}'s result: "
                    f"{exc}. "
                    + (
                        f"Resending the same result cannot change that, so it "
                        f"will not be retried. The result is kept at {kept} "
                        f"and this rollout's evidence at "
                        f"{self._trajectory_dir(rollout_id)}."
                        if retire else
                        f"The result is held in {kept} and will be resent once "
                        f"the cause is fixed."
                    ),
                    RuntimeWarning, stacklevel=2,
                )
            else:
                self._emit("report_failed", {
                    "rollout_id": rollout_id, "error": str(exc),
                })
                warnings.warn(
                    f"rollout {rollout_id}'s result could not be delivered "
                    f"({exc}); it is held in {path} and will be resent.",
                    RuntimeWarning, stacklevel=2,
                )
            return False
        except Exception as exc:
            # A transport is customer-supplied — ``FlywheelTransport`` is a
            # Protocol — so it can raise anything. Treated as transient and the
            # result KEPT: guessing that an unrecognised exception is permanent
            # would delete work on the strength of not understanding the error.
            # Caught here rather than left to escape, because escaping this
            # method is how a single bad transport stops the drain for every
            # other result it still holds.
            record["last_error"] = f"{type(exc).__name__}: {exc}"
            self._settle(path, record, delivered=False, retire=None)
            self._emit("report_failed", {
                "rollout_id": rollout_id, "error": record["last_error"],
            })
            warnings.warn(
                f"the transport raised {type(exc).__name__} delivering rollout "
                f"{rollout_id}'s result: {exc}. It is held in {path}.",
                RuntimeWarning, stacklevel=2,
            )
            return False
        self._permanent_refusals = 0
        self._settle(path, record, delivered=True, retire=None)
        self._emit("rollout_reported", {
            "rollout_id": rollout_id,
            "grade": (ack or {}).get("grade"),
            "duplicate": (ack or {}).get("duplicate", False),
            "exclusion": (record.get("result") or {}).get("exclusion", {}).get("reason"),
            "attempts": record["attempts"],
        })
        return True

    def _report(
        self, rollout: RolloutRequest, out: RolloutOutput, *, deadline: float
    ) -> dict[str, Any]:
        """One report call, bounded by the reserve when the transport takes one."""
        if getattr(self.transport, "accepts_delivery_deadline", False):
            return self.transport.report(rollout, out, deadline=deadline)
        return self.transport.report(rollout, out)

    # ── the outbox ─────────────────────────────────────────────────────────
    def _outbox_dir(self) -> Path:
        # Keyed by AGENT, because two flywheels sharing one artifacts_dir would
        # otherwise replay each other's results.
        agent = getattr(self.transport, "agent_id", None)
        slug = _fs_slug(agent) if isinstance(agent, str) and agent else "agent"
        return Path(self.artifacts_dir) / "_percepteye" / "_outbox" / slug

    def _spool_path(self, rollout_id: str) -> Path:
        return self._outbox_dir() / f"{_fs_slug(rollout_id)}.json"

    def _spool_write(self, path: Path, record: Mapping[str, Any]) -> None:
        """Write atomically: a half-written result is worse than none.

        NO BEARER CREDENTIAL REACHES THIS FILE, which is the part that had to
        be a mechanism rather than a file mode. The outbox sits under
        ``artifacts_dir`` at ``_percepteye/_outbox/<agent>/``, and the child is
        handed ``artifacts_dir/_percepteye/<rollout_id>`` as its cwd -- so this
        directory is ``../_outbox`` from a shell the customer's UNTRUSTED agent
        controls, running as this same uid. A lease token written here was
        readable by the agent it fences out, and travelled into any bundle that
        swept the artifacts tree. The token now lives in this process's memory
        (see ``_deliver_outcome``); what reaches disk is the result and the
        rollout id.

        0700/0600 is still applied and is still worth having -- it stops a
        co-tenant, a second service account and a group-readable default -- but
        it was never the answer to the agent itself, and nothing at this layer
        is: ``isolation`` is a process boundary and an environment scrub and
        says nothing about the filesystem. What is on disk is every concurrent
        rollout's complete RESULT, which the agent already authored.
        """
        with self._outbox_lock:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            try:
                # mkdir's mode applies only when it CREATES the directory, and
                # an outbox from an earlier version is already there.
                os.chmod(path.parent, 0o700)
            except OSError:
                pass
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(record, default=str), encoding="utf-8")
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
            os.replace(tmp, path)

    def _remembered_lease_token(self, rollout_id: str) -> str | None:
        """This process's copy of a rollout's lease token, if it still has one.

        ``None`` for anything spooled by an EARLIER process, which is correct
        rather than merely unavoidable: a lease that outlived its process is
        one the control plane's reaper has already cleared, so the token it
        would have supplied is refused anyway. The report goes without it and
        the refusal is recorded honestly.
        """
        with self._outbox_lock:
            return self._lease_tokens.get(rollout_id)

    def _settle(
        self, path: Path, record: Mapping[str, Any], *,
        delivered: bool, retire: str | None,
    ) -> Path:
        """Decide what becomes of one spooled outcome's FILE. The one place.

        Three fates and no fourth:

        * DELIVERED -- the control plane has it, so the copy goes.
        * RETIRED (``retire`` names why) -- this SDK will not send it again.
          The record is written back first, so the file an operator opens says
          what refused it, and is then renamed out of the ``*.json`` retry set.
          NEVER unlinked: "cannot be delivered" is a reason to stop trying, not
          a licence to destroy the only copy of a customer agent's work.
        * neither -- still in the retry set, with what we now know about it.

        Returns where the record lives now (the original path when a rename was
        not possible, so a warning quoting it still points at something real).
        """
        rollout_id = str(record.get("rollout_id") or "")
        if delivered:
            with self._outbox_lock:
                self._lease_tokens.pop(rollout_id, None)
                try:
                    path.unlink()
                except OSError:  # FileNotFoundError included: already gone
                    pass
            return path
        try:
            self._spool_write(path, record)
        except OSError:
            pass
        if retire is None:
            return path
        with self._outbox_lock:
            # The rollout is over for this SDK either way, and holding its
            # capability past that is exactly the exposure this file avoids.
            self._lease_tokens.pop(rollout_id, None)
        return _move_aside(path, retire)

    def _drain_outbox(self) -> int:
        """Deliver everything the outbox still holds. Returns how many landed.

        Called at the top of the claim loop — BEFORE claiming, because a result
        already produced is worth more than the next rollout — and once more on
        the way out. A rollout still in flight is skipped: its own worker owns
        that file until it is finished with it.
        """
        root = self._outbox_dir()
        if not root.is_dir():
            return 0
        delivered = 0
        try:
            paths = sorted(root.glob("*.json"))
        except OSError:
            return 0
        for path in paths:
            try:
                delivered += self._drain_one(path)
            except Exception as exc:
                # NO SINGLE FILE MAY STOP THE DRAIN. Everything below is
                # defensive already, but "defensive" is an argument and this is
                # a mechanism: whatever one unexpected file does, the results
                # behind it in this directory still get their turn, and the
                # claim loop this runs at the top of still reaches its claim.
                kept = _move_aside(path, "unreadable")
                self._emit("outcome_unrebuildable", {
                    "rollout_id": "", "path": str(kept),
                    "error": f"{type(exc).__name__}: {exc}",
                })
                warnings.warn(
                    f"a spooled result at {path} could not be processed "
                    f"({type(exc).__name__}: {exc}) and will not be retried. "
                    f"It is kept at {kept}.",
                    RuntimeWarning, stacklevel=2,
                )
        return delivered

    def _drain_one(self, path: Path) -> int:
        """Deliver ONE spooled outcome. Returns 1 if it landed, else 0."""
        record = _read_spool(path)
        if record is None:
            _move_aside(path, "unreadable")
            return 0
        rollout_id = str(record.get("rollout_id") or "")
        with self._inflight_lock:
            if rollout_id in self._inflight:
                return 0
        age = time.time() - float(record.get("first_seen_ts") or 0.0)
        if age > _OUTBOX_MAX_AGE_S:
            kept = _move_aside(path, "stale")
            self._emit("outcome_abandoned_in_outbox", {
                "rollout_id": rollout_id, "age_s": round(age, 1),
                "path": str(kept),
            })
            warnings.warn(
                f"rollout {rollout_id}'s result has been undeliverable for "
                f"{age / 86400.0:.1f} days and is no longer being retried. "
                f"It is kept at {kept}.",
                RuntimeWarning, stacklevel=2,
            )
            return 0
        delivered = 1 if self._deliver_record(
            path, record, deadline=time.time() + self._delivery_envelope_s(),
        ) else 0
        if delivered:
            # ONLY on delivery. "The file at this path is gone" was the same
            # predicate for a result the control plane accepted and a result it
            # refused forever, and the refused one is precisely where the local
            # evidence is the last copy there is.
            self._purge_trajectory(rollout_id)
        return delivered

    def _collect_artifacts(
        self, rollout: RolloutRequest, out: RolloutOutput,
    ) -> None:
        """Record the files the agent produced in the report that carries them.

        For a coding or REPL agent the ARTIFACT is the reward -- a diff, a
        built binary, a screenshot -- and a scorer that only sees final text is
        grading the commentary rather than the work.

        Every file is named with its digest (and size); the REPORT additionally
        inlines the content of SMALL TEXT artifacts — for a coding or REPL
        agent the files are the deliverable, and a grader that can verify a
        digest but never read a file can prove integrity while grading
        nothing. Two caps bound it: per-file (utf-8, ≤64KB) and a total per
        report, so a rollout that produced a 200MB core dump — or two hundred
        small ones — cannot bloat the report. Anything over a cap keeps the
        digest-string value: named, not inlined.

        The report is the ONLY artifact channel. This SDK once also POSTed
        each file to ``/rollouts/{id}/artifacts``, a route Mission Control
        never served (every call 404'd into an ``artifact_failed`` event), and
        the Node SDK never had it; Mission Control's report already carries
        each file's name, digest, size and small bytes.
        """
        # The LOCAL per-rollout directory, which is the one the child is told
        # about through PERCEPTEYE_TRAJECTORY_DIR and can therefore actually write
        # to. ``rollout.artifacts_dir`` is the SERVER's name for this rollout's
        # storage ("rollouts/rl_x") and is not a path on this machine.
        root = Path(self._trajectory_dir(rollout.rollout_id))
        if not root.is_dir():
            return
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
            if not path.is_file() or path.name == TOOL_CALLS_FILENAME:
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
                # collect is a fact about the rollout, and a scorer expecting
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
            if content is not None:
                inline_budget -= size
                out.artifacts[name] = {
                    "sha256": digest, "size_bytes": size, "content": content,
                }
            else:
                out.artifacts[name] = digest

    def _purge_trajectory(self, rollout_id: str) -> None:
        """Drop this rollout's trajectory directory once it is no longer needed.

        One directory per rollout accumulated for the life of the process, so a
        flywheel serving continuously grew its artifacts tree without bound.
        """
        try:
            shutil.rmtree(Path(self._trajectory_dir(rollout_id)), ignore_errors=True)
        except OSError:  # pragma: no cover - best effort, never fails a rollout
            pass

    def stop(self) -> None:
        self._stop = True

    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        """Hand one event to the customer's callback, and survive it.

        Guarded, because this is called from three different threads and one of
        them cannot afford to die. ``lease_lost`` and ``heartbeat_failed`` are
        emitted from inside the heartbeat thread's own exception handlers, so a
        callback that raised there exited the for, the while, and the thread --
        after which NO in-flight lease was renewed again, every subsequent
        rollout collected a 410 at report time, and the exception surfaced only
        through the thread excepthook. An observability hook that can silently
        stop lease renewal for the whole process is not one a docstring can
        make safe.

        The same rule as the plugins on the Node side: capturing evidence must
        never be able to change the behaviour being measured.
        """
        if not self._on_event:
            return
        try:
            self._on_event(kind, payload)
        except Exception:
            # Deliberately swallowed and not re-emitted. There is no logger in
            # this package, and routing a failed callback back through the
            # callback is a loop.
            pass


def _fs_slug(value: str) -> str:
    """A filename that cannot escape the outbox directory."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in value)
    return (safe.strip("._") or "unnamed")[:120]


def _wire_payload(out: RolloutOutput) -> dict[str, Any]:
    """The contract payload for this outcome, ALWAYS.

    ``to_wire`` raises ``ContractViolation`` for a value the wire cannot carry,
    and that exception is not a ``TransportError``: it used to escape
    ``_serve_one_inner`` entirely, stop the serve loop, and take the heartbeat
    thread — and therefore every concurrent lease — down with it. A payload we
    cannot send is a fact about the rollout, so it becomes one: the outcome is
    replaced by one that CAN be sent and says what happened to the original.

    AND SAYS WHAT WENT WITH IT. One unrepresentable field takes every other
    field down — tool calls, artifacts, reward, events, timing — because the
    salvage rebuilds from the two that are always sendable. Silently, a grader
    reading the row saw an empty rollout and an exception name; the fields that
    were populated and dropped are now named in the detail, so the difference
    between "the agent produced nothing" and "the SDK could not carry what it
    produced" is legible without this SDK's source.
    """
    try:
        return out.to_wire()
    except Exception as exc:
        # ``getattr``, because this function's whole contract is that it does
        # not raise: the object that just failed to serialise is the last one
        # to assume anything about.
        dropped = [
            name for name in (
                "tool_calls", "artifacts", "reward", "success",
                "llm_call_count", "events", "timing", "agent_fingerprint",
            ) if getattr(out, name, None)
        ]
        detail = "; ".join(filter(None, (
            out.exclusion_detail,
            f"the SDK could not represent this rollout's result on the "
            f"flywheel/1 wire: {type(exc).__name__}: {exc}",
            f"dropped by the salvage: {', '.join(dropped)}" if dropped else "",
        )))
        salvaged = RolloutOutput(
            final_text=out.final_text if isinstance(out.final_text, str) else "",
            exclusion_reason=out.exclusion_reason or "flywheel_crash",
            exclusion_detail=detail[:4000],
        )
        try:
            return salvaged.to_wire()
        except Exception:
            # Even the salvage was unrepresentable. Emit the minimum the
            # contract always accepts rather than nothing at all.
            return RolloutOutput(
                final_text="",
                exclusion_reason="flywheel_crash",
                exclusion_detail=f"unrepresentable result: {type(exc).__name__}",
            ).to_wire()


def _spooled_rollout(
    record: Mapping[str, Any], *, lease_token: str | None,
) -> RolloutRequest:
    """The minimum a report needs, rebuilt from a spooled outcome.

    ``report`` uses the rollout id and the lease token and nothing else. The
    token is PASSED IN rather than read from the record: it is a bearer
    credential and the record is a file the customer's agent can read, so the
    outbox stores it nowhere (see ``RolloutRunner._spool_write``). Every other
    field is blank because the outbox deliberately does not store it -- the
    per-rollout inference credential above all. Blanks, not placeholders that
    could be mistaken for real values.
    """
    return RolloutRequest(
        rollout_id=str(record.get("rollout_id") or ""),
        turn_input="",
        llm_url="",
        api_key="",
        artifacts_dir="",
        deadline_ts=float(record.get("deadline_ts") or 0.0),
        lease_token=lease_token,
    )


def _read_spool(path: Path) -> dict[str, Any] | None:
    """Read one spooled outcome, or ``None`` if it is not one we can send."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or record.get("schema") != _OUTBOX_SCHEMA:
        return None
    if not record.get("rollout_id") or not isinstance(record.get("result"), dict):
        return None
    # An outbox written by an SDK before the token came off disk still holds
    # one. Dropped on the way in, so it never re-enters memory and the next
    # write-back of this record leaves the credential nowhere.
    record.pop("lease_token", None)
    return record


def _move_aside(path: Path, why: str) -> Path:
    """Stop retrying a file, without deleting the work it holds.

    The name is RESERVED before the rename. ``os.replace`` onto a bare
    ``<id>.json.<why>`` would silently clobber it, so a rollout refused a
    second time -- a retried task, or a restart that re-spools the same id --
    would destroy the first retired record. That is precisely the silent loss
    this outbox exists to end, so retired names are uniquified rather than
    overwritten: the operator gets ``.refused-lease-lost``, then
    ``.refused-lease-lost.1``, and no copy is ever lost to a later one.
    """
    base = path.with_name(f"{path.name}.{why}")
    target, ordinal = base, 1
    while True:
        try:
            # O_EXCL is the reservation: it fails rather than truncating, so
            # two retirements can never collapse onto one file.
            os.close(os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600))
        except FileExistsError:
            target = base.with_name(f"{base.name}.{ordinal}")
            ordinal += 1
            continue
        except OSError:
            return path
        break
    try:
        os.replace(path, target)
    except OSError:
        return path
    return target


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
                # Carried across the process boundary. An agent that reports
                # its own tool calls resolved the identity in the child; the
                # parent must not be where that evidence is quietly lost.
                entity_ids=t.get("entity_ids") or {},
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


_ = TOOL_CALLS_FILENAME  # re-exported for adapters writing their own file
