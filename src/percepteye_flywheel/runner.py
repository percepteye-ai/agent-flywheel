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
) -> RolloutOutput:
    """Drive the agent once, in this process, under a rollout-scoped context."""
    started = time.time()
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
                with _cf.ThreadPoolExecutor(max_workers=1) as pool:
                    raw = pool.submit(ctx.run, resolved.fn, arg).result(timeout=timeout_s)
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


#: How often the heartbeat thread wakes to see what is due. Deliberately small
#: and fixed: the cadence that matters is PER ROLLOUT, because concurrent
#: rollouts can hold leases of different lengths, and a single global interval
#: computed once is wrong for all but one of them -- computed BEFORE anything
#: is in flight, it is wrong for all of them.
_BEAT_TICK_S = 1.0


class _InFlight:
    """One rollout being served right now, as the heartbeat thread sees it."""

    __slots__ = ("lease_lost", "next_beat_at", "proc", "rollout")

    def __init__(self, rollout: RolloutRequest, next_beat_at: float) -> None:
        self.rollout = rollout
        self.proc: subprocess.Popen[str] | None = None
        self.lease_lost = False
        self.next_beat_at = next_beat_at


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
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
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
        # Carried so the CHILD resolves the same callable the parent validated.
        # This was hardcoded to None on the subprocess payload, so a customer
        # who named an entrypoint with method= had it honoured at startup and
        # silently ignored on every rollout: the child re-probed and could run
        # a different method entirely.
        self._method = method
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
        timeout_s = min(self.rollout_timeout_s, budget)
        ledger_dir = self._ledger_dir()
        Path(ledger_dir, rollout.rollout_id).mkdir(parents=True, exist_ok=True)

        if self.isolation == "inprocess":
            return run_rollout_in_process(
                self.resolved, rollout, ledger_dir=ledger_dir, timeout_s=timeout_s
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
            [sys.executable, "-m", "percepteye_flywheel.child"],
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
            if out.exclusion_reason is not None:
                self.transport.abandon(
                    rollout,
                    reason=out.exclusion_reason,
                    detail=out.exclusion_detail or "",
                )
                self._emit("rollout_abandoned",
                           {"rollout_id": rollout.rollout_id, "reason": out.exclusion_reason})
                return
            ack = self.transport.report(rollout, out)
            self._emit("rollout_reported", {
                "rollout_id": rollout.rollout_id,
                "grade": ack.get("grade"),
                "duplicate": ack.get("duplicate", False),
            })
        except LeaseLost:
            self._emit("lease_lost", {"rollout_id": rollout.rollout_id})
        except TransportError as exc:
            # The rollout ran; only the report failed. The lease will lapse and
            # the server will hand the work to someone else, so this costs a
            # repeat rather than a hole -- but it is the expensive half we are
            # throwing away, so it is emitted loudly rather than swallowed.
            self._emit("report_failed", {"rollout_id": rollout.rollout_id, "error": str(exc)})

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
