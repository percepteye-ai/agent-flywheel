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
import subprocess
import sys
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
    ``RolloutOutput`` the harness returns, or in ``record_tool_call``.
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
            target = _load_spec(entrypoint)
            obj = target() if callable(target) and not _looks_like_agent(target) else target
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
        self._on_event = on_event
        self._stop = False

    # ── one rollout ────────────────────────────────────────────────────────
    def run_one(self, rollout: RolloutRequest) -> RolloutOutput:
        budget = rollout.seconds_remaining(time.time())
        timeout_s = min(self.rollout_timeout_s, budget) if budget > 0 else self.rollout_timeout_s
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
            "method": None,
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
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "percepteye_harness.child"],
                input=json.dumps(payload),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s + 15.0,
            )
        except subprocess.TimeoutExpired:
            return RolloutOutput(
                final_text="",
                exclusion_reason="deadline_exceeded",
                exclusion_detail=f"child exceeded {timeout_s:.0f}s and was killed",
                timing={"elapsed_s": time.time() - started},
            )
        if proc.returncode != 0:
            return RolloutOutput(
                final_text="",
                exclusion_reason="harness_crash",
                exclusion_detail=(proc.stderr or "")[-2000:],
                timing={"elapsed_s": time.time() - started},
            )
        try:
            out = _output_from_wire(json.loads(proc.stdout.strip().splitlines()[-1]))
        except Exception as exc:
            return RolloutOutput(
                final_text="",
                exclusion_reason="harness_crash",
                exclusion_detail=f"unparseable child result: {exc}",
                timing={"elapsed_s": time.time() - started},
            )
        out.timing.setdefault("elapsed_s", time.time() - started)
        _attach_ledger(out, ledger_dir, rollout.rollout_id)
        return out

    # ── the loop ───────────────────────────────────────────────────────────
    def serve_forever(self, *, max_rollouts: int | None = None) -> int:
        """Claim → run → report, until stopped. Returns rollouts completed."""
        done = 0
        while not self._stop:
            if max_rollouts is not None and done >= max_rollouts:
                break
            try:
                rollouts, poll_after_ms = self.transport.claim(self.concurrency)
            except TransportError as exc:
                self._emit("claim_failed", {"error": str(exc)})
                sleep_ms(5_000)
                continue
            if not rollouts:
                sleep_ms(poll_after_ms)
                continue
            for rollout in rollouts:
                self._serve_one(rollout)
                done += 1
        return done

    def _serve_one(self, rollout: RolloutRequest) -> None:
        self._emit("rollout_start", {"rollout_id": rollout.rollout_id})
        try:
            out = self.run_one(rollout)
        except Exception as exc:
            try:
                self.transport.abandon(
                    rollout, reason="harness_crash", detail=f"{type(exc).__name__}: {exc}"
                )
            except TransportError:
                pass
            self._emit("rollout_crash", {"rollout_id": rollout.rollout_id, "error": str(exc)})
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
            self._emit("report_failed", {"rollout_id": rollout.rollout_id, "error": str(exc)})

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
        harness_fingerprint=d.get("harness_fingerprint"),
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
