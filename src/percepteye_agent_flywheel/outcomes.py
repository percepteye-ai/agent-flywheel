"""Per-tool-call outcome capture.

``record_tool_call`` is the always-available ground truth. Framework adapters
are best-effort acceleration over this function; they are never a substitute
for it. If an adapter cannot classify a call, the customer can always say so
explicitly -- and saying "unknown" explicitly is a supported, first-class
answer, not a failure.

**Why a file and not a list.** The default isolation mode spawns one child
process per rollout, and the customer's agent runs in the child. In-memory
state and context variables do not cross that boundary, so a ledger held in
the parent would silently record nothing -- capture would appear to work while
producing an empty tool-call list, which reads downstream as "this agent made
no tool calls" rather than "we failed to look". Append-only JSONL under the
rollout's own directory is the one representation that works identically in
the parent, in a spawned child, and after a crash.
"""
from __future__ import annotations

import json
import os
import threading
import warnings
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from .contract import Outcome, ToolCallOutcome

__all__ = ["LEDGER_FILENAME", "OutcomeLedger", "current_rollout_id", "record_tool_call"]

LEDGER_FILENAME = "tool_calls.jsonl"

ROLLOUT_ID_ENV = "PERCEPTEYE_ROLLOUT_ID"
LEDGER_DIR_ENV = "PERCEPTEYE_LEDGER_DIR"

_current_rollout: ContextVar[str | None] = ContextVar(
    "percepteye_current_rollout", default=None
)
#: Where this rollout's ledger lives. A context variable rather than an
#: environment variable because in-process rollouts must NOT mutate the host
#: application's environment -- that is the global side effect this SDK exists
#: not to repeat. The child process gets the same value through its own
#: environment at spawn, which is a fresh process and therefore safe.
_current_ledger_dir: ContextVar[str | None] = ContextVar(
    "percepteye_current_ledger_dir", default=None
)
_write_lock = threading.Lock()
_warned_no_destination = False


def current_rollout_id() -> str | None:
    """The rollout this thread/task is serving, or None.

    Resolution order is context variable, then environment. The env var is what
    a spawned child inherits; the context variable is what an in-process
    concurrent flywheel sets. Both are per-rollout. Neither is per-process --
    a per-process identity collides the moment two rollouts run at once, which
    is precisely how an earlier SDK lost attribution.
    """
    rid = _current_rollout.get()
    if rid:
        return rid
    return os.environ.get(ROLLOUT_ID_ENV) or None


def _ledger_path(rollout_id: str) -> Path | None:
    """Where to append. Context variable first (in-process), env second (child).

    Returning None while a rollout IS active means capture would be silently
    discarded -- the failure mode this whole module is organised against -- so
    that case warns rather than passing quietly.
    """
    global _warned_no_destination
    base = _current_ledger_dir.get() or os.environ.get(LEDGER_DIR_ENV)
    if base:
        return Path(base) / rollout_id / LEDGER_FILENAME
    if not _warned_no_destination:
        _warned_no_destination = True
        warnings.warn(
            f"record_tool_call() was called for rollout {rollout_id!r} with no "
            f"ledger destination configured, so the outcome was DISCARDED. "
            f"Tool outcomes recorded outside a percepteye-agent-flywheel runner are "
            f"not captured. This warning is emitted once.",
            RuntimeWarning,
            stacklevel=3,
        )
    return None


def record_tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    outcome: Outcome,
    status_code: int | None,
    output: Any = None,
    error: str | None = None,
    error_class: str | None = None,
    latency_ms: float | None = None,
) -> None:
    """Record one executed tool call against the current rollout.

    ``outcome`` and ``status_code`` are keyword-only and REQUIRED. You cannot
    record half an outcome.

    The case this exists for, stated plainly: an application-level permission
    denial that the API returns as HTTP 200 is
    ``outcome="failed", status_code=200``. Reading the status alone would score
    that refused write as a landed mutation.

    Outside a rollout this is a no-op, so instrumentation can stay in
    production code paths unconditionally.
    """
    rid = current_rollout_id()
    if not rid:
        return
    rec = ToolCallOutcome(
        name=name,
        arguments=dict(arguments or {}),
        outcome=outcome,
        status_code=status_code,
        output=output,
        error=error,
        error_class=error_class,
        latency_ms=latency_ms,
    )
    path = _ledger_path(rid)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec.to_wire(), default=str, ensure_ascii=False)
    # Append-only, one line per call, flushed. A crash mid-rollout leaves every
    # call recorded up to that point rather than nothing.
    with _write_lock, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()


class OutcomeLedger:
    """Parent-side reader for one rollout's append-only ledger."""

    def __init__(self, ledger_dir: str | os.PathLike[str], rollout_id: str) -> None:
        self.path = Path(ledger_dir) / rollout_id / LEDGER_FILENAME

    def read(self) -> list[ToolCallOutcome] | None:
        """Every call recorded for this rollout, or None if nothing was recorded.

        ``None`` and ``[]`` mean different things and this preserves the
        difference: no ledger file at all means the agent never reported, and
        must not be rendered as an assertion that zero tool calls occurred.
        """
        if not self.path.exists():
            return None
        out: list[ToolCallOutcome] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(
                    ToolCallOutcome(
                        name=d["name"],
                        arguments=d.get("arguments") or {},
                        outcome=d["outcome"],
                        status_code=d.get("status_code"),
                        output=d.get("output"),
                        error=d.get("error"),
                        error_class=d.get("error_class"),
                        latency_ms=d.get("latency_ms"),
                    )
                )
        return out

    def unknown_count(self) -> int:
        calls = self.read() or []
        return sum(1 for c in calls if c.outcome == "unknown")


class _RolloutScope:
    """Binds a rollout and its ledger to the current context for one block.

    Both are set together. An earlier iteration bound only the id and left the
    destination to the environment, so in-process rollouts resolved a rollout
    but no ledger and dropped every recorded call while appearing to work.
    """

    def __init__(self, rollout_id: str, ledger_dir: str) -> None:
        self._rid = rollout_id
        self._ledger_dir = str(ledger_dir)
        self._tokens: list[Any] = []

    def __enter__(self) -> str:
        self._tokens = [
            _current_rollout.set(self._rid),
            _current_ledger_dir.set(self._ledger_dir),
        ]
        return self._rid

    def __exit__(self, *exc: Any) -> None:
        if self._tokens:
            _current_rollout.reset(self._tokens[0])
            _current_ledger_dir.reset(self._tokens[1])
            self._tokens = []
