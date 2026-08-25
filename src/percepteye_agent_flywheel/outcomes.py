"""Per-tool-call outcome capture.

``record_tool_call`` is the always-available ground truth. Framework adapters
are best-effort acceleration over this function; they are never a substitute
for it. If an adapter cannot classify a call, the customer can always say so
explicitly -- and saying "unknown" explicitly is a supported, first-class
answer, not a failure.

**Why a file and not a list.** The default isolation mode spawns one child
process per rollout, and the customer's agent runs in the child. In-memory
state and context variables do not cross that boundary, so a trajectory directory held in
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
from .errors import ContractViolation
from .evidence import extract_entity_ids

__all__ = ["TOOL_CALLS_FILENAME", "Trajectory", "current_rollout_id", "record_tool_call"]

TOOL_CALLS_FILENAME = "tool_calls.jsonl"

ROLLOUT_ID_ENV = "PERCEPTEYE_ROLLOUT_ID"
TRAJECTORY_DIR_ENV = "PERCEPTEYE_TRAJECTORY_DIR"

_current_rollout: ContextVar[str | None] = ContextVar(
    "percepteye_current_rollout", default=None
)
#: Where this rollout's trajectory lives. A context variable rather than an
#: environment variable because in-process rollouts must NOT mutate the host
#: application's environment -- that is the global side effect this SDK exists
#: not to repeat. The child process gets the same value through its own
#: environment at spawn, which is a fresh process and therefore safe.
_current_trajectory_dir: ContextVar[str | None] = ContextVar(
    "percepteye_current_trajectory_dir", default=None
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


def _trajectory_path(rollout_id: str) -> Path | None:
    """Where to append. Context variable first (in-process), env second (child).

    Returning None while a rollout IS active means capture would be silently
    discarded -- the failure mode this whole module is organised against -- so
    that case warns rather than passing quietly.
    """
    global _warned_no_destination
    base = _current_trajectory_dir.get() or os.environ.get(TRAJECTORY_DIR_ENV)
    if base:
        return Path(base) / TOOL_CALLS_FILENAME
    if not _warned_no_destination:
        _warned_no_destination = True
        warnings.warn(
            f"record_tool_call() was called for rollout {rollout_id!r} with no "
            f"trajectory directory configured, so the outcome was DISCARDED. "
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
    agent_name: str | None = None,
    tool_call_id: str | None = None,
    entity_ids: dict[str, str] | None = None,
) -> None:
    """Record one executed tool call against the current rollout.

    ``outcome`` and ``status_code`` are keyword-only and REQUIRED. You cannot
    record half an outcome.

    The case this exists for, stated plainly: an application-level permission
    denial that the API returns as HTTP 200 is
    ``outcome="failed", status_code=200``. Reading the status alone would score
    that refused write as a landed mutation.

    ``entity_ids`` names the entity a WRITE produced -- ``{"id": "10001"}``.
    Pass it when you know it; otherwise it is read out of ``output`` and the
    body itself is never required downstream. Only high-entropy fields are
    accepted, because a human ``name`` a different entity happens to share
    identifies the wrong resource.

    Outside a rollout this is a no-op, so instrumentation can stay in
    production code paths unconditionally.
    """
    rid = current_rollout_id()
    if not rid:
        return
    # Declared identity always wins. Extraction is best-effort acceleration
    # over an explicit answer, exactly as the adapters are over this function.
    ids = dict(entity_ids) if entity_ids else extract_entity_ids(output)
    rec = ToolCallOutcome(
        name=name,
        arguments=dict(arguments or {}),
        outcome=outcome,
        status_code=status_code,
        output=output,
        error=error,
        error_class=error_class,
        latency_ms=latency_ms,
        agent_name=agent_name,
        tool_call_id=tool_call_id,
        entity_ids=ids,
    )
    # Project into the customer's own traces BEFORE the write, so a rollout
    # with no trajectory directory still shows up in their telemetry.
    # Egress only: nothing reads these back, and a telemetry failure inside
    # cannot reach this function's caller.
    _emit_span(rec, rid)

    path = _trajectory_path(rid)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec.to_wire(), default=str, ensure_ascii=False)
    # Append-only, one line per call, flushed. A crash mid-rollout leaves every
    # call recorded up to that point rather than nothing.
    with _write_lock, path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()



def _emit_span(rec: ToolCallOutcome, rollout_id: str) -> None:
    """Mirror one outcome into OpenTelemetry, if the API is installed.

    Lazily imported so the package keeps its single runtime dependency: with
    the ``otel`` extra absent this costs one failed import, cached.
    """
    try:
        from .tracing import record_tool_call_span

        record_tool_call_span(
            rec.name, rec.arguments,
            outcome=rec.outcome,
            status_code=rec.status_code,
            error=rec.error,
            latency_ms=rec.latency_ms,
            agent_name=rec.agent_name,
            rollout_id=rollout_id,
            tool_call_id=rec.tool_call_id,
        )
    except Exception:
        return


class Trajectory:
    """Parent-side reader for one rollout's trajectory directory."""

    def __init__(self, trajectory_dir: str | os.PathLike[str]) -> None:
        self.path = Path(trajectory_dir) / TOOL_CALLS_FILENAME
        #: Records the last :meth:`read` could not parse. A truncated final
        #: line is the EXPECTED case, not an exotic one: the deadline kill
        #: lands mid-write, and once a plugin in another language appends here
        #: it stops being rare. Counted rather than raised, and counted rather
        #: than silently dropped -- a call we know executed and cannot read is
        #: not a call that did not happen.
        self.malformed = 0

    def read(self) -> list[ToolCallOutcome] | None:
        """Every call recorded for this rollout, or None if nothing was recorded.

        ``None`` and ``[]`` mean different things and this preserves the
        difference: no tool-call file at all means the agent never reported, and
        must not be rendered as an assertion that zero tool calls occurred.

        **One unreadable line costs that line and nothing else.** Before this
        guard a single truncated write -- which is what a SIGKILL on the
        deadline produces -- raised out of here, out of ``run_one``, and turned
        a COMPLETED rollout into ``flywheel_crash`` with every tool call lost.
        The count survives in :attr:`malformed` so the loss is reported rather
        than hidden; see ``_attach_trajectory``.
        """
        if not self.path.exists():
            return None
        self.malformed = 0
        out: list[ToolCallOutcome] = []
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    rec = ToolCallOutcome(
                        name=d["name"],
                        arguments=d.get("arguments") or {},
                        outcome=d["outcome"],
                        status_code=d.get("status_code"),
                        output=d.get("output"),
                        error=d.get("error"),
                        error_class=d.get("error_class"),
                        latency_ms=d.get("latency_ms"),
                        agent_name=d.get("agent_name"),
                        tool_call_id=d.get("tool_call_id"),
                        entity_ids=d.get("entity_ids") or {},
                    )
                except (ValueError, TypeError, KeyError, AttributeError,
                        ContractViolation):
                    # Truncated JSON, a non-object line, a missing required
                    # key, or a value the contract refuses. All four mean the
                    # same thing here -- this record is unreadable -- and none
                    # of them is a reason to discard the readable ones.
                    self.malformed += 1
                    continue
                out.append(rec)
        return out

    def unknown_count(self) -> int:
        calls = self.read() or []
        return sum(1 for c in calls if c.outcome == "unknown")


class _RolloutScope:
    """Binds a rollout and its trajectory directory to the context for one block.

    Both are set together. An earlier iteration bound only the id and left the
    destination to the environment, so in-process rollouts resolved a rollout
    but no destination and dropped every recorded call while appearing to work.
    """

    def __init__(self, rollout_id: str, trajectory_dir: str) -> None:
        self._rid = rollout_id
        self._trajectory_dir = str(trajectory_dir)
        self._tokens: list[Any] = []

    def __enter__(self) -> str:
        self._tokens = [
            _current_rollout.set(self._rid),
            _current_trajectory_dir.set(self._trajectory_dir),
        ]
        return self._rid

    def __exit__(self, *exc: Any) -> None:
        if self._tokens:
            _current_rollout.reset(self._tokens[0])
            _current_trajectory_dir.reset(self._tokens[1])
            self._tokens = []
