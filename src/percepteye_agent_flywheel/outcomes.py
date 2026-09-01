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
from typing import Any, Literal

from .contract import Outcome, ToolCallOutcome
from .errors import ContractViolation
from .evidence import extract_entity_ids

__all__ = [
    "TOOL_CALLS_FILENAME",
    "CaptureScope",
    "Trajectory",
    "current_capture",
    "current_rollout_id",
    "record_tool_call",
]

#: What a bound capture scope MEANS. Two values, and the distinction is the
#: point: both record through the same path into the same trajectory format,
#: so nothing downstream could tell them apart before this existed.
CaptureScope = Literal["rollout", "turn"]

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
#: Which KIND of scope the two variables above are bound to. `serve()` binds a
#: rollout; `attach().turn()` binds a production turn. They deliberately share
#: one trajectory format and one recording path -- the only thing that differs
#: is what the identity MEANS, and that is exactly the bit that was missing.
_current_scope_kind: ContextVar[str | None] = ContextVar(
    "percepteye_current_scope_kind", default=None
)
_warned_no_destination = False


def current_capture() -> tuple[CaptureScope, str] | None:
    """What this thread/task is capturing into, and under which identity.

    Three states, three answers, and none of them collapsed into another::

        ("rollout", rollout_id)  # inside a serve() rollout, either isolation
        ("turn",    turn_id)     # inside attach().turn(), in production
        None                     # neither

    **This is the function to branch on.** :func:`current_rollout_id` answers
    the narrower question "which trajectory am I appending to", and inside
    ``attach().turn()`` that is a TURN id -- so ``if current_rollout_id():``
    is true in production and takes the rollout branch. That is the whole
    reason this exists. There is no prefix convention to lean on instead: a
    rollout id is opaque by contract, and a caller may pass their own
    ``turn_id`` to :meth:`Attachment.turn`.

    A ``bool`` would have been the smaller API and the wrong one, because it
    answers "rollout?" with ``False`` for both production and no-capture-at-all
    -- the same absence-as-zero collapse this package refuses for
    ``outcome="unknown"`` and for ``tool_calls=None`` versus ``[]``.
    """
    rid = _current_rollout.get()
    if rid:
        kind = _current_scope_kind.get() or "rollout"
        return ("turn" if kind == "turn" else "rollout", rid)
    env = os.environ.get(ROLLOUT_ID_ENV) or None
    if env:
        # Only `build_child_env` writes this variable, and only for a rollout
        # child. `attach()` never writes the environment at all, so reaching it
        # here means a real rollout -- no new variable is needed to tell them
        # apart, and the "no feature flags" posture survives.
        return ("rollout", env)
    return None


def current_rollout_id() -> str | None:
    """The identity this thread/task is recording under, or None.

    Resolution order is context variable, then environment. The env var is what
    a spawned child inherits; the context variable is what an in-process
    concurrent flywheel sets. Neither is per-process -- a per-process identity
    collides the moment two rollouts run at once, which is precisely how an
    earlier SDK lost attribution.

    .. warning::
       Despite the name this is NOT a rollout-versus-production test. Inside
       ``attach().turn()`` it returns that TURN's id, because a production turn
       records through the same path. Branch on :func:`current_capture`
       instead, which distinguishes the two. The name is kept, and the
       behaviour with it, because :func:`record_tool_call` reads this to decide
       where to append -- narrowing it would silently disable production
       capture, trading a naming bug for a data-loss one.
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
    _emit_span(rec, rid, _scope_kind())

    path = _trajectory_path(rid)
    if path is None:
        return
    line = json.dumps(rec.to_wire(), default=str, ensure_ascii=False)
    _append(path, line)



class _Sink:
    """One open append handle per trajectory file, with its own lock.

    Two things were wrong with opening the file per call.

    It was SLOW: open + write + flush + close cost ~18us of the ~38us it took
    to record one outcome, and holding the handle instead costs ~1.2us -- the
    same work, 15x cheaper, for a file that is appended to hundreds of times
    and read once.

    And it did not scale. The lock was process-global and was held across the
    whole open/write/close, so every recording thread in the process queued
    behind every other one even when they were writing to DIFFERENT rollouts.
    Measured: ~20,000 records/second flat from 1 to 8 threads, with per-call
    latency 8.7x worse at 8. The contention was on the lock, not the GIL --
    with the write ablated the same code showed no inflation at any thread
    count. A lock per file makes concurrent turns independent, which is what
    they always were.

    The handle is opened LAZILY, on the first record. A turn that records
    nothing now costs no file and no directory -- see `Attachment.turn`.
    """

    __slots__ = ("_broken", "_fh", "_lock", "_path")

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._fh: Any = None
        self._broken = False

    def write(self, line: str) -> None:
        """Append one record. NEVER raises into the caller.

        Creating the directory moved from turn-entry to here, which moved the
        failure with it: an unwritable trajectory root used to be caught by
        `Attachment.turn` and skip capture for that turn, and would otherwise
        now surface inside `record_tool_call` in the middle of a customer's
        tool. A capture bug costs the observation and nothing else -- so the
        sink disables itself, says so once, and the turn carries on.
        """
        if self._broken:
            return
        try:
            self._write(line)
        except OSError as exc:
            self._broken = True
            warnings.warn(
                f"[percepteye] could not write the trajectory at "
                f"{self._path.parent}: {type(exc).__name__}: {exc}. Outcomes "
                f"for this trajectory are NOT being recorded. Capture is "
                f"disabled for it; nothing else is affected.",
                RuntimeWarning,
                stacklevel=4,
            )

    def _write(self, line: str) -> None:
        with self._lock:
            if self._fh is None:
                # Created here rather than at scope entry, so the directory is
                # a consequence of having something to put in it.
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._fh = self._path.open("a", encoding="utf-8")
            self._fh.write(line + "\n")
            # Flushed, never fsynced. Flushing puts the record in the OS page
            # cache, which survives this process dying -- a crashed agent, a
            # SIGKILL on a blown deadline -- and that is the failure this
            # capture path actually has to survive. fsync would additionally
            # survive the machine losing power, at roughly 1-10ms per record
            # against the ~20us the whole write costs now, which would make
            # recording slower than most of the tool calls being recorded.
            # The trade is deliberate: a machine crash loses recent outcomes.
            self._fh.flush()

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                try:
                    self._fh.close()
                finally:
                    self._fh = None


#: Open sinks, keyed by absolute path. Bounded by the number of trajectories
#: this process has written to and released as each scope exits, so it does not
#: grow with the number of records.
_sinks: dict[str, _Sink] = {}
_sinks_lock = threading.Lock()


#: Set once, the first time a sink is actually opened.
_atexit_registered = False


def _close_all_sinks() -> None:
    with _sinks_lock:
        sinks = list(_sinks.values())
        _sinks.clear()
    for sink in sinks:
        sink.close()


def _sink_for(path: Path) -> _Sink:
    global _atexit_registered
    key = str(path)
    with _sinks_lock:
        sink = _sinks.get(key)
        if sink is None:
            sink = _sinks[key] = _Sink(path)
            if not _atexit_registered:
                # Registered HERE rather than at module scope, so importing
                # this package remains completely inert -- no atexit hook, no
                # thread, no file, no socket. It costs one hook only in a
                # process that has actually recorded something.
                #
                # It exists for the environment-variable path: a child process
                # given PERCEPTEYE_ROLLOUT_ID has no scope to exit, so nothing
                # else would close its handle. Records are flushed on every
                # write, so this closes a handle rather than saving data.
                import atexit

                atexit.register(_close_all_sinks)
                _atexit_registered = True
        return sink


def _append(path: Path, line: str) -> None:
    _sink_for(path).write(line)


def _release(path_str: str | None) -> None:
    """Close and forget the sink for one trajectory, if it was ever opened."""
    if not path_str:
        return
    key = str(Path(path_str) / TOOL_CALLS_FILENAME)
    with _sinks_lock:
        sink = _sinks.pop(key, None)
    if sink is not None:
        sink.close()


def _scope_kind() -> CaptureScope:
    """Whether the bound identity is a rollout or a production turn.

    Defaults to "rollout" when nothing bound a kind, which is the pre-existing
    reading of a bare ``PERCEPTEYE_ROLLOUT_ID`` and keeps every existing caller
    of :class:`_RolloutScope` behaving exactly as it did.
    """
    return "turn" if _current_scope_kind.get() == "turn" else "rollout"


def _emit_span(rec: ToolCallOutcome, scope_id: str, kind: CaptureScope) -> None:
    """Mirror one outcome into OpenTelemetry, if the API is installed.

    ``kind`` is REQUIRED rather than defaulted. Defaulting it to "rollout"
    meant a future caller who forgot the argument would silently restore the
    bug this parameter exists to fix -- production turns stamped as rollouts in
    the customer's own traces -- and would do it with a green suite.

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
            rollout_id=scope_id if kind == "rollout" else None,
            turn_id=scope_id if kind == "turn" else None,
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

    def __init__(self, rollout_id: str, trajectory_dir: str, *,
                 kind: CaptureScope = "rollout") -> None:
        self._rid = rollout_id
        self._trajectory_dir = str(trajectory_dir)
        self._kind = kind
        self._tokens: list[Any] = []

    def __enter__(self) -> str:
        self._tokens = [
            _current_rollout.set(self._rid),
            _current_trajectory_dir.set(self._trajectory_dir),
            _current_scope_kind.set(self._kind),
        ]
        return self._rid

    def __exit__(self, *exc: Any) -> None:
        if self._tokens:
            _current_rollout.reset(self._tokens[0])
            _current_trajectory_dir.reset(self._tokens[1])
            _current_scope_kind.reset(self._tokens[2])
            self._tokens = []
        # The handle lives exactly as long as the scope that writes through it.
        # Without this the sink registry would grow one entry per turn for the
        # life of the process, which is the leak that holding a handle open
        # would otherwise trade for the speed.
        _release(self._trajectory_dir)
