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

from .contract import (
    Outcome,
    ToolCallOutcome,
    _production_identifier,
)
from .errors import ContractViolation
from .evidence import extract_entity_ids
from .execution_identity import is_lower_sha256

__all__ = [
    "AGENT_FINGERPRINT_FILENAME",
    "CONVERSATION_ID_FILENAME",
    "TASK_ID_FILENAME",
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

#: The turn record: what the agent was ASKED and what it ANSWERED.
#:
#: A SEPARATE FILE from the tool calls, and that separation is load-bearing.
#: :meth:`Trajectory.read` builds each row with required subscripts and counts
#: anything else as ``malformed``; that count becomes
#: ``tool_calls_omitted_count``, whose non-zero value forbids the training
#: grade. So a turn record written into ``tool_calls.jsonl`` would not be
#: politely ignored -- it would read as CORRUPTION and demote the rollout.
#:
#: TWO PRODUCERS, ONE FORMAT. The CLI recorder
#: (``hosts/scripts/percepteye_record.py``) writes these from host hook events
#: and is stdlib-only by contract, so it cannot import this module; ``attach()``
#: writes them through :func:`record_turn_text`. They must stay byte-compatible,
#: which is asserted by a test that reads BOTH producers' output back through
#: the one reader below rather than by a comment asking someone to remember.
TURNS_FILENAME = "turns.jsonl"

#: Where a turn directory records which conversation it belongs to.
CONVERSATION_ID_FILENAME = "conversation_id"

#: The caller's opaque task/case/work-item correlation for one production
#: turn.  This is evidence only: writing it grants no training eligibility and
#: the SDK never interprets it.  A sidecar survives delayed answers and process
#: restarts without coupling task identity to the agent's own turn id.
TASK_ID_FILENAME = "task_id"

#: A conflicting task binding is worse than no binding: it could attach advice
#: for one customer task to another.  The marker is monotonic so a later writer
#: cannot make a mixed turn look exact merely by winning the last write.
_TASK_ID_CONFLICT_FILENAME = "task_id.conflict"
#: SDK-authored execution identity captured beside one production turn. It is
#: a sidecar rather than Attachment state so an old, pending turn flushed by a
#: newer process can never be stamped with that process's execution identity.
AGENT_FINGERPRINT_FILENAME = "agent_fingerprint.json"

#: Monotonic fail-closed marker.  It is separate from the digest sidecar so a
#: concurrent first writer cannot overwrite a conflict after another process
#: observed its just-created, not-yet-written file.
_AGENT_FINGERPRINT_CONFLICT_FILENAME = "agent_fingerprint.conflict"

#: The two halves of a turn. ``task`` is what was asked, ``answer`` is what the
#: agent finally said -- the same two kinds the CLI recorder emits from
#: ``UserPromptSubmit`` and ``Stop``.
TURN_KINDS = ("task", "answer")

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
#: The conversation the OPEN scope belongs to, or ``None``.
#:
#: A context var rather than a read of the turn directory's
#: ``conversation_id`` file, because the one consumer is the tool-call span and
#: that runs on every tool call a customer's agent makes. A stat-and-read per
#: call is a cost this package has no business adding to somebody's hot path
#: for a value the scope already knew when it opened.
_current_conversation: ContextVar[str | None] = ContextVar(
    "percepteye_current_conversation", default=None
)
_current_task: ContextVar[str | None] = ContextVar(
    "percepteye_current_task", default=None
)
_current_execution_sha256: ContextVar[str | None] = ContextVar(
    "percepteye_current_execution_sha256", default=None
)
#: When the open scope is a production TURN whose agent the control plane
#: said may upload MODEL CONTENT (the second consent tier, see
#: :func:`~agent_flywheel.production.capture_verdict`): the turn's start, as
#: OTLP decimal nanoseconds. ``None`` everywhere else -- every rollout, every
#: turn without that consent, and outside any scope -- which is what makes the
#: adapters' model-boundary callbacks record NOTHING there.
_current_model_capture: ContextVar[str | None] = ContextVar(
    "percepteye_current_model_capture", default=None
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
            f"Tool outcomes recorded outside an agent-flywheel runner are "
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


def model_capture_active() -> bool:
    """Whether a model call observed NOW belongs in a turn's model boundary.

    True only inside ``attach().turn()`` for an agent the control plane said
    may upload model content. The adapters ask this FIRST, on every model
    callback, and return at once when it is false -- so a rollout, and a turn
    without that consent, pay one context-variable read and record nothing.
    """
    return _current_model_capture.get() is not None


#: Serializes appends to model-boundary files. ONE lock, not one per file like
#: the tool-call sinks: a model call takes seconds, so these writes are
#: nowhere near a hot path, and a record here can be megabytes -- which a
#: buffered write may split into several ``write(2)`` calls that a concurrent
#: writer (parallel tool calls finishing together) could interleave with.
_model_boundary_lock = threading.Lock()


def record_model_boundary(operation: dict[str, Any]) -> bool:
    """Append one observed model call or tool execution to the open turn.

    ``operation`` is one record of the model-boundary observation
    (:mod:`agent_flywheel.genai_otlp`): a ``chat`` with the messages the call
    was sent and what it answered, or an ``execute_tool`` with the arguments
    and the result, each with its own span id and clocks. The first record of
    a turn also writes the turn's trace id and its start.

    A NO-OP unless :func:`model_capture_active`. Returns whether anything was
    written, and NEVER raises: it runs inside a customer's model callback, and
    a capture bug costs the observation, never the call.
    """
    start = _current_model_capture.get()
    base = _current_trajectory_dir.get()
    if start is None or not base:
        return False
    from .genai_otlp import (
        OBSERVATION_FILENAME,
        new_span_id,
        new_trace_id,
        operation_record,
        trace_record,
    )

    try:
        line = operation_record(operation)
        path = Path(base) / OBSERVATION_FILENAME
        with _model_boundary_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            head = "" if path.exists() else trace_record(
                trace_id=new_trace_id(), span_id=new_span_id(), start=start,
            ) + "\n"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(head + line + "\n")
    except (OSError, TypeError, ValueError):
        return False
    _write_execution_fingerprint(Path(base), _current_execution_sha256.get())
    return True


def _record_model_boundary_end(trajectory_dir: str) -> None:
    """Close the turn's model boundary: when its scope ended. Never raises.

    Only a turn that already recorded something gets the line, so a turn whose
    agent called no model still leaves no file behind.
    """
    from .genai_otlp import OBSERVATION_FILENAME, end_record, now_unix_nano

    path = Path(trajectory_dir) / OBSERVATION_FILENAME
    try:
        with _model_boundary_lock:
            if path.exists():
                with path.open("a", encoding="utf-8") as fh:
                    fh.write(end_record(now_unix_nano()) + "\n")
    except OSError:
        return


def read_model_boundary(trajectory_dir: Any) -> dict[str, Any] | None:
    """The turn's observed model boundary, or ``None`` when none was recorded.

    Read back through the ONE reader both SDKs share
    (:func:`agent_flywheel.genai_otlp.observation_from_lines`).
    """
    from .genai_otlp import OBSERVATION_FILENAME, observation_from_lines

    try:
        raw = (Path(trajectory_dir) / OBSERVATION_FILENAME).read_text(
            encoding="utf-8")
    except OSError:
        return None
    return observation_from_lines(raw.splitlines())



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

    def write(self, line: str) -> bool:
        """Append one record. NEVER raises into the caller.

        Creating the directory moved from turn-entry to here, which moved the
        failure with it: an unwritable trajectory root used to be caught by
        `Attachment.turn` and skip capture for that turn, and would otherwise
        now surface inside `record_tool_call` in the middle of a customer's
        tool. A capture bug costs the observation and nothing else -- so the
        sink disables itself, says so once, and the turn carries on.
        """
        if self._broken:
            return False
        try:
            self._write(line)
            return True
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
            return False

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
    if _sink_for(path).write(line):
        _write_execution_fingerprint(path.parent, _current_execution_sha256.get())


def _write_execution_fingerprint(path: Path, execution_sha256: str | None) -> None:
    """Persist one exact SDK identity, invalidating a conflicting claim.

    Never raises. Capture cannot fail a customer turn. A second process may
    reuse a caller-chosen turn id; if it carries a different identity, leaving
    either digest in place would falsely bind half the turn, so a monotonic
    conflict marker makes readers refuse the identity.  Keeping that marker in
    a separate file also closes the create/write race between two processes.
    """
    if not is_lower_sha256(execution_sha256):
        return
    sidecar = path / AGENT_FINGERPRINT_FILENAME
    conflict = path / _AGENT_FINGERPRINT_CONFLICT_FILENAME
    expected = {"execution_sha256": execution_sha256}
    try:
        path.mkdir(parents=True, exist_ok=True)
        if conflict.exists():
            return
        try:
            with sidecar.open("x", encoding="utf-8") as output:
                output.write(json.dumps(expected, sort_keys=True))
            return
        except FileExistsError:
            pass
        try:
            current = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, TypeError, ValueError):
            current = None
        if current == expected:
            return
        # Never remove this marker.  A later process cannot turn a mixed turn
        # back into a single-identity turn merely by writing last.
        try:
            conflict.open("x", encoding="utf-8").close()
        except FileExistsError:
            pass
    except OSError:
        return


def _read_execution_fingerprint(path: Path) -> dict[str, str] | None:
    """Read only the exact sidecar shape authored by this SDK."""
    if (path / _AGENT_FINGERPRINT_CONFLICT_FILENAME).exists():
        return None
    try:
        value = json.loads(
            (path / AGENT_FINGERPRINT_FILENAME).read_text(encoding="utf-8")
        )
    except (OSError, TypeError, ValueError):
        return None
    if not isinstance(value, dict) or set(value) != {"execution_sha256"}:
        return None
    digest = value.get("execution_sha256")
    if (path / _AGENT_FINGERPRINT_CONFLICT_FILENAME).exists():
        return None
    return {"execution_sha256": digest} if is_lower_sha256(digest) else None


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
            conversation_id=_current_conversation.get(),
            task_id=_current_task.get(),
            tool_call_id=rec.tool_call_id,
            execution_sha256=_current_execution_sha256.get(),
        )
    except Exception:
        return


def record_turn_text(
    trajectory_dir: str | os.PathLike[str],
    kind: str,
    text: str,
    *,
    max_chars: int = 20_000,
    **ids: str,
) -> bool:
    """Append one turn record. Returns whether anything was written.

    Never raises. This runs inside a customer's production request, and the
    module's standing rule is that a capture bug costs the observation and
    never the turn.

    Empty text writes NOTHING rather than an empty record: "the agent answered
    with silence" and "we were not given the answer" are different claims, and
    only the second is true here.
    """
    if kind not in TURN_KINDS:
        raise ValueError(f"turn kind must be one of {TURN_KINDS}, got {kind!r}")
    if not isinstance(text, str) or not text.strip():
        return False
    record: dict[str, Any] = {"kind": kind, "text": text.strip()[:max_chars]}
    for key, value in ids.items():
        if key == "execution_sha256":
            continue
        if isinstance(value, str) and value:
            record[key] = value
    try:
        path = Path(trajectory_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / TURNS_FILENAME, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, TypeError, ValueError):
        return False
    _write_execution_fingerprint(path, ids.get("execution_sha256"))

    # The OTel projection rides the SAME call that writes the file, for the
    # same reason the tool-call path does: two call sites for one fact drift,
    # and a turn that reached the trajectory but not the customer's traces is
    # the drift nobody notices until they are comparing the two.
    #
    # Imported here, not at module scope: `tracing` reaches for the OTel API,
    # and a customer who never installed the extra should not pay an import for
    # a projection that will be a no-op.
    try:
        from .tracing import record_turn_span

        record_turn_span(
            **{("input_text" if kind == "task" else "output_text"): record["text"]},
            turn_id=ids.get("turn_id"),
            conversation_id=ids.get("conversation_id"),
            task_id=ids.get("task_id"),
            rollout_id=ids.get("rollout_id"),
            execution_sha256=ids.get("execution_sha256"),
        )
    except Exception:
        # A capture bug costs the observation and never the turn -- and the
        # file is already written, so this cannot cost that either.
        pass
    return True


def read_conversation_id(path: Any) -> str | None:
    """Which conversation this turn directory declared, or ``None``.

    Lives here beside the other turn-directory readers, not in the corpus
    projection that first needed it: the upload path reads it too, and a
    second copy would be a second answer to "where is the conversation id
    written". A missing or unreadable file is ``None`` -- a turn that never
    declared a conversation is a real state, not an error.
    """
    from pathlib import Path as _Path

    try:
        return (_Path(path) / CONVERSATION_ID_FILENAME).read_text(
            encoding="utf-8").strip() or None
    except OSError:
        return None


def _normalize_task_id(value: Any) -> str | None:
    """Return one safe opaque correlation id, or ``None``.

    The identifier is deliberately not parsed as a UUID or given any product
    semantics: customer agents use many task-id formats.  It is still kept
    conservative at the wire boundary.  Non-strings, leading/trailing
    whitespace, C0/C1 control characters and values beyond the control
    plane's 255-character field cannot be exact joins, so they remain absent
    rather than being silently rewritten into a different id. This calls the
    same predicate whose pattern is published in ``flywheel-1.json``.
    """
    return _production_identifier(value)


def _write_task_id(path: Path, task_id: Any) -> None:
    """Persist an exact task binding without ever failing the customer turn."""
    value = _normalize_task_id(task_id)
    if value is None:
        return
    sidecar = path / TASK_ID_FILENAME
    conflict = path / _TASK_ID_CONFLICT_FILENAME
    try:
        path.mkdir(parents=True, exist_ok=True)
        if conflict.exists():
            return
        try:
            with sidecar.open("x", encoding="utf-8") as output:
                output.write(value)
            return
        except FileExistsError:
            pass
        try:
            current = sidecar.read_text(encoding="utf-8")
        except OSError:
            current = None
        if current == value:
            return
        try:
            conflict.open("x", encoding="utf-8").close()
        except FileExistsError:
            pass
    except OSError:
        return


def read_task_id(path: Any) -> str | None:
    """Read an exact, non-conflicting caller task binding, or ``None``."""
    turn = Path(path)
    if (turn / _TASK_ID_CONFLICT_FILENAME).exists():
        return None
    try:
        raw = (turn / TASK_ID_FILENAME).read_text(encoding="utf-8")
    except OSError:
        return None
    value = _normalize_task_id(raw)
    if value != raw or (turn / _TASK_ID_CONFLICT_FILENAME).exists():
        return None
    return value


def read_turn_text(
    trajectory_dir: str | os.PathLike[str],
) -> dict[str, str] | None:
    """The turn's task and answer, or ``None`` if nothing was recorded.

    THE CONSUMER THIS FORMAT NEVER HAD. The CLI recorder has written
    ``turns.jsonl`` since it learned to subscribe to the turn events, and
    nothing has ever read it back -- a producer with no consumer, which is the
    shape this codebase keeps finding and the reason a trainable production
    turn was unreachable.

    Returns ``{"task": ..., "answer": ...}`` with only the halves actually
    present. ``None`` means no file, which is not the same as a turn that
    recorded neither half -- absence is never collapsed here either.

    The LAST record of each kind wins. A turn that was answered twice (a retry,
    a correction) ends with the answer that was actually returned.
    """
    path = Path(trajectory_dir) / TURNS_FILENAME
    if not path.exists():
        return None
    out: dict[str, str] = {}
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            # A truncated final write costs that record and nothing else --
            # the same rule Trajectory.read keeps.
            continue
        kind, text = rec.get("kind"), rec.get("text")
        if kind in TURN_KINDS and isinstance(text, str) and text:
            out[kind] = text
    return out


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
                 conversation_id: str | None = None,
                 task_id: str | None = None,
                 execution_sha256: str | None = None,
                 kind: CaptureScope = "rollout",
                 model_content: bool = False) -> None:
        self._rid = rollout_id
        self._trajectory_dir = str(trajectory_dir)
        self._kind = kind
        self._conversation_id = conversation_id or None
        self._task_id = _normalize_task_id(task_id)
        self._execution_sha256 = (
            execution_sha256 if is_lower_sha256(execution_sha256) else None
        )
        #: A PRODUCTION TURN's model boundary, and only with consent. A
        #: rollout never records one, whatever it is passed: it is claimed
        #: work on the SDK rollout lane, whose report is its own contract.
        self._model_content = bool(model_content) and kind == "turn"
        self._tokens: list[Any] = []

    def __enter__(self) -> str:
        started: str | None = None
        if self._model_content:
            from .genai_otlp import now_unix_nano

            started = now_unix_nano()
        self._tokens = [
            _current_rollout.set(self._rid),
            _current_trajectory_dir.set(self._trajectory_dir),
            _current_scope_kind.set(self._kind),
            _current_conversation.set(self._conversation_id),
            _current_task.set(self._task_id),
            _current_execution_sha256.set(self._execution_sha256),
            _current_model_capture.set(started),
        ]
        return self._rid

    def __exit__(self, *exc: Any) -> None:
        if self._tokens:
            _current_rollout.reset(self._tokens[0])
            _current_trajectory_dir.reset(self._tokens[1])
            _current_scope_kind.reset(self._tokens[2])
            _current_conversation.reset(self._tokens[3])
            _current_task.reset(self._tokens[4])
            _current_execution_sha256.reset(self._tokens[5])
            _current_model_capture.reset(self._tokens[6])
            self._tokens = []
        if self._model_content:
            _record_model_boundary_end(self._trajectory_dir)
        # The handle lives exactly as long as the scope that writes through it.
        # Without this the sink registry would grow one entry per turn for the
        # life of the process, which is the leak that holding a handle open
        # would otherwise trade for the speed.
        _release(self._trajectory_dir)
