"""``attach()`` -- capture in the process that serves your users.

:func:`~agent_flywheel.serve` contributes CAPACITY: it claims tasks from
the control plane and runs your agent against them in a child process. This
module contributes EVIDENCE instead, from the turns your agent is already serving.
Same adapters, same tri-state, same trajectory file, same absence contract --
scoped around a live user turn rather than a claimed rollout.

**It captures. It never reroutes.** Those are different jobs and mixing them
would be a correctness bug, not an inconvenience: pointing a production agent
at a new endpoint is a decision with user-visible consequences, and it belongs
to :func:`~agent_flywheel.apply_current_policy`, where the customer
makes it explicitly. ``attach`` binds the adapters with no endpoint at all,
which the adapters handle by design -- ``LangChainAdapter._route`` returns
immediately when it was handed no chat model, and the ADK adapter ignores the
URL entirely.

Three rules govern every line here, and they are the same three the rest of the
package keeps:

* a capture bug costs the observation and nothing else -- never the turn;
* an unreachable control plane never stops the agent serving its users;
* absence is never collapsed into zero.

**Uploading is OPT-IN, per agent, and off by default.** A production turn is
your END USERS' own message and the answer you gave them, so nothing here
decides to send it. :meth:`Attachment.flush` uploads only when the control
plane said this agent is enabled for capture -- a per-agent switch an operator
sets, which the registration response reports back. Until then turns
accumulate as trajectory directories under the root you name and go nowhere,
which is exactly what they did before the route existed.

**When it uploads.** ``flush()`` sends turns that have an answer; ``close()``
flushes once more and sends the rest, including turns whose answer never
arrived -- those carry real tool calls and the control plane stores them,
flagged and excluded from anything needing a completion. Nothing is sent at
turn-scope exit, because :meth:`Attachment.record_answer` is callable after
the scope closes and an upload there would ship every turn before its answer
existed.

**What it never does.** It never raises into your process, and it never blocks
a turn on the network: a control plane that is unreachable leaves the turns on
disk for the next flush.
"""
from __future__ import annotations

import atexit
import os
import shutil
import threading
import time
import uuid
import warnings
import weakref
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# NOT imported at module scope. `_http` pulls in httpx, which is 83-87% of
# this package's import time and ~88% of its resident memory -- and a process
# that only reads the promoted policy, or only captures locally, never opens a
# socket at all. Deferring it means those customers stop paying for a transport
# they do not use. `policy.py` has always done this; these were the paths that
# forced it anyway.
if TYPE_CHECKING:
    from ._http import ControlPlaneClient
from .contract import (
    _PRODUCTION_IDENTIFIER_MAX_CHARS,
    _PRODUCTION_TURN_BATCH_MAX_ITEMS,
    _production_agent_identifier,
    _production_identifier,
    _production_turn_identifier,
)
from .declaration import Describe
from .entrypoint import InputShape
from .errors import ConfigurationError, TransportError
from .execution_identity import normalize_attested_components
from .outcomes import (
    AGENT_FINGERPRINT_FILENAME,
    CONVERSATION_ID_FILENAME,
    Trajectory,
    _normalize_task_id,
    _read_execution_fingerprint,
    _release,
    _RolloutScope,
    _write_execution_fingerprint,
    _write_task_id,
    read_conversation_id,
    read_task_id,
    read_turn_text,
    record_turn_text,
)
from .policy import missing_credentials_cause

__all__ = ["Attachment", "CaptureVerdict", "attach", "capture_verdict"]

ENV_CONTROL_PLANE_URL = "PERCEPTEYE_CONTROL_PLANE_URL"
ENV_API_KEY = "PERCEPTEYE_API_KEY"


@dataclass(frozen=True)
class CaptureVerdict:
    """May captured turns be uploaded, and if not, WHY NOT."""

    #: Whether the control plane said yes, exactly.
    may_upload: bool
    #: The sentence explaining a ``False``. ``None`` when and only when
    #: ``may_upload`` is true -- so a caller that shows this to an operator
    #: always has something to show.
    reason: str | None
    #: Whether the control plane ANSWERED the question at all, as opposed to
    #: sending a body with no ``capture`` block in it. Both are ``may_upload
    #: = False``, and they are different facts: the re-poll on the flush
    #: cadence must not overwrite a specific refusal the server gave at
    #: startup ("this key lacks the scope") with the generic one, just because
    #: one poll came back empty during a blip.
    answered: bool = True


def capture_verdict(body: Any) -> CaptureVerdict:
    """THE consent predicate. Ask this, never one field of it.

    :param body: a registration response, or a ``GET /agents/{id}/capture``
        body -- both carry the answer under ``capture``. Anything else is
        treated as "the control plane did not answer".

    THE SERVER DECIDES; this only reads the verdict. ``capture.turns`` is
    computed in the control plane from BOTH the per-agent consent flag (opt-in;
    unset is not consent) AND whether this flywheel key carries the scope that
    permits reporting turns. Re-deriving either half here would be a second
    answer to a settled question.

    ``turns is True``, NOT TRUTHINESS, and the difference is the whole point:
    this gate protects an end user's own message and the answer we gave them,
    and a consent question is answered by the exact value. ``{"turns":
    "false"}`` -- a JSON string, which an older intake or a proxy can produce
    -- is truthy in Python, so ``bool(cap.get("turns"))`` read a refusal as
    consent and began uploading. The Node SDK's ``captureVerdict`` has always
    asked ``=== true`` and has a regression test for that exact payload;
    ``tests/test_capture_verdict_parity.py`` now drives the same table through
    both, so the two packages cannot answer this differently again.

    ABSENCE IS NOT CONSENT, and it is not a silent refusal either. A
    registration that failed, or a control plane too old to send ``capture`` at
    all, yields ``False`` with a sentence saying so -- ``upload_refusal`` is
    documented as "the sentence explaining a False", and a ``None`` there names
    no cause for the two commonest refusals there are.
    """
    cap = body.get("capture") if isinstance(body, Mapping) else None
    if not isinstance(cap, Mapping):
        return CaptureVerdict(
            may_upload=False,
            reason=(
                "the control plane did not answer whether this agent may "
                "upload production turns, so nothing is uploaded. Turns are "
                "still captured locally."
            ),
            answered=False,
        )
    if cap.get("turns") is True:
        return CaptureVerdict(may_upload=True, reason=None)
    served = cap.get("reason")
    return CaptureVerdict(
        may_upload=False,
        # The server's own sentence when it sent one -- it names the switch and
        # who can flip it, which anything written here could only paraphrase.
        reason=(
            served.strip()
            if isinstance(served, str) and served.strip()
            else "production capture is not enabled for this agent."
        ),
    )


#: How many turns pass between automatic prunes when `retention_turns` is set.
#: Amortises one directory scan over many turns; the ceiling is therefore
#: `retention_turns + _PRUNE_EVERY`, not `retention_turns` exactly.
_PRUNE_EVERY = 256

#: Turns between automatic upload attempts.
#:
#: THE PRODUCER THIS PATH NEVER HAD. ``flush()`` existed, ``close()`` called it
#: and ``__exit__`` called ``close()`` -- and the documented usage is
#: ``px = attach(...)`` held for the life of the process with a ``with
#: px.turn(...)`` per request, which never exits a scope on the Attachment
#: itself. So a long-running server following this package's own README
#: captured turns forever and uploaded none of them. An upload path whose only
#: trigger is a shutdown the process never performs is not a path.
#:
#: Smaller than ``_PRUNE_EVERY`` because the two answer different questions:
#: pruning is a directory scan protecting disk, and 256 turns of delay costs
#: nothing; uploading is how the data becomes reachable at all, and a turn that
#: sits unsent is a turn training cannot see. 32 sits under the 64-turn batch
#: so an ordinary flush is one request.
#:
#: WHAT THIS COSTS A LOW-TRAFFIC AGENT, said plainly rather than left to be
#: discovered: an agent serving a handful of turns a day reaches 32 slowly, so
#: its turns leave at the next process exit rather than the same afternoon.
#: That is accepted. The alternative is a timer thread running for the life of
#: a customer's serving process to move data nobody is waiting on, and an
#: unasked-for background thread is a worse thing to put in someone else's
#: process than a delay.
_FLUSH_EVERY = 32

#: How long ``close()`` waits for an in-flight periodic flush before giving up.
#:
#: ``close()`` runs from the ``atexit`` hook, and a periodic flush may be
#: holding the upload lock with a request that can take the HTTP client's full
#: timeout. Blocking on it unbounded would let an unresponsive control plane
#: keep a customer's process from exiting -- a far worse outcome than the one
#: it prevents, because a turn that is not sent now is still on disk and the
#: next process to attach uploads it.
_CLOSE_FLUSH_GRACE_S = 5.0


def _started_at(d: Path) -> float:
    """When this turn STARTED, for ordering a conversation.

    THE ``conversation_id`` FILE'S mtime, NOT THE DIRECTORY'S -- and that
    distinction is the whole point. A directory's mtime advances every time a
    file is CREATED inside it, which this module already knows and already
    designed around once: see ``_UPLOADED_INDEX``, which is one file at the
    root rather than a marker in each turn for exactly this reason. But a
    turn's own capture creates files inside its directory long after the turn
    began -- ``record_answer`` creates ``turns.jsonl`` whenever ``input_text``
    was not passed to :meth:`Attachment.turn` (both are documented optional),
    and the first ``record_tool_call`` creates the trajectory file. So the
    directory's mtime says "last written to", and ordering a conversation by
    it puts a turn whose answer arrived late AFTER the turn that followed it:
    the conversation is uploaded in the wrong order and MC stores the wrong
    ``turn_index`` verbatim.

    The ``conversation_id`` file is written once, at scope entry, and never
    touched again. It exists for exactly the turns that have an index at all,
    so nothing new has to be written to get this -- and a turn that predates
    this ordering, or whose file cannot be read, still falls back to the
    directory mtime rather than losing its place entirely.
    """
    try:
        return (d / CONVERSATION_ID_FILENAME).stat().st_mtime
    except OSError:
        pass
    try:
        return d.stat().st_mtime
    except OSError:
        return 0.0


#: How ``_http.ControlPlaneClient.request`` spells a terminal 403. It formats
#: every 4xx that is not 408/429 as ``"{method} {path} -> HTTP {code}: {detail}"``
#: and raises ``TransportError``; the status is not carried structurally, so
#: this is the seam between the two. If that format ever changes, the failure
#: is a refusal read as a blip again -- which is what
#: ``test_a_revoked_agent_stops_uploading`` is there to catch.
_REFUSED_MARKER = "-> HTTP 403"


def _capture_refusal(exc: BaseException) -> str | None:
    """The intake's sentence when it REFUSED these turns, else ``None``.

    A 403 is a DECISION -- an operator turned capture off, or the key lost the
    ``turn.report`` scope -- and will not become a 2xx by being sent again.
    ``_flush_locked`` used to swallow it into the same ``failed += len(chunk)``
    as a timeout, which is one wrong cause named for every failure: the process
    re-POSTed the same batch of end-user prompts and answers every cadence
    forever, with nothing in the customer's logs and ``upload_refusal`` still
    reading whatever startup said.

    Anything else -- a timeout, a 5xx, a 429 that outlasted the retry ladder --
    is a blip and stays retryable. Failing to recognise a refusal costs a
    pointless retry; mistaking a blip for a refusal would stop a consenting
    customer's uploads, so this matches narrowly and errs towards retrying.
    """
    if not isinstance(exc, TransportError):
        return None
    text = str(exc)
    if _REFUSED_MARKER not in text:
        return None
    detail = text.partition(_REFUSED_MARKER)[2].lstrip(": ").strip()
    return detail or "the control plane refused this agent's turns (HTTP 403)."


def _recorded_anything(
    text: dict[str, str], calls: Any, *, task_id: str | None = None,
) -> bool:
    """Did this turn capture anything at all?

    ONE ANSWER, TWO ASKERS. ``_turn_wire`` returns ``None`` when this is False
    -- there is nothing to upload -- and ``prune`` uses it to tell a deleted
    directory that held DATA nobody sent from one that could never have been
    sent at all. Asked twice with two spellings, the second asker charges an
    empty directory as lost end-user data and sends the operator hunting for a
    broken upload path.
    """
    # A valid caller correlation is itself turn-native evidence.  Dropping a
    # task-only turn here made ``turn(task_id=...)`` create a durable sidecar
    # that neither upload nor pruning considered a record.  Keep the neutral
    # identifier on the wire; Mission Control can decide that a content-free
    # turn contributes no usable observation.
    return bool(text.get("task") or text.get("answer") or calls or task_id)


class Attachment:
    """A bound capture session for one process. Returned by :func:`attach`.

    Construct it through :func:`attach`; the adapters are already bound by the
    time you hold one.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        trajectory_root: str,
        policy: dict[str, Any],
        unbinds: list[Any],
        adapters_bound: tuple[str, ...],
        retention_turns: int | None = None,
        control_plane_url: str | None = None,
        api_key: str | None = None,
        may_upload: bool = False,
        upload_refusal: str | None = None,
        agent_fingerprint: Mapping[str, str] | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.trajectory_root = trajectory_root
        #: What the control plane said this agent should be running, or ``{}``
        #: when nothing is deployed OR the control plane could not be reached.
        #: Both read the same on purpose: keep your current configuration.
        #: INFORMATIONAL here -- ``attach`` does not act on it. Applying it is
        #: ``apply_current_policy``'s job and the customer's decision.
        self.policy = policy
        #: Which adapters actually bound. ``()`` is a real answer -- an agent
        #: using no supported framework captures nothing automatically, and
        #: saying so is better than implying coverage that is not there.
        self.adapters_bound = adapters_bound
        self._unbinds = unbinds
        self._closed = False
        #: When set, `prune(keep_last=...)` runs itself. Checked every
        #: _PRUNE_EVERY turns rather than every turn, because pruning is a
        #: directory scan and doing one per user request would cost more than
        #: the capture it is cleaning up after.
        self._retention_turns = retention_turns
        self._turns_since_prune = 0
        self._turns_since_flush = 0
        self._lock = threading.Lock()
        #: SEPARATE FROM ``_lock`` on purpose. ``_lock`` guards the turn
        #: counters and is taken on the request path; an upload holding it
        #: would put the network between a user and their answer, which is the
        #: one thing this class promises never to do. This one serializes
        #: uploads against each other so two flushes cannot send the same turn
        #: twice before either has written its marker.
        self._flush_lock = threading.Lock()
        self._cp_url = control_plane_url
        self._api_key = api_key
        #: What the control plane SAID at registration, not what we assume.
        #: Capture is opt-in per agent because a turn carries this agent's end
        #: users' own messages -- so the default is False, and finding out by
        #: uploading would mean sending the data whose permission is in
        #: question.
        self.may_upload = may_upload
        #: Why not, in a sentence, when it is False and the control plane said.
        self.upload_refusal = upload_refusal
        #: Exact SDK-reconciled execution identity, when attach() was given
        #: enough behavior observation and opaque adapter evidence to prove
        #: one. It is persisted per turn; absent stays absent.
        self.agent_fingerprint = dict(agent_fingerprint or {}) or None
        self._execution_sha256 = (
            self.agent_fingerprint.get("execution_sha256")
            if self.agent_fingerprint is not None else None
        )
        #: Whether the shutdown drain is already registered. Consent can arrive
        #: at startup OR later, and can be taken away and re-granted, so the
        #: hook is claimed once rather than once per grant.
        self._exit_flush_registered = False

    @contextmanager
    def turn(self, *, turn_id: str | None = None,
             conversation_id: str | None = None,
             task_id: str | None = None,
             input_text: str | None = None) -> Iterator[str]:
        """Scope one user turn. Tool calls inside it land in its own trajectory.

        Yields the turn id, which the caller should keep: it is the join key
        for any outcome that arrives later than the turn does -- the draft was
        kept an hour afterwards, the pull request merged the next morning.

        A turn boundary is DECLARED rather than inferred. Guessing where one
        request ends and the next begins is the same class of guess this SDK
        refuses everywhere else, and getting it wrong silently merges two
        users' tool calls into one record.

        Nothing in here can fail the turn. A trajectory that cannot be written
        disables capture for itself and says so once; the body still runs.

        **A turn with neither content nor declared correlation leaves nothing
        behind.** Content creates the directory on its first write. A supplied
        ``conversation_id`` or valid ``task_id`` is itself evidence and gets a
        sidecar at scope entry. Before that distinction, every scope created a
        directory unconditionally, so an agent that answered from context
        without recording content or correlation still cost one directory --
        and at a million turns that was a third of a gigabyte and a million
        inodes for turns that had nothing to say.

        ``input_text`` records what the user ASKED, and :meth:`record_answer`
        records what the agent finally said. Both are OPTIONAL and both must be
        passed by the caller, because nothing here can see them: the adapters
        declare ``reports_outcomes=True`` and observe tool calls, not turn
        content, so a prompt and a completion are facts only the customer's own
        code holds. Without them a turn records its tool calls and nothing
        else, which is exactly what it did before -- and is why tool outcomes
        stay zero-instrumentation while turn text is opt-in.

        ``task_id`` is an optional opaque correlation supplied by the caller.
        The SDK persists and forwards it but never interprets it: it is not a
        success label, a policy identity, a hint, or permission to train.  A
        control-plane consumer may use it as exact join evidence after its own
        validation.  Invalid or conflicting values stay absent rather than
        being guessed or rewritten, and cannot fail the served turn.
        """
        # An invalid caller id disables only that correlation, never the
        # customer request. Generate a fresh wire-safe id and return it from
        # the context manager so delayed outcomes still have an exact join.
        tid = (
            _production_turn_identifier(turn_id)
            or f"t_{uuid.uuid4().hex[:16]}"
        )
        effective_conversation_id = _production_identifier(conversation_id)
        path = Path(self.trajectory_root) / tid
        validated_task_id = _normalize_task_id(task_id)
        if effective_conversation_id or validated_task_id:
            # Correlation sidecars need the directory up front because they
            # exist whether or not a tool is ever called.
            try:
                path.mkdir(parents=True, exist_ok=True)
                if effective_conversation_id:
                    (path / CONVERSATION_ID_FILENAME).write_text(
                        effective_conversation_id, encoding="utf-8")
                _write_task_id(path, validated_task_id)
                _write_execution_fingerprint(path, self._execution_sha256)
            except OSError:
                pass
        # Read back what was actually persisted.  In particular, a reused turn
        # id carrying a different task creates a conflict marker; continuing
        # with the caller's latest value here would make the OTLP path claim a
        # binding that the upload path correctly refused.
        effective_task_id = read_task_id(path)
        if input_text:
            record_turn_text(path, "task", input_text,
                             turn_id=tid,
                             conversation_id=effective_conversation_id or "",
                             task_id=effective_task_id or "",
                             execution_sha256=self._execution_sha256 or "")
        # ONE counter block for both cadences, because they are the same
        # question -- "is periodic maintenance due?" -- asked twice. They were
        # split while only pruning existed and the flush counter was added
        # beside it; two lock acquisitions per turn on the request path for one
        # answer is the kind of duplication that drifts.
        with self._lock:
            self._turns_since_prune += 1
            self._turns_since_flush += 1
            prune_due = (self._retention_turns is not None
                         and self._turns_since_prune >= _PRUNE_EVERY)
            if prune_due:
                self._turns_since_prune = 0
            flush_due = self._turns_since_flush >= _FLUSH_EVERY
            if flush_due:
                self._turns_since_flush = 0
        if prune_due:
            self.prune(keep_last=self._retention_turns)
        if flush_due:
            self._flush_soon()

        with _RolloutScope(tid, str(path), kind="turn",
                           conversation_id=effective_conversation_id,
                           task_id=effective_task_id,
                           execution_sha256=self._execution_sha256):
            yield tid

    def record_answer(self, turn_id: str, text: str) -> bool:
        """Record what the agent finally answered for one turn.

        SEPARATE FROM ``turn(input_text=...)`` because the answer does not
        exist yet when the turn opens. Callable after the scope has closed,
        for the same reason :meth:`turn` yields its id: an outcome can arrive
        later than the turn does.

        Returns whether anything was written, and never raises -- a capture bug
        costs the observation and never the turn.
        """
        safe_turn_id = _production_turn_identifier(turn_id)
        if safe_turn_id is None:
            return False
        path = Path(self.trajectory_root) / safe_turn_id
        sidecar = path / AGENT_FINGERPRINT_FILENAME
        # A pre-existing turn with no identity may have been captured by an
        # older process. Never relabel it as this process merely because its
        # delayed answer arrived now. A turn created by this attachment has a
        # sidecar on its first task/call/conversation write; an answer-only
        # turn has no directory yet and can be labelled safely.
        execution_sha256 = (
            self._execution_sha256
            if not path.exists() or sidecar.exists()
            else None
        )
        return record_turn_text(
            path, "answer", text, turn_id=safe_turn_id,
            task_id=read_task_id(path) or "",
            execution_sha256=execution_sha256 or "",
        )

    def read_turn(self, turn_id: str) -> dict[str, str] | None:
        """The task and answer recorded for one turn, or ``None``.

        ``None`` means nothing was recorded, which is not the same as a turn
        that recorded neither half -- the distinction absence is never
        collapsed into.
        """
        safe_turn_id = _production_turn_identifier(turn_id)
        if safe_turn_id is None:
            return None
        return read_turn_text(Path(self.trajectory_root) / safe_turn_id)

    def read(self, turn_id: str) -> Any:
        """The tool calls captured for one turn, or ``None`` if none were.

        ``None`` and ``[]`` mean different things here for the same reason they
        do everywhere else: no file means the turn reported nothing, and must
        not be rendered as an assertion that zero tool calls occurred.
        """
        safe_turn_id = _production_turn_identifier(turn_id)
        if safe_turn_id is None:
            return None
        return Trajectory(Path(self.trajectory_root) / safe_turn_id).read()

    def prune(
        self,
        *,
        keep_last: int | None = None,
        older_than_s: float | None = None,
    ) -> int:
        """Delete recorded turns from ``trajectory_root``. Returns how many.

        The trajectory is append-only and nothing else removes it, so without
        this a long-running production process accumulates one directory per
        RECORDING turn forever -- roughly 2.4 GB and a million inodes per
        million turns at typical record sizes, and far more if tool outputs are
        large, because ``output`` is stored verbatim.

        This is a method rather than a default because the trajectory is the
        customer's data, sitting in a directory the customer named, and a
        library that silently deletes it would be worse than one that grows.
        Pass ``retention_turns=`` to :func:`attach` to have it called for you.

        Args:
            keep_last: keep this many most-recent turns, delete the rest.
            older_than_s: delete turns last modified longer ago than this.

        Both may be given; a turn is deleted if EITHER says so. Passing neither
        deletes nothing and returns 0 -- an unqualified "prune" would be a
        request to delete everything, and that is not a thing to infer.

        Never raises. A turn that cannot be read or removed is left alone and
        does not stop the others; retention failing must not take down the
        process it was protecting.
        """
        if keep_last is None and older_than_s is None:
            return 0

        root = Path(self.trajectory_root)
        try:
            entries = [e for e in root.iterdir() if e.is_dir()]
        except OSError:
            return 0

        def mtime(e: Path) -> float:
            try:
                return e.stat().st_mtime
            except OSError:
                return 0.0

        # Newest first, so "keep_last" is a slice rather than a sort per turn.
        entries.sort(key=mtime, reverse=True)

        # UPLOADED TURNS ARE DELETED FIRST, and this is a correctness fix, not
        # a nicety. Retention deletes by mtime and knows nothing about upload
        # state, while the periodic flush deliberately HOLDS BACK turns that
        # have no answer yet -- so on an agent with `retention_turns` set, a
        # turn whose answer was still coming could be deleted before it was
        # ever sent. Silent, permanent, and invisible: the turn simply never
        # existed as far as the control plane is concerned.
        #
        # The same NUMBER of turns is still removed, so the disk guarantee the
        # customer asked for is unchanged; only the CHOICE of victims moves.
        # Among equally-doomed turns, one already at the control plane costs
        # nothing to delete and one still pending costs the observation.
        #
        # Asked only when uploading is actually on. With no consent there is no
        # such thing as a pending turn, and reading the index would be work in
        # service of a distinction that does not exist.
        if self.may_upload:
            already = self._uploaded_ids()
            entries.sort(key=lambda e: (e.name not in already, mtime(e)),
                         reverse=True)

        doomed: list[Path] = []
        if keep_last is not None:
            doomed.extend(entries[max(0, keep_last):])
        if older_than_s is not None:
            cutoff = time.time() - older_than_s
            # THE ONES ``keep_last`` KEPT, and every turn when it was not
            # given. This was ``entries[:max(0, keep_last or 0)]``, which for
            # the documented standalone use -- ``prune(older_than_s=...)`` with
            # no ``keep_last`` -- is ``entries[:0]``: nothing was ever
            # considered, so a time-based retention window deleted nothing and
            # returned 0, reading as "nothing was old enough". The tail is
            # already doomed by the branch above, and ``dict.fromkeys`` below
            # dedupes the overlap.
            head = entries if keep_last is None else entries[:max(0, keep_last)]
            doomed.extend(e for e in head if mtime(e) < cutoff)

        removed = 0
        unsent = 0
        pending = self._uploaded_ids() if self.may_upload else None
        for e in dict.fromkeys(doomed):
            # NOT "absent from the index" -- that is half the question. A turn
            # that recorded NOTHING never enters the index because
            # ``_turn_wire`` returns None for it and every flush skips it, so
            # counting it here charged an empty directory as lost end-user
            # data. ``turn(conversation_id=...)`` creates one such directory
            # per turn for an agent that answers from context without calling
            # a tool and passes neither ``input_text`` nor an answer -- both
            # documented optional -- which is a steady drip of a warning whose
            # two remedies are both wrong.
            if (pending is not None and e.name not in pending
                    and self._holds_a_record(e)):
                unsent += 1
            # Release any handle we still hold on it first, or on a platform
            # that refuses to unlink an open file this would silently no-op.
            _release(str(e))
            try:
                shutil.rmtree(e)
                removed += 1
            except OSError:
                continue
        if unsent:
            # SAID OUT LOUD, once per prune that does it. Retention winning
            # over an unsent turn is a legitimate outcome -- the customer
            # asked for a bounded directory and we are honouring it -- but it
            # is data the control plane will never see, and an operator who
            # is not told will look for those turns upstream.
            warnings.warn(
                f"agent-flywheel: retention deleted {unsent} captured turn(s) "
                f"that had not been uploaded -- including any still waiting "
                f"on an answer, which the periodic flush holds back. They are "
                f"gone. Raise retention_turns, or check why uploads are not "
                f"succeeding (Attachment.upload_refusal says if consent is "
                f"the reason).",
                RuntimeWarning,
                stacklevel=2,
            )
        if removed:
            # The upload index describes this tree, so it is bounded by this
            # tree. Left alone it would grow for the life of the process while
            # the directory it describes stays capped -- which is the opposite
            # of what a retention setting is for.
            self._forget_uploaded({
                e.name for e in entries if e not in set(doomed)
            })
        return removed

    #: Where upload state is kept: ONE FILE at the trajectory root, holding the
    #: ids already accepted by the control plane, newline-delimited.
    #:
    #: NOT A MARKER INSIDE THE TURN DIRECTORY, which is where this started and
    #: which was a real defect. Creating a file inside a directory bumps that
    #: DIRECTORY's mtime, and mtime is the ordering key for both readers of
    #: this tree: ``export.iter_records`` sorts by it to keep a conversation's
    #: turns in the order they happened, and ``prune(keep_last=N)`` uses it to
    #: decide which turns are newest. Marking a turn uploaded would therefore
    #: have floated it to the end of its own conversation and made retention
    #: delete the wrong turns -- a capture bug caused entirely by the
    #: bookkeeping for the upload.
    #:
    #: A FILE and not a directory, because neither reader filters dotfiles:
    #: they take everything under the root that ``is_dir()``, so a
    #: ``.uploaded/`` directory would have been scanned as a turn.
    #:
    #: Per TURN, which is safe here and would not have been for a
    #: conversation-shaped upload: each turn is its own row keyed on its own
    #: id, so sending turn 4 later does not re-send or truncate turns 1-3.
    _UPLOADED_INDEX = ".uploaded"

    def _uploaded_ids(self) -> set[str]:
        """Turn ids the control plane has already accepted."""
        try:
            raw = (Path(self.trajectory_root) / self._UPLOADED_INDEX).read_text(
                encoding="utf-8")
        except OSError:
            return set()
        return {line.strip() for line in raw.splitlines() if line.strip()}

    def _mark_uploaded(self, turn_ids: list[str]) -> None:
        """Append accepted ids. Append-only, so a crash mid-write can lose the
        record of an upload but never invent one -- and a lost record costs a
        re-send the server dedupes, never a duplicate row."""
        if not turn_ids:
            return
        try:
            with (Path(self.trajectory_root) / self._UPLOADED_INDEX).open(
                    "a", encoding="utf-8") as fh:
                fh.write("".join(t + "\n" for t in turn_ids))
        except OSError:
            # The turns landed; we just cannot remember that. A re-send is
            # deduped server-side, so this costs bandwidth, never correctness.
            pass

    def _forget_uploaded(self, keep: set[str]) -> None:
        """Drop ids whose turn directory no longer exists.

        Called from ``prune``. Without it the index grows for the life of the
        process while the tree it describes stays bounded -- an unbounded file
        is not an acceptable price for retention that exists to bound things.
        """
        try:
            path = Path(self.trajectory_root) / self._UPLOADED_INDEX
            current = self._uploaded_ids() & keep
            path.write_text("".join(t + "\n" for t in sorted(current)),
                            encoding="utf-8")
        except OSError:
            pass

    def _holds_a_record(self, d: Path) -> bool:
        """Whether this turn directory holds anything an upload could carry.

        THE SAME QUESTION :meth:`_turn_wire` ANSWERS WITH ``None``, asked
        without building the wire -- ``_turn_wire`` also computes
        ``turn_index``, which scans the whole root, and ``prune`` asks this
        once per deleted directory.

        Unreadable counts as HELD. A turn we cannot parse may well be a real
        turn, and the warning this feeds is about data that is about to be
        destroyed: the honest direction to fail is towards telling the
        operator.
        """
        try:
            return _recorded_anything(
                read_turn_text(d) or {},
                Trajectory(d).read(),
                task_id=read_task_id(d),
            )
        except Exception:
            return True

    def _turn_wire(self, d: Path) -> dict[str, Any] | None:
        """One captured turn in the intake's wire shape, or None if empty.

        TURN-NATIVE, not corpus-shaped. What crosses the boundary is what the
        agent observed -- the user's message, the answer, the calls -- and the
        projection into any training-row format happens on the far side, where
        it can change without a customer redeploying.
        """
        turn_id = _production_turn_identifier(d.name)
        if turn_id is None:
            raise ValueError(
                "captured turn id is not representable on the flywheel/1 wire"
            )
        text = read_turn_text(d) or {}
        traj = Trajectory(d)
        calls = traj.read()
        answer = text.get("answer")
        task_id = read_task_id(d)
        if not _recorded_anything(text, calls, task_id=task_id):
            return None
        # THE TRI-STATE, asked of the trajectory FILE rather than inferred.
        # ``[] if self.adapters_bound else None`` answered a different
        # question: whether an adapter bound at ATTACH time, which says
        # nothing about whether THIS turn recorded anything. A turn whose
        # trajectory was never written would have been uploaded as a positive
        # assertion that the agent made zero calls -- an observation nobody
        # made, and the one transformation this SDK exists to refuse.
        #
        # File present, no rows  -> ``[]``  (it ran and called nothing)
        # File absent            -> ``None`` (nothing was recorded for it)
        if calls:
            tool_calls: list[dict[str, Any]] | None = [c.to_wire() for c in calls]
        elif traj.path.exists():
            tool_calls = []
        else:
            tool_calls = None
        wire: dict[str, Any] = {
            "turn_id": turn_id,
            "input_text": text.get("task"),
            "final_text": answer,
            "tool_calls": tool_calls,
            "tool_calls_omitted_count": traj.malformed,
        }
        cid = _production_identifier(read_conversation_id(d))
        if cid:
            wire["conversation_id"] = cid
            wire["turn_index"] = self._turn_index(d, cid)
        if task_id:
            wire["task_id"] = task_id
        fingerprint = _read_execution_fingerprint(d)
        if fingerprint is not None:
            wire["agent_fingerprint"] = fingerprint
        return wire

    def _turn_index(self, turn_dir: Path, conversation_id: str) -> int:
        """This turn's position within its conversation.

        DERIVED FROM WHEN THE TURN STARTED rather than from a counter. A
        counter held in the Attachment would restart at zero on every process
        restart and on every flush, and the control plane indexes on
        ``(org, agent, conversation_id, turn_index)``: every turn landing at 0
        makes a conversation unorderable, which is the one thing the index
        exists to prevent.

        Computed over every turn of the conversation ON DISK, not just the
        ones being uploaded now, so a turn sent in a later flush keeps the
        position it actually had. It shifts only if an earlier turn is pruned
        away, which is the honest answer: the earlier turn is gone.
        """
        root = Path(self.trajectory_root)
        siblings: list[tuple[float, str]] = []
        try:
            entries = list(root.iterdir())
        except OSError:
            return 0
        for d in entries:
            if not d.is_dir():
                continue
            if read_conversation_id(d) != conversation_id:
                continue
            siblings.append((_started_at(d), d.name))
        siblings.sort()
        for i, (_, name) in enumerate(siblings):
            if name == turn_dir.name:
                return i
        return 0

    def _flush_soon(self) -> None:
        """Upload in the background. Returns immediately; never raises.

        OFF THE REQUEST PATH, which is the whole reason this is not simply a
        call to :meth:`flush` from :meth:`turn`. This package promises that an
        upload never blocks a turn, and a customer's user is waiting on the
        other end of the one that triggered it. So the network happens on a
        daemon thread and the request returns at once.

        AT MOST ONE IN FLIGHT. The thread takes ``_flush_lock`` without
        blocking and gives up if another flush already holds it -- rather than
        queueing, because a queued flush would find the same turns the running
        one is about to mark as sent. Whatever it skips, the next cadence
        picks up.

        A daemon thread is used deliberately: it must never hold a customer's
        interpreter open at exit. Anything it would have sent is sent by the
        ``atexit`` hook instead, which runs before daemon threads are killed.
        """
        # "IS THERE ANY POINT STARTING A THREAD?" -- and deliberately NOT
        # "have we already been told yes". This used to short-circuit on
        # ``may_upload``, which quietly defeated the whole late-consent path:
        # the thread that re-reads the verdict never started, so an agent told
        # "no" at startup could never learn otherwise, and only its own test
        # revealed it.
        #
        # A process with no control plane configured has nothing to ask and
        # nothing to send, so it starts nothing. The authoritative consent
        # check stays in `_flush_locked`, which every path goes through.
        if self._closed or not (self._cp_url and self._api_key):
            return

        def _run() -> None:
            if not self._flush_lock.acquire(blocking=False):
                return
            try:
                # The consent re-check lives in `_flush_locked`, which every
                # path goes through -- it used to live here, so a manual
                # `flush()` or `close()` after an operator granted consent
                # still refused while only the 32-turn cadence noticed.
                self._flush_locked(include_unanswered=False, batch_size=64)
            except Exception:
                # `_flush_locked` already swallows everything; this is the
                # thread-boundary backstop, because an exception escaping a
                # thread's target prints a traceback into a customer's logs
                # under our name for a failure that cost only the observation.
                pass
            finally:
                self._flush_lock.release()

        try:
            threading.Thread(
                target=_run, name="percepteye-flush", daemon=True,
            ).start()
        except RuntimeError:
            # Interpreter shutting down: threads can no longer be started.
            # The atexit hook is the flush that matters at this point.
            pass

    def _grant_upload(self) -> None:
        """Consent has landed. THE ONE PLACE ``may_upload`` turns True.

        THE SHUTDOWN FLUSH IS REGISTERED HERE, not at the end of :func:`attach`
        where the startup verdict was read, because consent granted after
        startup is a first-class case -- it is the entire reason the verdict is
        a re-readable GET. Registering only on the startup answer meant a
        late-consent process had NO exit hook, and ``close()`` is the only
        caller that passes ``include_unanswered=True``: the periodic cadence
        deliberately holds unanswered turns back, so every turn still waiting
        on its answer, plus the whole sub-cadence tail, never left that
        process at all.

        Registered ONCE. The hook is idempotent through ``close()``, but a
        second registration would still be a second entry pinning a second
        weakref for the life of the interpreter, and consent can be granted
        more than once now that a terminal refusal takes it away.
        """
        self.may_upload = True
        self.upload_refusal = None
        if not self._exit_flush_registered:
            self._exit_flush_registered = True
            _flush_at_exit(self)

    def _refuse_upload(self, reason: str) -> None:
        """The intake terminally refused these turns. Stop sending, say why.

        THE ONE PLACE ``may_upload`` turns False after it was True, and it says
        so out loud because nothing else would: ``upload_refusal`` would
        otherwise still read whatever startup said, the operator would see
        repeated 403s in the control plane's logs with nothing in the
        customer's, and the customer would see a serving process making an
        outbound request every cadence that can never succeed.
        """
        self.may_upload = False
        self.upload_refusal = reason
        try:
            warnings.warn(
                f"agent-flywheel: the control plane refused this agent's captured "
                f"turns and uploading has stopped: {reason} The turns stay on "
                f"disk (Attachment.upload_refusal repeats this), and uploading "
                f"resumes on its own if capture is enabled again.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:  # -W error turns the line above into a raise
            pass

    def _recheck_capture(self) -> None:
        """Re-read the capture verdict. Never raises; fails closed.

        A READ, which is the whole reason this is safe to do periodically:
        before ``GET /agents/{id}/capture`` existed the only way to ask was to
        re-register, and a registration that omits ``discovered_agent`` records
        introspection as ``disabled`` -- so polling for permission would have
        repeatedly disabled the agent's own description.

        ONLY EVER TURNS CAPTURE ON, never off. A control plane that cannot
        answer returns ``{}``, and reading that as a revocation would stop a
        consenting customer's uploads on a network blip. Revocation is enforced
        where it belongs: the intake refuses. ``_flush_locked`` reads that
        refusal -- a 403 is a decision, not a blip -- stops sending, and leaves
        the turns on disk; this method is then how consent is picked back up if
        it is re-granted.
        """
        if not self._cp_url or not self._api_key:
            return
        client = None
        try:
            from ._http import ControlPlaneClient
            from .transport import AttachTransport

            client = ControlPlaneClient(self._cp_url, self._api_key)
            verdict = capture_verdict(
                AttachTransport(client, self.agent_id).capture_current(self.agent_id))
            if verdict.may_upload:
                self._grant_upload()
            elif verdict.answered:
                # Keep the reason current even while the answer stays no: an
                # operator who fixes consent but not the key's scope should see
                # the second sentence, not the first one from startup.
                #
                # ONLY WHEN IT ANSWERED. A poll that came back with no capture
                # block is a blip, and overwriting a specific refusal with the
                # generic one would lose the sentence the operator needs --
                # the same argument that stops an empty answer revoking
                # consent two lines up.
                self.upload_refusal = verdict.reason
        except Exception:
            # A capture bug costs the observation, never the turn.
            pass
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def flush(self, *, include_unanswered: bool = False,
              batch_size: int = 64) -> dict[str, int]:
        """Upload captured turns to the control plane. Returns counts.

        Serialized against the automatic background flush, so a manual call
        cannot race it into sending the same turn twice.

        RE-ASKS FOR CONSENT while the answer is still no, exactly as the
        background cadence does -- this is the documented escape hatch for
        "an operator just enabled capture, push the backlog now", and it used
        to answer ``not_permitted`` because only the 32-turn cadence asked.
        """
        with self._flush_lock:
            return self._flush_locked(
                include_unanswered=include_unanswered, batch_size=batch_size)

    def _flush_locked(self, *, include_unanswered: bool = False,
                      batch_size: int = 64) -> dict[str, int]:
        """The upload itself. Caller holds ``_flush_lock``.

        WHY NOT AT TURN-SCOPE EXIT, which is where an earlier design put it:
        ``record_answer`` is callable AFTER the scope closes, and the package's
        own example records the answer after the ``with`` block. Uploading on
        exit therefore shipped a turn whose ``final_text`` did not exist yet --
        every turn, for the documented usage.

        So the default holds back a turn that has no answer, because the answer
        may still be coming. ``close()`` passes ``include_unanswered=True``:
        at shutdown nothing more is coming, and a turn that made real tool
        calls and never recorded an answer is honest data rather than a
        validation error -- the intake stores it, flags it, and excludes it
        from objectives that need a completion.

        Never raises. A control plane that is unreachable must not take down a
        process that is serving users; the turns stay on disk and the next
        flush retries them.
        """
        sent = skipped = failed = 0
        refused = False
        if not self._cp_url or not self._api_key:
            return {"sent": 0, "skipped": 0, "failed": 0, "not_permitted": 1}
        if not self.may_upload:
            # AN OPERATOR MAY HAVE SAID YES SINCE STARTUP. Consent is a human
            # decision made in Mission Control, on their clock, not ours -- and
            # before the verdict was readable this cost a restart to notice.
            #
            # ASKED HERE, not in the background cadence, because this is the
            # seam every path goes through: the cadence, a manual `flush()`,
            # `close()` and the atexit drain. It lived in `_flush_soon` alone,
            # so a customer told "push the backlog now" called `flush()`, got
            # `not_permitted` and sent nothing -- while the docs offer both as
            # the explicit escape hatch.
            #
            # Asked only while the answer is NO, so it stops entirely once
            # permission is granted.
            self._recheck_capture()
            if not self.may_upload:
                return {"sent": 0, "skipped": 0, "failed": 0,
                        "not_permitted": 1}

        root = Path(self.trajectory_root)
        if not root.is_dir():
            return {"sent": 0, "skipped": 0, "failed": 0}

        already = self._uploaded_ids()
        pending: list[tuple[Path, dict[str, Any]]] = []
        try:
            entries = sorted(root.iterdir(), key=lambda p: p.name)
        except OSError:
            return {"sent": 0, "skipped": 0, "failed": 0}
        for d in entries:
            if not d.is_dir() or d.name in already:
                continue
            # PER DIRECTORY. The scan used to sit outside the try below, so a
            # truncated trajectory -- which this package documents as the
            # EXPECTED case, not an exotic one, because a deadline kill cuts
            # the final line -- raised straight out of a method whose docstring
            # says it never raises, inside a customer's shutdown path. One
            # unreadable turn now costs that turn.
            try:
                wire = self._turn_wire(d)
            except Exception:
                failed += 1
                continue
            if wire is None:
                continue
            if wire.get("final_text") is None and not include_unanswered:
                skipped += 1
                continue
            pending.append((d, wire))

        if not pending:
            return {"sent": 0, "skipped": skipped, "failed": 0}

        client = None
        try:
            from ._http import ControlPlaneClient
            from .transport import AttachTransport

            client = ControlPlaneClient(self._cp_url, self._api_key)
            tr = AttachTransport(client, self.agent_id)
            requested_batch_size = (
                batch_size
                if isinstance(batch_size, int) and not isinstance(batch_size, bool)
                else 1
            )
            wire_batch_size = min(
                max(1, requested_batch_size),
                _PRODUCTION_TURN_BATCH_MAX_ITEMS,
            )
            for i in range(0, len(pending), wire_batch_size):
                chunk = pending[i:i + wire_batch_size]
                try:
                    tr.report_turns(self.agent_id, [w for _, w in chunk])
                except Exception as exc:
                    refusal = _capture_refusal(exc)
                    if refusal is not None:
                        # NOT A TRANSPORT FAILURE, and treating it as one is
                        # what made this loop re-POST the same end-user message
                        # bodies to a control plane that had been told not to
                        # accept them, every cadence, for the life of the
                        # process. Stop, say why in the words the intake used,
                        # and leave the turns on disk. `_recheck_capture` picks
                        # consent back up if it is re-granted.
                        self._refuse_upload(refusal)
                        refused = True
                        failed += len(pending) - sent
                        break
                    # Whole chunk unacknowledged. Leave every marker unwritten
                    # so the next flush retries it: the intake dedupes on
                    # turn_id, so a retry of a batch that DID land is answered
                    # as a duplicate rather than stored twice.
                    failed += len(chunk)
                    continue
                self._mark_uploaded([d.name for d, _ in chunk])
                sent += len(chunk)
        except Exception:
            failed += len(pending) - sent
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        counts = {"sent": sent, "skipped": skipped, "failed": failed}
        if refused:
            counts["not_permitted"] = 1
        return counts

    def close(self) -> None:
        """Unbind the adapters. Idempotent, and never raises.

        THE UPLOAD HAPPENS OUTSIDE THE LOCK, deliberately. ``turn()`` takes the
        same lock to advance its prune counter, so flushing while holding it
        would park a live user request behind a network call for as long as the
        control plane took to answer -- in a threaded server, on the shutdown
        path, which is exactly when latency is least affordable. The lock is
        held only long enough to claim the close and take the unbind list.

        Claiming ``_closed`` FIRST is what keeps this idempotent across
        threads: a second caller returns immediately rather than flushing
        again. And a double flush would be harmless anyway -- the uploaded
        markers make it find nothing -- so the ordering costs nothing to get
        right.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            unbinds, self._unbinds = self._unbinds, []

        # LAST CHANCE, and the only place unanswered turns are sent: nothing
        # more is coming after this. Before unbinding, so a turn completed
        # during shutdown is still on disk to find.
        #
        # It goes through `_flush_locked`, so it re-asks for consent while the
        # answer is still no -- for a low-traffic agent that never reached the
        # cadence, this shutdown is the only chance a grant made an hour ago
        # ever gets to be noticed.
        #
        # BOUNDED, because this also runs from the ``atexit`` hook. A periodic
        # flush may be in flight holding ``_flush_lock``, and each of its
        # requests can take the client's full 30s timeout; waiting on it
        # without a bound would let an unresponsive control plane hold a
        # customer's process open at shutdown. That is a worse failure than the
        # one it avoids: a turn we do not send now is still on disk, and the
        # next process that attaches uploads it. So give the running flush a
        # short grace period and then let go.
        if self._flush_lock.acquire(timeout=_CLOSE_FLUSH_GRACE_S):
            try:
                self._flush_locked(include_unanswered=True)
            except Exception:
                # Shutdown must not raise. The turns remain on disk.
                pass
            finally:
                self._flush_lock.release()
        for u in unbinds:
            try:
                u()
            except Exception:
                # Unbinding is cleanup. A framework that changed shape under
                # us must not take the customer's shutdown with it.
                pass

    def __enter__(self) -> Attachment:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _reconcile_attached_execution(
    entrypoint: Any,
    *,
    input_shape: InputShape,
    method: str | None,
    adapters_bound: tuple[str, ...],
    trajectory_root: str,
    attested_execution_components: Mapping[str, str],
    discovery_enabled: bool = True,
) -> tuple[Any, dict[str, str] | None]:
    """Use the rollout reconciler without claiming it owns production calls.

    ``discovery_enabled`` is the customer's consent, asked the way serve()
    asks it: when off, the entrypoint is still built and validated, but the
    agent is not introspected and no execution identity is derived from it.
    """
    if isinstance(entrypoint, (list, tuple)):
        raise ConfigurationError(
            "attach(execution_entrypoint=) requires an import spec or the live "
            "Python agent object; attach does not launch or observe a command process"
        )

    # Imports stay local: attachment-only users who do not request execution
    # identity should not pay for discovery/runner modules on their hot path.
    from . import __version__
    from .contract import CONTRACT_VERSION
    from .introspection import introspect
    from .runner import RolloutRunner

    runner = RolloutRunner(
        entrypoint,
        transport=None,  # type: ignore[arg-type]  # identity-only; never runs
        input_shape=input_shape,
        artifacts_dir=trajectory_root,
        isolation="inprocess",
        concurrency=1,
        method=method,
        adapters=list(adapters_bound),
        reports_tool_calls=bool(adapters_bound),
        attested_execution_components=attested_execution_components,
    )
    described = introspect(runner.resolved.fn) if discovery_enabled else None
    digest = runner.configure_execution_identity(
        described,
        discovery_enabled=discovery_enabled,
        sdk=f"agent-flywheel-python/{__version__}",
        contract_version=CONTRACT_VERSION,
    )
    return described, ({"execution_sha256": digest} if digest is not None else None)


def attach(
    *,
    agent_id: str,
    trajectory_root: str,
    control_plane_url: str | None = None,
    api_key: str | None = None,
    adapters: list[str] | str | None = "auto",
    register: bool = True,
    retention_turns: int | None = None,
    execution_entrypoint: Any | None = None,
    execution_input_shape: InputShape = "text",
    execution_method: str | None = None,
    attested_execution_components: Mapping[str, str] | None = None,
    describe: Describe | None = None,
) -> Attachment:
    """Bind capture in THIS process, for an agent already serving users.

    One call at startup::

        import agent_flywheel as fw

        px = fw.attach(agent_id="support-agent", trajectory_root="/var/lib/percepteye")

        with px.turn() as turn_id:
            reply = agent.invoke(user_text)      # your agent, unmodified

    ``trajectory_root`` is REQUIRED and that is deliberate. Every captured turn
    lands there as a directory, and it is the customer's copy: uploading does
    not delete it, and without ``retention_turns`` nothing does. Naming the
    directory is how retention becomes a decision somebody made rather than a
    surprise.

    UPLOADING IS AUTOMATIC AND CONSENTED. When the control plane says this
    agent may capture -- opt-in per agent, because these are the customer's END
    USERS' messages -- answered turns are sent every ``_FLUSH_EVERY`` turns on
    a background thread, and an ``atexit`` hook sends the rest at shutdown.
    Until consent is given, and whenever the control plane cannot be reached,
    they simply accumulate. :attr:`Attachment.may_upload` is the live answer
    and :attr:`Attachment.upload_refusal` is the sentence explaining a False.

    ``retention_turns=N`` keeps only the N most recently written turns, pruning
    on an interval so it costs one directory scan per 256 turns rather than one
    per turn. Unset, NOTHING is deleted and the trajectory grows without bound:
    roughly 2.4 GB and a million inodes per million recording turns, and more
    if tool outputs are large. That default is deliberate -- the trajectory is
    your data in a directory you named, and a library that silently deleted it
    would be worse than one that grows -- but it is a decision, so make it.

    ``register=False`` skips the control-plane call entirely, for a process
    that should make no outbound connection at all.

    ``execution_entrypoint`` optionally names the exact Python agent this
    attachment observes. Together with ``attested_execution_components`` it
    lets the SDK run the same fail-closed discovery and canonical reconciliation
    used for rollouts, then persist ``agent_fingerprint.execution_sha256`` on
    each production turn and its OpenTelemetry spans. Components are opaque
    lowercase SHA-256 attestations of implementation/state discovery cannot
    observe. A missing, incomplete, or lossy description emits no identity;
    components alone never do. Command processes are refused because attach()
    neither launches nor encloses their execution. The agent keeps serving in
    every case except an invalid explicit identity configuration.

    ``describe`` is the agent's own description, exactly as for ``serve()``
    (a mapping of ``system_prompt`` / ``tools`` / ``model``, or a zero-argument
    callable returning one). With ``execution_entrypoint`` it wins over
    reflection where given; without one it is the whole description this
    registration sends -- the way a production-only agent tells the control
    plane its prompt and tools. It is the customer's word, not an observation,
    so it anchors no execution identity; ``PERCEPTEYE_INTROSPECT=0`` wins over
    it, with a warning.

    Nothing here raises on a control-plane failure. ``agent_id`` missing or
    ``trajectory_root`` missing raise -- those are configuration mistakes in
    front of the person making them -- but an unreachable control plane yields
    an attachment with ``policy == {}``, because an agent must keep serving its
    users whatever we happen to think of our own availability.
    """
    if _production_agent_identifier(agent_id) is None:
        raise ConfigurationError(
            "attach() requires agent_id to match the production identifier "
            "grammar: non-empty, control-free, without edge whitespace, and "
            "without path separators or dot segments, with at most "
            f"{_PRODUCTION_IDENTIFIER_MAX_CHARS} characters"
        )
    if not trajectory_root or not isinstance(trajectory_root, str):
        raise ConfigurationError(
            "attach() requires trajectory_root -- the directory captured turns "
            "accumulate in. It is not defaulted because nothing deletes them "
            "yet, and where a customer's data lands is not ours to assume."
        )
    normalized_components = normalize_attested_components(
        attested_execution_components
    )
    if "execution_sha256" in normalized_components:
        raise ConfigurationError(
            "attested execution components must not author execution_sha256"
        )
    if execution_entrypoint is None and normalized_components:
        raise ConfigurationError(
            "attested_execution_components requires execution_entrypoint so the "
            "SDK can reconcile evidence with the observed agent"
        )
    try:
        os.makedirs(trajectory_root, exist_ok=True)
    except OSError as exc:
        raise ConfigurationError(
            f"attach() cannot use trajectory_root={trajectory_root!r}: {exc}"
        ) from exc

    # ── adapters, bound with NO endpoint ────────────────────────────────────
    # The empty strings are load-bearing, not placeholders. bind_selected
    # forwards them to each adapter's bind(), and both built-ins treat "no
    # endpoint" as "do not touch routing": LangChainAdapter._route returns at
    # once when self.llm is None, and the ADK adapter never reads the URL. So
    # this installs observation and cannot re-point a single production call.
    unbinds: list[Any] = []
    bound: tuple[str, ...] = ()
    try:
        from .binding import bind_selected

        report, unbinds = bind_selected(adapters, llm_url="", api_key="")
        bound = tuple(report.bound or ())
    except Exception:
        # A framework that cannot be bound is a smaller loss than a process
        # that will not start. serve() raises here instead, and correctly: a
        # rollout with no capture is a wasted rollout, while a production turn
        # with no capture is still a served user.
        unbinds, bound = [], ()

    # THE CUSTOMER'S CONSENT, read once and asked the way serve() asks it.
    # With an entrypoint this used to be ignored: the agent was introspected
    # and its description -- system prompt included -- registered under
    # PERCEPTEYE_INTROSPECT=0, and an unrecognised agent's null then recorded
    # `unreadable` rather than the opt-out.
    from . import _env_true
    from . import declaration as _declaration

    introspection_consented = _env_true("PERCEPTEYE_INTROSPECT", default=True)
    if describe is not None and not introspection_consented:
        warnings.warn(
            "PERCEPTEYE_INTROSPECT is off, and it wins: the description "
            "declared with describe= is not sent.",
            RuntimeWarning, stacklevel=2,
        )
    # The customer's own declaration (see declaration.py), validated at
    # startup like everything else here.
    declared = _declaration.resolve(describe) if introspection_consented else None
    described = None
    agent_fingerprint: dict[str, str] | None = None
    if execution_entrypoint is not None:
        try:
            described, agent_fingerprint = _reconcile_attached_execution(
                execution_entrypoint,
                input_shape=execution_input_shape,
                method=execution_method,
                adapters_bound=bound,
                trajectory_root=trajectory_root,
                attested_execution_components=normalized_components,
                discovery_enabled=introspection_consented,
            )
        except Exception:
            # Unlike observation failures inside introspect(), invalid entrypoint
            # or attestation configuration is a caller mistake. Undo the global
            # adapter binding before surfacing it.
            for unbind in unbinds:
                try:
                    unbind()
                except Exception:
                    pass
            raise
    if declared is not None:
        # What the customer declared wins where they gave it; with no
        # entrypoint it is the whole description. And it is their word, not an
        # observation of what runs, so it anchors no execution identity.
        described = _declaration.apply(declared, reflected=described,
                                       agent_name=agent_id)
        agent_fingerprint = None

    # ── the control plane, entirely optional ────────────────────────────────
    policy: dict[str, Any] = {}
    register_error: Exception | None = None
    url = control_plane_url or os.environ.get(ENV_CONTROL_PLANE_URL)
    key = api_key or os.environ.get(ENV_API_KEY)
    may_upload = False
    # NAMED, not None. `upload_refusal` is documented as "the sentence
    # explaining a False", and the ways to get here without ever asking the
    # control plane are distinct problems with distinct remedies. One message
    # covering all of them -- or, as it was, no message at all -- sends an
    # operator to check the wrong thing.
    #
    # WHICH VARIABLE is asked through the same helper `policy_source` and the
    # CLI ask, so "configured" means one thing across the package.
    cause = missing_credentials_cause(url, key)
    if not register:
        upload_refusal: str | None = (
            "attach(register=False) was asked for, so the control plane was "
            "never asked whether this agent may upload production turns. "
            "Turns are still captured locally."
        )
    elif cause:
        upload_refusal = (
            f"{cause}, so the control plane was never asked whether this agent "
            f"may upload production turns. Turns are still captured locally."
        )
    else:
        # A control plane we are ABOUT to ask. Anything that stops us reaching
        # its answer -- a refused registration, an unreachable host, a client
        # that would not build -- leaves this sentence standing, which is the
        # true one for every such path. The successful path overwrites it.
        upload_refusal = capture_verdict(None).reason
    if register and url and key:
        client: ControlPlaneClient | None = None
        try:
            from . import __version__
            from ._http import ControlPlaneClient
            from .contract import CONTRACT_VERSION
            from .transport import AttachTransport

            client = ControlPlaneClient(url, key)
            tr = AttachTransport(client, agent_id)
            try:
                registration = {
                    "agent_id": agent_id,
                    # This process serves users; it does not contribute capacity.
                    # `serve()` sends "rollout". Sending the same value from
                    # here made the field carry zero bits -- intake could not
                    # tell a training donor from a production reader, which is
                    # the one question a `mode` on a registration exists to
                    # answer.
                    "mode": "production",
                    "sdk": f"agent-flywheel-python/{__version__}",
                    "contract_version": CONTRACT_VERSION,
                    "reports_tool_calls": bool(bound),
                    "isolation": "inprocess",
                    "concurrency": 1,
                    "input_shape": (
                        execution_input_shape
                        if execution_entrypoint is not None
                        else "text"
                    ),
                    "entrypoint": "attach",
                }
                if introspection_consented:
                    # A description when there is an agent to read, else
                    # `null` -- "not described". Never an absent key: that is
                    # the customer opting out, which the control plane records
                    # as `disabled`, the destructive state that stops the
                    # description a serve() registration stored from driving
                    # workflow generation. The Node plugin sends null for the
                    # same reason. Under PERCEPTEYE_INTROSPECT=0 the key IS
                    # omitted: that opt-out is real.
                    registration["discovered_agent"] = (
                        described.to_wire() if described is not None else None
                    )
                reg = tr.register(registration)
                # WHAT THE SERVER SAID, not what we assume, and asked through
                # the ONE predicate `_recheck_capture` asks -- see
                # :func:`capture_verdict`. This site used to ask
                # `bool(cap.get("turns"))` while the late-consent path three
                # hundred lines up asked `is True`, so the same registration
                # payload could grant consent here and refuse it there.
                verdict = capture_verdict(reg)
                may_upload = verdict.may_upload
                upload_refusal = verdict.reason
            except Exception as exc:
                # A REFUSED registration must not cost the last mile.
                #
                # `policy_current` is a bare GET scoped by API key -- it does
                # not put agent_id on the wire and does not require a
                # registration to have landed -- so there is no reason a
                # rejected register should also cost this process its trained
                # model. It used to: one `except` covered both calls, so
                # anything intake disliked about the payload silently demoted
                # production to the untrained policy.
                register_error = exc
            policy = tr.policy_current() or {}
        except Exception:
            policy = {}
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    # Warned LAST, outside the try that owns the network calls, so raising
    # here cannot cost the policy read -- and guarded, because under
    # `-W error` a warning IS an exception and this function must never fail a
    # customer's startup. Under that setting the warning is lost; the policy,
    # which is the part that matters, is not.
    #
    # Only when the policy read SUCCEEDED, which is what distinguishes the two
    # causes without parsing an error string: the network demonstrably works,
    # so intake refused this payload specifically. A register that failed
    # because the control plane was unreachable is left silent, exactly as it
    # was before -- an offline laptop should not print a warning about intake.
    if register_error is not None and policy:
        try:
            warnings.warn(
                f"[percepteye] the control plane refused this registration for "
                f"agent_id={agent_id!r}: {type(register_error).__name__}: "
                f"{register_error}. The policy read succeeded, so capture and "
                f"the trained policy are both unaffected. The likely cause is "
                f"an intake that has not been updated for this SDK version. "
                f"Capture continues locally either way; turns are uploaded "
                f"only when the control plane says this agent is enabled for "
                f"it.",
                RuntimeWarning,
                stacklevel=2,
            )
        except Exception:  # -W error turns the line above into a raise
            pass

    px = Attachment(
        agent_id,
        trajectory_root=trajectory_root,
        policy=policy,
        unbinds=unbinds,
        adapters_bound=bound,
        retention_turns=retention_turns,
        control_plane_url=url,
        api_key=key,
        may_upload=may_upload,
        upload_refusal=upload_refusal,
        agent_fingerprint=agent_fingerprint,
    )
    if may_upload:
        # Through the same seam late consent goes through, so the hook is
        # registered by exactly one rule: "may_upload just became True".
        px._grant_upload()
    return px


def _flush_at_exit(attachment: Attachment) -> None:
    """Send whatever is still on disk when the process shuts down.

    THE OTHER HALF OF THE PRODUCER. The periodic cadence in :meth:`turn` holds
    back turns that have no answer yet, because one may still be coming, and it
    only runs while requests are arriving. Neither is true at shutdown: nothing
    more is coming for any turn, and no further turn will trigger a cadence. So
    the last thing the process does is a flush that includes the unanswered
    ones -- exactly what :meth:`close` does, which is what a caller using
    ``with attach(...)`` already got and a caller holding the attachment for
    the life of the process never did.

    A WEAK REFERENCE, so registering this cannot be what keeps a customer's
    Attachment alive for the life of the interpreter. If the object is already
    gone, so is anything it would have sent -- and its turns are still on disk
    for the next process to pick up.

    Registered only when the control plane has said this agent may upload --
    AT STARTUP OR LATER, from :meth:`Attachment._grant_upload`, which is the
    single place that answer turns True. An agent without consent would do
    nothing here but cost a hook, and would make every process that merely
    imported the SDK look like one that talks to us at exit; an agent granted
    consent an hour in needs this hook exactly as much as one granted it at
    startup, and registering only on the startup verdict left that process
    with no shutdown flush at all.

    ``close()`` is idempotent, so a caller who closes explicitly, uses the
    context manager, AND gets this hook flushes exactly once.
    """
    ref = weakref.ref(attachment)

    def _drain() -> None:
        px = ref()
        if px is None:
            return
        try:
            px.close()
        except Exception:
            # An interpreter that is shutting down is a hostile place to do
            # I/O. Losing the final batch costs the observation; raising here
            # would print a traceback under our name as the customer's process
            # exits, for turns that are still safely on disk.
            pass

    atexit.register(_drain)
