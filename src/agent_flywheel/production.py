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
import hashlib
import json
import os
import shutil
import threading
import time
import uuid
import warnings
import weakref
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

# NOT imported at module scope. `_http` pulls in httpx, which is 83-87% of
# this package's import time and ~88% of its resident memory -- and a process
# that only reads the promoted policy, or only captures locally, never opens a
# socket at all. Deferring it means those customers stop paying for a transport
# they do not use. `policy.py` has always done this; these were the paths that
# forced it anyway.
if TYPE_CHECKING:
    from ._http import ControlPlaneClient
    from .contract import RulebookReceipt
from . import declaration as _declaration
from .contract import (
    _PRODUCTION_IDENTIFIER_MAX_CHARS,
    _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS,
    _PRODUCTION_TURN_BATCH_MAX_ITEMS,
    _TURN_FEEDBACK_BATCH_MAX_ITEMS,
    _production_agent_identifier,
    _production_identifier,
    _production_turn_identifier,
    names_only_an_episode,
    production_outcome,
    rfc3339_unix_nanos,
    turn_feedback,
    turn_feedback_on_a_span,
)
from .control_plane import resolve_control_plane_url
from .declaration import Describe
from .entrypoint import InputShape
from .errors import (
    ConfigurationError,
    InvalidFeedback,
    InvalidOutcome,
    TransportError,
)
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
    read_model_boundary,
    read_task_id,
    read_turn_text,
    record_turn_text,
)
from .policy import missing_credentials_cause
from .transport import (
    RULEBOOK_SETTLE_GRACE_S,
    RulebookUpload,
    declare_rulebook_in_background,
    declared_rulebook,
)

__all__ = ["Attachment", "CaptureVerdict", "attach", "capture_verdict"]

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
    #: THE SECOND CONSENT TIER: may a turn ALSO carry what its model calls
    #: saw and said -- the system prompt, the whole conversation history, the
    #: tool definitions and every tool result -- as its ``genai_trace``?
    #: Separate from ``may_upload`` because it is a different grant: the end
    #: user's message and the answer are one thing, the customer's prompt and
    #: everything their tools returned are another. True only where
    #: ``may_upload`` is, since the trace rides on the turn. Off by default,
    #: like the first tier: the control plane's ``capture.model_content`` must
    #: be exactly ``true``.
    may_capture_model_content: bool = False
    #: The sentence explaining a ``False`` above; ``None`` exactly when it is
    #: ``True``.
    model_content_reason: str | None = (
        "the control plane did not say this agent may upload model content.")


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

    THE SECOND TIER, ``capture.model_content``, is read by the same rules: the
    exact value, only where turns may flow, and the server's own sentence
    (``model_content_reason``) when it withheld it. It decides whether the
    adapters RECORD what a model call saw at all -- not merely whether it is
    uploaded -- because that content is the customer's own prompt and
    everything their tools returned, and recording it before anyone consented
    would make a later grant retroactive.
    """
    cap = body.get("capture") if isinstance(body, Mapping) else None
    if not isinstance(cap, Mapping):
        unanswered = (
            "the control plane did not answer whether this agent may "
            "upload production turns, so nothing is uploaded. Turns are "
            "still captured locally."
        )
        return CaptureVerdict(
            may_upload=False, reason=unanswered, answered=False,
            model_content_reason=unanswered,
        )
    if cap.get("turns") is True:
        # THE SECOND TIER, asked the way the first is: the exact value, and
        # never granted by anything the server did not say. `"true"` is not
        # consent here either.
        if cap.get("model_content") is True:
            return CaptureVerdict(
                may_upload=True, reason=None,
                may_capture_model_content=True, model_content_reason=None,
            )
        served = cap.get("model_content_reason")
        return CaptureVerdict(
            may_upload=True, reason=None,
            model_content_reason=(
                served.strip()
                if isinstance(served, str) and served.strip()
                else _MODEL_CONTENT_NOT_ENABLED
            ),
        )
    served = cap.get("reason")
    # The server's own sentence when it sent one -- it names the switch and
    # who can flip it, which anything written here could only paraphrase.
    reason = (
        served.strip()
        if isinstance(served, str) and served.strip()
        else "production capture is not enabled for this agent."
    )
    # A trace rides on a turn, so a refused turn refuses its trace for the
    # same reason -- and says THAT reason, not a second one.
    return CaptureVerdict(may_upload=False, reason=reason,
                          model_content_reason=reason)


#: How the turn intake names a trace it refused for want of model-content
#: consent (the control plane's ``governance.MODEL_CONTENT_NOT_ENABLED``).
_MODEL_CONTENT_REFUSED = "model_content_capture_not_enabled"

#: Every name the turn intake refuses a trace by that means "stop recording
#: model content": no consent, or a turn attributed to no Product (a shared
#: or undeclared agent, ``turn_genai_trace.GENAI_TRACE_TURN_NOT_ATTRIBUTED``),
#: whose trace the intake never keeps. The capture verdict withholds the tier
#: for both, and a re-read picks it back up when either is fixed. A
#: non-conformant trace is not here: that one is this renderer's bug, not a
#: decision anyone made about this agent.
_MODEL_CONTENT_REFUSALS = frozenset({
    _MODEL_CONTENT_REFUSED,
    "genai_trace_turn_not_attributed_to_a_product",
})

#: The model-content refusal when the control plane granted turns, withheld
#: model content and sent no sentence of its own.
_MODEL_CONTENT_NOT_ENABLED = (
    "model-content capture is not enabled for this agent, so the system "
    "prompt, the conversation history and the tool results its model calls "
    "saw are neither recorded nor uploaded."
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

#: What :attr:`Attachment.rulebook_refusal` says while the control plane has
#: not yet answered ``attach(rules=...)``.
_RULEBOOK_PENDING = (
    "the control plane has not answered the rules passed as rules= yet; "
    "Attachment.wait_for_rulebook() waits for its answer."
)


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


#: A reported outcome waiting to be uploaded is ONE FILE in the trajectory
#: root, ``.outcome.<digest of its id>.json``. A file and not a directory for
#: the reason ``Attachment._UPLOADED_INDEX`` is: every directory under the
#: root is read as a turn. One file per outcome, written whole and renamed
#: into place, so two processes sharing a root never rewrite each other's
#: pending outcomes; the Node SDK writes the same names.
_OUTCOME_FILE_PREFIX = ".outcome."
_OUTCOME_FILE_SUFFIX = ".json"
#: Per-turn feedback waits the same way, under its own name:
#: ``.feedback.<digest of its id>.json``. The outcomes' week and count bounds
#: apply to it.
_FEEDBACK_FILE_PREFIX = ".feedback."
#: How long an outcome that could not be delivered is kept. Bounded, because
#: an outbox nothing can drain (a control plane that has no ``/outcomes``
#: route yet) must not grow for the life of a deployment; a week, the Node
#: SDK's rollout outbox bound.
_OUTCOME_MAX_AGE_S = 7 * 24 * 3600.0
#: How many outcomes are kept while consent is still UNKNOWN (the verdict read
#: has not been answered yet). Bounded by count as well as age, because a week
#: of a busy agent's reports is not a small outbox; past it the oldest go, out
#: loud. The Node SDK's ``OUTCOME_MAX_KEPT_UNANSWERED``.
_OUTCOME_MAX_KEPT_UNANSWERED = 1000
#: How many requests one flush may spend splitting refused ``/outcomes``
#: batches. Isolating one bad record of 64 costs 12; the bound stops a refusal
#: the split did not recognise as whole-batch from costing two requests per
#: outcome, each charged against the control plane's rate limit. The Node
#: SDK's ``OUTCOME_SPLIT_MAX_REQUESTS``.
_OUTCOME_SPLIT_MAX_REQUESTS = 32


def _outcome_file(root: str | os.PathLike[str], outcome_id: str) -> Path:
    """Where the outcome with this id waits. A digest of the id, never the id:
    a caller's id may be 255 characters, and a file name may not."""
    digest = hashlib.sha256(outcome_id.encode("utf-8")).hexdigest()[:32]
    return Path(root) / (
        f"{_OUTCOME_FILE_PREFIX}{digest}{_OUTCOME_FILE_SUFFIX}")


def _feedback_file(root: str | os.PathLike[str], feedback_id: str) -> Path:
    """Where the feedback with this id waits (:func:`_outcome_file`)."""
    digest = hashlib.sha256(feedback_id.encode("utf-8")).hexdigest()[:32]
    return Path(root) / (
        f"{_FEEDBACK_FILE_PREFIX}{digest}{_OUTCOME_FILE_SUFFIX}")


def _rfc3339(value: Any, refusal: type[Exception] = InvalidOutcome) -> Any:
    """``occurred_at`` as the wire spells it. ``None`` is now; an aware
    ``datetime`` becomes UTC with milliseconds and ``Z`` (what the Node SDK's
    ``Date.toISOString()`` writes); a string goes through untouched for the
    validator to judge. A naive ``datetime`` is refused, as ``refusal``: it
    names no instant.
    """
    if value is None:
        value = datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise refusal(
                "occurred_at must carry a timezone: a naive datetime does "
                "not say when")
        return value.astimezone(timezone.utc).isoformat(
            timespec="milliseconds").replace("+00:00", "Z")
    return value


#: How ``_http.ControlPlaneClient.request`` spells a terminal 403. It formats
#: every 4xx that is not 408/429 as ``"{method} {path} -> HTTP {code}: {detail}"``
#: and raises ``TransportError``; the status is not carried structurally, so
#: this is the seam between the two. If that format ever changes, the failure
#: is a refusal read as a blip again -- which is what
#: ``test_a_revoked_agent_stops_uploading`` is there to catch.
_REFUSED_MARKER = "-> HTTP 403"


#: A 4xx that is about the ROUTE or the KEY, never the records in an
#: ``/outcomes`` batch: an absent route (404, 405 -- a control plane older
#: than ``POST /outcomes``), a key it does not accept (401), a gone resource
#: (410). The Node SDK's ``outcomeRouteRefused``.
_OUTCOME_ROUTE_REFUSALS = frozenset({401, 404, 405, 410})
#: 4xx that are neither: 403 is the consent refusal
#: (:func:`_capture_refusal`); 408 and 429 are blips the client retried.
_OUTCOME_NOT_A_BATCH_REFUSAL = frozenset({403, 408, 429})


def _outcome_route_refused(status: Any) -> bool:
    """Whether an ``/outcomes`` answer refused the route or the key, not the
    records: sending stops for this flush and the outcomes stay."""
    return status in _OUTCOME_ROUTE_REFUSALS


#: The 4xx about the BODY'S SIZE (a proxy's limit, or the control plane's):
#: never a refusal of what the batch says, so a smaller part may pass.
_OUTCOME_TOO_LARGE = 413


def _outcome_refused_for_size(status: Any) -> bool:
    """Whether an ``/outcomes`` refusal was about how LARGE the body was.
    Both halves of a batch too large for a proxy answer it in the same
    words, so it is never read as a refusal of the batch: the split goes on,
    and a record that draws it alone is too large on its own. The Node SDK's
    ``outcomeRefusedForSize``."""
    return status == _OUTCOME_TOO_LARGE


def _outcome_wire_bytes(wire: Mapping[str, Any]) -> int:
    """A record's size as the wire spells it: compact JSON, UTF-8 -- the
    Node SDK's ``Buffer.byteLength(JSON.stringify(wire))``."""
    return len(json.dumps(wire, separators=(",", ":"),
                          ensure_ascii=False).encode("utf-8"))


def _outcome_batch_refused(status: Any) -> bool:
    """Whether an ``/outcomes`` answer refused THIS BATCH: any other 4xx
    (400, 409, 413, 422 ...). The batch is split to find the records it
    refused, or -- both halves refused alike for what they say, never for
    their size (:func:`_outcome_refused_for_size`) -- kept as a refusal of
    the batch itself (``Attachment._flush_outcomes``). The Node SDK's
    ``outcomeBatchRefused``."""
    return (isinstance(status, int) and not isinstance(status, bool)
            and 400 <= status < 500
            and status not in _OUTCOME_ROUTE_REFUSALS
            and status not in _OUTCOME_NOT_A_BATCH_REFUSAL)


#: A 4xx that says THE ROUTE IS NOT THERE: a control plane older than it
#: answers 404, or 405 where another method has the path. The Node SDK's
#: ``routeAbsent``.
_ROUTE_ABSENT = frozenset({404, 405})


def _unanswered_outcomes_dropped(dropped: int) -> str:
    """The one sentence for outcomes dropped by the count bound while the
    consent question is unanswered -- its cause, which is not a refusal: the
    control plane has not answered (it could not be reached, or did not say).
    The Node SDK's ``unansweredOutcomesDropped``."""
    return (
        f"agent-flywheel: {dropped} reported outcome(s) were dropped, oldest "
        "first: the control plane has not answered whether this agent may "
        "upload production traffic (it could not be reached, or did not say), "
        f"and at most {_OUTCOME_MAX_KEPT_UNANSWERED} outcomes that wait on "
        "that answer are kept until it does.")


def _route_absent(status: Any) -> bool:
    """Whether an answer says the route does not exist on this control
    plane. Every other 4xx is about the key or the records, and must never
    be read as this: per-turn feedback deletes its queue on it
    (``Attachment._flush_feedback``)."""
    return status in _ROUTE_ABSENT


class _ReportKind(NamedTuple):
    """What ``Attachment._send_reports`` needs to know about one of the two
    report routes: how to say it, where the reply lists its records, and how
    to send a batch."""

    #: "reported outcomes"
    many: str
    #: "reported outcome(s)"
    counted: str
    #: "an outcome"
    one: str
    #: The key of the reply's per-record list.
    results: str
    #: The key of a record's id, in the wire and in the reply.
    id: str
    #: The route's batch limit.
    batch_max: int
    #: The ``AttachTransport`` method that sends one batch.
    send: str


_OUTCOMES = _ReportKind(
    "reported outcomes", "reported outcome(s)", "an outcome",
    "outcomes", "outcome_id", _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS,
    "report_outcomes")
_FEEDBACK = _ReportKind(
    "per-turn feedback reports", "per-turn feedback report(s)",
    "a feedback report", "feedback", "feedback_id",
    _TURN_FEEDBACK_BATCH_MAX_ITEMS, "report_turn_feedback")


def _reply_records(reply: Any, kind: _ReportKind) -> list[Mapping[str, Any]]:
    """The per-record answers of an acknowledged batch; ``[]`` for a reply
    that carries none. Never raises."""
    results = reply.get(kind.results) if isinstance(reply, Mapping) else None
    if not isinstance(results, list):
        return []
    return [r for r in results if isinstance(r, Mapping)]


def _refused_by_name(reply: Any, kind: _ReportKind) -> list[str]:
    """Each record of an acknowledged batch the control plane did NOT record
    (``accepted: false``), as ``<its id> (<the control plane's word>)``. The
    Node SDK's ``refusedByName``. Never raises."""
    return [
        f"{r.get(kind.id) or 'unnamed'} ({r.get('error') or 'refused'})"
        for r in _reply_records(reply, kind) if r.get("accepted") is False
    ]


def _notes_not_stored(reply: Any, kind: _ReportKind) -> list[str]:
    """Each record the control plane recorded WITHOUT its note
    (``note.stored: false``), as ``<its id> (<the control plane's word>)``.
    The Node SDK's ``notesNotStored``. Never raises."""
    out = []
    for r in _reply_records(reply, kind):
        note = r.get("note")
        if (r.get("accepted") is True and isinstance(note, Mapping)
                and note.get("stored") is False):
            out.append(f"{r.get(kind.id) or 'unnamed'} "
                       f"({note.get('error') or 'not stored'})")
    return out


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
    and the turn recorded no model boundary either -- there is nothing to
    upload -- and ``prune`` (through ``_holds_a_record``, which asks the same
    two questions) uses it to tell a deleted directory that held DATA nobody
    sent from one that could never have been sent at all. Asked twice with
    two spellings, the second asker charges an
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
        consent_refused: bool = False,
        agent_fingerprint: Mapping[str, str] | None = None,
        may_capture_model_content: bool = False,
        model_content_refusal: str | None = None,
        rulebook: RulebookReceipt | None = None,
        rulebook_refusal: str | None = None,
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
        #: Whether the control plane SAID NO -- an answered verdict, or the
        #: intake's 403 -- as opposed to not having answered yet (a
        #: registration that failed on a blip). ``may_upload`` is False in
        #: both, and they are different facts: an outcome reported while the
        #: answer is UNKNOWN is kept for the flush that re-asks, and only an
        #: explicit no makes :meth:`report_outcome` a no-op.
        self._consent_refused = bool(consent_refused and not may_upload)
        #: The cause of the last whole-batch refusal said out loud, per kind
        #: of report (:meth:`_warn_whole_batch_refusal`), so it is said once.
        self._batch_refusal_said: dict[str, str] = {}
        #: The control plane refused an outcome that names only an episode
        #: while production capture is not enabled -- one that predates the
        #: episode join. Said once; no more are sent on those terms by this
        #: process (:meth:`_refuse_episode_outcomes`).
        self._episode_outcomes_refused = False
        #: The control plane will not take per-turn feedback from this
        #: process: it has no such route (404, 405 -- what was waiting was
        #: dropped) or it refused this key (403 -- what was waiting stays on
        #: disk). Said once; after it :meth:`report_turn_feedback` is a no-op.
        self._turn_feedback_refused = False
        #: THE SECOND CONSENT TIER, as the control plane last said it: may a
        #: turn also carry what its model calls saw and said -- the system
        #: prompt, the whole conversation history, the tool definitions and
        #: every tool result -- as one OTel GenAI trace? It decides whether the
        #: adapters RECORD that content at all, turn by turn, and whether a
        #: recorded turn uploads it. Off by default, like the first tier.
        self.may_capture_model_content = bool(
            may_capture_model_content and may_upload)
        #: Why not, in a sentence, when it is False.
        self.model_content_refusal = (
            None if self.may_capture_model_content
            else (model_content_refusal or upload_refusal)
        )
        #: What ``attach(rules=...)`` came to, as ONE value -- read through
        #: :attr:`rulebook` and :attr:`rulebook_refusal`, and replaced whole
        #: when the declaration settles in the background, so neither
        #: attribute is ever read half-updated against the other.
        self._rulebook_upload = RulebookUpload(rulebook, rulebook_refusal)
        #: The declaration in flight, or ``None`` -- see
        #: :meth:`wait_for_rulebook`.
        self._rulebook_declaration: threading.Thread | None = None
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

    @property
    def rulebook(self) -> RulebookReceipt | None:
        """What the control plane did with ``attach(rules=...)``: its receipt
        (``status``, ``version``, ``content_sha256``, ``rule_count``), or
        ``None`` when no rules were passed, none registered, or the control
        plane has not answered yet (:meth:`wait_for_rulebook`)."""
        return self._rulebook_upload.receipt

    @property
    def rulebook_refusal(self) -> str | None:
        """Why the rules passed as ``rules=`` are not confirmed in force, in a
        sentence -- the control plane's refusal naming its cause, a failure
        that leaves it unknown, a call that never asked it, or an answer still
        on its way. ``None`` when they registered (:attr:`rulebook` says how)
        or none were passed. ``attach()`` declares rules exactly once."""
        return self._rulebook_upload.refusal

    def wait_for_rulebook(self, timeout: float | None = None) -> bool:
        """Wait up to ``timeout`` seconds (``None``: as long as it takes) for
        the control plane's answer to ``attach(rules=...)``.

        True once :attr:`rulebook` and :attr:`rulebook_refusal` say what it
        did -- at once when no declaration is in flight. ``attach()`` returns
        WITHOUT waiting, so that a slow rulebook route costs the rules and
        nothing else; this is for the caller who wants the answer anyway.
        """
        declaration = self._rulebook_declaration
        if declaration is None:
            return True
        declaration.join(timeout)
        return not declaration.is_alive()

    def _declare_rulebook(self, transport: Any, request: Any,
                          client: ControlPlaneClient) -> None:
        """Declare ``attach(rules=...)`` in the background; the one caller is
        :func:`attach`, once this attachment exists to hold the answer.

        The declaration OWNS ``client`` from here and closes it when the
        control plane answers. Until then :attr:`rulebook_refusal` says the
        answer is on its way -- never ``None``, which would read as
        "registered".
        """
        self._rulebook_upload = RulebookUpload(None, _RULEBOOK_PENDING)

        def _settle(upload: RulebookUpload) -> None:
            self._rulebook_upload = upload
            try:
                client.close()
            except Exception:
                pass

        self._rulebook_declaration = declare_rulebook_in_background(
            transport, request, _settle)

    @contextmanager
    def turn(self, *, turn_id: str | None = None,
             conversation_id: str | None = None,
             task_id: str | None = None,
             input_text: str | None = None) -> Iterator[str]:
        """Scope one user turn. Tool calls inside it land in its own trajectory.

        Yields the turn id, which the caller should keep: it is the join key
        for any outcome that arrives later than the turn does -- the draft was
        kept an hour afterwards, the pull request merged the next morning --
        and :meth:`report_outcome` is how that outcome is reported.

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
        passed by the caller: they are the END USER's message and the answer
        the customer's code returned, which no adapter can tell apart from the
        rest of a model call. Without them a turn records its tool calls and
        nothing else, which is exactly what it did before -- and is why tool
        outcomes stay zero-instrumentation while turn text is opt-in.

        WHAT THE MODEL SAW is a different fact and a different consent. When
        the control plane grants this agent the model-content tier
        (:attr:`may_capture_model_content`), the adapters also OBSERVE every
        model call inside this scope -- its system instructions, its whole
        input, the tool definitions it was offered, what it answered, and each
        tool's result -- and the turn uploads that as one OTel GenAI trace
        (:mod:`agent_flywheel.genai_otlp`). Observe-only: no call, response or
        result is changed. Without the grant nothing of it is recorded at all.

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

        # The model-content tier is read HERE, once per turn: a grant or a
        # revocation that lands mid-turn applies from the next turn, so one
        # turn's model boundary is never half recorded.
        with _RolloutScope(tid, str(path), kind="turn",
                           conversation_id=effective_conversation_id,
                           task_id=effective_task_id,
                           execution_sha256=self._execution_sha256,
                           model_content=self.may_capture_model_content):
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

    def report_outcome(
        self,
        outcome: str,
        *,
        turn_id: str | None = None,
        conversation_id: str | None = None,
        value: float | None = None,
        note: str | None = None,
        occurred_at: datetime | str | None = None,
        outcome_id: str | None = None,
        episode_id: str | None = None,
    ) -> str | None:
        """Tell the control plane how a turn, a conversation or an episode
        turned out.

        The outcome that arrives later than the turn does: the case was
        ``"resolved"``, ``"escalated"`` to a person, ``"reopened"`` or
        ``"abandoned"``, or the end user gave a ``"thumbs_up"`` or
        ``"thumbs_down"``. Name what it is the outcome of with the ``turn_id``
        :meth:`turn` yielded, the ``conversation_id`` you passed it, or both.
        Report as many as happen -- resolved today, reopened next week -- each
        is recorded and none replaces another.

        FOR AN EPISODE THE PLATFORM RAN (an environment or simulation
        rollout), pass ``episode_id``: the ``rollout_id`` your entrypoint was
        handed with it (``current_rollout_id()`` inside the rollout). The
        report joins that episode's reward evidence as a weak reading. An
        outcome that names ONLY an episode carries nothing about your end
        users' traffic, so it is queued and sent whether or not production
        capture is enabled; it needs only a control plane to send to. A
        control plane that predates the join refuses it, which is said once,
        and no more are sent on those terms by this process.

        ``value`` is an optional number of your own (an order value, a
        satisfaction score); ``note`` an optional short remark, at most 1,000
        characters, masked for personal data by the control plane before it
        is stored; ``occurred_at`` when it happened (an aware ``datetime`` or
        an RFC 3339 string), now when omitted. ``outcome_id`` is this
        report's idempotency key: leave it out and one is minted; pass your
        own stable id to make your own retries of the same report collapse.

        A WEAK LABEL, not a grade. It is your report of what happened next;
        it does not score the turn and nothing is trained on it directly.

        Returns the outcome's id once it is queued for upload, or ``None``
        when nothing was queued. IT IS A NO-OP ONCE THE CONTROL PLANE HAS SAID
        PRODUCTION CAPTURE IS NOT ENABLED for this agent, or with no control
        plane to ask: an outcome is a fact about your end users' traffic, so
        it is sent on exactly the terms a turn is, and nothing is kept locally
        for an agent that may not send it. While the answer is still UNKNOWN
        (the registration that asks failed on a blip) the outcome is kept, and
        reporting it starts the flush that asks again; it is sent once the
        answer is yes. At most 1,000 are kept while it is unknown; past that
        the oldest are dropped, with a warning.

        Queued, then uploaded off the calling thread with the turns' own
        retry behaviour: the outcome waits on disk under ``trajectory_root``
        until the control plane acknowledges it, survives a restart, and is
        given up on (with a warning) only after a week undelivered, or at
        once, with a warning naming the cause, when the control plane refuses
        that record (a 4xx about the record; the rest of the batch is sent).
        A refusal of the whole batch -- every part of it refused alike --
        keeps the outcomes for a later flush instead.

        Raises :class:`~agent_flywheel.errors.InvalidOutcome` -- before
        anything is stored, capture enabled or not -- for a word outside the
        vocabulary, no turn, conversation or episode, or an id, number, note
        or time the wire cannot carry. Nothing else here raises.
        """
        wire: dict[str, Any] = {
            "outcome_id": (
                f"o_{uuid.uuid4().hex}" if outcome_id is None else outcome_id
            ),
            "outcome": outcome,
            "occurred_at": _rfc3339(occurred_at),
        }
        for name, given in (
            ("turn_id", turn_id), ("conversation_id", conversation_id),
            ("episode_id", episode_id), ("value", value), ("note", note),
        ):
            if given is not None:
                wire[name] = given
        wire = production_outcome(wire)
        if not self.may_upload:
            if not (self._cp_url and self._api_key):
                # Nothing to ask and nothing to send to.
                return None
            if names_only_an_episode(wire):
                # Not about end users' traffic: consent is not its question.
                if self._episode_outcomes_refused:
                    return None
            elif self._consent_refused:
                # An explicit no: nothing will ever say yes to this outcome,
                # so nothing is kept. While the answer is merely UNKNOWN it
                # is kept, and ``_flush_soon`` below re-asks.
                return None
        path = _outcome_file(self.trajectory_root, wire["outcome_id"])
        partial = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(
                json.dumps(wire, sort_keys=True), encoding="utf-8")
            os.replace(partial, path)
        except OSError:
            # Capture costs the observation, never the caller.
            return None
        if not self.may_upload and not names_only_an_episode(wire):
            # Only an outcome consent is the question for waits on its
            # answer; one naming only an episode is sent regardless.
            self._cap_unanswered_outcomes()
        self._flush_soon()
        return str(wire["outcome_id"])

    def report_turn_feedback(
        self,
        signal: str,
        *,
        episode_id: str | None = None,
        turn_index: int | None = None,
        occurred_at: datetime | str | None = None,
        feedback_id: str | None = None,
        turn_id: str | None = None,
    ) -> str | None:
        """Tell the control plane what the user did with one turn of an
        episode -- or with one production turn, named by ``turn_id``.

        ``signal`` is what your application already knows: the user
        ``"accepted"`` the answer, ``"corrected"`` it, ``"repeated"`` the
        request, ``"escalated"`` to a person or ``"abandoned"`` the task.
        ``episode_id`` is the ``rollout_id`` your entrypoint was handed with
        the episode (``current_rollout_id()`` inside it) and ``turn_index``
        the zero-based position of the turn in it. ``occurred_at`` is when it
        happened (an aware ``datetime`` or an RFC 3339 string), now when
        omitted. ``feedback_id`` is this report's idempotency key: leave it
        out and one is minted. Report as many as happen -- corrected, then
        accepted -- each is its own record.

        A WEAK LABEL, and not an outcome: it does not score the turn, it is
        stored apart from reported outcomes, and nothing is trained on it
        directly.

        Returns the feedback's id once it is queued for upload, or ``None``
        when nothing was queued: with no control plane to send to, or once
        the control plane has refused feedback (said once, and nothing more
        is queued or sent by this process: one that predates the route
        answers 404, and what was waiting is dropped; a 403 is its decision
        about this key, and what was waiting stays on disk for its week). A
        report it refuses by name is said by its id and not sent again.
        It needs no production-capture consent: it carries ids, one word and
        a time. Queued as one file under ``trajectory_root`` and uploaded off
        the calling thread; kept across a blip for at most a week.

        A PRODUCTION TURN (``turn_id=``, the id :meth:`turn` yielded, in
        place of ``episode_id`` and ``turn_index``) has no episode for that
        route to join. Its feedback goes where its turn went: an event on the
        turn's trace, through your TracerProvider
        (``tracing.record_turn_feedback_span``: a span linked to the turn's,
        carrying ``percepteye.turn.feedback`` at ``occurred_at``), which an
        OTLP export to the control plane delivers with the turn. It returns
        the feedback's id once that span is recorded, and ``None`` when
        nothing records it (no OpenTelemetry, no provider): nothing is queued
        for it, because no route takes it.

        Raises :class:`~agent_flywheel.errors.InvalidFeedback` -- before
        anything is stored -- for a word outside the vocabulary (the message
        names the allowed ones), an episode id, turn index, turn id or time
        the wire cannot carry, or a turn named both ways. Nothing else here
        raises, and no answer of the control plane ever reaches the caller.
        """
        if turn_id is not None:
            return self._report_turn_feedback_on_its_span(
                signal, turn_id=turn_id, named_an_episode=(
                    episode_id is not None or turn_index is not None),
                occurred_at=occurred_at, feedback_id=feedback_id)
        wire = turn_feedback({
            "feedback_id": (
                f"f_{uuid.uuid4().hex}" if feedback_id is None
                else feedback_id
            ),
            "episode_id": episode_id,
            "turn_index": turn_index,
            "signal": signal,
            "occurred_at": _rfc3339(occurred_at, InvalidFeedback),
        })
        if self._turn_feedback_refused or not (
                self._cp_url and self._api_key):
            return None
        path = _feedback_file(self.trajectory_root, wire["feedback_id"])
        partial = path.with_name(path.name + ".tmp")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            partial.write_text(
                json.dumps(wire, sort_keys=True), encoding="utf-8")
            os.replace(partial, path)
        except OSError:
            # Capture costs the observation, never the caller.
            return None
        self._flush_soon()
        return str(wire["feedback_id"])

    def _report_turn_feedback_on_its_span(
        self, signal: str, *, turn_id: str, named_an_episode: bool,
        occurred_at: datetime | str | None, feedback_id: str | None,
    ) -> str | None:
        """:meth:`report_turn_feedback` for a production turn: validated by
        ``contract.turn_feedback_on_a_span``, emitted by
        ``tracing.record_turn_feedback_span``. The id once recorded, else
        ``None``."""
        if named_an_episode:
            raise InvalidFeedback(
                "feedback names one turn: a production turn's turn_id, or an "
                "episode's episode_id and turn_index, not both")
        wire = turn_feedback_on_a_span({
            "feedback_id": (
                f"f_{uuid.uuid4().hex}" if feedback_id is None
                else feedback_id
            ),
            "turn_id": turn_id,
            "signal": signal,
            "occurred_at": _rfc3339(occurred_at, InvalidFeedback),
        })
        from .tracing import record_turn_feedback_span

        recorded = record_turn_feedback_span(
            turn_id=wire["turn_id"], signal=wire["signal"],
            occurred_at_unix_nano=rfc3339_unix_nanos(wire["occurred_at"]),
            feedback_id=wire["feedback_id"])
        return str(wire["feedback_id"]) if recorded else None

    def _flush_feedback(self) -> dict[str, int]:
        """Upload the per-turn feedback waiting on disk. Caller holds
        ``_flush_lock``. Never raises.

        THE OUTCOMES' RULES, through the outcomes' own sender
        (:meth:`_send_reports`): an acknowledged batch is deleted, and a
        report the control plane refused by name inside it is said by name
        and not sent again; a blip leaves what is waiting for the next flush,
        bounded by the outcomes' week and count; a 403 is its decision (a key
        without the scope, a key bound elsewhere), so sending stops for this
        process and what is waiting stays on disk; a 401 or 410 stops this
        flush and it stays; any other 4xx is about the records, so the batch
        is split and only the report it refused is dropped.

        ONE DIFFERENCE: AN ABSENT ROUTE (:func:`_route_absent`, 404 or 405)
        is a control plane that predates per-turn feedback, and nothing will
        ever take what is waiting. It is dropped, said once, and
        :meth:`report_turn_feedback` is a no-op for the rest of the process.

        Counts, only when feedback was waiting: ``feedback_sent``,
        ``feedback_failed`` (still on disk), ``feedback_dropped``.
        """
        files = self._queued_files(_FEEDBACK_FILE_PREFIX)
        now = time.time()

        def _age(path: Path) -> float:
            try:
                return path.stat().st_mtime
            except OSError:
                return 0.0

        files.sort(key=lambda p: (_age(p), p.name))
        stale = [p for p in files if now - _age(p) > _OUTCOME_MAX_AGE_S]
        excess = len(files) - len(stale) - _OUTCOME_MAX_KEPT_UNANSWERED
        fresh = [p for p in files if p not in stale]
        dropped = 0
        for path in stale + fresh[:max(excess, 0)]:
            try:
                path.unlink()
                dropped += 1
            except OSError:
                pass
        pending: list[tuple[Path, dict[str, Any]]] = []
        for path in fresh[max(excess, 0):]:
            try:
                pending.append((path, turn_feedback(
                    json.loads(path.read_text(encoding="utf-8")))))
            except (OSError, ValueError, InvalidFeedback):
                # Unreadable or not ours to send; one file costs that file.
                continue
        if not pending or self._turn_feedback_refused:
            return {"feedback_dropped": dropped} if dropped else {}
        sent, failed, refused = self._send_reports(
            pending, _FEEDBACK,
            on_refusal=self._refuse_turn_feedback,
            on_absent=self._turn_feedback_absent)
        return {"feedback_sent": sent, "feedback_failed": failed,
                "feedback_dropped": dropped + refused}

    def _refuse_turn_feedback(self, reason: str) -> None:
        """The control plane refused per-turn feedback (a 403: its decision
        about this key). Said ONCE; what is waiting stays on disk for its
        week, and no more is queued or sent by this process. Never raises."""
        if self._turn_feedback_refused:
            return
        self._turn_feedback_refused = True
        self._warn(
            "agent-flywheel: the control plane refused per-turn feedback "
            f"({reason}). What is waiting stays on disk and no more is sent "
            "by this process.")

    def _turn_feedback_absent(self, reason: str) -> None:
        """The control plane has no per-turn feedback route (404, 405): it
        predates it. Said ONCE; the sender has dropped what was waiting, and
        no more is queued or sent by this process. Never raises."""
        if self._turn_feedback_refused:
            return
        self._turn_feedback_refused = True
        self._warn(
            "agent-flywheel: the control plane did not take per-turn "
            f"feedback ({reason}); it may predate it. What was waiting is "
            "dropped and no more is sent by this process.")

    def _queued_files(self, prefix: str) -> list[Path]:
        """The files with this prefix waiting in the trajectory root, in name
        order. Never raises: an unreadable root is an empty queue."""
        try:
            return sorted(
                p for p in Path(self.trajectory_root).iterdir()
                if p.name.startswith(prefix)
                and p.name.endswith(_OUTCOME_FILE_SUFFIX) and p.is_file()
            )
        except OSError:
            return []

    def _outcome_files(self) -> list[Path]:
        """The outcome files waiting in the trajectory root, in name order.
        Never raises: an unreadable root is an empty queue."""
        return self._queued_files(_OUTCOME_FILE_PREFIX)

    def _cap_unanswered_outcomes(self) -> None:
        """While consent is UNKNOWN, keep at most
        :data:`_OUTCOME_MAX_KEPT_UNANSWERED` of the outcomes that wait on it:
        the oldest (by when they were written, then by name) go, with a
        warning naming that cause. An outcome that names only an episode
        (``contract.names_only_an_episode``) does not wait on consent, so it
        is neither counted nor dropped here. Never raises."""
        files = self._outcome_files()
        if len(files) <= _OUTCOME_MAX_KEPT_UNANSWERED:
            return
        waiting: list[Path] = []
        for path in files:
            try:
                wire = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not (isinstance(wire, Mapping) and names_only_an_episode(wire)):
                waiting.append(path)
        excess = len(waiting) - _OUTCOME_MAX_KEPT_UNANSWERED
        if excess <= 0:
            return

        def _age(path: Path) -> tuple[float, str]:
            try:
                return (path.stat().st_mtime, path.name)
            except OSError:
                return (0.0, path.name)

        dropped = 0
        for path in sorted(waiting, key=_age)[:excess]:
            try:
                path.unlink()
                dropped += 1
            except OSError:
                continue
        if dropped:
            self._warn(_unanswered_outcomes_dropped(dropped))

    def _expire_outcomes(self) -> list[Path]:
        """Drop the outcomes undelivered for a week, out loud; the rest, in
        name order. Runs on EVERY flush, whatever consent says: the bound used
        to live inside the upload, which runs only once consent is yes, so an
        outcome kept while the answer was unknown -- or after a refusal --
        never expired. Never raises."""
        kept: list[Path] = []
        expired = 0
        now = time.time()
        for path in self._outcome_files():
            try:
                if now - path.stat().st_mtime > _OUTCOME_MAX_AGE_S:
                    path.unlink()
                    expired += 1
                    continue
            except OSError:
                continue
            kept.append(path)
        if expired:
            self._warn(
                f"agent-flywheel: {expired} reported outcome(s) could not be "
                f"delivered for a week and were dropped.")
        return kept

    def _refuse_episode_outcomes(self, reason: str) -> None:
        """The control plane refused outcomes that name only an episode, for
        want of production capture: it predates the episode join (or has it
        switched off). Said ONCE; they stay on disk, go with the rest if
        capture is enabled, and no more are queued or sent on those terms by
        this process. Never raises."""
        if self._episode_outcomes_refused:
            return
        self._episode_outcomes_refused = True
        self._warn(
            "agent-flywheel: the control plane did not take outcomes "
            f"reported for an episode ({reason}); it may predate them. They "
            "stay on disk and no more are sent by this process unless "
            "production capture is enabled.")

    def _flush_outcomes(
        self, files: list[Path], *, episodes_only: bool = False,
    ) -> dict[str, int]:
        """Upload the outcomes waiting on disk (``files``, already past
        :meth:`_expire_outcomes`). Caller holds ``_flush_lock`` and has
        established :attr:`may_upload` -- or passes ``episodes_only``, which
        sends just the outcomes that name an episode and nothing else
        (``contract.names_only_an_episode``), the ones consent is not the
        question for; a 403 then stops those alone
        (:meth:`_refuse_episode_outcomes`). Never raises. The rules are
        :meth:`_send_reports`'s.

        Counts: ``outcomes_sent``, the outcomes the control plane recorded;
        ``outcomes_failed``, the outcomes still on disk for a later flush,
        each counted once; ``outcomes_dropped``, the records the control
        plane refused.
        """
        pending: list[tuple[Path, dict[str, Any]]] = []
        for path in files:
            try:
                pending.append((path, production_outcome(
                    json.loads(path.read_text(encoding="utf-8")))))
            except (OSError, ValueError, InvalidOutcome):
                # Unreadable or not ours to send; one file costs that file.
                continue
        if episodes_only:
            pending = [p for p in pending if names_only_an_episode(p[1])]
        if not pending:
            return {}
        sent, failed, refused = self._send_reports(
            pending, _OUTCOMES,
            on_refusal=(self._refuse_episode_outcomes if episodes_only
                        else self._refuse_upload))
        return {"outcomes_sent": sent, "outcomes_failed": failed,
                "outcomes_dropped": refused}

    def _send_reports(
        self,
        pending: list[tuple[Path, dict[str, Any]]],
        kind: _ReportKind,
        *,
        on_refusal: Any,
        on_absent: Any = None,
    ) -> tuple[int, int, int]:
        """Send the reports waiting on disk -- outcomes or per-turn feedback,
        ``kind`` says which -- and return ``(sent, failed, dropped)``: the
        ones the control plane recorded, the ones still on disk for a later
        flush (each counted once), the ones it refused. Caller holds
        ``_flush_lock``. Never raises.

        ONE SENDER FOR BOTH ROUTES. Feedback had its own, which read every
        final 4xx as "the route is not there" and deleted the queue for it.

        THE TURNS' RULES: a batch the control plane acknowledged is deleted;
        a 403 is its decision, so sending stops (``on_refusal`` is told its
        words) and the reports stay on disk; a blip (no answer, a 5xx, a 408
        or 429 that outlasted the retries) leaves the batch for the next
        flush.

        A REPORT REFUSED BY NAME inside an acknowledged batch
        (``accepted: false`` -- a conflicting replay, an unknown episode)
        would be refused again: it is not sent again, it counts as dropped
        and not as sent, and it is said BY ITS ID with the control plane's
        own word (:func:`_refused_by_name`). An outcome recorded without its
        note is said the same way (:func:`_notes_not_stored`).

        A 4xx ABOUT THE RECORDS IS FINAL FOR THEM
        (:func:`_outcome_batch_refused`). A refused batch is split in halves,
        and BOTH halves are tried before either is split again: the good ones
        are sent, and each bad one, once it stands alone, is dropped with a
        warning naming the control plane's own cause.

        Unless both halves are refused with the same status and the same
        words: then the refusal was about THE BATCH, not a record in it -- a
        422 from a control plane on another contract minor, a proxy's 400 --
        and splitting further would drop every report at two requests each.
        Sending stops for this flush, they stay on disk (the week's expiry
        still applies), and the cause is said once.

        Except a 413 (:func:`_outcome_refused_for_size`): it is about the
        body's SIZE, so both halves of a batch too large for a proxy answer
        it alike, and smaller parts may still pass. The split goes on; a
        record that draws a 413 standing alone is too large on its own, and
        is dropped with a warning naming the 413 and its size.

        Splitting is bounded per flush (:data:`_OUTCOME_SPLIT_MAX_REQUESTS`);
        what the bound leaves unsent stays for the next flush.

        A 4xx about the ROUTE or the KEY (401, 404, 405, 410) is not the
        records' fault either: sending stops for this flush and they stay on
        disk, said out loud. Unless the caller passes ``on_absent`` and the
        route is ABSENT (:func:`_route_absent`): then everything waiting is
        dropped, counted as dropped, and ``on_absent`` is told the words.
        """
        size = kind.batch_max
        Batch = list[tuple[Path, dict[str, Any]]]
        # Every GROUP still to send, in order: a batch on its own, or the two
        # halves of one refused batch, which are tried together. A refused
        # half's own halves go back to the front.
        queue: list[list[Batch]] = [
            [pending[i:i + size]] for i in range(0, len(pending), size)]
        sent = failed = split_requests = absent = 0
        dropped: list[str] = []
        by_name: list[str] = []
        noteless: list[str] = []
        client = None

        def _left() -> int:
            return sum(len(b) for group in queue for b in group)

        def _forget(batch: Batch) -> None:
            for path, _wire in batch:
                try:
                    path.unlink()
                except OSError:
                    # Resent next flush; the intake answers a duplicate.
                    pass

        try:
            from ._http import ControlPlaneClient
            from .transport import AttachTransport

            client = ControlPlaneClient(self._cp_url, self._api_key)
            send = getattr(AttachTransport(client, self.agent_id), kind.send)
            while queue:
                group = queue.pop(0)
                halves = len(group) == 2
                if halves and (split_requests + 2
                               > _OUTCOME_SPLIT_MAX_REQUESTS):
                    failed += sum(len(b) for b in group) + _left()
                    queue.clear()
                    self._warn(
                        f"agent-flywheel: the control plane refused "
                        f"{kind.many}, and finding which ones reached "
                        f"{_OUTCOME_SPLIT_MAX_REQUESTS} requests in this "
                        "flush; the rest stay on disk for a later flush.")
                    break
                refused: list[tuple[Batch, tuple[int, str]]] = []
                stop = False
                for index, batch in enumerate(group):
                    if halves:
                        split_requests += 1
                    try:
                        reply = send(
                            self.agent_id, [wire for _, wire in batch])
                    except Exception as exc:
                        # Kept by a stop: this half, the half not tried
                        # yet, a half already refused beside it, the queue.
                        beside = ([r for b in group[index + 1:] for r in b]
                                  + [r for b, _ in refused for r in b])
                        untried = len(beside)
                        refusal = _capture_refusal(exc)
                        if refusal is not None:
                            on_refusal(refusal)
                            failed += len(batch) + untried + _left()
                            queue.clear()
                            stop = True
                            break
                        status = getattr(exc, "status_code", None)
                        if on_absent is not None and _route_absent(status):
                            gone = batch + beside + [
                                r for g in queue for b in g for r in b]
                            _forget(gone)
                            absent += len(gone)
                            queue.clear()
                            on_absent(str(exc))
                            stop = True
                            break
                        if _outcome_batch_refused(status):
                            refused.append((batch, (status, str(exc))))
                            continue
                        if _outcome_route_refused(status):
                            failed += len(batch) + untried + _left()
                            queue.clear()
                            # A 410 is a ``LeaseLost``, whose words are the
                            # detail alone: the status is named either way.
                            said = str(exc)
                            if f"HTTP {status}" not in said:
                                said = f"HTTP {status}: {said}"
                            self._warn(
                                "agent-flywheel: the control plane did not "
                                f"take {kind.many} ({said}); they stay "
                                "on disk and are sent again on a later "
                                "flush.")
                            stop = True
                            break
                        failed += len(batch)
                        continue
                    _forget(batch)
                    named = _refused_by_name(reply, kind)[:len(batch)]
                    by_name.extend(named)
                    noteless.extend(_notes_not_stored(reply, kind))
                    sent += len(batch) - len(named)
                if stop:
                    break
                if (halves and len(refused) == 2
                        and refused[0][1] == refused[1][1]
                        and not _outcome_refused_for_size(refused[0][1][0])):
                    failed += sum(len(b) for b, _ in refused) + _left()
                    queue.clear()
                    self._warn_whole_batch_refusal(kind, refused[0][1][1])
                    break
                split: list[list[Batch]] = []
                for batch, (status, cause) in refused:
                    if len(batch) == 1:
                        _forget(batch)
                        if _outcome_refused_for_size(status):
                            cause = (
                                f"{cause} ({kind.one} of "
                                f"{_outcome_wire_bytes(batch[0][1])} bytes, "
                                "too large to send even alone)")
                        dropped.append(cause)
                    else:
                        half = len(batch) // 2
                        split.append([batch[:half], batch[half:]])
                queue[:0] = split
        except Exception:
            failed += _left()
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass
        if dropped:
            self._warn(
                f"agent-flywheel: the control plane refused {len(dropped)} "
                f"{kind.counted}, so they were dropped and the rest of "
                f"their batch was sent: {'; '.join(sorted(set(dropped)))}")
        if by_name:
            self._warn(
                f"agent-flywheel: the control plane did not record "
                f"{len(by_name)} {kind.counted}; they will not be sent "
                f"again: {', '.join(by_name)}")
        if noteless:
            self._warn(
                f"agent-flywheel: the control plane recorded {len(noteless)} "
                f"{kind.counted} without the note: {', '.join(noteless)}")
        return sent, failed, len(dropped) + len(by_name) + absent

    def _warn_whole_batch_refusal(self, kind: _ReportKind, cause: str) -> None:
        """Say ONCE per kind and cause that the control plane refuses whole
        batches: it holds for every flush until one side is upgraded, and a
        warning per flush would be a week of the same line."""
        if self._batch_refusal_said.get(kind.many) == cause:
            return
        self._batch_refusal_said[kind.many] = cause
        self._warn(
            "agent-flywheel: the control plane refused whole batches of "
            f"{kind.many} ({cause}), which is about the batch, not a "
            "record in it; they stay on disk, are sent again on a later "
            "flush, and are dropped only after a week undelivered.")

    @staticmethod
    def _warn(message: str) -> None:
        try:
            warnings.warn(message, RuntimeWarning, stacklevel=3)
        except Exception:  # -W error turns the line above into a raise
            pass

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
            ) or read_model_boundary(d) is not None
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
        cid = _production_identifier(read_conversation_id(d))
        # WHAT THE MODEL SAW AND SAID, when an adapter observed it -- and
        # only while the control plane still says this agent may upload it.
        # Rendered from the turn's own record every time, never cached: the
        # record is the one source, so a retried upload of this turn carries
        # the same trace (the intake reads a different one as a conflict).
        # Its conversation id is the wire's, read from the same file, because
        # the intake refuses a trace that names any other.
        genai_trace = None
        if self.may_capture_model_content:
            from .genai_otlp import render_turn_trace

            genai_trace = render_turn_trace(
                read_model_boundary(d), agent_name=self.agent_id,
                conversation_id=cid,
            )
        if (not _recorded_anything(text, calls, task_id=task_id)
                and genai_trace is None):
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
        if cid:
            wire["conversation_id"] = cid
            wire["turn_index"] = self._turn_index(d, cid)
        if task_id:
            wire["task_id"] = task_id
        fingerprint = _read_execution_fingerprint(d)
        if fingerprint is not None:
            wire["agent_fingerprint"] = fingerprint
        if genai_trace is not None:
            # OMITTED, never null, when there is none: a turn that observed no
            # model call keeps exactly the bytes it always had.
            wire["genai_trace"] = genai_trace
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
        self._consent_refused = False
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
        self._consent_refused = True
        # A trace rides on a turn: a refused turn refuses its model content
        # too, for the same reason.
        self.may_capture_model_content = False
        self.model_content_refusal = reason
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
            if verdict.answered:
                # The second tier is read off the same answer, both ways: this
                # is a verdict the control plane GAVE, not a blip, so a grant
                # turns model-content recording on from the next turn and a
                # refusal keeps it off with the server's current sentence.
                self.may_capture_model_content = (
                    verdict.may_capture_model_content and self.may_upload)
                self.model_content_refusal = (
                    None if self.may_capture_model_content
                    else verdict.model_content_reason)
            if not verdict.may_upload and verdict.answered:
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
                self._consent_refused = True
        except Exception:
            # A capture bug costs the observation, never the turn.
            pass
        finally:
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def _note_model_content_refusal(self, reply: Any) -> None:
        """Stop recording model content the intake says it may not take.

        THE REVOCATION PATH for the second tier. The intake keeps a turn and
        refuses its trace BY NAME (:data:`_MODEL_CONTENT_REFUSALS`) when an
        operator has withdrawn model-content consent, or when the agent's
        turns are attributed to no Product (it became shared, or no Product
        declares it) -- so the turn itself is not a failure and this is the
        only place the refusal is visible. Left unread, every later turn
        would record the customer's prompt and tool results to disk and send
        them to be discarded. Never raises.
        """
        try:
            results = reply.get("turns") if isinstance(reply, Mapping) else None
            for result in results or ():
                trace = result.get("genai_trace") if isinstance(
                    result, Mapping) else None
                if (isinstance(trace, Mapping)
                        and trace.get("error") in _MODEL_CONTENT_REFUSALS):
                    detail = trace.get("detail")
                    self.may_capture_model_content = False
                    self.model_content_refusal = (
                        detail if isinstance(detail, str) and detail.strip()
                        else _MODEL_CONTENT_NOT_ENABLED)
                    try:
                        warnings.warn(
                            "agent-flywheel: the control plane refused this "
                            "agent's model content, so it is no longer "
                            "recorded or uploaded: "
                            f"{self.model_content_refusal}",
                            RuntimeWarning, stacklevel=2,
                        )
                    except Exception:  # -W error turns the line into a raise
                        pass
                    return
        except Exception:
            return

    def flush(self, *, include_unanswered: bool = False,
              batch_size: int = 64) -> dict[str, int]:
        """Upload captured turns, and any reported outcomes still waiting, to
        the control plane. Returns counts: the turns' ``sent`` / ``skipped`` /
        ``failed``, plus ``outcomes_sent`` / ``outcomes_failed`` when outcomes
        were waiting.

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
        """Everything a flush sends: the captured turns, then the reported
        outcomes. Caller holds ``_flush_lock``. Never raises.

        ONE SEAM FOR BOTH, because the consent question is one question: the
        turn half asks (and re-asks) it, and outcomes go only when its answer
        is yes. The counts are the turns'; ``outcomes_sent`` and
        ``outcomes_failed`` are added only when outcomes were waiting.

        EXCEPT WHAT IS NOT ABOUT END USERS' TRAFFIC, which consent is not the
        question for: an outcome that names only an episode the platform ran,
        and per-turn feedback (:meth:`_flush_feedback`). Those go whenever
        there is a control plane to send to.

        THE WEEK'S EXPIRY RUNS ON EVERY FLUSH, consent or not
        (:meth:`_expire_outcomes`): an outcome kept while the answer is
        unknown, or after a refusal, is bounded like any other.
        """
        counts = self._flush_turns_locked(
            include_unanswered=include_unanswered, batch_size=batch_size)
        waiting = self._expire_outcomes()
        if self.may_upload and waiting:
            counts.update(self._flush_outcomes(waiting))
        elif (waiting and not self._episode_outcomes_refused
                and self._cp_url and self._api_key):
            counts.update(self._flush_outcomes(waiting, episodes_only=True))
        if self._cp_url and self._api_key:
            counts.update(self._flush_feedback())
        return counts

    def _flush_turns_locked(self, *, include_unanswered: bool = False,
                            batch_size: int = 64) -> dict[str, int]:
        """The turn upload itself. Caller holds ``_flush_lock``.

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
        if self.may_upload and not self.may_capture_model_content:
            # THE SECOND TIER IS RE-ASKED THE SAME WAY, and for the same
            # reason: an operator may grant model content to an agent that is
            # already uploading turns, and nothing else would tell this
            # process. One read per flush while the answer is no, and none
            # once it is yes. A revocation is learned from the intake's own
            # refusal instead (`_note_model_content_refusal`).
            self._recheck_capture()
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
                    reply = tr.report_turns(self.agent_id, [w for _, w in chunk])
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
                self._note_model_content_refusal(reply)
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
        # The rules' declaration, if the control plane has not answered it
        # yet: a process closing moments after attach() would otherwise kill
        # it with the interpreter. Bounded the same way, and for the same
        # reason; rules it does not settle are declared by the next start.
        self.wait_for_rulebook(RULEBOOK_SETTLE_GRACE_S)
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
    rules: Sequence[str] | None = None,
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

    MODEL CONTENT IS A SECOND, SEPARATE CONSENT. What a turn's model calls saw
    and said -- the system prompt, the conversation history, the tool
    definitions and every tool result -- is observed by the LangChain and ADK
    adapters, and recorded and uploaded as the turn's ``genai_trace``, only
    while the control plane grants the model-content tier.
    :attr:`Attachment.may_capture_model_content` is the live answer and
    :attr:`Attachment.model_content_refusal` the sentence explaining a False.
    Off until an operator turns it on; a later grant is noticed on the flush
    cadence, and a revocation the intake names stops it at once.

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
    every case except an invalid explicit identity configuration or an
    invalid ``describe``.

    ``describe`` is the agent's own description, exactly as for ``serve()``
    (a mapping of ``system_prompt`` / ``tools`` / ``model``, or a zero-argument
    callable returning one, called once the agent is built). With
    ``execution_entrypoint`` it wins over reflection where given; without one
    it is the whole description this registration sends -- the way a
    production-only agent tells the control plane its prompt and tools. One
    that leaves ``tools`` out does not erase the tools ``serve()`` reflected
    for the same agent: the control plane keeps a catalogue a description did
    not observe. It is the customer's word, not an observation, so it anchors
    no execution identity; ``PERCEPTEYE_INTROSPECT=0`` wins over it, with a
    warning.

    ``rules`` are the workflow rules of the Product this agent belongs to, as
    for ``serve()``: sent once, after the registration, to
    ``POST /agents/{agent_id}/rulebook``, and filed by the control plane under
    the one active Product that declares this agent. ``None`` sends nothing;
    ``[]`` retracts the rulebook in force; the same rules on the next start
    register nothing, and the last writer wins. They are declared IN THE
    BACKGROUND, after the policy read, and this returns without waiting for
    the answer: a refused, undeliverable or slow upload costs the rules and
    nothing else -- never the registration, the capture verdict, the policy
    read or the time this call takes. :attr:`Attachment.rulebook` is the
    control plane's receipt and :attr:`Attachment.rulebook_refusal` the
    sentence explaining why there is none (yet, until it answers);
    :meth:`Attachment.wait_for_rulebook` waits for the answer. Not a list of
    strings raises :class:`ConfigurationError`.

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
            "unless you ask (retention_turns=, prune()), and where a "
            "customer's data lands is not ours to assume."
        )
    # The Node SDK's `retentionTurns` accepts the same values and drops any
    # other with the same sentence (`capture.js:retentionTurnsFrom`); here a
    # wrong value raises, before capture starts, as every attach() argument
    # does. A bool is not a count, though Python calls it an int.
    if retention_turns is not None and not (
            isinstance(retention_turns, int)
            and not isinstance(retention_turns, bool)
            and retention_turns >= 0):
        raise ConfigurationError(
            "retention_turns must be a whole number of turns, 0 or more, "
            f"not {retention_turns!r}"
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
    if describe is not None and not callable(describe):
        # A static declaration is checked NOW, with the rest of the
        # configuration and before any adapter is bound, so a typo leaves
        # nothing behind -- and whatever PERCEPTEYE_INTROSPECT says, as serve()
        # does, so lifting the opt-out cannot turn it into a refusal to start.
        _declaration.resolve(describe)
    # The rules' SHAPE, checked with the rest of the configuration for the
    # same reason. What the control plane thinks of them is its answer, and
    # arrives as `rulebook_refusal`, never as a raise.
    rulebook_request = declared_rulebook(rules)
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
    # once when llm_url is empty, and the ADK adapter never reads the URL. So
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

    introspection_consented = _env_true("PERCEPTEYE_INTROSPECT", default=True)
    if describe is not None and not introspection_consented:
        try:
            warnings.warn(
                "PERCEPTEYE_INTROSPECT is off, and it wins: the description "
                "declared with describe= is not sent.",
                RuntimeWarning, stacklevel=2,
            )
        except Exception:  # -W error turns the line above into a raise
            pass
    described = None
    declared = None
    agent_fingerprint: dict[str, str] | None = None
    try:
        if execution_entrypoint is not None:
            described, agent_fingerprint = _reconcile_attached_execution(
                execution_entrypoint,
                input_shape=execution_input_shape,
                method=execution_method,
                adapters_bound=bound,
                trajectory_root=trajectory_root,
                attested_execution_components=normalized_components,
                discovery_enabled=introspection_consented,
            )
        # The customer's own declaration (see declaration.py), resolved once
        # the agent is built, as serve() does -- and only with consent, so a
        # callable is not run for a description that will not be sent.
        if introspection_consented:
            declared = _declaration.resolve(describe)
    except Exception:
        # Unlike observation failures inside introspect(), invalid entrypoint,
        # attestation or declaration configuration is a caller mistake. Undo
        # the global adapter binding before surfacing it.
        for unbind in unbinds:
            try:
                unbind()
            except Exception:
                pass
        raise
    if declared is not None and declared.describes:
        # What the customer declared wins where they gave it; with no
        # entrypoint it is the whole description. And it is their word, not an
        # observation of what runs, so it anchors no execution identity.
        # (``api_spec`` alone describes nothing and changes neither.)
        described = _declaration.apply(declared, reflected=described,
                                       agent_name=agent_id)
        agent_fingerprint = None

    # ── the control plane, entirely optional ────────────────────────────────
    policy: dict[str, Any] = {}
    register_error: Exception | None = None
    url = resolve_control_plane_url(control_plane_url)
    # THE ONE RULE: the argument, the variable, then the key stored for this
    # agent (``credentials.resolve_api_key``).
    from .credentials import resolve_api_key

    _resolved_key = resolve_api_key(
        api_key, agent_id=agent_id, control_plane_url=url,
    )
    key = _resolved_key.value if _resolved_key else None
    may_upload = False
    #: Only an ANSWERED no (see ``Attachment._consent_refused``).
    consent_refused = False
    may_capture_model_content = False
    model_content_refusal: str | None = None
    # NAMED, not None. `upload_refusal` is documented as "the sentence
    # explaining a False", and the ways to get here without ever asking the
    # control plane are distinct problems with distinct remedies. One message
    # covering all of them -- or, as it was, no message at all -- sends an
    # operator to check the wrong thing.
    #
    # WHICH VARIABLE is asked through the same helper `policy_source` and the
    # CLI ask, so "configured" means one thing across the package.
    cause = missing_credentials_cause(
        key, agent_id=agent_id, control_plane_url=control_plane_url,
    )
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
    # The same rule for the rules: NAMED for every way they can go unsent,
    # each with its own remedy, and replaced by what the declaration did when
    # one runs (`Attachment._declare_rulebook`).
    rulebook_refusal: str | None = None
    if rulebook_request is not None:
        if not register:
            rulebook_refusal = (
                "attach(register=False) was asked for, so the rules passed as "
                "rules= were never sent to the control plane."
            )
        elif cause:
            rulebook_refusal = (
                f"{cause}, so the rules passed as rules= were never sent to "
                f"the control plane."
            )
        else:
            rulebook_refusal = (
                "the control plane client could not be built, so the rules "
                "passed as rules= were never sent to it."
            )
    #: The transport the rules are declared through and the client it owns,
    #: once the Attachment exists to hold the answer -- set only when the
    #: policy read is done.
    declare_through: tuple[Any, ControlPlaneClient] | None = None
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
                if declared is not None:
                    # Where the agent's API spec is published, when describe=
                    # says (declaration.py): a reference, never the document.
                    registration.update(declared.registration_fields())
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
                consent_refused = verdict.answered and not verdict.may_upload
                may_capture_model_content = verdict.may_capture_model_content
                model_content_refusal = verdict.model_content_reason
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
            # The rules go AFTER the policy read and OFF this call's return
            # path (below, once the Attachment exists): rules the control
            # plane refuses, or cannot hear, or answers slowly, must cost the
            # rules and nothing else -- not the policy already read, not the
            # capture verdict, not the customer's startup.
            if rulebook_request is not None:
                declare_through = (tr, client)
        except Exception:
            policy = {}
        finally:
            # A declaration still to run OWNS the client and closes it itself.
            if client is not None and declare_through is None:
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
        consent_refused=consent_refused,
        agent_fingerprint=agent_fingerprint,
        may_capture_model_content=may_capture_model_content,
        model_content_refusal=model_content_refusal,
        rulebook_refusal=rulebook_refusal,
    )
    if may_upload:
        # Through the same seam late consent goes through, so the hook is
        # registered by exactly one rule: "may_upload just became True".
        px._grant_upload()
    if declare_through is not None:
        transport, owned = declare_through
        px._declare_rulebook(transport, rulebook_request, owned)
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
