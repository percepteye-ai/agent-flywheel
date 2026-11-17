"""The flywheel/1 wire types.

Generated shape is pinned by ``schema/flywheel-1.json``, which this repository
owns. Server implementations pin against the published schema; the schema is
never derived from a server type.

The organising principle of this module: **the dangerous shape is
unconstructible.** Not documented, not linted -- unconstructible. Every rule
below exists because its absence produced fabricated training data.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Literal

#: The one vocabulary a refusal names a value's type in, shared with
#: ``describe=`` and with the Node SDK's ``jsonType`` -- ``got null``, never
#: ``got NoneType``: an operator configuring rules in JSON wrote no Python.
from .declaration import _json_type, _json_whole_number
from .errors import ContractViolation, InvalidFeedback, InvalidOutcome

#: Imported rather than restated so the contract validates against exactly
#: the field set the extractor produces -- one definition, never a twin
#: that can drift out from under the other half.
from .evidence import IDENTITY_FIELDS as _IDENTITY_FIELDS
from .execution_identity import is_lower_sha256

__all__ = [
    "CONTRACT_VERSION",
    "PRODUCTION_OUTCOMES",
    "TURN_FEEDBACK_SIGNALS",
    "CHECK_ESTABLISHES",
    "CHECK_STATUSES",
    "ExclusionReason",
    "MAX_REPORTED_CHECKS",
    "Outcome",
    "RolloutOutput",
    "RolloutRequest",
    "RulebookReceipt",
    "RulebookRequest",
    "RulebookStatus",
    "TERMINATION_CLASSES",
    "ToolCallOutcome",
]

CONTRACT_VERSION = "flywheel/1"

#: Exact limits published by ``schema/flywheel-1.json`` for the production
#: turn wire.  The schema remains the authority; these names keep each runtime
#: producer from restating a different boundary.
_PRODUCTION_IDENTIFIER_MAX_CHARS = 255
_PRODUCTION_TURN_BATCH_MAX_ITEMS = 64

#: One exact grammar for opaque identifiers on the production-turn wire.
#: Leading/trailing whitespace and C0/C1 controls are forbidden; internal
#: spacing and non-ASCII customer identifiers otherwise remain opaque.
#: These strings are asserted verbatim against the published JSON Schema.
_PRODUCTION_IDENTIFIER_PATTERN = (
    r"^(?!\s)(?!.*\s$)[^\u0000-\u001F\u007F-\u009F]+$"
)
_PRODUCTION_TURN_IDENTIFIER_PATTERN = (
    r"^(?!\.{1,2}$)(?!\s)(?!.*\s$)(?!.*[\\/])"
    r"[^\u0000-\u001F\u007F-\u009F]+$"
)
_PRODUCTION_AGENT_IDENTIFIER_PATTERN = _PRODUCTION_TURN_IDENTIFIER_PATTERN
_PRODUCTION_IDENTIFIER_RE = re.compile(_PRODUCTION_IDENTIFIER_PATTERN)
_PRODUCTION_TURN_IDENTIFIER_RE = re.compile(
    _PRODUCTION_TURN_IDENTIFIER_PATTERN
)


def _well_formed_text(value: str) -> bool:
    """Whether ``value`` is Unicode text a JSON parser takes: no LONE
    SURROGATE. Python strings can hold one (``"\\ud800"``), and the wire
    escapes it, but the control plane's parser refuses it -- a 422 for the
    whole batch, then a record dropped by the split. Refused at the call
    instead, so what is refused at the call and at the wire cannot differ.
    The Node SDK's ``String.prototype.isWellFormed``."""
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _production_identifier(value: Any) -> str | None:
    """Return an exact identifier representable on the production-turn wire."""
    if (
        not isinstance(value, str)
        or not value
        or not _well_formed_text(value)
        or len(value) > _PRODUCTION_IDENTIFIER_MAX_CHARS
        or _PRODUCTION_IDENTIFIER_RE.fullmatch(value) is None
    ):
        return None
    return value


def _production_turn_identifier(value: Any) -> str | None:
    """Return an exact wire id that is also one safe directory segment."""
    if (
        not isinstance(value, str)
        or not value
        or not _well_formed_text(value)
        or len(value) > _PRODUCTION_IDENTIFIER_MAX_CHARS
        or _PRODUCTION_TURN_IDENTIFIER_RE.fullmatch(value) is None
    ):
        return None
    return value


def _production_agent_identifier(value: Any) -> str | None:
    """Return an exact wire id safe as the agent URL path segment."""
    return _production_turn_identifier(value)


#: THE closed vocabulary of ``Attachment.report_outcome``: how a production
#: turn or conversation actually turned out. Asserted verbatim against
#: ``schema/flywheel-1.json`` (``ProductionOutcome.outcome``), which the
#: control plane's intake and the Node SDK carry too.
PRODUCTION_OUTCOMES: tuple[str, ...] = (
    "resolved", "escalated", "reopened", "abandoned",
    "thumbs_up", "thumbs_down",
)
#: A note is a short remark, not a transcript (the schema's ``maxLength``,
#: counted in code points).
_PRODUCTION_OUTCOME_NOTE_MAX_CHARS = 1000
_PRODUCTION_OUTCOME_BATCH_MAX_ITEMS = 64
#: An RFC 3339 instant WITH its offset. A time with no offset is not an
#: instant, and guessing the zone would move the outcome. ASCII digits only:
#: Python's ``\d`` also matches other scripts' digits, which no server parses.
_RFC3339_PATTERN = (
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?"
    r"(Z|[+-][0-9]{2}:[0-9]{2})$"
)
_RFC3339_RE = re.compile(_RFC3339_PATTERN)
_OCCURRED_AT_REFUSAL = (
    "occurred_at must be an RFC 3339 time with its offset, such as "
    "2026-11-10T09:00:00Z")


def _is_rfc3339_instant(text: Any) -> bool:
    """Whether ``text`` is an RFC 3339 instant that EXISTS: the pattern, and
    then every field in its range -- a real calendar day (``2026-02-30`` is
    not one), an hour of 0-23 (``T24:00`` is not one), no leap second, an
    offset under a day. A shape alone let an impossible time reach the
    control plane, which refused the whole batch around it. The Node SDK's
    ``isRfc3339Instant`` applies the same arithmetic."""
    if not isinstance(text, str) or _RFC3339_RE.fullmatch(text) is None:
        return False
    year, month, day = int(text[0:4]), int(text[5:7]), int(text[8:10])
    hour, minute, second = int(text[11:13]), int(text[14:16]), int(text[17:19])
    if not (1 <= year and 1 <= month <= 12):
        return False
    leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    days = (31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
    if not (1 <= day <= days[month - 1] and hour <= 23 and minute <= 59
            and second <= 59):
        return False
    if not text.endswith("Z"):
        if int(text[-5:-3]) > 23 or int(text[-2:]) > 59:
            return False
    return True


def _is_finite_wire_number(value: Any) -> bool:
    """Whether ``value`` is a number the wire carries as a finite float: not a
    bool, not NaN or an infinity, and not an int too large for a float (the
    control plane reads ``value`` as one, and refused the batch around it)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False
_OUTCOME_WIRE_KEYS = (
    "outcome_id", "outcome", "turn_id", "conversation_id", "episode_id",
    "value", "note", "occurred_at",
)


def production_outcome(wire: Any) -> dict[str, Any]:
    """One reported outcome exactly as the wire carries it, or
    :class:`InvalidOutcome` naming what is wrong.

    THE one validator: ``Attachment.report_outcome`` builds its record
    through it, and ``AttachTransport.report_outcomes`` passes every outcome
    through it again before a batch leaves, so what is refused at the call
    and what is refused at the wire cannot differ. The Node SDK's
    ``productionOutcome`` applies the same rules with the same messages.

    Absent optional fields are OMITTED, never null. Nothing is rewritten: an
    id, a note or a time that does not fit is refused, not cleaned into a
    different one.
    """
    if not isinstance(wire, dict):
        raise InvalidOutcome(
            f"an outcome must be an object, got {_json_type(wire)}")
    unknown = sorted(set(wire) - set(_OUTCOME_WIRE_KEYS))
    if unknown:
        raise InvalidOutcome(
            "an outcome has no field "
            + json.dumps(unknown[0], ensure_ascii=False))
    outcome = wire.get("outcome")
    if not isinstance(outcome, str) or outcome not in PRODUCTION_OUTCOMES:
        # The value as JSON spells it, so both SDKs print the same refusal.
        got = (json.dumps(outcome, ensure_ascii=False)
               if isinstance(outcome, str) else _json_type(outcome))
        raise InvalidOutcome(
            f"outcome must be one of {', '.join(PRODUCTION_OUTCOMES)}; "
            f"got {got}")
    if _production_turn_identifier(wire.get("outcome_id")) is None:
        raise InvalidOutcome("outcome_id is not a valid identifier")
    out: dict[str, Any] = {
        "outcome_id": wire["outcome_id"], "outcome": outcome,
    }
    if wire.get("turn_id") is not None:
        if _production_turn_identifier(wire["turn_id"]) is None:
            raise InvalidOutcome("turn_id is not a valid turn identifier")
        out["turn_id"] = wire["turn_id"]
    if wire.get("conversation_id") is not None:
        if _production_identifier(wire["conversation_id"]) is None:
            raise InvalidOutcome(
                "conversation_id is not a valid identifier")
        out["conversation_id"] = wire["conversation_id"]
    if wire.get("episode_id") is not None:
        # The ``rollout_id`` the platform handed the agent with the episode.
        if _production_identifier(wire["episode_id"]) is None:
            raise InvalidOutcome("episode_id is not a valid identifier")
        out["episode_id"] = wire["episode_id"]
    if not {"turn_id", "conversation_id", "episode_id"} & set(out):
        raise InvalidOutcome(
            "an outcome names the turn_id, the conversation_id or the "
            "episode_id it is the outcome of")
    value = wire.get("value")
    if value is not None:
        if not _is_finite_wire_number(value):
            raise InvalidOutcome("value must be a finite number")
        out["value"] = value
    note = wire.get("note")
    if note is not None:
        if (not isinstance(note, str) or not note
                or len(note) > _PRODUCTION_OUTCOME_NOTE_MAX_CHARS):
            raise InvalidOutcome(
                "note must be a non-empty string of at most "
                f"{_PRODUCTION_OUTCOME_NOTE_MAX_CHARS} characters")
        if not _well_formed_text(note):
            raise InvalidOutcome("note must be well-formed Unicode text")
        out["note"] = note
    occurred_at = wire.get("occurred_at")
    if not _is_rfc3339_instant(occurred_at):
        raise InvalidOutcome(_OCCURRED_AT_REFUSAL)
    out["occurred_at"] = occurred_at
    return out


def names_only_an_episode(outcome: Mapping[str, Any]) -> bool:
    """Whether a reported outcome names an episode and nothing else. Such an
    outcome is about a platform-run episode, not the agent's end users'
    traffic, so it is sent without the production-capture consent a turn's
    or a conversation's outcome needs."""
    return ("episode_id" in outcome and "turn_id" not in outcome
            and "conversation_id" not in outcome)


#: THE closed vocabulary of ``Attachment.report_turn_feedback``: what the
#: user did with one turn. Asserted verbatim against
#: ``schema/flywheel-1.json`` (``TurnFeedback.signal``), which the control
#: plane's intake and the Node SDK carry too.
TURN_FEEDBACK_SIGNALS: tuple[str, ...] = (
    "accepted", "corrected", "repeated", "escalated", "abandoned",
)
#: A turn index is a position inside one episode (the schema's ``maximum``).
_TURN_FEEDBACK_TURN_INDEX_MAX = 100000
_TURN_FEEDBACK_BATCH_MAX_ITEMS = 64
_FEEDBACK_WIRE_KEYS = (
    "feedback_id", "episode_id", "turn_index", "signal", "occurred_at",
)


def _feedback_head(wire: Any, keys: tuple[str, ...]) -> str:
    """The checks every per-turn feedback report shares, in their order --
    an object, no field outside ``keys``, a ``signal`` from the vocabulary, a
    ``feedback_id`` -- and the signal. :func:`turn_feedback` and
    :func:`turn_feedback_on_a_span` add what names the turn."""
    if not isinstance(wire, dict):
        raise InvalidFeedback(
            f"feedback must be an object, got {_json_type(wire)}")
    unknown = sorted(set(wire) - set(keys))
    if unknown:
        raise InvalidFeedback(
            "feedback has no field "
            + json.dumps(unknown[0], ensure_ascii=False))
    signal = wire.get("signal")
    if not isinstance(signal, str) or signal not in TURN_FEEDBACK_SIGNALS:
        got = (json.dumps(signal, ensure_ascii=False)
               if isinstance(signal, str) else _json_type(signal))
        raise InvalidFeedback(
            f"signal must be one of {', '.join(TURN_FEEDBACK_SIGNALS)}; "
            f"got {got}")
    if _production_turn_identifier(wire.get("feedback_id")) is None:
        raise InvalidFeedback("feedback_id is not a valid identifier")
    return signal


def turn_feedback(wire: Any) -> dict[str, Any]:
    """One per-turn feedback report exactly as the wire carries it, or
    :class:`InvalidFeedback` naming what is wrong.

    THE one validator, as :func:`production_outcome` is for outcomes:
    ``Attachment.report_turn_feedback`` builds its record through it and
    ``AttachTransport.report_turn_feedback`` passes every report through it
    again before a batch leaves. The Node SDK's ``turnFeedback`` applies the
    same rules with the same messages.
    """
    signal = _feedback_head(wire, _FEEDBACK_WIRE_KEYS)
    if _production_identifier(wire.get("episode_id")) is None:
        raise InvalidFeedback("episode_id is not a valid identifier")
    turn_index = _json_whole_number(wire.get("turn_index"))
    if turn_index is None or not (
            0 <= turn_index <= _TURN_FEEDBACK_TURN_INDEX_MAX):
        raise InvalidFeedback(
            "turn_index must be a whole number from 0 to "
            f"{_TURN_FEEDBACK_TURN_INDEX_MAX}")
    if not _is_rfc3339_instant(wire.get("occurred_at")):
        raise InvalidFeedback(_OCCURRED_AT_REFUSAL)
    return {
        "feedback_id": wire["feedback_id"], "episode_id": wire["episode_id"],
        "turn_index": turn_index, "signal": signal,
        "occurred_at": wire["occurred_at"],
    }


#: What feedback on a PRODUCTION TURN carries: the turn's own id in place of
#: an episode and a position. It is not sent to ``POST /turn-feedback``
#: (whose turns are an episode's); it is an event on that turn's trace.
_SPAN_FEEDBACK_KEYS = ("feedback_id", "turn_id", "signal", "occurred_at")


def turn_feedback_on_a_span(wire: Any) -> dict[str, Any]:
    """Per-turn feedback on a production turn, ``{feedback_id, turn_id,
    signal, occurred_at}``, or :class:`InvalidFeedback` naming what is wrong:
    :func:`turn_feedback`'s vocabulary, ids and time, with the turn named by
    the ``turn_id`` its span carries (``percepteye.turn.id``). What
    ``tracing.record_turn_feedback_span`` emits. The Node SDK's
    ``turnFeedbackOnASpan`` applies the same rules with the same messages."""
    signal = _feedback_head(wire, _SPAN_FEEDBACK_KEYS)
    if _production_turn_identifier(wire.get("turn_id")) is None:
        raise InvalidFeedback("turn_id is not a valid turn identifier")
    if not _is_rfc3339_instant(wire.get("occurred_at")):
        raise InvalidFeedback(_OCCURRED_AT_REFUSAL)
    return {
        "feedback_id": wire["feedback_id"], "turn_id": wire["turn_id"],
        "signal": signal, "occurred_at": wire["occurred_at"],
    }


def rfc3339_unix_nanos(value: str) -> int:
    """An RFC 3339 instant (one :func:`_is_rfc3339_instant` takes) as Unix
    nanoseconds, exactly: the fraction's first nine digits, the offset
    applied. What an OTel event's ``timeUnixNano`` carries. The Node SDK's
    ``rfc3339UnixNanos``."""
    import calendar

    m = re.fullmatch(
        r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})"
        r"(?:\.([0-9]+))?(Z|([+-])([0-9]{2}):([0-9]{2}))", value)
    if m is None:
        raise InvalidFeedback(_OCCURRED_AT_REFUSAL)
    year, month, day, hour, minute, second = (int(g) for g in m.groups()[:6])
    seconds = calendar.timegm((year, month, day, hour, minute, second))
    if m.group(8) != "Z":
        offset = int(m.group(10)) * 3600 + int(m.group(11)) * 60
        seconds -= offset if m.group(9) == "+" else -offset
    fraction = int(((m.group(7) or "") + "000000000")[:9])
    return seconds * 1_000_000_000 + fraction


Outcome = Literal["ok", "failed", "unknown"]

ExclusionReason = Literal[
    "flywheel_crash",
    "substrate_unavailable",
    "credential_expired",
    "deadline_exceeded",
    #: The agent could not be STARTED — the binary is missing, the runtime
    #: refused it, the import failed. Nothing ran, so there is nothing to
    #: learn from and the work is worth retrying elsewhere.
    "entrypoint_error",
    #: The agent RAN and exited non-zero. Its own failure, not the
    #: substrate's: a crash after doing work, or a completion gate it could
    #: not satisfy. Distinct from ``entrypoint_error`` because the two want
    #: opposite handling — one is retryable and untrainable, the other is a
    #: real attempt whose failure is exactly what training must see. Sharing
    #: one label sent a whole afternoon's triage at the process launcher
    #: while the detail string read "quality gate still failing".
    "agent_exit_nonzero",
    "cancelled",
    "capacity",
]

_VALID_OUTCOMES = ("ok", "failed", "unknown")
_STATUS_KEYS = ("status_code", "status", "http_status", "code", "statusCode")
_HTTP_IN_TEXT = re.compile(r"\bHTTP\s*\d{3}\b", re.I)


def _scrub_status(output: Any) -> Any:
    """Remove machine-readable status from a tool result body.

    Applied when and only when ``outcome == "unknown"``. Withholding the
    ``status_code`` field alone is not enough: a server-side scorer that
    recovers a status by parsing the result envelope will find it there and
    treat an unobserved call as a landed one. Absence of evidence must not be
    recoverable as evidence of success through a side channel.

    The human-readable content is preserved -- the trajectory still reads the
    same to a judge, it simply carries no status the scorer can mistake for an
    outcome.
    """
    if isinstance(output, str):
        try:
            body = json.loads(output)
        except Exception:
            return _HTTP_IN_TEXT.sub("", output)
        scrubbed = _scrub_status(body)
        return json.dumps(scrubbed) if not isinstance(scrubbed, str) else scrubbed
    if isinstance(output, dict):
        out = {k: v for k, v in output.items() if k not in _STATUS_KEYS}
        return {
            k: (_HTTP_IN_TEXT.sub("", v).strip() if isinstance(v, str) else _scrub_status(v))
            for k, v in out.items()
        }
    if isinstance(output, list):
        return [_scrub_status(v) for v in output]
    return output


@dataclass(frozen=True)
class ToolCallOutcome:
    """One executed tool call.

    ``outcome`` and ``status_code`` are both REQUIRED and non-defaultable. That
    is the whole point: a partially populated call is the dangerous shape, so
    it must be impossible to build one by forgetting a keyword.

    ``outcome="unknown"`` is legal, honest, and expected. It is the correct
    answer whenever the framework did not expose a real verdict -- most
    commonly a tool configured to swallow its own exceptions and return a
    string that reads like a normal result. It is NEVER rounded up to "ok".

    When the outcome is unknown, any status is discarded, from the field and
    from the body alike. A status may confirm or contradict an observed
    outcome; it may never manufacture one.
    """

    name: str
    arguments: dict[str, Any]
    outcome: Outcome
    status_code: int | None
    output: Any = None
    error: str | None = None
    error_class: str | None = None
    latency_ms: float | None = None
    #: Which agent in a multi-agent system made this call. ``None`` means
    #: unattributed, which is the honest answer for a single agent and for any
    #: framework that does not name its sub-agents -- it is NOT a claim that
    #: one agent made everything. Training a planner on a coder's tool calls is
    #: the multi-agent version of scoring an unobserved call as a success.
    agent_name: str | None = None
    #: The id the MODEL assigned to this call -- LangChain's
    #: ``ToolMessage.tool_call_id``, ADK's ``function_call_id``, the OpenAI
    #: ``tool_calls[].id``. It is the join key that lets the control plane say
    #: WHICH completion a call followed: the same id reappears on the
    #: ``role="tool"`` message of the agent's next request, so attribution
    #: needs no clocks and survives concurrent tools. ``None`` means the
    #: framework exposed no id; the call still counts, it just cannot be
    #: placed more precisely than "somewhere in this rollout".
    tool_call_id: str | None = None
    #: The identity a WRITE produced, filed under the field it came from --
    #: ``{"id": "10001", "key": "PROJ-42"}``. This is what lets the entity be
    #: found again later, and it is deliberately not the response body: the
    #: body is customer production data, the identity is a few opaque strings.
    #: Extract, then discard.
    #:
    #: Empty means no identity evidence, which is an abstention rather than a
    #: failure -- exactly the rule ``unknown`` follows.
    entity_ids: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ContractViolation("ToolCallOutcome.name must be a non-empty string")
        if self.outcome not in _VALID_OUTCOMES:
            raise ContractViolation(
                f"ToolCallOutcome.outcome must be one of {_VALID_OUTCOMES}, "
                f"got {self.outcome!r}. There is no default: report what you "
                f"observed, or report 'unknown'."
            )
        sc = self.status_code
        if sc is not None:
            if isinstance(sc, bool) or not isinstance(sc, int):
                raise ContractViolation(
                    f"ToolCallOutcome.status_code must be an int or None, got {sc!r}"
                )
            if not (100 <= sc <= 599):
                raise ContractViolation(f"status_code out of range: {sc}")

        if self.entity_ids:
            if not isinstance(self.entity_ids, dict):
                raise ContractViolation(
                    f"ToolCallOutcome.entity_ids must be a dict, "
                    f"got {type(self.entity_ids).__name__}"
                )
            for k, v in self.entity_ids.items():
                # Weak fields are refused BY NAME rather than merely
                # discouraged. A created resource routinely echoes a human
                # `name` that a different entity shares by coincidence, and
                # matching on it confirms the wrong resource -- a false
                # positive dressed as a verified success, which is worse than
                # no evidence at all.
                if str(k).strip().lower() not in _IDENTITY_FIELDS:
                    raise ContractViolation(
                        f"ToolCallOutcome.entity_ids key {k!r} is not a "
                        f"high-entropy identity field {_IDENTITY_FIELDS}. "
                        f"Low-entropy fields identify the wrong entity."
                    )
                if not isinstance(v, str) or not v.strip():
                    raise ContractViolation(
                        f"ToolCallOutcome.entity_ids[{k!r}] must be a "
                        f"non-empty string, got {v!r}"
                    )

        if self.outcome == "unknown":
            # Enforced, not requested. An unknown outcome carrying a 2xx is the
            # single shape that turns an honest capture into fabricated data.
            object.__setattr__(self, "status_code", None)
            object.__setattr__(self, "output", _scrub_status(self.output))

    def to_wire(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "arguments": self.arguments or {},
            "outcome": self.outcome,
            "status_code": self.status_code,
        }
        for k in ("output", "error", "error_class", "latency_ms", "agent_name",
                  "tool_call_id"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        # Deliberately NOT scrubbed for an `unknown` outcome, unlike `output`.
        # `_scrub_status` exists so an unobserved call cannot have a status
        # recovered from its body and be read as landed. An identity claims
        # nothing about the outcome -- it names an entity so that something
        # ELSE can go and observe one. Withholding it would suppress the only
        # evidence capable of turning an honest `unknown` into an honest `ok`.
        if self.entity_ids:
            d["entity_ids"] = dict(self.entity_ids)
        return d


@dataclass(frozen=True)
class RolloutRequest:
    """One unit of work, handed to the agent by the control plane.

    There is deliberately no ``environment`` field. The only configuration
    channel into an agent is subprocess environment at spawn -- configuration
    crosses into a FRESH CHILD, never into a running parent.
    """

    rollout_id: str
    turn_input: str
    llm_url: str
    api_key: str
    artifacts_dir: str
    deadline_ts: float
    #: When the LEASE lapses, as opposed to when the work is due. Separate
    #: numbers for separate jobs: the deadline bounds the work, the lease is
    #: liveness and a heartbeat renews it. Optional because the beat falls back
    #: to the lease we requested when a server does not say what it granted --
    #: but a server that clamps our request and does not tell us is exactly how
    #: a heartbeat ends up slower than the lease it is meant to keep alive.
    lease_expires_ts: float | None = None
    lease_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, d: dict[str, Any]) -> RolloutRequest:
        missing = [
            k
            for k in ("rollout_id", "turn_input", "llm_url", "api_key",
                      "artifacts_dir", "deadline_ts")
            if k not in d
        ]
        if missing:
            raise ContractViolation(f"RolloutRequest missing required field(s): {missing}")
        if "environment" in d:
            raise ContractViolation(
                "RolloutRequest carries an 'environment' key. This channel was "
                "removed: configuration reaches the agent through subprocess "
                "env at spawn, never by mutating a running process."
            )
        return cls(
            rollout_id=str(d["rollout_id"]),
            turn_input=str(d["turn_input"]),
            llm_url=str(d["llm_url"]),
            api_key=str(d["api_key"]),
            artifacts_dir=str(d["artifacts_dir"]),
            deadline_ts=float(d["deadline_ts"]),
            lease_expires_ts=(
                float(d["lease_expires_ts"])
                if d.get("lease_expires_ts") is not None else None
            ),
            lease_token=d.get("lease_token"),
            metadata=dict(d.get("metadata") or {}),
        )

    def seconds_remaining(self, now: float) -> float:
        """Time budget from the SERVER's deadline, never from a local env var.

        This is the rollout's WORK budget, and it is not the lease. The control
        plane used to send the lease expiry here, so an agent's budget was
        silently whatever the lease happened to be -- and once heartbeats
        existed the two disagreed outright, because the lease extended while
        the rollout ran and this number did not move. It now moves for exactly
        one reason: time passing.
        """
        return max(0.0, self.deadline_ts - now)


#: The state a reported check is in. Only ``measured`` carries a ``value``: a
#: check that could not run is never a zero.
CHECK_STATUSES = (
    "measured", "unavailable", "error", "abstained", "not_applicable",
)
#: What a reported check proves.
CHECK_ESTABLISHES = (
    "action_taken", "state_reached", "output_matched",
    "message_communicated", "call_returned_without_error",
)
#: How an episode ended. The agent states the class; who is answerable for it
#: is decided by the platform, from its own records.
TERMINATION_CLASSES = (
    "completed", "turn_limit", "token_limit", "time_limit", "error",
    "cancelled",
)
#: Most checks one report may carry.
MAX_REPORTED_CHECKS = 200
#: The largest integer seed: the largest a JSON number carries exactly in
#: every language (JavaScript's ``Number.MAX_SAFE_INTEGER``).
MAX_SAFE_SEED = 2**53 - 1
_CHECK_KEYS = frozenset({"id", "status", "value", "basis", "required",
                         "establishes"})


def finite_number(value: Any) -> float | None:
    """``value`` as a float when it is a finite number and not a bool, else
    None. Never raises: an integer too large for a float is not finite."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _short_text(value: Any, limit: int, what: str) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ContractViolation(
            f"{what} must be a non-empty string of at most {limit} characters")
    return value


def checks_to_wire(checks: Any) -> list[dict[str, Any]]:
    """The wire form of the checks an agent's own environment ran on an
    episode: ``[{"id", "status", "value", "basis", "required",
    "establishes"}]``, only ``id`` and ``status`` required. Raises
    :class:`ContractViolation` naming the row and the field at fault.

    AT MOST :data:`MAX_REPORTED_CHECKS` ROWS, THE FIRST ONES. A longer list
    used to be refused whole, which cost the episode every check it ran for
    stating one too many. The rows past the limit are not carried and not
    read; :func:`checks_over_limit` counts them, and the report states the
    count as ``checks_dropped`` (:meth:`RolloutOutput.to_wire`)."""
    if not isinstance(checks, (list, tuple)):
        raise ContractViolation("checks must be a list")
    rows: list[dict[str, Any]] = []
    for i, row in enumerate(checks[:MAX_REPORTED_CHECKS]):
        where = f"checks[{i}]"
        if not isinstance(row, Mapping):
            raise ContractViolation(f"{where} must be an object")
        extra = set(row) - _CHECK_KEYS
        if extra:
            raise ContractViolation(
                f"{where} carries unknown key(s) {sorted(map(str, extra))}")
        out: dict[str, Any] = {
            "id": _short_text(row.get("id"), 200, f"{where}.id")}
        status = row.get("status")
        if status not in CHECK_STATUSES:
            raise ContractViolation(
                f"{where}.status must be one of {list(CHECK_STATUSES)}")
        out["status"] = status
        value = row.get("value")
        if value is not None:
            number = finite_number(value)
            if number is None:
                raise ContractViolation(
                    f"{where}.value must be a finite number or None")
            if status != "measured":
                raise ContractViolation(
                    f"{where}.value is stated but its status is {status!r}; "
                    f"only a measured check has a value")
            out["value"] = number
        if row.get("basis") is not None:
            out["basis"] = _short_text(row["basis"], 200, f"{where}.basis")
        if row.get("required") is not None:
            if not isinstance(row["required"], bool):
                raise ContractViolation(f"{where}.required must be a bool")
            out["required"] = row["required"]
        if row.get("establishes") is not None:
            if row["establishes"] not in CHECK_ESTABLISHES:
                raise ContractViolation(
                    f"{where}.establishes must be one of "
                    f"{list(CHECK_ESTABLISHES)}")
            out["establishes"] = row["establishes"]
        rows.append(out)
    return rows


def checks_over_limit(checks: Any) -> int:
    """How many of ``checks`` :func:`checks_to_wire` leaves behind: the rows
    past :data:`MAX_REPORTED_CHECKS`. 0 for anything that is not a list."""
    if not isinstance(checks, (list, tuple)):
        return 0
    return max(len(checks) - MAX_REPORTED_CHECKS, 0)


def termination_to_wire(termination: Any) -> dict[str, Any]:
    """The wire form of how the episode ended: ``{"class", "detail"}``."""
    if not isinstance(termination, Mapping):
        raise ContractViolation("termination must be an object")
    extra = set(termination) - {"class", "detail"}
    if extra:
        raise ContractViolation(
            f"termination carries unknown key(s) {sorted(map(str, extra))}")
    if termination.get("class") not in TERMINATION_CLASSES:
        raise ContractViolation(
            f"termination.class must be one of {list(TERMINATION_CLASSES)}")
    out: dict[str, Any] = {"class": termination["class"]}
    if termination.get("detail") is not None:
        out["detail"] = _short_text(
            termination["detail"], 500, "termination.detail")
    return out


def seeds_to_wire(seeds: Any) -> dict[str, Any]:
    """The wire form of the seeds the episode ran under:
    ``{"environment", "sampling"}``, each an integer or a short string.

    ONE RULE WITH THE NODE SDK's ``seedsToWire``, which reads the same
    ``outcome.json``: a string is 1 to 128 characters; an integer is one a
    JSON number carries exactly in every language (``|n| <= 2**53 - 1``), and
    a float that is such an integer (``7.0``, which is how JavaScript and
    many JSON writers spell 7) is that integer. A bool, a fraction, NaN and a
    larger integer are refused. Python took any ``int`` and no ``float``, so
    the two SDKs disagreed on ``7.0`` and on ``2**60``."""
    if not isinstance(seeds, Mapping):
        raise ContractViolation("seeds must be an object")
    extra = set(seeds) - {"environment", "sampling"}
    if extra:
        raise ContractViolation(
            f"seeds carries unknown key(s) {sorted(map(str, extra))}")
    out: dict[str, Any] = {}
    for key in ("environment", "sampling"):
        value = seeds.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            out[key] = _short_text(value, 128, f"seeds.{key}")
            continue
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or value != value or abs(value) > MAX_SAFE_SEED
                or value != int(value)):
            raise ContractViolation(
                f"seeds.{key} must be an integer or a string")
        out[key] = int(value)
    return out


#: Every key ``RolloutOutput.to_wire()`` may emit. Anything derived from the
#: inference wire or from the rollout record is absent by construction: this is
#: a closed dataclass, so an agent cannot assert a field it has no way to
#: know. The server validates independently, because a client-side check binds
#: only honest clients.
WIRE_KEYS = frozenset({
    "final_text", "tool_calls", "tool_calls_omitted_count", "artifacts",
    "events", "timing", "llm_call_count", "reward", "success",
    "agent_fingerprint", "exclusion", "checks", "checks_dropped",
    "termination", "seeds",
})


@dataclass
class RolloutOutput:
    """What the agent reports back. Flywheel-authored fields only.

    ``tool_calls`` defaults to ``None``, not ``[]``. The distinction is
    load-bearing: ``None`` means "not reported", ``[]`` means "I observed zero
    tool calls", which is a positive assertion. An agent that reports nothing
    must not accidentally assert zero, so the default is the honest one.
    """

    final_text: str
    tool_calls: list[ToolCallOutcome] | None = None
    tool_calls_omitted_count: int = 0
    llm_call_count: int | None = None
    #: name → digest string, OR — for small UTF-8 files — an inline body
    #: ``{"sha256", "size_bytes", "content"}``. The inline form exists because
    #: for a coding/REPL agent the FILES are the deliverable: a grader that
    #: can verify a digest but never read the file proves integrity while
    #: grading nothing. Values stay strings for binary/over-cap files.
    artifacts: dict[str, Any] = field(default_factory=dict)
    reward: float | None = None
    success: bool | None = None
    exclusion_reason: ExclusionReason | None = None
    exclusion_detail: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    agent_fingerprint: dict[str, Any] | None = None
    #: What the agent's own environment checked on this episode, beyond the
    #: one ``reward``/``success``: see :func:`checks_to_wire`. ``None`` is
    #: "not reported"; ``[]`` is not sent either (it states nothing).
    checks: list[dict[str, Any]] | None = None
    #: How many checks this episode ran that the report does NOT carry: the
    #: rows past :data:`MAX_REPORTED_CHECKS`, and every row of a ``checks``
    #: the wire could not carry at all (``runner._wire_payload``). Sent as
    #: ``checks_dropped`` when positive, so a reader of the stored episode can
    #: tell "these were all its checks" from "these are the ones that fit".
    #: Only a positive integer is stated; anything else is not sent.
    checks_dropped: int = 0
    #: How the episode ended, ``{"class", "detail"}``, the class one of
    #: :data:`TERMINATION_CLASSES`. A statement, not a verdict: the platform
    #: decides from its own records whether the ending was the policy's.
    termination: dict[str, Any] | None = None
    #: The seeds the episode ran under, ``{"environment", "sampling"}``, so a
    #: replica can be told from a repeat.
    seeds: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
        if self.llm_call_count is not None and (
            type(self.llm_call_count) is not int or self.llm_call_count < 0
        ):
            raise ContractViolation(
                "llm_call_count must be a nonnegative integer or None"
            )
        d: dict[str, Any] = {
            "final_text": self.final_text,
            "tool_calls": (
                None if self.tool_calls is None
                else [tc.to_wire() for tc in self.tool_calls]
            ),
            "tool_calls_omitted_count": self.tool_calls_omitted_count,
            "artifacts": dict(self.artifacts),
            "events": list(self.events),
            "timing": dict(self.timing),
        }
        for k in ("llm_call_count", "reward", "success", "agent_fingerprint"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
        if self.exclusion_reason is not None:
            d["exclusion"] = {
                "reason": self.exclusion_reason,
                "detail": self.exclusion_detail,
            }
        dropped = self.checks_dropped
        if type(dropped) is not int or dropped < 0:
            dropped = 0
        # An empty list states nothing; anything else that is not a list is
        # refused by ``checks_to_wire``, never read as nothing -- ``{}`` and
        # ``""`` were, while the Node SDK refused them.
        if self.checks is not None and not (
                isinstance(self.checks, (list, tuple)) and not self.checks):
            d["checks"] = checks_to_wire(self.checks)
            dropped += checks_over_limit(self.checks)
        if dropped:
            d["checks_dropped"] = dropped
        if self.termination is not None:
            d["termination"] = termination_to_wire(self.termination)
        if self.seeds is not None:
            seeds = seeds_to_wire(self.seeds)
            if seeds:
                d["seeds"] = seeds
        extra = set(d) - WIRE_KEYS
        if extra:
            raise ContractViolation(
                f"RolloutOutput would emit unknown key(s) {sorted(extra)}"
            )
        return d


@dataclass(frozen=True)
class RulebookRequest:
    """The body of ``POST /agents/{agent_id}/rulebook``: the workflow rules of
    the Product this agent belongs to, one prose rule per string, in force as a
    whole. An empty list retracts them.

    STRINGS ONLY, IN ORDER. A rule's id is derived from its text by the control
    plane, never authored here, and the list is the rulebook as written. A
    ``set`` is refused because its iteration order can change between
    processes, and rules re-declared on every start in a different order would
    register a new version each time.

    SHAPE ONLY. What makes a rule acceptable -- its length, a duplicate, JSON
    text, a control character, a credential value, how many there may be -- is
    the control plane's to decide, and it refuses by cause and rule index. A
    second copy of those bounds here would be one more thing to drift, so this
    refuses only what cannot be a list of rules at all.
    """

    #: Stored as a tuple, whichever of a list or a tuple was passed.
    rules: Sequence[str]

    def __post_init__(self) -> None:
        rules = self.rules
        if not isinstance(rules, (list, tuple)):
            hint = (
                " (a single string is one rule: pass it inside a list)"
                if isinstance(rules, str) else ""
            )
            raise ContractViolation(
                "rules must be a list of strings, one rule per string; got "
                f"{_json_type(rules)}{hint}"
            )
        for index, rule in enumerate(rules):
            if not isinstance(rule, str):
                raise ContractViolation(
                    f"rule {index} must be a string; got {_json_type(rule)}"
                )
        object.__setattr__(self, "rules", tuple(rules))

    def to_wire(self) -> dict[str, Any]:
        return {"rules": list(self.rules)}


#: What a rulebook registration did. The order is the schema's enum order.
RulebookStatus = Literal["registered", "retracted", "unchanged"]
_RULEBOOK_STATUSES: tuple[str, ...] = ("registered", "retracted", "unchanged")


@dataclass(frozen=True)
class RulebookReceipt:
    """The control plane's answer to a :class:`RulebookRequest`.

    ``status`` is ``registered`` (a new version carrying rules), ``retracted``
    (a new version carrying none) or ``unchanged`` (the rules equal the ones in
    force, so nothing was registered -- the answer every restart that
    re-declares the same rules gets). ``content_sha256`` names the version's
    bytes. The version fields are ``None`` only for an ``unchanged``
    retraction of a Product that never had rules.
    """

    status: RulebookStatus
    product_id: str
    version_id: str | None
    version: int | None
    content_sha256: str | None
    parent_sha256: str | None
    rule_count: int

    @classmethod
    def from_wire(cls, d: Any) -> RulebookReceipt:
        if not isinstance(d, dict):
            raise ContractViolation("a rulebook receipt must be a JSON object")
        missing = [f.name for f in fields(cls) if f.name not in d]
        if missing:
            raise ContractViolation(
                f"RulebookReceipt missing required field(s): {missing}"
            )
        wrong = []
        if d["status"] not in _RULEBOOK_STATUSES:
            wrong.append("status")
        if not isinstance(d["product_id"], str) or not d["product_id"]:
            wrong.append("product_id")
        if d["version_id"] is not None and not isinstance(d["version_id"], str):
            wrong.append("version_id")
        # WHOLE NUMBERS AS JSON HAS THEM: `2.0` is the schema's integer and
        # the Node SDK's `2`, so it is read -- and kept -- as `2`. Refusing it
        # made the same answer a receipt there and "cannot read" here.
        version = (None if d["version"] is None
                   else _json_whole_number(d["version"]))
        if d["version"] is not None and (version is None or version < 1):
            wrong.append("version")
        for name in ("content_sha256", "parent_sha256"):
            if d[name] is not None and not is_lower_sha256(d[name]):
                wrong.append(name)
        rule_count = _json_whole_number(d["rule_count"])
        if rule_count is None or rule_count < 0:
            wrong.append("rule_count")
        if wrong:
            raise ContractViolation(
                f"RulebookReceipt carries invalid field(s): {wrong}"
            )
        return cls(**{
            **{f.name: d[f.name] for f in fields(cls)},
            "version": version, "rule_count": rule_count,
        })
