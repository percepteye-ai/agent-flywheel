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
import re
from collections.abc import Sequence
from dataclasses import dataclass, field, fields
from typing import Any, Literal

from .errors import ContractViolation

#: Imported rather than restated so the contract validates against exactly
#: the field set the extractor produces -- one definition, never a twin
#: that can drift out from under the other half.
from .evidence import IDENTITY_FIELDS as _IDENTITY_FIELDS
from .execution_identity import is_lower_sha256

__all__ = [
    "CONTRACT_VERSION",
    "ExclusionReason",
    "Outcome",
    "RolloutOutput",
    "RolloutRequest",
    "RulebookReceipt",
    "RulebookRequest",
    "RulebookStatus",
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


def _production_identifier(value: Any) -> str | None:
    """Return an exact identifier representable on the production-turn wire."""
    if (
        not isinstance(value, str)
        or not value
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
        or len(value) > _PRODUCTION_IDENTIFIER_MAX_CHARS
        or _PRODUCTION_TURN_IDENTIFIER_RE.fullmatch(value) is None
    ):
        return None
    return value


def _production_agent_identifier(value: Any) -> str | None:
    """Return an exact wire id safe as the agent URL path segment."""
    return _production_turn_identifier(value)

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


#: Every key ``RolloutOutput.to_wire()`` may emit. Anything derived from the
#: inference wire or from the rollout record is absent by construction: this is
#: a closed dataclass, so an agent cannot assert a field it has no way to
#: know. The server validates independently, because a client-side check binds
#: only honest clients.
WIRE_KEYS = frozenset({
    "final_text", "tool_calls", "tool_calls_omitted_count", "artifacts",
    "events", "timing", "llm_call_count", "reward", "success",
    "agent_fingerprint", "exclusion",
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
                f"{type(rules).__name__}{hint}"
            )
        for index, rule in enumerate(rules):
            if not isinstance(rule, str):
                raise ContractViolation(
                    f"rule {index} must be a string; got {type(rule).__name__}"
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
        version = d["version"]
        if version is not None and (type(version) is not int or version < 1):
            wrong.append("version")
        for name in ("content_sha256", "parent_sha256"):
            if d[name] is not None and not is_lower_sha256(d[name]):
                wrong.append(name)
        if type(d["rule_count"]) is not int or d["rule_count"] < 0:
            wrong.append("rule_count")
        if wrong:
            raise ContractViolation(
                f"RulebookReceipt carries invalid field(s): {wrong}"
            )
        return cls(**{f.name: d[f.name] for f in fields(cls)})
