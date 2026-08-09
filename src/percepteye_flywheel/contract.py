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
from dataclasses import dataclass, field
from typing import Any, Literal

from .errors import ContractViolation

__all__ = [
    "CONTRACT_VERSION",
    "ExclusionReason",
    "Outcome",
    "RolloutOutput",
    "RolloutRequest",
    "ToolCallOutcome",
]

CONTRACT_VERSION = "flywheel/1"

Outcome = Literal["ok", "failed", "unknown"]

ExclusionReason = Literal[
    "flywheel_crash",
    "substrate_unavailable",
    "credential_expired",
    "deadline_exceeded",
    "entrypoint_error",
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
        for k in ("output", "error", "error_class", "latency_ms"):
            v = getattr(self, k)
            if v is not None:
                d[k] = v
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
            lease_token=d.get("lease_token"),
            metadata=dict(d.get("metadata") or {}),
        )

    def seconds_remaining(self, now: float) -> float:
        """Time budget from the SERVER's deadline, never from a local env var."""
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
    artifacts: dict[str, str] = field(default_factory=dict)
    reward: float | None = None
    success: bool | None = None
    exclusion_reason: ExclusionReason | None = None
    exclusion_detail: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    timing: dict[str, Any] = field(default_factory=dict)
    agent_fingerprint: dict[str, Any] | None = None

    def to_wire(self) -> dict[str, Any]:
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
