"""Typed errors.

Every error here is raised at REGISTRATION, before any rollout exists, or is a
transport-level fault. None of them fires mid-rollout: a failure to *emit* a
rollout must raise loudly, but an exception thrown by the customer's agent
*inside* a rollout is caught and reported as a failed rollout, never raised.

That split is deliberate. Fail open on the customer's application path -- never
break their agent. Fail closed on the training path -- a silently dropped
rollout is not missing data, it is a missing member of a comparison group,
which biases every other member's score.
"""
from __future__ import annotations

__all__ = [
    "ConfigurationError",
    "ContractViolation",
    "EntrypointError",
    "FlywheelError",
    "LeaseLost",
    "ModeNotAvailable",
    "RoutingNotInstallable",
    "TransportError",
]


class FlywheelError(Exception):
    """Base for everything this package raises."""


class ConfigurationError(FlywheelError):
    """The SDK cannot start. Raised by configure()/serve() at import or startup.

    Misconfiguration must fail at init, loudly. A previous SDK shipped a
    health-check function with zero call sites and defaulted a missing API key
    to the literal string "dev-key", so a misconfigured deployment ran happily
    and reported nothing for weeks.
    """


class EntrypointError(ConfigurationError):
    """The customer's agent object exposes no callable we can drive.

    Raised at REGISTRATION. Resolving mid-rollout would turn a typo into a
    missing member of a comparison group, discovered hours later.
    """


class RoutingNotInstallable(ConfigurationError):
    """We cannot guarantee the agent's LLM calls reach our endpoint.

    There are two binding outcomes and no third: BOUND, or this. There is no
    degraded-routing state, because the failure is silent and its destination
    is a live third party -- the rollout completes, the agent returns a
    plausible final_text, and nothing distinguishes it from a clean rollout.

    Carries machine-readable ``reason`` and ``evidence`` plus a printed remedy.
    """

    def __init__(self, reason: str, evidence: str, remedy: str = "") -> None:
        self.reason = reason
        self.evidence = evidence
        self.remedy = remedy
        msg = f"[{reason}] {evidence}"
        if remedy:
            msg += f"\n\n  remedy: {remedy}"
        super().__init__(msg)


class ContractViolation(FlywheelError):
    """A value that cannot be represented on the flywheel/1 wire.

    Raised by the contract types themselves, at construction, so the dangerous
    shape is unconstructible rather than merely discouraged.
    """


class TransportError(FlywheelError):
    """The control plane could not be reached, or answered unusably.

    ``status_code`` is the HTTP status when there WAS one, and ``None`` when
    the request never produced a response at all. Callers need the difference:
    a 403 will refuse the identical payload forever, while a connection reset
    will not, and a caller deciding whether to keep a result for redelivery
    cannot tell those apart from the message text without parsing English.

    ``ambiguous`` is True when the control plane MAY HAVE PERFORMED the
    request although this error was raised: some attempt failed after the
    request could have reached it -- a read timeout, a dropped connection, a
    5xx other than 503 -- or it answered 2xx with a body that could not be
    read. False means no attempt can have been performed: each was never sent,
    or was refused (a 4xx, or a 503 saying it was not handled). ONLY THE CLIENT
    KNOWS, because only it saw every attempt: a retried request whose LAST
    attempt was never sent may have landed on the first, and the message names
    the last one alone. A caller about to tell someone "that did not happen"
    asks this.
    """

    def __init__(self, message: str, *, status_code: int | None = None,
                 ambiguous: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.ambiguous = ambiguous


class LeaseLost(TransportError):
    """This rollout's lease is gone: stop working on it.

    STOP WORK, NOT DISCARD THE WORK -- this used to say "stop work and
    discard", and the delivery path read it literally, deleting a spooled
    result the moment the control plane answered 410. Only an identity-sealed
    rollout's report route answers 410 at all, and it answers it forever once
    the reaper has cleared the lease, so "discard" named the one case where the
    local copy was the last one. The ROW is lost; the result is kept, named,
    and reported out loud (see ``RolloutRunner._settle``).
    """


class ModeNotAvailable(ConfigurationError):
    """The requested mode is not available in this contract version."""
