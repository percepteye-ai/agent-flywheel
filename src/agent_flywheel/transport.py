"""Transports.

``attach`` (pull) is the one that ships. The flywheel dials out, claims work,
runs it, reports back. No inbound connection, no listening socket, no public
DNS, no TLS termination on the customer's side, no firewall change. For a
flywheel inside a corporate network that is the difference between "install a
package" and "file a change request with the network team".

``serve`` (push) is deliberately absent rather than half-present. The
``FlywheelTransport`` protocol exists so it can be added without the runner
learning which one is in use.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable
from urllib.parse import quote

# NOT imported at module scope. `_http` pulls in httpx, which is 83-87% of
# this package's import time and ~88% of its resident memory -- and a process
# that only reads the promoted policy, or only captures locally, never opens a
# socket at all. Deferring it means those customers stop paying for a transport
# they do not use. `policy.py` has always done this; these were the paths that
# forced it anyway.
if TYPE_CHECKING:
    from ._http import ControlPlaneClient
from .contract import (
    _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS,
    _PRODUCTION_TURN_BATCH_MAX_ITEMS,
    _TURN_FEEDBACK_BATCH_MAX_ITEMS,
    RolloutOutput,
    RolloutRequest,
    RulebookReceipt,
    RulebookRequest,
    _path_segment,
    _production_agent_identifier,
    _production_identifier,
    _production_turn_identifier,
    production_outcome,
    turn_feedback,
)
from .errors import ConfigurationError, ContractViolation, TransportError
from .execution_identity import is_lower_sha256
from .outcomes import _normalize_task_id

__all__ = [
    "AttachTransport",
    "FlywheelTransport",
    "RulebookUpload",
    "declare_rulebook_in_background",
    "declared_rulebook",
    "rulebook_refusal",
]


def _body_digest(value: Any) -> str:
    """The digest an idempotency key carries for a request body.

    ONE DERIVATION for every keyed body, and the Node SDK's twin reproduces it
    byte for byte (``stableStringify``: sorted keys, Python's separators,
    ``ensure_ascii`` escapes), so the same body gets the same key from either
    package. A second spelling here is how the two would stop agreeing.
    """
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:32]


class RulebookUpload(NamedTuple):
    """What :meth:`AttachTransport.register_rulebook` did. Exactly one field
    is set: the control plane's receipt, or the sentence naming why there is
    none."""

    receipt: RulebookReceipt | None
    refusal: str | None


def rulebook_refusal(exc: BaseException) -> str:
    """The sentence saying why a rulebook declaration came back with no
    receipt.

    ONE BUILDER, and the Node SDK's ``rulebookRefusal`` builds the same
    sentence from the same error, so a customer reads one answer whichever
    package their agent runs. The error's message is kept whole because it
    carries the ACTUAL cause: the client puts the route and status in it, then
    the control plane's own refusal -- ``rulebook_invalid`` with the rule's
    cause and index, ``rulebook_unattributed`` with the binding status that
    attributes the agent to no single Product, a key without the scope -- or,
    when no answer came at all, the transport failure. One message for every
    cause would send an operator to fix the wrong thing.

    "NOT REGISTERED" ONLY WHEN IT IS KNOWN. A failure after the request may
    have landed -- a read timeout, a dropped connection, a 502 from a proxy
    whose upstream finished -- is :attr:`TransportError.ambiguous`, and the
    rules may be in force: saying the rulebook is unchanged would send an
    operator to believe the old one still is. That failure is named as
    unknown, with the one remedy that is safe either way.
    """
    if isinstance(exc, TransportError) and exc.ambiguous:
        return (
            f"whether this agent's rules were registered is unknown: {exc}. "
            f"The control plane may have received them before the failure; "
            f"declaring the same rules again is safe, because rules equal to "
            f"the ones in force are answered unchanged."
        )
    return (
        f"this agent's rules were not registered: {exc}. The rulebook in "
        f"force, if any, is unchanged."
    )


#: How long a caller about to close the client a declaration is using waits
#: for its answer: ``serve()`` when its loop returns, :meth:`Attachment.close`.
#: BOUNDED for the reason ``close()``'s flush grace is: an unresponsive
#: control plane must not hold a customer's process open at exit. Rules not
#: declared by then are declared by the next start, which re-declares them.
RULEBOOK_SETTLE_GRACE_S = 5.0


def declare_rulebook_in_background(
    transport: AttachTransport,
    request: RulebookRequest,
    settle: Callable[[RulebookUpload], None],
) -> threading.Thread | None:
    """Declare ``request`` OFF the caller's startup path, and hand what
    happened to ``settle``.

    THE ONE WAY ``serve()`` and ``attach()`` declare rules. A rulebook route
    that is slow or hung -- the client's four attempts of up to 30s each --
    must cost the rules and nothing else: not ``attach()`` returning the
    policy it has already read, not ``serve()``'s policy read or claim loop,
    not a customer's startup. Declared inline, it cost all of those for up to
    two minutes. The Node plugin declares them off every chain its other
    startup reads wait on, for the same reason.

    On a DAEMON thread, so it never holds an interpreter open at exit; a
    caller that is about to close the client this uses joins the returned
    thread for :data:`RULEBOOK_SETTLE_GRACE_S` first. Returns ``None`` when
    the interpreter can no longer start threads (it is shutting down): the
    declaration then runs here, and has settled by the time this returns.

    Never raises, like :meth:`AttachTransport.register_rulebook`; a
    ``settle`` that raises is swallowed at the thread boundary, which must
    not print a traceback under this package's name for a failure that cost
    only the report of the rules.
    """
    def _run() -> None:
        try:
            upload = transport.register_rulebook(request)
        except Exception as exc:  # backstop; the method never raises
            upload = RulebookUpload(None, rulebook_refusal(exc))
        try:
            settle(upload)
        except Exception:
            pass

    thread = threading.Thread(
        target=_run, name="percepteye-rulebook", daemon=True)
    try:
        thread.start()
    except RuntimeError:
        _run()
        return None
    return thread


def declared_rulebook(rules: Any) -> RulebookRequest | None:
    """``rules=`` as ``serve()`` and ``attach()`` take it, checked at startup.

    ``None`` sends nothing -- the rulebook in force, whoever registered it, is
    left alone. Anything else must be a list of strings, and a value that is
    not one is a configuration mistake in front of the person making it, so it
    raises here, before anything connects, rather than surfacing later as rules
    that never registered.
    """
    if rules is None:
        return None
    try:
        return RulebookRequest(rules)
    except ContractViolation as exc:
        raise ConfigurationError(str(exc)) from None


@runtime_checkable
class FlywheelTransport(Protocol):
    """How work arrives and results leave. The runner never learns which.

    Two OPTIONAL extensions, both read with a ``getattr`` default so an
    implementation that predates them keeps working unchanged:
    ``delivery_envelope_s`` (how long one call can take, so the runner can
    reserve it) and ``accepts_delivery_deadline`` (whether ``report`` takes a
    ``deadline=`` keyword). A transport that
    declares neither is served with the runner's floor margin.
    """

    def claim(self, max_rollouts: int) -> tuple[list[RolloutRequest], int]:
        """Return (rollouts, poll_after_ms). An empty list is normal."""
        ...

    def heartbeat(self, rollout: RolloutRequest, *, progress: str | None = None) -> None:
        ...

    def report(self, rollout: RolloutRequest, output: RolloutOutput) -> dict[str, Any]:
        ...

    def abandon(self, rollout: RolloutRequest, *, reason: str, detail: str = "") -> None:
        ...


#: Statuses on ``GET /agents/{id}/capture`` that are the control plane's
#: answer, not a blip: a key it does not accept (401), a key or agent it will
#: not answer for (403), no such agent or no such route (404 -- a control
#: plane older than the read). The Node SDK's ``CAPTURE_READ_REFUSALS``.
CAPTURE_READ_REFUSALS = frozenset({401, 403, 404})


def capture_read_refusal(what: str) -> dict[str, Any]:
    """The verdict-read body for an answer that grants nothing: an answered
    refusal (``capture.turns`` false) whose reason names what the control
    plane said, so :func:`~agent_flywheel.production.capture_verdict` reads
    it as an answer. The Node SDK's ``captureReadRefusal``, word for word."""
    return {"capture": {"turns": False, "reason": (
        "the control plane did not grant this agent production capture when "
        f"asked ({what}), so nothing is uploaded. Turns are still captured "
        "locally.")}}


class AttachTransport:
    """Outbound-only pull transport.

    Claim is a long poll. An empty claim returns 200 with an empty list and a
    server-chosen ``poll_after_ms`` -- never a bare 204, which carries no
    backoff information and leaves the client to invent a fixed sleep.
    """

    #: DECLARED, never probed. ``FlywheelTransport`` is a Protocol a third
    #: party may implement, and calling ``report(..., deadline=...)`` on one
    #: that predates the keyword is a TypeError at the last step of a rollout.
    #: The runner reads this flag and passes the deadline only to a transport
    #: that says it takes one.
    accepts_delivery_deadline = True

    def __init__(
        self,
        client: ControlPlaneClient,
        agent_id: str,
        *,
        wait_ms: int = 25_000,
        lease_ms: int = 1_800_000,
    ) -> None:
        self._c = client
        self.agent_id = agent_id
        self.wait_ms = max(0, min(wait_ms, 25_000))
        self.lease_ms = max(1_000, min(lease_ms, 1_800_000))
        # Set only after an authenticated registration succeeds.  Older
        # servers/SDK clients naturally leave it absent and can continue to
        # serve ordinary, unbound rollouts.
        self.agent_execution_sha256: str | None = None

    @property
    def delivery_envelope_s(self) -> float:
        """Forwarded verbatim from the client that will spend it.

        The runner talks to a transport and never to the HTTP client, so the
        number it reserves has to arrive through here. Forwarded rather than
        recomputed: two copies of this arithmetic is exactly how a reserve and
        the cost it reserves for drift apart.
        """
        return self._c.delivery_envelope_s

    def health(self) -> dict[str, Any]:
        """Called at startup. Misconfiguration must fail here, not on rollout 1."""
        return self._c.request("GET", "/health")

    def register(self, spec: dict[str, Any]) -> dict[str, Any]:
        response = self._c.request(
            "POST", "/agents", json_body=spec,
            idempotency_key=f"reg-{quote(self.agent_id, safe='-._~')}",
        )
        execution_sha = spec.get("agent_execution_sha256")
        self.agent_execution_sha256 = (
            execution_sha if is_lower_sha256(execution_sha) else None
        )
        return response

    def register_rulebook(
        self, request: RulebookRequest | Sequence[str],
    ) -> RulebookUpload:
        """Declare the workflow rules of the Product this agent belongs to:
        ``POST /agents/{agent_id}/rulebook``. ``[]`` retracts them.

        THE PRODUCT IS THE CONTROL PLANE'S TO NAME. The wire carries none: the
        control plane attributes the rules to the one active Product that
        declares this agent, and refuses when no Product does or more than one
        does, rather than picking.

        FAILS OPEN, like :meth:`policy_current`: rules that did not register
        must not stop an agent from starting or serving, so nothing the
        control plane does or fails to do raises from here. What happened is
        RETURNED instead -- the receipt, or the sentence naming why there is
        none (:func:`rulebook_refusal`). Only a ``request`` that is not a list
        of strings raises, :class:`ContractViolation`, before anything is
        sent: that is a caller's mistake, not the control plane's answer.

        SAFE TO REPLAY. The control plane answers rules equal to the ones in
        force ``unchanged`` and registers nothing, so the client's default
        retry of an ambiguous failure -- a request that may already have
        landed -- is left on. The idempotency key is the turn batch's
        derivation (:func:`_body_digest`) over the body, so both SDKs send the
        same key for the same rules; the route's own ``unchanged`` rule, not
        the key, is what makes the replay harmless.
        """
        if not isinstance(request, RulebookRequest):
            request = RulebookRequest(request)
        body = request.to_wire()
        try:
            # Inside the try: an agent id that cannot be a path segment is
            # named as why there is no receipt, like any other cause.
            agent = _path_segment(self.agent_id, "agent_id")
            reply = self._c.request(
                "POST",
                f"/agents/{agent}/rulebook",
                json_body=body,
                idempotency_key=f"rulebook-{agent}-{_body_digest(body)}",
            )
        except Exception as exc:  # see docstring
            return RulebookUpload(None, rulebook_refusal(exc))
        try:
            return RulebookUpload(RulebookReceipt.from_wire(reply), None)
        except ContractViolation as exc:
            # The control plane ANSWERED 2xx, so "not registered" would be a
            # guess. A receipt outside the published contract is named as
            # what it is: an answer this package cannot read.
            return RulebookUpload(None, (
                f"the control plane answered this agent's rules with a receipt "
                f"this SDK cannot read ({exc}), so whether they are in force "
                f"is unknown."
            ))

    def policy_current(self) -> dict[str, Any]:
        """What policy this agent should be running, per the control plane.

        The last mile of the flywheel: training produces a better model, and
        this is how an agent learns about it. A READ — the agent is TOLD its
        arm and never chooses one, which is what keeps results attributable.

        Never raises into the caller's startup. A control plane that cannot
        answer must not stop an agent from running, so an empty answer means
        "keep doing what you were doing" — exactly the behaviour before this
        endpoint existed.
        """
        try:
            return self.policy_current_or_raise()
        except Exception:  # see docstring
            return {}

    def policy_current_or_raise(self) -> dict[str, Any]:
        """The same read, with a failure left to the caller.

        For :class:`~agent_flywheel.policy.PolicySource`, the one reader that
        must tell an unreachable control plane from one that answered. Its
        failed refresh keeps the last good snapshot and says so through
        ``last_refresh_ok`` and ``stale_for_s``; read through the fail-open
        method above, the failure arrived as ``{}``, a SUCCESSFUL answer of
        nothing deployed, so an outage reported a fresh, healthy policy.
        """
        return self._c.request("GET", "/policy/current") or {}

    def prompt_current(self) -> dict[str, Any]:
        """The approved optimized system prompt for this agent, or ``{}``.

        The other half of the last mile. ``policy_current`` says which MODEL to
        run; training also produces a system prompt, co-optimized with that
        model and gated by the same kind of human approval — and until this
        existed an agent could learn only THAT the prompt had moved, never what
        it moved to. ``/policy/current``'s ``prompt_generation`` names which
        approved prompt is served, and :class:`~agent_flywheel.PolicySource`
        reads this only when that differs from the one it holds.

        A SEPARATE call from ``policy_current`` on purpose. That one is polled
        and identity-sized; a system prompt is unbounded text, so putting it in
        the poll would make every tick carry a payload the caller already has.

        Same fail-open contract as everything else here: ``{}`` means keep the
        prompt you have. A control plane that cannot answer must never blank a
        running agent's instructions.
        """
        try:
            return self._c.request("GET", "/prompt/current") or {}
        except Exception:  # see docstring
            return {}

    def claim(self, max_rollouts: int) -> tuple[list[RolloutRequest], int]:
        claim_body: dict[str, Any] = {
            "agent_id": self.agent_id,
            "max": max(1, min(max_rollouts, 16)),
            "wait_ms": self.wait_ms,
            "lease_ms": self.lease_ms,
        }
        if self.agent_execution_sha256 is not None:
            claim_body["agent_execution_sha256"] = self.agent_execution_sha256
        body = self._c.request(
            "POST",
            "/rollouts/claim",
            json_body=claim_body,
            idempotency_key=str(uuid.uuid4()),
            timeout=(self.wait_ms / 1000.0) + 10.0,
            # Claiming is NOT replayable. The control plane does not honour
            # Idempotency-Key on this route, so resending a claim that may
            # already have been processed leases a SECOND batch while the first
            # is held by nobody -- those rollouts then sit until their leases
            # lapse. Only a failure that never reached the server is retried.
            retry_if_ambiguous=False,
        )
        raw = body.get("rollouts") or []
        poll_after = int(body.get("poll_after_ms") or 1000)
        return [RolloutRequest.from_wire(r) for r in raw], poll_after

    def heartbeat(self, rollout: RolloutRequest, *, progress: str | None = None) -> None:
        payload: dict[str, Any] = {
            "lease_token": rollout.lease_token,
            "extend_ms": min(self.lease_ms, 600_000),
        }
        if progress:
            payload["progress"] = progress
        self._c.request(
            "POST", f"/rollouts/{rollout.rollout_id}/heartbeat", json_body=payload
        )

    def report(
        self,
        rollout: RolloutRequest,
        output: RolloutOutput,
        *,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        headers = (
            {"X-Percepteye-Lease-Token": rollout.lease_token}
            if rollout.lease_token else None
        )
        return self._c.request(
            "POST",
            f"/rollouts/{_path_segment(rollout.rollout_id, 'rollout_id')}"
            "/result",
            json_body=output.to_wire(),
            # The server keys on (rollout_id, attempt); an identical replay is
            # acknowledged as a duplicate rather than double-counted.
            idempotency_key=f"result-{rollout.rollout_id}",
            headers=headers,
            # Bounded by what the runner RESERVED for delivery. This call used
            # to pass no timeout at all -- unlike claim, one line above -- so a
            # report attempted with the runner's ten-second margin left ran the
            # client's full four-attempt envelope instead.
            deadline=deadline,
        )

    def capture_current(self, agent_id: str) -> dict[str, Any]:
        """May this key upload this agent's production turns, right now?

        A READ, and that is the point. The verdict used to be reachable only
        through ``POST /agents`` -- an upsert -- so re-asking cost a write, and
        a registration that omits ``discovered_agent`` is recorded as
        introspection ``disabled``, which stops that agent's description
        driving workflow generation. Asking a permission question should not
        change an unrelated column.

        What it makes possible: an operator enables capture on a running
        agent, and the agent notices. Before this, consent granted after
        startup was invisible until the process restarted.

        FAILS CLOSED. An unreachable control plane returns ``{}``, which the
        caller reads as "no", so a network blip can never be mistaken for
        permission. That is the opposite of :meth:`policy_current`, which fails
        OPEN because keeping the current policy is the safe answer there -- and
        the two must not be made to look alike, because their safe directions
        are opposite.

        A DEFINITIVE ANSWER IS AN ANSWER, though. A 401, 403 or 404 on this
        read, or a 2xx body with no ``capture`` block in it, is the control
        plane deciding, not a blip: it will say the same thing to the next
        read. Returned as ``{}`` it read as "consent unknown", so an outcome
        reported meanwhile was kept on disk for good and every report re-asked.
        Each is returned as an answered refusal naming what was said
        (:func:`capture_read_refusal`); only an exception with no answer, or
        a status outside :data:`CAPTURE_READ_REFUSALS`, is ``{}``.
        """
        if _production_agent_identifier(agent_id) is None:
            return {}
        try:
            body = self._c.request(
                "GET", f"/agents/{_path_segment(agent_id, 'agent_id')}/capture")
        except TransportError as exc:
            status = exc.status_code
            if status in CAPTURE_READ_REFUSALS:
                return capture_read_refusal(str(exc))
            return {}
        except Exception:  # see docstring
            return {}
        if isinstance(body, dict) and isinstance(body.get("capture"), dict):
            return body
        return capture_read_refusal(
            "it answered with no capture verdict in the body")

    def report_turns(
        self, agent_id: str, turns: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Upload a batch of captured production turns.

        RETRYING AN AMBIGUOUS SEND IS SAFE HERE, unlike a claim. The server
        keys each turn on (org, agent, turn_id) and answers an identical
        resubmission as a duplicate; a claim, by contrast, would hand out a
        second piece of work. So this leaves the client's default retry
        behaviour alone rather than disabling it.

        The idempotency key covers the BATCH, so a network-level replay of the
        same bytes is collapsed before the per-turn rule is reached.
        """
        if _production_agent_identifier(agent_id) is None:
            raise ContractViolation(
                "production turn agent_id does not match the published "
                "production agent identifier grammar"
            )
        if (
            not isinstance(turns, list)
            or not 1 <= len(turns) <= _PRODUCTION_TURN_BATCH_MAX_ITEMS
        ):
            raise ContractViolation(
                "production turn batches must contain between 1 and "
                f"{_PRODUCTION_TURN_BATCH_MAX_ITEMS} turns"
            )
        for index, turn in enumerate(turns):
            if not isinstance(turn, dict):
                raise ContractViolation(
                    f"production turn {index} must be an object"
                )
            if _production_turn_identifier(turn.get("turn_id")) is None:
                raise ContractViolation(
                    f"production turn {index} has an invalid turn_id"
                )
            if (
                "conversation_id" in turn
                and _production_identifier(turn.get("conversation_id")) is None
            ):
                raise ContractViolation(
                    f"production turn {index} has an invalid conversation_id"
                )
            # An explicit ``"task_id": None`` is refused too: the wire says
            # "no task id" by omitting the key, and the intake refuses a null.
            if (
                "task_id" in turn
                and _normalize_task_id(turn["task_id"]) is None
            ):
                raise ContractViolation(
                    f"production turn {index} has an invalid task_id"
                )
        digest = _body_digest(turns)
        return self._c.request(
            "POST",
            "/turns",
            json_body={"agent_id": agent_id, "turns": turns},
            idempotency_key=(
                f"turns-{quote(agent_id, safe='-._~')}-{digest}"
            ),
        )

    def report_outcomes(
        self, agent_id: str, outcomes: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Upload a batch of reported outcomes: ``POST /outcomes``.

        How production turns or conversations actually turned out, each in
        the shape :func:`~agent_flywheel.contract.production_outcome`
        validates -- which every outcome is passed through again here, so a
        batch that leaves is one the published schema accepts.

        RETRYING IS SAFE, as it is for :meth:`report_turns`: the control
        plane keys each outcome on its ``outcome_id`` and answers a replay as
        a duplicate. The idempotency key is the turn batch's derivation over
        the outcomes, so both SDKs send the same key for the same batch.
        """
        if _production_agent_identifier(agent_id) is None:
            raise ContractViolation(
                "production outcome agent_id does not match the published "
                "production agent identifier grammar"
            )
        if (
            not isinstance(outcomes, list)
            or not 1 <= len(outcomes) <= _PRODUCTION_OUTCOME_BATCH_MAX_ITEMS
        ):
            raise ContractViolation(
                "production outcome batches must contain between 1 and "
                f"{_PRODUCTION_OUTCOME_BATCH_MAX_ITEMS} outcomes"
            )
        body = [production_outcome(outcome) for outcome in outcomes]
        return self._c.request(
            "POST",
            "/outcomes",
            json_body={"agent_id": agent_id, "outcomes": body},
            idempotency_key=(
                f"outcomes-{quote(agent_id, safe='-._~')}-{_body_digest(body)}"
            ),
        )

    def report_turn_feedback(
        self, agent_id: str, feedback: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Upload a batch of per-turn feedback: ``POST /turn-feedback``.

        What the user did with one turn of one episode, each in the shape
        :func:`~agent_flywheel.contract.turn_feedback` validates -- which
        every report is passed through again here.

        RETRYING IS SAFE, as it is for :meth:`report_outcomes`: the control
        plane keys each report on its ``feedback_id`` and answers a replay as
        a duplicate. The idempotency key is the outcome batch's derivation
        over the feedback, so both SDKs send the same key for the same batch.
        """
        if _production_agent_identifier(agent_id) is None:
            raise ContractViolation(
                "turn feedback agent_id does not match the published "
                "production agent identifier grammar"
            )
        if (
            not isinstance(feedback, list)
            or not 1 <= len(feedback) <= _TURN_FEEDBACK_BATCH_MAX_ITEMS
        ):
            raise ContractViolation(
                "turn feedback batches must contain between 1 and "
                f"{_TURN_FEEDBACK_BATCH_MAX_ITEMS} reports"
            )
        body = [turn_feedback(report) for report in feedback]
        return self._c.request(
            "POST",
            "/turn-feedback",
            json_body={"agent_id": agent_id, "feedback": body},
            idempotency_key=(
                "turn-feedback-"
                f"{quote(agent_id, safe='-._~')}-{_body_digest(body)}"
            ),
        )

    def abandon(self, rollout: RolloutRequest, *, reason: str, detail: str = "") -> None:
        """Give the work back, typed.

        This is not a low reward. It is 'do not train on this' -- a substrate
        fault, an expired credential, a crash. Reporting a crashed rollout as a
        zero-scoring one teaches the policy that its own behaviour caused an
        outcome it had nothing to do with.
        """
        self._c.request(
            "POST",
            f"/rollouts/{_path_segment(rollout.rollout_id, 'rollout_id')}"
            "/abandon",
            json_body={
                "lease_token": rollout.lease_token,
                "reason": reason,
                "detail": detail[:2000],
            },
            idempotency_key=f"abandon-{rollout.rollout_id}",
        )


def sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise TransportError(msg)
