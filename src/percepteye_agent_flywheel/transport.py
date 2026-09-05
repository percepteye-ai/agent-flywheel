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
import time
import uuid
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

# NOT imported at module scope. `_http` pulls in httpx, which is 83-87% of
# this package's import time and ~88% of its resident memory -- and a process
# that only reads the promoted policy, or only captures locally, never opens a
# socket at all. Deferring it means those customers stop paying for a transport
# they do not use. `policy.py` has always done this; these were the paths that
# forced it anyway.
if TYPE_CHECKING:
    from ._http import ControlPlaneClient
from .contract import RolloutOutput, RolloutRequest
from .errors import TransportError

__all__ = ["AttachTransport", "FlywheelTransport"]


@runtime_checkable
class FlywheelTransport(Protocol):
    """How work arrives and results leave. The runner never learns which."""

    def claim(self, max_rollouts: int) -> tuple[list[RolloutRequest], int]:
        """Return (rollouts, poll_after_ms). An empty list is normal."""
        ...

    def heartbeat(self, rollout: RolloutRequest, *, progress: str | None = None) -> None:
        ...

    def report(self, rollout: RolloutRequest, output: RolloutOutput) -> dict[str, Any]:
        ...

    def abandon(self, rollout: RolloutRequest, *, reason: str, detail: str = "") -> None:
        ...


class AttachTransport:
    """Outbound-only pull transport.

    Claim is a long poll. An empty claim returns 200 with an empty list and a
    server-chosen ``poll_after_ms`` -- never a bare 204, which carries no
    backoff information and leaves the client to invent a fixed sleep.
    """

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

    def health(self) -> dict[str, Any]:
        """Called at startup. Misconfiguration must fail here, not on rollout 1."""
        return self._c.request("GET", "/health")

    def register(self, spec: dict[str, Any]) -> dict[str, Any]:
        return self._c.request(
            "POST", "/agents", json_body=spec,
            idempotency_key=f"reg-{self.agent_id}",
        )

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
            return self._c.request("GET", "/policy/current") or {}
        except Exception:  # see docstring
            return {}

    def prompt_current(self) -> dict[str, Any]:
        """The approved optimized system prompt for this agent, or ``{}``.

        The other half of the last mile. ``policy_current`` says which MODEL to
        run; training also produces a system prompt, co-optimized with that
        model and gated by the same kind of human approval — and until this
        existed an agent could learn only THAT the prompt had moved, never what
        it moved to. ``PolicySnapshot.serving_generation`` changes for a prompt
        change, and there was nothing to fetch in response.

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
        body = self._c.request(
            "POST",
            "/rollouts/claim",
            json_body={
                "agent_id": self.agent_id,
                "max": max(1, min(max_rollouts, 16)),
                "wait_ms": self.wait_ms,
                "lease_ms": self.lease_ms,
            },
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

    def report(self, rollout: RolloutRequest, output: RolloutOutput) -> dict[str, Any]:
        return self._c.request(
            "POST",
            f"/rollouts/{rollout.rollout_id}/result",
            json_body=output.to_wire(),
            # The server keys on (rollout_id, attempt); an identical replay is
            # acknowledged as a duplicate rather than double-counted.
            idempotency_key=f"result-{rollout.rollout_id}",
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
        """
        try:
            return self._c.request(
                "GET", f"/agents/{agent_id}/capture") or {}
        except Exception:  # see docstring
            return {}

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
        digest = hashlib.sha256(
            json.dumps(turns, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:32]
        return self._c.request(
            "POST",
            "/turns",
            json_body={"agent_id": agent_id, "turns": turns},
            idempotency_key=f"turns-{agent_id}-{digest}",
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
            f"/rollouts/{rollout.rollout_id}/abandon",
            json_body={
                "lease_token": rollout.lease_token,
                "reason": reason,
                "detail": detail[:2000],
            },
            idempotency_key=f"abandon-{rollout.rollout_id}",
        )

    def register_artifact(
        self, rollout: RolloutRequest, *, name: str, size_bytes: int,
        sha256: str, content_type: str = "application/octet-stream",
        role: str = "output",
    ) -> dict[str, Any]:
        return self._c.request(
            "POST",
            f"/rollouts/{rollout.rollout_id}/artifacts",
            json_body={
                "name": name, "content_type": content_type,
                "size_bytes": size_bytes, "sha256": sha256, "role": role,
            },
            idempotency_key=f"art-{rollout.rollout_id}-{sha256[:16]}-{name}",
        )


def sleep_ms(ms: int) -> None:
    if ms > 0:
        time.sleep(ms / 1000.0)


def require(cond: bool, msg: str) -> None:
    if not cond:
        raise TransportError(msg)
