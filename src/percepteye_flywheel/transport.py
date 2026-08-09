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

import time
import uuid
from typing import Any, Protocol, runtime_checkable

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
            "POST", "/flywheeles", json_body=spec,
            idempotency_key=f"reg-{self.agent_id}",
        )

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
            # Replay-safe: re-claiming with the same key inside the lease
            # returns the same set rather than a second, overlapping one.
            idempotency_key=str(uuid.uuid4()),
            timeout=(self.wait_ms / 1000.0) + 10.0,
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
