"""percepteye-flywheel -- run your own agent against Percepteye rollouts.

    import percepteye_flywheel as ph

    ph.serve(
        "my_package.agent:build",       # importable, so a child can rebuild it
        agent_id="my-coding-agent",
        input_shape="text",
        control_plane_url="https://api.percepteye.com/api/flywheel/v1",
    )

Outbound only. The flywheel dials out, claims a rollout, runs it, reports back.
No listening socket, no inbound firewall rule, no public DNS.

Two things are worth knowing before you read further.

**There is one mode.** ``rollout`` -- Percepteye supplies the model endpoint,
your flywheel supplies the agent loop. Modes that do not change what the SDK
actually does are not offered, because an earlier SDK shipped five of them, of
which four were byte-identical strings in one payload field the server never
read.

**Reporting "I don't know" is a supported answer.** When your tool wrapper
cannot tell whether a call succeeded, say ``outcome="unknown"``. It is
recorded as unknown and the verifiers that depend on it abstain. It is never
rounded up. This matters more than it sounds: the alternative is a reward
signal that scores a failed call as a landed one, and a policy trained to
believe it.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any, Literal

from ._http import ControlPlaneClient
from .contract import (
    CONTRACT_VERSION,
    ExclusionReason,
    Outcome,
    RolloutOutput,
    RolloutRequest,
    ToolCallOutcome,
)
from .entrypoint import InputShape, resolve_entrypoint
from .errors import (
    ConfigurationError,
    ContractViolation,
    EntrypointError,
    FlywheelError,
    LeaseLost,
    ModeNotAvailable,
    RoutingNotInstallable,
    TransportError,
)
from .outcomes import OutcomeLedger, current_rollout_id, record_tool_call
from .runner import RolloutRunner
from .transport import AttachTransport, FlywheelTransport

__version__ = "0.1.0"

__all__ = [
    "CONTRACT_VERSION",
    "AttachTransport",
    "ConfigurationError",
    "ContractViolation",
    "EntrypointError",
    "ExclusionReason",
    "FlywheelError",
    "FlywheelTransport",
    "InputShape",
    "LeaseLost",
    "ModeNotAvailable",
    "Outcome",
    "OutcomeLedger",
    "RolloutOutput",
    "RolloutRequest",
    "RolloutRunner",
    "RoutingNotInstallable",
    "ToolCallOutcome",
    "TransportError",
    "__version__",
    "current_rollout_id",
    "record_tool_call",
    "resolve_entrypoint",
    "serve",
]

#: Every environment variable this package reads. There are no feature flags,
#: and no variable selects behaviour -- these are addresses and credentials.
ENV_CONTROL_PLANE_URL = "PERCEPTEYE_CONTROL_PLANE_URL"
ENV_API_KEY = "PERCEPTEYE_API_KEY"
ENV_ROLLOUT_ID = "PERCEPTEYE_ROLLOUT_ID"      # set by us, for the child
ENV_LEDGER_DIR = "PERCEPTEYE_LEDGER_DIR"      # set by us, for the child

FlywheelMode = Literal["rollout"]


def serve(
    entrypoint: Any,
    *,
    agent_id: str,
    input_shape: InputShape,
    mode: FlywheelMode = "rollout",
    transport: Literal["attach"] = "attach",
    control_plane_url: str | None = None,
    api_key: str | None = None,
    artifacts_dir: str | None = None,
    isolation: Literal["subprocess", "inprocess"] = "subprocess",
    concurrency: int = 1,
    rollout_timeout_s: float = 900.0,
    reports_tool_calls: bool = False,
    method: str | None = None,
    max_rollouts: int | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
) -> int:
    """Claim rollouts and run them until stopped. Returns rollouts completed.

    Everything that can be wrong with the configuration is checked here,
    before the first rollout: the entrypoint resolves, the credential exists,
    the control plane answers, and the isolation mode is compatible with the
    entrypoint you passed.

    Args:
        entrypoint: ``"module:attr"`` (required for subprocess isolation, so a
            child can rebuild the agent) or a constructed object
            (``isolation="inprocess"`` only).
        input_shape: how ``turn_input`` reaches your agent -- ``"text"``,
            ``"messages"`` or ``"dict"``. Declared, never guessed.
        reports_tool_calls: assert that this flywheel reports a per-call
            outcome for every tool call it makes. Without it, the reward
            surface available to your rollouts is materially smaller, because
            the verifiers that grade "did this call land" have nothing to
            read and correctly abstain.
    """
    if mode != "rollout":
        raise ModeNotAvailable(
            f"mode={mode!r} is not available in contract {CONTRACT_VERSION}. "
            f"See https://docs.percepteye.com/flywheel/modes"
        )
    if transport != "attach":
        raise ModeNotAvailable(
            f"transport={transport!r} is not available in contract {CONTRACT_VERSION}."
        )

    url = control_plane_url or os.environ.get(ENV_CONTROL_PLANE_URL)
    if not url:
        raise ConfigurationError(
            f"control_plane_url is required (or set {ENV_CONTROL_PLANE_URL})"
        )
    key = api_key or os.environ.get(ENV_API_KEY)
    if not key:
        raise ConfigurationError(f"an API key is required (or set {ENV_API_KEY})")

    adir = artifacts_dir or os.path.join(os.getcwd(), ".percepteye-artifacts")
    os.makedirs(adir, exist_ok=True)

    client = ControlPlaneClient(url, key)
    tr = AttachTransport(client, agent_id)

    # Fail at startup, loudly. A health check nobody calls is not a health
    # check; an earlier SDK shipped one with zero call sites.
    health = tr.health()
    served = health.get("contract_versions_supported") or [health.get("contract_version")]
    if CONTRACT_VERSION not in [v for v in served if v]:
        raise ConfigurationError(
            f"control plane speaks {served}, this SDK speaks {CONTRACT_VERSION}. "
            f"Upgrade percepteye-flywheel."
        )

    runner = RolloutRunner(
        entrypoint,
        transport=tr,
        input_shape=input_shape,
        artifacts_dir=adir,
        isolation=isolation,
        concurrency=concurrency,
        rollout_timeout_s=rollout_timeout_s,
        method=method,
        on_event=on_event,
    )

    tr.register({
        "agent_id": agent_id,
        "mode": mode,
        "sdk": f"percepteye-flywheel-python/{__version__}",
        "contract_version": CONTRACT_VERSION,
        "reports_tool_calls": bool(reports_tool_calls),
        "isolation": isolation,
        "concurrency": concurrency,
        "input_shape": input_shape,
        "entrypoint": runner.resolved.name,
    })

    try:
        return runner.serve_forever(max_rollouts=max_rollouts)
    finally:
        client.close()
