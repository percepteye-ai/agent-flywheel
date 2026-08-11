"""percepteye-agent-flywheel -- run your own agent against Percepteye rollouts.

    import percepteye_agent_flywheel as ph

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
import warnings
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
from .introspection import DiscoveredAgent, ToolDefinition
from .introspection import introspect as introspect_agent
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
from .isolation import unroutable_providers_present
from .outcomes import OutcomeLedger, current_rollout_id, record_tool_call
from .runner import RolloutRunner
from .transport import AttachTransport, FlywheelTransport

__version__ = "0.1.0"

__all__ = [
    "CONTRACT_VERSION",
    "AttachTransport",
    "ConfigurationError",
    "ContractViolation",
    "DiscoveredAgent",
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
    "ToolDefinition",
    "TransportError",
    "__version__",
    "current_rollout_id",
    "introspect_agent",
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


def _env_true(name: str, *, default: bool) -> bool:
    """An env kill switch that only a DELIBERATE value can flip.

    Unset means the default. ``0``/``false``/``no``/``off`` disable; anything
    else is treated as enabled, so a typo cannot silently switch capture off
    and leave the operator believing it is on.
    """
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


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
    adapters: list[str] | str | None = "auto",
    introspect: bool = True,
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
        adapters: framework integrations, by NAME, bound inside whichever
            process runs your agent. ``"auto"`` (the default) binds every
            built-in whose framework is installed, so a LangChain or LangGraph
            agent reports per-call outcomes without you writing anything.
            Pass ``[]`` to bind none, or ``["langchain"]`` to require one --
            a NAMED adapter that cannot bind raises, because naming it asserts
            that it is how your calls are captured.
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
            f"Upgrade percepteye-agent-flywheel."
        )

    # Provider configuration we cannot point at the gateway. Reported here, in
    # front of the person who set it, rather than a run later as a
    # training-grade exclusion nobody can explain. A warning and not a refusal:
    # a name on this list is evidence, not proof -- the agent may not use that
    # provider at all -- and the control plane's reported-vs-served call count
    # is the check that actually decides.
    stripped = unroutable_providers_present()
    if stripped:
        warnings.warn(
            f"{', '.join(stripped)} set in this environment. The Percepteye "
            f"gateway speaks the OpenAI completions API, so these cannot be "
            f"routed through it, and they are REMOVED from each rollout's "
            f"child process. An agent built on one of those providers will "
            f"raise rather than sample from the live provider unobserved. "
            f"Use an OpenAI-compatible client to have your completions "
            f"captured.",
            RuntimeWarning,
            stacklevel=2,
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
        adapters=adapters,
        on_event=on_event,
    )

    # A command entrypoint has no resolved object -- ``resolved`` is None by
    # construction (it is already its own process). Reading ``.name`` off it
    # raised AttributeError here, so every non-Python agent failed at
    # registration, before its first rollout. No test caught it because the
    # tests build a RolloutRunner directly and never go through serve().
    entrypoint_label = (
        runner.resolved.name if runner.resolved is not None
        else " ".join(getattr(runner, "_command", None) or []) or "command"
    )

    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "mode": mode,
        "sdk": f"percepteye-agent-flywheel-python/{__version__}",
        "contract_version": CONTRACT_VERSION,
        "reports_tool_calls": bool(reports_tool_calls),
        "isolation": isolation,
        "concurrency": concurrency,
        "input_shape": input_shape,
        "entrypoint": entrypoint_label,
    }

    # ── agent discovery (ADP), from inside the process ──────────────────────
    # ON by default, with two ways to refuse: ``introspect=False`` here, or
    # ``PERCEPTEYE_INTROSPECT=0`` in the environment, for an operator who
    # cannot edit the call site. Both are documented prominently in the
    # README, because this ships your system prompt and tool catalogue to the
    # control plane and a default that surprises someone is worse than no
    # default at all.
    #
    # ``discovered_agent: None`` is sent DELIBERATELY when we could not look.
    # Omitting the key would make "not introspected" indistinguishable from
    # "introspected, found nothing" -- the same absence-as-zero collapse the
    # tool-outcome contract exists to prevent, one layer up.
    if introspect and _env_true("PERCEPTEYE_INTROSPECT", default=True):
        described = introspect_agent(
            runner.resolved.fn if runner.resolved is not None else None
        )
        payload["discovered_agent"] = described.to_wire() if described else None

    tr.register(payload)

    try:
        return runner.serve_forever(max_rollouts=max_rollouts)
    finally:
        client.close()
