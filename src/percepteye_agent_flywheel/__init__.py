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
from collections.abc import Callable, Mapping
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
from .entrypoint import TASK, InputShape, resolve_entrypoint
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
from .introspection import DiscoveredAgent, ToolDefinition
from .introspection import introspect as introspect_agent
from .isolation import unroutable_providers_present
from .outcomes import Trajectory, current_rollout_id, record_tool_call
from .policy import PolicySnapshot, PolicySource, policy_source
from .production import Attachment, attach
from .runner import RolloutRunner
from .transport import AttachTransport, FlywheelTransport

__version__ = "0.1.0"

__all__ = [
    "CONTRACT_VERSION",
    "TASK",
    "AttachTransport",
    "Attachment",
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
    "PolicySnapshot",
    "PolicySource",
    "RolloutOutput",
    "RolloutRequest",
    "RolloutRunner",
    "RoutingNotInstallable",
    "ToolCallOutcome",
    "ToolDefinition",
    "Trajectory",
    "TransportError",
    "__version__",
    "apply_current_policy",
    "attach",
    "current_policy",
    "current_prompt",
    "current_rollout_id",
    "endpoint_kwargs",
    "introspect_agent",
    "policy_source",
    "record_tool_call",
    "resolve_entrypoint",
    "serve",
]

#: Every environment variable this package reads. There are no feature flags,
#: and no variable selects behaviour -- these are addresses and credentials.
ENV_CONTROL_PLANE_URL = "PERCEPTEYE_CONTROL_PLANE_URL"
ENV_API_KEY = "PERCEPTEYE_API_KEY"
ENV_ROLLOUT_ID = "PERCEPTEYE_ROLLOUT_ID"      # set by us, for the child
ENV_TRAJECTORY_DIR = "PERCEPTEYE_TRAJECTORY_DIR"  # set by us, for the child

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


def current_policy(
    *,
    agent_id: str,
    control_plane_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Ask the control plane what model this agent should be running.

    For the customer's OWN agent process — the one serving their users —
    which is a different process from :func:`serve`. ``serve`` donates
    capacity for training; this is how the production agent picks up what
    that training produced.

    Returns ``{}`` when the control plane cannot answer, and callers should
    read that as "keep your current configuration". An unreachable control
    plane must never stop a customer's agent from serving its users.

    What comes back is an ENDPOINT, because that is the only form an agent
    can act on — an artifact URI is not callable. It is shaped for an
    OpenAI-compatible client::

        pol = fw.current_policy(agent_id="coding-agent")
        ep = pol.get("endpoint")
        if ep:
            client = OpenAI(base_url=ep["base_url"],
                            api_key=os.environ[ep["api_key_env"]])
            model = ep["model"]

    ``endpoint`` is absent when the control plane has nothing deployed to
    hand you — keep your current configuration, do not fall back to
    something you guessed.
    """
    url = control_plane_url or os.environ.get(ENV_CONTROL_PLANE_URL)
    key = api_key or os.environ.get(ENV_API_KEY)
    if not url or not key:
        return {}
    client = ControlPlaneClient(base_url=url, api_key=key)
    try:
        from .transport import AttachTransport

        return AttachTransport(client, agent_id=agent_id).policy_current()
    finally:
        client.close()


def current_prompt(
    *,
    agent_id: str,
    control_plane_url: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """The approved system prompt this agent should be running, or ``{}``.

    The other half of :func:`current_policy`. That one answers "which model";
    training also produces a system PROMPT, co-optimized with that model and
    gated by the same human approval — and until this existed an agent could
    learn only that the prompt had moved, never what it moved to. The snapshot
    in :class:`PolicySource` says as much in its own comment: its generation
    changes for a prompt change, "which this snapshot does not carry and a
    client rebuild would not apply".

    Returns the deliverable::

        {"name": ..., "version": 3, "sha256": ..., "text": "...",
         "candidate_kind": "replacement", "status": "approved",
         "model_prompt_checksum": ..., "stale_baseline": false}

    ``{}`` means KEEP THE PROMPT YOU HAVE — nothing has passed the approval
    gate for this agent, or the control plane could not answer. It never means
    "use no prompt", and an unreachable control plane must never blank a
    running agent's instructions.

    **Deliberately not applied for you, and that is the difference from**
    :func:`apply_current_policy`. Swapping a model endpoint is invisible to an
    agent's behaviour; swapping its system prompt changes what it DOES. That is
    a decision for the person who owns the agent, so this hands back the text
    and stops. Read ``sha256`` to tell whether it differs from what you are
    running.

    Two fields are worth checking before you apply it:

    * ``model_prompt_checksum`` — the prompt identity stamped on the MODEL at
      certification. The contract is a PAIR: this prompt was approved against
      that checkpoint, and applying it over a different one is not the thing
      that was approved.
    * ``stale_baseline`` — true when the prompt you supplied as the baseline
      has changed since this was certified. The deliverable still stands; it
      was just shaped against a version of your prompt you no longer ship.
    """
    url = control_plane_url or os.environ.get(ENV_CONTROL_PLANE_URL)
    key = api_key or os.environ.get(ENV_API_KEY)
    if not url or not key:
        return {}
    client = ControlPlaneClient(base_url=url, api_key=key)
    try:
        from .transport import AttachTransport

        tr = AttachTransport(client, agent_id=agent_id)
        ask = getattr(tr, "prompt_current", None)
        return (ask() if callable(ask) else {}) or {}
    finally:
        client.close()


def endpoint_kwargs(
    policy: Mapping[str, Any] | None, *, api_key: str | None = None,
) -> dict[str, Any]:
    """The ONE place a ``/policy/current`` body becomes client kwargs.

    Extracted because there are now two callers — :func:`apply_current_policy`
    and :class:`PolicySource` — and "a caller who unpacks it slightly
    differently is a caller sampling from something other than what was
    trained" applies at least as much between two of OUR surfaces as it does
    between two of the customer's.

    ``{}`` means nothing to apply: no endpoint, no ``base_url``, or no
    credential we can authenticate with. That last one is deliberate — an
    endpoint we cannot authenticate to is not usable, and returning it would
    produce 401s at the caller's first request rather than a legible "nothing
    to apply" here.
    """
    if not isinstance(policy, Mapping):
        return {}
    ep = policy.get("endpoint")
    if not isinstance(ep, Mapping) or not ep.get("base_url"):
        return {}
    key_env = str(ep.get("api_key_env") or ENV_API_KEY)
    key = api_key or os.environ.get(key_env) or os.environ.get(ENV_API_KEY)
    if not key:
        return {}
    return {
        "base_url": str(ep["base_url"]),
        "api_key": str(key),
        "model": str(ep.get("model") or ""),
    }


def apply_current_policy(
    *,
    agent_id: str,
    control_plane_url: str | None = None,
    api_key: str | None = None,
    set_env: bool = False,
) -> dict[str, Any]:
    """Turn :func:`current_policy` into arguments you can hand a client.

    The last mile. ``current_policy`` already returns the right answer, but
    every caller had to unpack it the same way, and a caller who unpacks it
    slightly differently is a caller sampling from something other than what
    was trained. Returns kwargs shaped for an OpenAI-compatible client::

        kw = fw.apply_current_policy(agent_id="coding-agent")
        if kw:
            client, model = OpenAI(**{k: v for k, v in kw.items()
                                      if k != "model"}), kw["model"]

    Returns ``{}`` when nothing is deployed or the control plane cannot
    answer — keep your current configuration. Same fail-open as
    ``current_policy``, for the same reason: an unreachable control plane must
    never stop a customer's agent from serving its users.

    NOT FOR :func:`serve`, and that is a correctness boundary rather than a
    style preference. A ``serve`` rollout samples through the per-rollout
    GATEWAY url, which is what attributes the completion to that rollout and
    captures it — and the gateway already rewrites the model field to the
    policy under test. Pointing a donor agent at the policy endpoint directly
    would bypass capture entirely and silently produce a run with nothing to
    train on. Use this in your PRODUCTION agent, the one serving your users.

    ``set_env=True`` additionally exports ``OPENAI_BASE_URL`` /
    ``OPENAI_API_KEY`` for libraries that read the environment rather than
    accept arguments. Off by default: mutating the process environment is a
    side effect a caller should ask for.
    """
    pol = current_policy(
        agent_id=agent_id, control_plane_url=control_plane_url, api_key=api_key,
    )
    out = endpoint_kwargs(pol, api_key=api_key)
    if not out:
        return {}
    if set_env:
        os.environ["OPENAI_BASE_URL"] = out["base_url"]
        os.environ["OPENAI_API_KEY"] = out["api_key"]
    return out


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
    #: How this agent's CLI spells its completion-gate option, e.g.
    #: ``"--autonomous-gate"``. Required before any gate is passed: the flag is
    #: a fact about the binary and cannot be guessed, and handing an unknown
    #: option to an arbitrary agent would break it. The gate COMMAND itself is
    #: a property of the task and arrives per rollout. Leave unset for an agent
    #: with no gate concept — it then behaves exactly as before.
    gate_flag: str | None = None,
    #: The environment variables THIS agent reads its endpoint from, when they
    #: are not the OpenAI/Anthropic names set by default. Declared, never
    #: sniffed: an agent that reads neither default looks identical from here
    #: to one that is correctly routed, right up until the completions arrive
    #: at the live provider instead of the gateway.
    endpoint_env: tuple[str, str] | None = None,
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
        gate_flag=gate_flag,
        endpoint_env=endpoint_env,
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
        # str() each part: a TASK sentinel is not a string, and joining it
        # raised TypeError here — the same class of bug the line above records
        # having shipped once, where reading .name off a command entrypoint
        # failed every non-Python agent at registration.
        else " ".join(str(a) for a in (getattr(runner, "_command", None) or []))
        or "command"
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

    # ── the last mile ───────────────────────────────────────────────────────
    # Training produces a better model; this is where the agent finds out.
    # The registration response used to be discarded and nothing ever asked
    # what to run, so a trained adapter reached the customer's process only
    # by a human editing their config.
    #
    # Resolved ONCE at startup rather than polled: a policy that changes
    # under a running agent is the same confound the control plane refuses on
    # its own training runs. A change lands on the next restart, and the
    # dashboard says so.
    # getattr, not a direct call: ``policy_current`` is not on the
    # FlywheelTransport Protocol, so a third-party or older transport need
    # not implement it. The last mile must not break an agent whose
    # transport predates it — the same fail-open the method itself keeps.
    _ask = getattr(tr, "policy_current", None)
    policy = _ask() if callable(_ask) else {}
    if policy:
        # INFORMATIONAL ONLY, deliberately. A serve rollout samples through the
        # per-rollout GATEWAY url — that is what attributes and captures the
        # completion, and the gateway already rewrites the model field to the
        # policy under test. Applying the endpoint here would bypass capture
        # and produce a run with nothing to train on. Production agents call
        # apply_current_policy() instead.
        _ep = policy.get("endpoint") if isinstance(policy.get("endpoint"), dict) else {}
        print(
            f"[percepteye] policy: {policy.get('arm', 'base')} — "
            f"{policy.get('reason', '')}"
            + (f" → {_ep.get('model')} at {_ep.get('base_url')}"
               if _ep.get("base_url") else ""),
            flush=True,
        )

    try:
        return runner.serve_forever(max_rollouts=max_rollouts)
    finally:
        client.close()
