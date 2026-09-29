"""agent-flywheel -- run your own agent against PerceptEye rollouts.

    import agent_flywheel as fw

    fw.serve(
        "my_package.agent:build",       # importable, so a child can rebuild it
        agent_id="my-coding-agent",
        input_shape="text",
        control_plane_url="https://launch.percepteye.ai/api/flywheel/v1",
    )

Outbound only. The flywheel dials out, claims a rollout, runs it, reports back.
No listening socket, no inbound firewall rule, no public DNS.

Two things are worth knowing before you read further.

**There are two modes, and they are a deployment switch.** ``training`` --
Percepteye supplies the model endpoint, your flywheel supplies the agent loop
-- and ``production``, in which ``serve()`` returns 0 without connecting to
anything. Set ``PERCEPTEYE_AGENT_MODE`` and the same image does either, because
"stop contributing capacity" is an operational decision and it should not require
editing the source to express it.

That is the ONLY axis this word names here. An earlier SDK shipped five modes
of which four were byte-identical strings in one payload field the server
never read, and the rule that came out of it stands: a mode has to change what
the SDK does. These two do -- one claims rollouts and the other opens no
socket.

On the wire, ``serve()`` registers as ``"rollout"`` -- that field names the
protocol this process speaks, and it is the value intake has always received
from a training donor. ``attach()`` registers as ``"production"``, because a
training donor and a production reader are the one distinction a ``mode`` on a
registration exists to draw, and while both sent ``"rollout"`` the field
answered nothing. So the wire carries two values and this parameter carries
two modes, but they are not the same axis and do not have to agree.

**Reporting "I don't know" is a supported answer.** When your tool wrapper
cannot tell whether a call succeeded, say ``outcome="unknown"``. It is
recorded as unknown and the verifiers that depend on it abstain. It is never
rounded up. This matters more than it sounds: the alternative is a reward
signal that scores a failed call as a landed one, and a policy trained to
believe it.
"""
from __future__ import annotations

import logging
import os
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

from . import declaration as _declaration
from ._version import __version__
from .contract import (
    CONTRACT_VERSION,
    ExclusionReason,
    Outcome,
    RolloutOutput,
    RolloutRequest,
    RulebookReceipt,
    ToolCallOutcome,
)
from .declaration import Describe
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
from .event_stream import CommandOutputAdapter
from .execution_identity import ExecutionChild, ExecutionSnapshot
from .host_description import WARMUP_PROMPT, warm_up_command

# Re-exported, not used in this module. The redundant alias is the
# explicit-re-export form: it keeps `from agent_flywheel import
# read_discovered_agent` working -- which it does today -- without adding the
# name to __all__ and thereby claiming it as a supported public surface.
from .host_description import read_discovered_agent as read_discovered_agent
from .introspection import DiscoveredAgent, ToolDefinition
from .introspection import introspect as introspect_agent
from .isolation import unroutable_providers_present
from .outcomes import (
    CaptureScope,
    Trajectory,
    current_capture,
    current_rollout_id,
    record_tool_call,
)
from .policy import (
    ARCHIVE_REF_PREFIXES,
    PolicySnapshot,
    PolicySource,
    is_archive_ref,
    missing_credentials_cause,
    policy_source,
)
from .production import Attachment, CaptureVerdict, attach, capture_verdict
from .prompt import (
    SERVABLE_PROMPT_STATUSES,
    PromptDecision,
    prompt_decision,
)
from .runner import DEFAULT_ROLLOUT_TIMEOUT_S, RolloutRunner
from .transport import (
    RULEBOOK_SETTLE_GRACE_S,
    AttachTransport,
    FlywheelTransport,
    RulebookUpload,
    declare_rulebook_in_background,
    declared_rulebook,
)

#: This package's logger. It has no handler and never installs one -- it
#: emits into whatever logging the customer's process already configured,
#: exactly as the OpenTelemetry extra emits into their TracerProvider.
#: Used for the answers a caller CANNOT see in the return value: `{}` from
#: `endpoint_kwargs` and `current_prompt` is the same value for "nothing is
#: deployed" and "you never told me where the control plane is", and only
#: one of those is a thing an operator can fix.
_log = logging.getLogger(__name__)

__all__ = [
    "ARCHIVE_REF_PREFIXES",
    "CONTRACT_VERSION",
    "SERVABLE_PROMPT_STATUSES",
    "TASK",
    "AttachTransport",
    "Attachment",
    "CaptureScope",
    "CaptureVerdict",
    "CommandOutputAdapter",
    "ConfigurationError",
    "ContractViolation",
    "DiscoveredAgent",
    "EntrypointError",
    "ExclusionReason",
    "ExecutionChild",
    "ExecutionSnapshot",
    "FlywheelError",
    "FlywheelTransport",
    "InputShape",
    "LeaseLost",
    "ModeNotAvailable",
    "Outcome",
    "PolicySnapshot",
    "PolicySource",
    "PromptDecision",
    "RolloutOutput",
    "RolloutRequest",
    "RolloutRunner",
    "RoutingNotInstallable",
    "RulebookReceipt",
    "ToolCallOutcome",
    "ToolDefinition",
    "Trajectory",
    "TransportError",
    "__version__",
    "apply_current_policy",
    "attach",
    "capture_verdict",
    "current_capture",
    "current_mode",
    "current_policy",
    "current_prompt",
    "current_rollout_id",
    "endpoint_kwargs",
    "introspect_agent",
    "is_archive_ref",
    "policy_source",
    "prompt_decision",
    "record_tool_call",
    "resolve_entrypoint",
    "resolve_mode",
    "serve",
]

#: Every environment variable this package reads. Two of them select
#: behaviour -- PERCEPTEYE_AGENT_MODE and PERCEPTEYE_INTROSPECT -- and the rest are
#: addresses and credentials.
#:
#: PERCEPTEYE_AGENT_MODE names what the AGENT is for -- training or serving
#: users -- rather than naming a knob on this SDK, which is why "agent" is in
#: it. It carries the prefix because an unprefixed AGENT_MODE is a name we
#: would not own: it is plausible enough that a customer already exports it for
#: something unrelated, and an unrecognised value RAISES here, so a collision
#: would stop a flywheel over a word that was never addressed to us. The prefix
#: buys the right to refuse a value we cannot parse.
#:
#: Being inside the namespace also means `build_child_env` strips it from a
#: rollout child along with everything else PERCEPTEYE_, which is correct: it
#: configures the process that DONATES capacity, and a child does not call
#: serve().
ENV_CONTROL_PLANE_URL = "PERCEPTEYE_CONTROL_PLANE_URL"
ENV_API_KEY = "PERCEPTEYE_API_KEY"
ENV_MODE = "PERCEPTEYE_AGENT_MODE"
ENV_ROLLOUT_ID = "PERCEPTEYE_ROLLOUT_ID"      # set by us, for the child
ENV_TRAJECTORY_DIR = "PERCEPTEYE_TRAJECTORY_DIR"  # set by us, for the child

#: What this PROCESS is for. ``"rollout"`` is the legacy spelling of
#: ``"training"`` and still means exactly that.
#:
#: This used to be ``Literal["rollout"]`` -- one value, which `serve()` checked
#: and rejected everything else, so the parameter carried zero bits and the
#: field of the same name on the registration payload carried zero bits with
#: it. It is now the deployment switch, because the alternative was a SECOND
#: thing called "mode" sitting next to a vestigial first one.
FlywheelMode = Literal["training", "production", "rollout"]

#: Accepted spellings, resolved to the two real modes. Deliberately a small
#: closed set: an unrecognised value RAISES rather than guessing, because the
#: two ways of guessing wrong are "silently contribute capacity the operator meant
#: to switch off" and "silently contribute nothing while they believe training is
#: running". Neither is recoverable by the operator, because both look exactly
#: like success.
_MODE_ALIASES: dict[str, str] = {
    "training": "training",
    "train": "training",
    "rollout": "training",       # the legacy value, unchanged in meaning
    "rollouts": "training",
    "fine-tuning": "training",
    "finetuning": "training",
    "improving": "training",
    "production": "production",
    "prod": "production",
    "serving": "production",
}


def _mode_from(raw: Any, *, source: str) -> str:
    """One spelling to one mode, or a refusal that names where it came from."""
    key = str(raw).strip().lower()
    if key not in _MODE_ALIASES:
        raise ConfigurationError(
            f"{source}{raw!r} is not a mode. Use 'training' (this process "
            f"does training work: it claims rollouts and runs them) or "
            f"'production' (it does not). Recognised spellings: "
            f"{', '.join(sorted(_MODE_ALIASES))}."
        )
    return _MODE_ALIASES[key]


def _resolve_mode_source(explicit: str | None = None) -> tuple[str, str]:
    """The mode, and which of ``argument``/``environment``/``default`` set it.

    An EMPTY explicit argument falls through to the environment rather than
    being honoured as a value. ``mode=cfg.get("percepteye_mode", "")`` is the
    ordinary way a config layer says "not set", and treating it as a value
    silently pinned the process to "training" without ever reading
    ``PERCEPTEYE_AGENT_MODE`` -- so an operator who had flipped the switch to stop
    contributing kept contributing, on their own credentials, with nothing printed.
    That is the exact failure ``_MODE_ALIASES`` calls unrecoverable, arrived at
    through the one input that skipped the closed-set check entirely.
    """
    if explicit is not None and str(explicit).strip() != "":
        return _mode_from(explicit, source="mode="), "argument"
    env = os.environ.get(ENV_MODE)
    if env is not None and str(env).strip() != "":
        return _mode_from(env, source=f"{ENV_MODE}="), "environment"
    return "training", "default"


def resolve_mode(explicit: str | None = None) -> str:
    """Which mode this process is in: ``"training"`` or ``"production"``.

    Precedence is explicit argument, then ``PERCEPTEYE_AGENT_MODE``, then
    ``"training"``. The argument wins so that a caller who has hard-coded a
    mode gets what they wrote; the environment provides the default in the
    ordinary case where nobody passed one, which is what lets an operator flip
    a deployment WITHOUT editing code. An empty or whitespace argument counts
    as "nobody passed one".

    Raises:
        ConfigurationError: for any value that is not a recognised mode. There
            is no safe guess -- see ``_MODE_ALIASES``.
    """
    return _resolve_mode_source(explicit)[0]


def current_mode() -> str:
    """The mode this process would serve in, from the environment alone.

    ``"training"`` or ``"production"``. Read it to log which way a deployment
    is configured, or to assert it in a smoke test. Raises for an unrecognised
    ``PERCEPTEYE_AGENT_MODE``, exactly as :func:`serve` does, so a typo surfaces
    wherever it is first read rather than only at startup.
    """
    return resolve_mode()


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
    which is a different process from :func:`serve`. ``serve`` contributes
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
    url = control_plane_url or os.environ.get(ENV_CONTROL_PLANE_URL) or ""
    key = api_key or os.environ.get(ENV_API_KEY) or ""
    cause = missing_credentials_cause(url, key)
    if cause:
        _log.warning(
            "agent-flywheel: not reading the served policy for agent_id=%r: %s, so "
            "the control plane was never asked. This agent keeps the model it "
            "is on.", agent_id, cause,
        )
        return {}
    from ._http import ControlPlaneClient

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
    and stops.

    **Do not re-derive whether it may be applied — ask** :func:`prompt_decision`.
    This docstring used to list the fields to check and the order to check them
    in, which made every caller reimplement a rule with no mechanism behind it.
    The rule is a function now::

        served = fw.current_prompt(agent_id="my-agent")
        decision = fw.prompt_decision(served)
        if decision.apply:
            system_prompt = decision.text          # never None here
        else:
            log.info("keeping the current prompt: %s", decision.reason)

    It answers the same four questions the Node SDK's ``promptDecision`` asks,
    in the same order and with the same sentences, so the two packages refuse
    the same bundles for the same stated reason. See
    :mod:`agent_flywheel.prompt`.
    """
    url = control_plane_url or os.environ.get(ENV_CONTROL_PLANE_URL) or ""
    key = api_key or os.environ.get(ENV_API_KEY) or ""
    cause = missing_credentials_cause(url, key)
    if cause:
        # SAID, because `prompt_decision({})` blames the control plane for
        # serving no approved prompt -- true of the empty answer and wrong
        # about why, which sends an operator to the dashboard to look for an
        # approval that was never the problem.
        _log.warning(
            "agent-flywheel: not reading the served prompt for agent_id=%r: %s, so "
            "the control plane was never asked. This agent keeps the prompt it "
            "assembles itself.", agent_id, cause,
        )
        return {}
    from ._http import ControlPlaneClient

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
    # PRECEDENCE, and the order is the contract:
    #
    #   1. an explicit ``api_key=`` argument — the caller's own override always
    #      wins, so a customer who wants full control keeps it;
    #   2. ``endpoint.api_key`` from the control plane, when it sent one;
    #   3. the environment variable the control plane NAMED;
    #   4. ``PERCEPTEYE_API_KEY``.
    #
    # (2) exists for one case, and without it that case fails silently. A
    # checkpoint promotion normally moves only ``model`` — base_url and the
    # credential stay put, so an agent upgrades with nothing new. They do NOT
    # stay put when the control plane moves an agent to a DIFFERENT provider:
    # then base_url and the credential both change, and an agent cannot read an
    # env var it was never given. Without (2) this function returns ``{}``, the
    # caller keeps its previous configuration, and the agent FREEZES on its
    # last policy — receiving no further upgrades and reporting nothing wrong.
    #
    # The value is never persisted or logged here; it goes straight into the
    # client kwargs the caller builds, exactly like a key read from the
    # environment.
    key_env = str(ep.get("api_key_env") or ENV_API_KEY)
    key = (
        api_key
        or str(ep.get("api_key") or "").strip()
        or os.environ.get(key_env)
        or os.environ.get(ENV_API_KEY)
    )
    if not key:
        return {}
    model = str(ep.get("model") or "")
    # AN ARCHIVE URI IS NOT A MODEL NAME, and this is the model half of the
    # pair rule `prompt_decision` enforces for the prompt half. The control
    # plane already refuses to serve one; re-asserted here on the same argument
    # as `SERVABLE_PROMPT_STATUSES`, because a candidate whose checkpoint has
    # been uploaded but not yet deployed behind a serving endpoint would
    # otherwise be handed to `OpenAI(model="gs://...")` and 404 on every
    # completion. Refusing keeps the champion, which is what every other gate
    # on this path fails closed to.
    #
    # SAID OUT LOUD, because silence here is indistinguishable from "nothing is
    # deployed": `{}` is this function's ordinary answer and the caller is
    # documented to read it as "keep your current configuration". An operator
    # whose promotion never arrives needs the sentence, not the empty dict.
    if is_archive_ref(model):
        _log.warning(
            "agent-flywheel: NOT applying the served policy for this agent: the "
            "served model %r is a stored-artifact reference, not a name an "
            "endpoint can serve; that address would 404 on every completion. "
            "Keeping the model you are on.", model,
        )
        return {}
    return {
        "base_url": str(ep["base_url"]),
        "api_key": str(key),
        "model": model,
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


def _describe_via_host(
    command: list[Any], *, artifacts_dir: str, agent_id: str,
    execution_snapshot: ExecutionSnapshot | None = None,
    endpoint_env: tuple[str, str] | None = None,
    declared: _declaration.Declaration | None = None,
) -> DiscoveredAgent | None:
    """Have the host describe a command agent, by running it once.

    Returns None when the host produced nothing, having said why. The failure
    is silent everywhere else -- the plugin's hook simply never fires -- and an
    operator who is not told here finds out as an empty workflow list an hour
    later, with nothing in between naming the cause. ``declared`` is what
    ``describe=`` already said, so the warning states what registers instead.
    """
    # The TASK sentinel marks where a rollout's task goes in the argv. There is
    # no rollout here, so it is filled with the warm-up prompt: an argv still
    # holding the sentinel would be passed to the binary as the literal string.
    argv = [WARMUP_PROMPT if a is TASK else str(a) for a in command]
    warm_dir = os.path.join(artifacts_dir, "_describe")

    snapshot = execution_snapshot
    snapshot_child: ExecutionChild | None = None
    try:
        env = None
        cwd = None
        if snapshot is not None:
            # Registration must describe the same scaffold the rollouts use.
            # Letting this one preflight read ambient HOME would register a
            # prompt/tool identity that no trainable child actually ran.
            try:
                snapshot_child = snapshot.prepare_child("describe")
            except Exception as exc:
                raise ConfigurationError(
                    "execution snapshot could not prepare the discovery child: "
                    f"{type(exc).__name__}"
                ) from exc
            env = dict(os.environ)
            cwd = warm_dir
            try:
                rollout_environment = dict(snapshot_child.environment)
                discovery_environment = dict(
                    snapshot_child.discovery_environment
                )
            except Exception as exc:
                raise ConfigurationError(
                    f"{snapshot.adapter_name} execution child environments "
                    "must be string mappings"
                ) from exc
            if any(
                not isinstance(name, str) or not isinstance(value, str)
                for mapping in (rollout_environment, discovery_environment)
                for name, value in mapping.items()
            ):
                raise ConfigurationError(
                    f"{snapshot.adapter_name} execution child environments "
                    "must contain only string names and values"
                )
            if any(
                rollout_environment.get(name) != value
                for name, value in discovery_environment.items()
            ):
                raise ConfigurationError(
                    f"{snapshot.adapter_name} discovery environment must be a "
                    "non-secret subset of its rollout environment"
                )
        else:
            discovery_environment = None
        described = warm_up_command(
            argv, trajectory_dir=warm_dir, agent_name=agent_id, env=env, cwd=cwd,
            endpoint_env=endpoint_env,
            isolation_environment=(
                discovery_environment
            ),
        )
    finally:
        if snapshot_child is not None:
            lifecycle_failure: Exception | None = None
            try:
                assert snapshot is not None
                snapshot.attest_child(snapshot_child)
            except Exception as exc:
                lifecycle_failure = exc
            try:
                snapshot.release_child(snapshot_child)
            except Exception as exc:
                lifecycle_failure = lifecycle_failure or exc
            if lifecycle_failure is not None:
                raise ConfigurationError(
                    "execution snapshot lifecycle failed during command-agent "
                    f"discovery ({type(lifecycle_failure).__name__}); "
                    "registration was refused"
                ) from lifecycle_failure
    if described is not None:
        n = described.tool_definitions
        print(
            f"[percepteye] described {agent_id} from the host: "
            f"{len(n) if n is not None else 'no'} tool(s), "
            f"{len(described.system_prompt or '')} chars of system prompt"
            # WHICH SOURCE answered, named rather than inferred. The two carry
            # different amounts -- the wire drops read-only annotations and
            # sub-agents -- so an operator reading a thin catalogue needs to
            # know whether that is the agent or the vantage point.
            f" (via {'the wire' if described.source_type == 'host_wire' else 'the host plugin'})."
        )
        return described

    if declared is None:
        consequence = (
            "This agent will register as introspection_state=unreadable, no "
            "workflows will be generated for it, and every phase downstream of "
            "that (reward suites, graded rollouts, training) has nothing to "
            "work from.")
    else:
        consequence = "It registers only what describe= declared" + (
            "; with no tools declared, no workflows will be generated for it."
            if declared.tools is None else ".")
    warnings.warn(
        f"Could not describe {agent_id!r}: the agent ran, its host wrote no "
        f"description, and no completion request it made declared any tools. "
        f"{consequence}\n"
        f"Two things to check, in this order:\n"
        f"  1. Does the agent reach its provider from an env var? The warm-up "
        f"points OPENAI_BASE_URL / OPENAI_API_BASE / ANTHROPIC_BASE_URL at a "
        f"local endpoint and reads the tool catalogue off the request. An "
        f"agent that resolves its endpoint some other way never arrives, and "
        f"is also not routable for rollouts -- pass endpoint_env= to serve().\n"
        f"  2. Does it declare any tools at all? A CLI agent whose extensions "
        f"are configured per-profile can start with an empty catalogue; goose, "
        f"for one, sends zero tools unless an extension is enabled.\n"
        f"For OpenClaw specifically, the usual cause is the conversation-hook "
        f"opt-in: the host silently refuses to register `llm_input` for a "
        f"non-bundled plugin unless the config sets\n"
        f"  plugins.entries.agent-flywheel.hooks."
        f"allowConversationAccess = true",
        RuntimeWarning,
        stacklevel=3,
    )
    return None


def _rulebook_line(upload: RulebookUpload) -> str:
    """The one line ``serve()`` prints about the rules it declared.

    Printed, like the policy line, because a donor's console is where its
    operator looks at startup -- and a refusal must be as visible as a
    registration, or rules that never took effect look exactly like rules
    that did.
    """
    receipt, refusal = upload
    if receipt is None:
        # NOT "not registered": the sentence says whether that is known, and
        # a failure after the rules may have landed says it is not.
        return f"[percepteye] rulebook NOT confirmed — {refusal}"
    if receipt.status == "unchanged":
        what = "unchanged, already in force"
    else:
        what = receipt.status
    version = (f" as version {receipt.version}"
               if receipt.version is not None else "")
    return (
        f"[percepteye] rulebook: {what}{version} — {receipt.rule_count} "
        f"rule(s) for Product {receipt.product_id}"
    )


def serve(
    entrypoint: Any,
    *,
    agent_id: str,
    input_shape: InputShape,
    mode: FlywheelMode | None = None,
    transport: Literal["attach"] = "attach",
    control_plane_url: str | None = None,
    api_key: str | None = None,
    artifacts_dir: str | None = None,
    isolation: Literal["subprocess", "inprocess"] = "subprocess",
    concurrency: int = 1,
    #: THE RUNNER'S CONSTANT, not a copy of its value. This is the number an
    #: identity-sealed adapter setting is sized against, and while it was
    #: spelled here as a literal there were two of them: an adapter derived its
    #: frozen ceiling from the runner's constant, every production rollout was
    #: bounded by this one, and changing either left the agent's sealed
    #: identity sized against arithmetic the runner no longer performed --
    #: silently, because nothing compared them.
    rollout_timeout_s: float = DEFAULT_ROLLOUT_TIMEOUT_S,
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
    command_output_adapter: CommandOutputAdapter | None = None,
    execution_snapshot: ExecutionSnapshot | None = None,
    harness_snapshot: ExecutionSnapshot | None = None,
    attested_execution_components: Mapping[str, str] | None = None,
    volatile_command_values: Sequence[str] | None = None,
    reports_tool_calls: bool = False,
    method: str | None = None,
    adapters: list[str] | str | None = "auto",
    introspect: bool = True,
    describe: Describe | None = None,
    rules: Sequence[str] | None = None,
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
        mode: ``"training"`` to contribute capacity, ``"production"`` to do
            nothing. Omit it -- which is the point -- and ``PERCEPTEYE_AGENT_MODE``
            decides, so the same deployed image serves or does not without an
            edit. Unset means ``"training"``, preserving the behaviour of every
            existing call. ``"rollout"`` is the legacy spelling of
            ``"training"``. In production mode this returns 0 before opening
            any connection, and prints one line saying so; it does NOT raise,
            because a switch that crashes the service it was meant to quiet is
            not a switch. An unrecognised value raises
            :class:`ConfigurationError` rather than guessing.
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
        execution_snapshot: framework-neutral execution-adapter lifecycle.
            The adapter supplies opaque exact component digests, prepares an
            isolated child view, and attests/releases it after every rollout.
            Command and import-spec children receive its environment mapping;
            an in-process object requires an empty mapping because process-wide
            environment mutation is not isolated. An incomplete adapter never
            yields an OPSD-eligible execution identity.
        harness_snapshot: deprecated compatibility alias for
            ``execution_snapshot``. Prime-specific construction lives in the
            optional ``adapters.prime_agent`` module; core treats it like any
            other execution adapter.
        command_output_adapter: optional command-specific stdout projection.
            Core never guesses a host vocabulary. An adapter may project final
            text, independently observed model-call count, tool outcomes, and
            explicit exclusion evidence.
        attested_execution_components: framework-neutral opaque evidence from
            an execution adapter, as stable names mapped to lowercase 64-hex
            SHA-256 values. The SDK interprets none of the names: it validates,
            sorts, and folds them into ``agent_fingerprint.execution_sha256``
            with the discovered prompt/tools/sub-agents/model and serve config.
            An adapter supplying a digest is asserting it covers mutable state
            discovery cannot observe; do not put secrets or free-form values in
            this mapping. Without at least one adapter-owned component (or a
            component-providing ``execution_snapshot``/``harness_snapshot``),
            the SDK withholds the
            OPSD digest rather than claiming discovery identifies unobserved
            implementation code.
        volatile_command_values: command options whose VALUE is transport
            rather than identity, named exactly as the command spells them --
            ``("--daemon-socket",)`` for an agent launched with a per-process
            socket path. The option token stays in the execution identity and
            only its value is dropped, so an agent that runs with a socket is
            still distinct from one that does not, while a restart that moves
            the socket keeps the same ``execution_sha256``. That matters
            because an OPSD run binds the digest recorded at preparation and
            refuses every episode attested with a different one: an undeclared
            PID-bearing argument invalidates the prepared run on the next
            restart. Nothing is detected -- an option that is not declared is
            identity, including one that looks like a temp path -- and a name
            that is absent from the command, or that names the program itself,
            raises :class:`ConfigurationError` rather than quietly covering
            nothing. Command entrypoints only; an importable agent has no argv.
        describe: your agent's own description, for when nothing can read it
            -- a custom agent that is no framework the SDK reflects on, or a
            prompt reflection cannot reach. A mapping of ``system_prompt``,
            ``tools`` (the list you already send the model: OpenAI function
            specs, Anthropic tools, or ``{name, description, input_schema,
            read_only}``) and ``model``, or a zero-argument callable returning
            one, called once the agent is built. What you declare wins; what
            you leave out still comes from reflection, and leaving ``tools``
            out never means "no tools" (that is ``tools=[]``). For a command
            agent, declaring all three replaces the warm-up; declaring less
            still runs it for the rest. A declared description never
            anchors an execution identity -- it is your word, not an
            observation. ``introspect=False`` with ``describe=`` is a
            contradiction and raises; ``PERCEPTEYE_INTROSPECT=0`` wins over
            it, with a warning.
        rules: the workflow rules of the Product this agent belongs to, one
            prose rule per string, sent once at startup, after registration,
            to ``POST /agents/{agent_id}/rulebook``. The control plane files
            them under the one active Product that declares this agent, as
            that Product's rulebook in force -- training guidance for the
            decisions a rule applies to; nothing this agent is served changes.
            ``None`` (the default) sends nothing and leaves the rulebook in
            force alone; ``[]`` RETRACTS it. Code that declares rules
            re-declares them on every start -- the same rules register
            nothing, different ones are a new version, and the last writer
            wins, whether that is this call or the Product's own API. Not a
            list of strings raises :class:`ConfigurationError` here; anything
            the control plane refuses (a rule it will not accept, named by
            cause and index; an agent no single Product declares) or cannot
            hear is printed in one line and never stops the flywheel. They
            are declared in the background, so a slow rulebook route holds
            back neither the policy read nor the first claim. Rules
            never ride ``describe=``: a declaration withholds the agent's
            execution identity, and these are the Product's rules, not the
            agent's description.
        max_rollouts: stop after this many rollouts COMPLETE, and return. The
            only clean exit this function has: nothing in this package handles
            SIGINT or SIGTERM, so otherwise it runs until the process is
            killed and any in-flight rollout is orphaned rather than abandoned,
            to be reclaimed by lease expiry. Note the count is completions, not
            successes -- a crash and a graded pass both increment it.
        on_event: ``Callable[[str, dict], None]``, called SYNCHRONOUSLY on
            whichever thread produced the event -- the serving thread for
            ``claim_*``, a pool worker for the per-rollout kinds, and the
            heartbeat thread for ``lease_lost``/``heartbeat_failed`` -- so a
            slow callback slows that thread. Exceptions raised by the callback
            are SWALLOWED: it is an observability hook and must not be able to
            change what it observes. Keep it cheap and non-blocking; the
            heartbeat thread renews every in-flight lease. Kinds:
            ``claim_empty``, ``claim_failed``,
            ``rollout_start``, ``rollout_reported``, ``rollout_crash``,
            ``report_failed``, ``artifact_failed``, ``lease_lost``,
            ``heartbeat_failed``. Without it a flywheel is silent, and an empty
            queue is indistinguishable from a broken claim route --
            ``claim_empty`` versus ``claim_failed`` is what separates them.
    """
    # ── the deployment switch ────────────────────────────────────────────────
    # Resolved FIRST, before the credential checks and before any socket, so
    # that a production deployment needs neither a control-plane URL nor a key
    # to start clean. Turning training off must not require being configured
    # for it.
    resolved_mode, _mode_source = _resolve_mode_source(mode)

    # An override has to be announced in BOTH directions. The production
    # branch below prints when code says stop; this prints when code says GO
    # while the environment said stop -- which is the direction an operator
    # actually cares about, and the one that was silent. Flipping the variable
    # and seeing the flywheel keep claiming, with nothing on stdout to explain
    # it, is the same silence this function refuses everywhere else.
    if _mode_source == "argument":
        _env_raw = os.environ.get(ENV_MODE)
        if _env_raw is not None and str(_env_raw).strip() != "":
            try:
                _env_mode = _mode_from(_env_raw, source=f"{ENV_MODE}=")
            except ConfigurationError as exc:
                # Not fatal: the argument is authoritative and this process has
                # been told what to do. But an unreadable switch must not pass
                # unremarked, or the next operator to rely on it is misled.
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
                _env_mode = None
            if _env_mode is not None and _env_mode != resolved_mode:
                print(
                    f"[percepteye] mode={resolved_mode} — set by mode= in "
                    f"code, overriding {ENV_MODE}={_env_mode}. Remove mode= "
                    f"from the call to let this deployment decide.",
                    flush=True,
                )

    if resolved_mode == "production":
        # Return, never raise: this is a running service, and a switch that
        # crashes the process it was meant to quiet is not a switch. Return 0
        # because zero rollouts is the literal truth.
        #
        # And never SILENTLY: a no-op serve() that printed nothing is the exact
        # failure this package refuses everywhere else -- the operator believes
        # capacity is being contributed and nothing says otherwise. One line, on
        # the way out, naming what decided it.
        print(
            f"[percepteye] mode=production — not claiming rollouts. "
            f"(set {ENV_MODE}=training to contribute capacity"
            + (", or remove mode= from this call"
               if _mode_source == "argument" else "")
            + ")",
            flush=True,
        )
        return 0
    if transport != "attach":
        raise ModeNotAvailable(
            f"transport={transport!r} is not available in contract {CONTRACT_VERSION}."
        )

    if describe is not None and not introspect:
        raise ConfigurationError(
            "describe= declares a description and introspect=False says to "
            "send none; drop one of them"
        )
    if describe is not None and not callable(describe):
        # A static declaration is checked NOW, before anything connects: a
        # typo here otherwise surfaces as a run that generated nothing.
        _declaration.resolve(describe)
    # Checked NOW for the same reason: rules that are not a list of strings
    # would otherwise surface as a rulebook that never registered.
    rulebook = declared_rulebook(rules)

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

    if execution_snapshot is not None and harness_snapshot is not None:
        raise ConfigurationError(
            "pass execution_snapshot or the deprecated harness_snapshot alias, not both"
        )
    if harness_snapshot is not None:
        warnings.warn(
            "serve(harness_snapshot=...) is deprecated; use execution_snapshot=...",
            DeprecationWarning,
            stacklevel=2,
        )
    resolved_execution_snapshot = (
        execution_snapshot if execution_snapshot is not None else harness_snapshot
    )

    from ._http import ControlPlaneClient

    client = ControlPlaneClient(url, key)
    tr = AttachTransport(client, agent_id)

    # Fail at startup, loudly. A health check nobody calls is not a health
    # check; an earlier SDK shipped one with zero call sites.
    health = tr.health()
    served = health.get("contract_versions_supported") or [health.get("contract_version")]
    if CONTRACT_VERSION not in [v for v in served if v]:
        raise ConfigurationError(
            f"control plane speaks {served}, this SDK speaks {CONTRACT_VERSION}. "
            f"Upgrade agent-flywheel."
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
            f"{', '.join(stripped)} set in this environment. The PerceptEye "
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
        command_output_adapter=command_output_adapter,
        execution_snapshot=resolved_execution_snapshot,
        method=method,
        adapters=adapters,
        on_event=on_event,
        reports_tool_calls=reports_tool_calls,
        attested_execution_components=attested_execution_components,
        volatile_command_values=volatile_command_values,
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
        # The WIRE value, which is not the deployment mode and never was.
        # `attach()` sends "production" on this same field; a donor sends
        # "rollout", which names the protocol it speaks and is what intake has
        # always received here. Reaching this line at all means
        # resolved_mode == "training".
        "mode": "rollout",
        "sdk": f"agent-flywheel-python/{__version__}",
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
    discovery_enabled = introspect and _env_true("PERCEPTEYE_INTROSPECT", default=True)
    if describe is not None and not discovery_enabled:
        warnings.warn(
            "PERCEPTEYE_INTROSPECT is off, and it wins: the description "
            "declared with describe= is not sent.",
            RuntimeWarning, stacklevel=2,
        )
    described = None
    # The customer's own declaration, resolved AFTER the build so a callable
    # sees an agent whose tools are registered. See declaration.py.
    declared = _declaration.resolve(describe) if discovery_enabled else None
    if discovery_enabled:
        described = introspect_agent(
            runner.resolved.fn if runner.resolved is not None else None
        )
        # A COMMAND agent has no object to reflect on, so the line above always
        # returns None for one and the agent registers `unreadable` forever --
        # which blocks workflow generation, and with it every downstream phase.
        # Ask the HOST instead: run the agent once, let its own plugin observe
        # the assembled prompt and the tool list it sends to the model, and
        # read that. See host_description.py for why this cannot wait for a
        # rollout (the launch gate and the first rollout are mutually
        # blocking).
        #
        # Not when the customer declared ALL of it: then the warm-up has
        # nothing to add. A partial declaration still runs it, because what
        # they left out comes from observation -- skipping it registered a
        # declared model with no tools and no prompt.
        if (described is None and getattr(runner, "_command", None)
                and not (declared is not None and declared.complete)):
            described = _describe_via_host(
                runner._command, artifacts_dir=adir, agent_id=agent_id,
                execution_snapshot=runner.execution_snapshot,
                endpoint_env=endpoint_env, declared=declared,
            )
        described = _declaration.apply(declared, reflected=described,
                                       agent_name=agent_id)
        payload["discovered_agent"] = described.to_wire() if described else None

    # The rollout fingerprint is authored here, after discovery, from the same
    # framework-neutral facts registration sends plus exact pre-projection
    # discovery evidence and any adapter-attested opaque components. A disabled,
    # unreadable, or lossy description deliberately yields no digest, so an OPSD
    # consumer cannot mistake an unknown scaffold for an exact identity.
    #
    # A DECLARED description yields none either: it is the customer's word
    # about their agent, not an observation of what runs.
    runner.configure_execution_identity(
        described,
        discovery_enabled=discovery_enabled and declared is None,
        sdk=payload["sdk"],
        contract_version=CONTRACT_VERSION,
    )

    # Neutral worker evidence, not agent logic and not a model identity.  The
    # control plane uses this authenticated registration fact to select the
    # exact rollout worker whose end-to-end latency was qualified.  It still
    # verifies every returned trajectory carries the same digest; registration
    # is the pre-dispatch identity, not a substitute for observed evidence.
    if runner.execution_sha256 is not None:
        payload["agent_execution_sha256"] = runner.execution_sha256

    tr.register(payload)

    # ── the Product's workflow rules, when the code declares them ───────────
    # Once, after the registration and never inside it: a registration is
    # persisted whole on every restart, and rules are the Product's, not the
    # agent's. IN THE BACKGROUND: a slow or hung rulebook route must cost the
    # rules and nothing else, and inline it held the policy read and the claim
    # loop for its whole retry envelope. The declaration never raises (a
    # rulebook that did not register must not stop a donor), so the line it
    # prints -- one write, so it cannot interleave with another line -- is the
    # whole of what it can do to this process: say what happened.
    declaring = None
    if rulebook is not None:
        declaring = declare_rulebook_in_background(
            tr, rulebook,
            lambda upload: print(f"{_rulebook_line(upload)}\n", end="",
                                 flush=True),
        )

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
        # The declaration shares this client: give an answer still on its way
        # a bounded moment before closing it, rather than cutting it off.
        if declaring is not None:
            declaring.join(RULEBOOK_SETTLE_GRACE_S)
        client.close()
