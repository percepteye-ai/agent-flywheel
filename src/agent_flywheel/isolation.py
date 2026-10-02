"""Per-rollout isolation, and the environment handed to the child.

Two rollouts running at once must not be able to see each other's endpoint,
credential or trajectory directory. The default is one child process per rollout, with the
rollout's configuration crossing into a FRESH process at spawn.

That direction matters. An earlier SDK wrote server-supplied values into the
running host process's ``os.environ`` and restored them afterwards. It was
both dangerous -- a global mutation in a live application -- and useless,
because the agent's LLM client had already read its base URL when it was
constructed, long before the poll loop started.

**The child's environment is scrubbed, never copied wholesale.** The parent
holds the long-lived control-plane key. Splatting the parent environment into
a child that runs customer agent code -- frequently model-authored code with a
shell -- hands the strongest credential in the system to the least trusted
process in it. Every credential this SDK mints is per-rollout and expiring;
the one that is not must never leave the parent.
"""
from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

__all__ = [
    "PERCEPTEYE_NAMESPACE",
    "ROUTING_VARS",
    "UNROUTABLE_PROVIDER_VARS",
    "build_child_env",
    "is_environment_overlay",
    "overlay_environment",
    "unroutable_providers_present",
]

PERCEPTEYE_NAMESPACE = "PERCEPTEYE_"

#: The only names in our namespace a caller may set through ``extra``.
#: Everything else in it is ours and must not be forgeable from outside --
#: the control-plane credential lives in this namespace.
_SETTABLE_BY_CALLER = (
    # The turn input, for a command entrypoint that reads it from the
    # environment rather than stdin.
    "PERCEPTEYE_TURN_INPUT",
)

#: Names the child is given for this rollout, and nothing else from our side.
#: These are the clients whose base URL our gateway can actually serve: it
#: speaks OpenAI ``/chat/completions``, so pointing a client at it only works
#: if the client speaks that.
ROUTING_VARS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
)

#: Provider configuration we CANNOT redirect, removed from the child.
#:
#: Setting the two OpenAI variables above does nothing for an agent built on
#: Gemini, Azure OpenAI or Bedrock: those clients read their own names, so the
#: agent sailed past our gateway and sampled from the live provider. Every
#: completion it served was then invisible to us -- no logprobs, no policy
#: identity, and a trajectory that cannot back an on-policy objective while
#: looking exactly like one that can.
#:
#: Removing them is deliberate and is not the same as removing a customer's
#: other credentials. Their Jira token is theirs and their agent needs it; the
#: MODEL endpoint is the one thing this product replaces. An agent that finds
#: no provider configured raises at construction, in front of the person who
#: configured it, which is the outcome we want -- silent off-gateway sampling
#: is the one we do not.
#:
#: This list names what we know. It is not a proof of completeness, and it is
#: not the only defence: the control plane compares the calls an agent reports
#: against the calls the gateway actually served, and a mismatch excludes the
#: rollout from training however the traffic escaped.
UNROUTABLE_PROVIDER_VARS = (
    # Azure OpenAI — same SDK, different endpoint shape; OPENAI_BASE_URL is
    # ignored once these are set.
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_AD_TOKEN",
    "OPENAI_API_TYPE",
    "OPENAI_API_VERSION",
    # Google Gemini / Vertex.
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_GENAI_USE_VERTEXAI",
    # NVIDIA NIM / build.nvidia.com — the inference path NemoClaw configures
    # for OpenClaw. NemoClaw sets it up ONCE at onboarding, via
    # ``openshell provider create --credential NVIDIA_API_KEY=… --config
    # OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1``, so the routing
    # lives in the OpenShell provider registry rather than the process
    # environment. Our per-rollout OPENAI_BASE_URL therefore does not win,
    # and the agent samples from NVIDIA cloud while the control plane
    # believes it is sampling the policy under training. Removing the key is
    # the reliable half, and the startup report names it.
    "NVIDIA_API_KEY",
    "NVIDIA_NIM_API_KEY",
    # Bedrock. Deliberately only the Bedrock-specific token: AWS_ACCESS_KEY_ID
    # is the same credential a customer's agent uses for S3, and taking that
    # would break agents that never touch a model through it.
    "AWS_BEARER_TOKEN_BEDROCK",
    # OpenAI-compatible resellers. Their base URL is redirectable in principle,
    # but only if the agent reads it from the environment, and each SDK spells
    # that differently. Removing the key is the reliable half.
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "OPENROUTER_API_KEY",
    "FIREWORKS_API_KEY",
    "DEEPSEEK_API_KEY",
    "PERPLEXITY_API_KEY",
    "COHERE_API_KEY",
    "XAI_API_KEY",
)

#: Passed through to the child even under a strict allowlist -- without these a
#: Python process generally cannot start.
_BASELINE_PASSTHROUGH = (
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TZ", "TMPDIR", "TEMP",
    "SYSTEMROOT", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
)


def _is_our_namespace(name: str) -> bool:
    """Names owned by Percepteye and therefore subject to our scrub policy."""
    return name.startswith(PERCEPTEYE_NAMESPACE)


def is_environment_overlay(overlay: Mapping[object, object]) -> bool:
    """Whether ``overlay`` is one an adapter may hand a child process.

    Every name a string, mapped to the string to set it to, or to ``None`` to
    REMOVE it from the child. Removal is not an empty value: a process told
    ``PYTHONPATH=""`` still sees the name, and what an empty value means is up
    to whoever reads it.
    """
    return all(
        isinstance(name, str) and (value is None or isinstance(value, str))
        for name, value in overlay.items()
    )


def overlay_environment(
    env: dict[str, str], overlay: Mapping[str, str | None],
) -> None:
    """Apply an adapter's overlay to ``env`` in place: set, or remove on None."""
    for name, value in overlay.items():
        if value is None:
            env.pop(name, None)
        else:
            env[name] = value


def build_child_env(
    *,
    rollout_id: str,
    llm_url: str,
    api_key: str,
    trajectory_dir: str,
    parent_env: Mapping[str, str] | None = None,
    allowlist: Iterable[str] | None = None,
    extra: Mapping[str, str | None] | None = None,
    endpoint_env: tuple[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for one rollout's child process.

    Two policies:

    * default -- inherit the customer's environment, minus our namespace. Their
      agent needs their credentials; that is the entire premise of running it
      in their process. Ours is removed.
    * ``allowlist`` -- inherit only the named variables (plus a small baseline
      needed to start a process at all). For customers who want the child to
      carry nothing they did not name.

    In both policies the rules below are non-negotiable and not configurable:

    1. No variable in the Percepteye namespace is inherited from the parent.
       The control-plane key lives there and stays in the parent.
    2. Adapter-owned ``extra`` values are overlaid, except for our protected
       namespace. They can supply process layout such as a private HOME but
       cannot override rollout identity or routing. A ``None`` value removes
       that name from the child, inherited or baseline alike -- an interpreter
       variable an adapter's runtime must not see, say.
    3. Two Percepteye variables are then SET, scoped to this rollout: the
       rollout id and the trajectory directory.
    4. The per-rollout LLM endpoint and its single-use key are set last, so
       they win over anything inherited or adapter-supplied.
    """
    src = dict(os.environ if parent_env is None else parent_env)
    if allowlist is not None:
        keep = set(allowlist) | set(_BASELINE_PASSTHROUGH)
        env = {k: v for k, v in src.items() if k in keep}
    else:
        env = dict(src)

    # (1) Our namespace never crosses the boundary by inheritance.
    for name in [k for k in env if _is_our_namespace(k)]:
        env.pop(name, None)

    # (2) Provider configuration we cannot point at our gateway. Left in
    # place, an agent built on one of these routes straight past us and
    # samples from the live provider -- producing a trajectory that cannot
    # back an on-policy objective while looking exactly like one that can.
    for name in UNROUTABLE_PROVIDER_VARS:
        env.pop(name, None)

    # (2b) Adapter-owned execution layout. Do this before rollout identity and
    # routing so no adapter can replace the SDK's scoped identifiers or send a
    # completion around the gateway with an ambient/provider credential.
    if extra:
        overlay_environment(env, {
            k: v for k, v in extra.items()
            if not _is_our_namespace(k) or k in _SETTABLE_BY_CALLER
        })

    # (3) Rollout-scoped, and nothing else from our side.
    env["PERCEPTEYE_ROLLOUT_ID"] = rollout_id
    env["PERCEPTEYE_TRAJECTORY_DIR"] = trajectory_dir

    # (4) Routing last so it beats anything inherited. Note this only wins over
    # ENVIRONMENT: a client constructed with an explicit base_url argument in
    # the customer's own code beats every one of these, silently, and lands at
    # a live third party. Detecting that is what binding is for.
    env["OPENAI_BASE_URL"] = llm_url
    env["OPENAI_API_BASE"] = llm_url
    env["OPENAI_API_KEY"] = api_key
    env["ANTHROPIC_BASE_URL"] = llm_url
    env["ANTHROPIC_API_KEY"] = api_key

    # (4b) Optional adapter-declared endpoint names. Some clients read neither
    # default above; the mapping is explicit rather than guessed. Additive:
    # defaults stay because a process may use more than one compatible client.
    if endpoint_env:
        url_var, key_var = endpoint_env
        env[str(url_var)] = llm_url
        env[str(key_var)] = api_key

    return env


def unroutable_providers_present(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Provider configuration in this environment we cannot route.

    Reported at startup so a misconfiguration is visible to the person who
    made it, rather than showing up a run later as a training-grade exclusion
    nobody can explain.
    """
    src = os.environ if env is None else env
    return tuple(n for n in UNROUTABLE_PROVIDER_VARS if src.get(n))
