"""Per-rollout isolation, and the environment handed to the child.

Two rollouts running at once must not be able to see each other's endpoint,
credential or ledger. The default is one child process per rollout, with the
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

__all__ = ["PERCEPTEYE_NAMESPACE", "ROUTING_VARS", "build_child_env"]

PERCEPTEYE_NAMESPACE = "PERCEPTEYE_"

#: Names the child is given for this rollout, and nothing else from our side.
ROUTING_VARS = (
    "OPENAI_BASE_URL",
    "OPENAI_API_BASE",
    "OPENAI_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_API_KEY",
)

#: Passed through to the child even under a strict allowlist -- without these a
#: Python process generally cannot start.
_BASELINE_PASSTHROUGH = (
    "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "TZ", "TMPDIR", "TEMP",
    "SYSTEMROOT", "PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
)


def _is_our_secret(name: str) -> bool:
    """Names that belong to Percepteye and must never reach customer code."""
    return name.startswith(PERCEPTEYE_NAMESPACE)


def build_child_env(
    *,
    rollout_id: str,
    llm_url: str,
    api_key: str,
    ledger_dir: str,
    parent_env: Mapping[str, str] | None = None,
    allowlist: Iterable[str] | None = None,
    extra: Mapping[str, str] | None = None,
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
    2. Exactly two Percepteye variables are then SET, scoped to this rollout:
       the rollout id and the ledger directory.
    3. The per-rollout LLM endpoint and its single-use key are set last, so
       they win over anything inherited.
    """
    src = dict(os.environ if parent_env is None else parent_env)

    if allowlist is not None:
        keep = set(allowlist) | set(_BASELINE_PASSTHROUGH)
        env = {k: v for k, v in src.items() if k in keep}
    else:
        env = dict(src)

    # (1) Our namespace never crosses the boundary by inheritance.
    for name in [k for k in env if _is_our_secret(k)]:
        env.pop(name, None)

    # (2) Rollout-scoped, and nothing else from our side.
    env["PERCEPTEYE_ROLLOUT_ID"] = rollout_id
    env["PERCEPTEYE_LEDGER_DIR"] = ledger_dir

    # (3) Routing last so it beats anything inherited. Note this only wins over
    # ENVIRONMENT: a client constructed with an explicit base_url argument in
    # the customer's own code beats every one of these, silently, and lands at
    # a live third party. Detecting that is what binding is for.
    env["OPENAI_BASE_URL"] = llm_url
    env["OPENAI_API_BASE"] = llm_url
    env["OPENAI_API_KEY"] = api_key
    env["ANTHROPIC_BASE_URL"] = llm_url
    env["ANTHROPIC_API_KEY"] = api_key

    if extra:
        for k, v in extra.items():
            if _is_our_secret(k) and k not in ("PERCEPTEYE_ROLLOUT_ID",
                                               "PERCEPTEYE_LEDGER_DIR"):
                continue
            env[k] = v

    return env
