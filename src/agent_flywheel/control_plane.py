"""Where the control plane is: ONE answer for every reader in this package.

``PERCEPTEYE_CONTROL_PLANE_URL`` used to have no default here while the Node
SDK defaulted it to the production origin, so the same customer, following the
same onboarding, got a working agent from one SDK and an agent that refused to
start (``serve()``) or silently sent nothing (``attach()``) from the other. The
two SDKs ship in unison; this is the Node package's rule
(``src/config.js:resolveConfig``), read the same way:

1. the value passed in code (``control_plane_url=``; Node: the plugin config's
   ``controlPlaneUrl``), when it is a non-blank string;
2. ``PERCEPTEYE_CONTROL_PLANE_URL``, when it is a non-blank string;
3. :data:`DEFAULT_CONTROL_PLANE_URL`, the production origin the dashboard
   serves this API from.

A value is used trimmed of surrounding whitespace, and a blank one falls
through to the next, as Node's ``firstString`` does.
``tests/test_control_plane_url_parity.py`` drives both packages over one table
and asserts the same answer.
"""
from __future__ import annotations

import os
from collections.abc import Mapping

__all__ = [
    "DEFAULT_CONTROL_PLANE_URL",
    "ENV_CONTROL_PLANE_URL",
    "GET_A_KEY",
    "SIGN_UP_URL",
    "resolve_control_plane_url",
]

ENV_CONTROL_PLANE_URL = "PERCEPTEYE_CONTROL_PLANE_URL"

#: The live API: the dashboard at https://launch.percepteye.ai serves the
#: Flywheel API under ``/api/flywheel/v1`` on its own origin, and it is the URL
#: the dashboard shows when it issues a key. The Node SDK's default, byte for
#: byte.
DEFAULT_CONTROL_PLANE_URL = "https://launch.percepteye.ai/api/flywheel/v1"

#: Where a developer with no key gets one: the dashboard's screen for
#: connecting an agent with this SDK, on the default's own origin. Signed out,
#: it offers sign-in and "Create an account" and keeps its address through
#: both, so a new developer lands where the key is issued rather than on Home.
#: Every refusal a missing ``PERCEPTEYE_API_KEY`` causes names it, because that
#: refusal is the moment the developer needs it. They used to name only the
#: variable, which says what is missing and not where it comes from. The Node
#: SDK's ``SIGN_UP_URL``, byte for byte.
SIGN_UP_URL = "https://launch.percepteye.ai/onboard/agent-sdk"

#: The remedy for a missing key, one wording for every message that needs it.
GET_A_KEY = f"sign up free at {SIGN_UP_URL} and connect this agent to get its key"


def _non_blank(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def resolve_control_plane_url(
    explicit: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """The control-plane URL to use: the argument, the variable, the default."""
    env = os.environ if environ is None else environ
    return (
        _non_blank(explicit)
        or _non_blank(env.get(ENV_CONTROL_PLANE_URL))
        or DEFAULT_CONTROL_PLANE_URL
    )
