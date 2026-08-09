"""A minimal agent used by the subprocess round-trip test.

Importable by design: subprocess isolation rebuilds the agent in the child, so
the entrypoint must be reachable as ``module:attr``.
"""
from __future__ import annotations

import os

from percepteye_harness import record_tool_call


class EchoAgent:
    def invoke(self, task: str) -> str:
        # A real, observed success.
        record_tool_call("lookup", {"q": task}, outcome="ok", status_code=200)
        # An application-level refusal returned as HTTP 200 -- the pair that
        # makes "read the status" wrong.
        record_tool_call(
            "charge", {"amount": 1}, outcome="failed", status_code=200,
            error="app_auth_error",
        )
        # A tool that swallowed its own exception and returned success-shaped
        # text. The honest report is unknown; the status must not survive.
        record_tool_call(
            "opaque", {}, outcome="unknown", status_code=200,
            output='{"status_code": 200, "note": "done"}',
        )
        return f"handled:{task}"


def build() -> EchoAgent:
    return EchoAgent()


class LeakProbe:
    """Reports what the child can actually see of the parent's environment."""

    def invoke(self, task: str) -> str:
        seen = sorted(k for k in os.environ if k.startswith("PERCEPTEYE_"))
        return "|".join([
            "percepteye_vars=" + ",".join(seen),
            "control_key=" + os.environ.get("PERCEPTEYE_API_KEY", "<absent>"),
            "customer_key=" + os.environ.get("JIRA_API_TOKEN", "<absent>"),
            "openai_base=" + os.environ.get("OPENAI_BASE_URL", "<absent>"),
            "openai_key=" + os.environ.get("OPENAI_API_KEY", "<absent>"),
        ])


def build_leak_probe() -> LeakProbe:
    return LeakProbe()


class Boom:
    def invoke(self, task: str) -> str:
        raise RuntimeError("agent exploded")


def build_boom() -> Boom:
    return Boom()


class Slow:
    """Blocks far longer than any timeout the test sets, so a timeout that is
    not enforced shows up as a hang rather than a pass."""

    def invoke(self, task: str) -> str:
        import time

        time.sleep(120)
        return "should never be reported"


def build_slow() -> Slow:
    return Slow()
