"""A minimal agent used by the subprocess round-trip test.

Importable by design: subprocess isolation rebuilds the agent in the child, so
the entrypoint must be reachable as ``module:attr``.
"""
from __future__ import annotations

import os

from percepteye_agent_flywheel import record_tool_call


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


class Medium:
    """Sleeps long enough that serving three of them sequentially is plainly
    distinguishable from serving them at once."""

    def invoke(self, task: str) -> str:
        import time

        time.sleep(2.0)
        return f"medium:{task}"


def build_medium() -> Medium:
    return Medium()


class NamedMethod:
    """Has BOTH a custom entrypoint and a decoy the prober would find first.

    If ``method=`` does not reach the child, the child falls back to probing,
    finds ``invoke``, and returns the decoy -- so the test fails on the value
    rather than on an exception.
    """

    def handle_task(self, task: str) -> str:
        return f"named:{task}"

    def invoke(self, task: str) -> str:
        return f"decoy:{task}"


def build_named_method() -> NamedMethod:
    return NamedMethod()


class UninstrumentedLangChainAgent:
    """An agent with NO Percepteye code in it at all.

    Not one import, not one ``record_tool_call``. If outcomes come back from
    this, they came from the adapter binding the framework's own callbacks --
    which is the entire claim of zero-instrumentation capture.
    """

    def invoke(self, task: str) -> str:
        from langchain_core.tools import tool

        @tool
        def lookup(q: str) -> str:
            """Look something up."""
            return f"found:{q}"

        @tool
        def explode(q: str) -> str:
            """Always fails."""
            raise RuntimeError("upstream refused")

        out = lookup.invoke({"q": task})
        try:
            explode.invoke({"q": task})
        except Exception:
            pass
        return str(out)


def build_uninstrumented() -> UninstrumentedLangChainAgent:
    return UninstrumentedLangChainAgent()


class ArtifactWriter:
    """Writes files where a real coding agent would: its own artifacts dir."""

    def invoke(self, task: str) -> str:
        import os
        import pathlib

        root = pathlib.Path(os.environ["PERCEPTEYE_LEDGER_DIR"]) / os.environ[
            "PERCEPTEYE_ROLLOUT_ID"]
        (root / "out").mkdir(parents=True, exist_ok=True)
        (root / "result.txt").write_text(f"done: {task}\n")
        (root / "out" / "patch.diff").write_text("--- a\n+++ b\n@@ -1 +1 @@\n-x\n+y\n")
        record_tool_call("write", {"n": 2}, outcome="ok", status_code=200)
        return "wrote 2 files"


def build_artifact_writer() -> ArtifactWriter:
    return ArtifactWriter()
