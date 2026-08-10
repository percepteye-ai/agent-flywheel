"""Serve prime-agent rollouts. No Python integration — it is a TypeScript binary.

    PERCEPTEYE_CONTROL_PLANE_URL=... PERCEPTEYE_API_KEY=... \
      python examples/prime_agent/serve_prime.py

prime-agent is served through the COMMAND entrypoint, which is the path that
exists for exactly this: an agent that is a process rather than an importable
Python object. It needs no Percepteye code, and the only thing added to it is
one extension that points its completions at the per-rollout endpoint.
"""
from __future__ import annotations

import os
import pathlib

import percepteye_agent_flywheel as fw

HERE = pathlib.Path(__file__).resolve().parent

#: `-p` is print mode: one task, non-interactive, answer on stdout. The task
#: also arrives on stdin, which prime-agent merges into the prompt — the SDK
#: sends it both ways because CLIs differ and guessing wrongly is silent.
#:
#: The autonomous limits are bounded deliberately. A rollout that runs until it
#: decides to stop is a rollout whose cost we do not control, and the episode
#: budget is the only clock the control plane can enforce from outside.
ARGV = [
    "prime-agent",
    "-p",
    "--mode", "json",              # structured events, not prose
    "-e", str(HERE / "route.ts"),  # the routing extension
    "--autonomous",
    "--autonomous-max-turns", os.environ.get("PRIME_MAX_TURNS", "12"),
    "--autonomous-max-continuations", "2",
]


def main() -> int:
    return fw.serve(
        ARGV,
        agent_id=os.environ.get("PERCEPTEYE_AGENT_ID", "prime-agent"),
        input_shape="text",
        # prime-agent's own tool results live in its JSON event stream, which
        # nothing here parses yet. Claiming outcomes we do not extract would be
        # the assertion this whole contract refuses, so this stays False until
        # the event parser lands.
        reports_tool_calls=False,
        concurrency=int(os.environ.get("PRIME_CONCURRENCY", "1")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
