"""Child-process entry point: run exactly one rollout and print the result.

``python -m agent_flywheel.child`` reads one JSON payload on stdin and
writes one JSON line on stdout. Nothing else is written to stdout, so the
parent can always parse the last line; the agent's own printing goes to
stderr's side of the pipe and is captured for diagnostics.

This module is why per-tool-call capture works under the default isolation
mode at all. Binding, context variables and in-memory state do not cross a
process boundary, so a design that binds only in the parent would record
nothing while appearing to work -- and an empty tool-call list reads
downstream as "this agent used no tools", not as "we failed to look". The
child re-establishes the rollout scope on its own side, and the file it
writes is a file, which the parent merges.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from .contract import RolloutRequest
from .entrypoint import resolve_entrypoint


def main(argv: list[str] | None = None) -> int:
    raw = sys.stdin.read()
    if not raw.strip():
        print(json.dumps({"final_text": "", "exclusion": {
            "reason": "flywheel_crash", "detail": "child received no payload"}}))
        return 0

    payload: dict[str, Any] = json.loads(raw)
    spec: str = payload["spec"]
    rollout = RolloutRequest.from_wire(payload["rollout"])
    trajectory_dir: str = payload["trajectory_dir"]
    input_shape = payload.get("input_shape") or "text"

    # Import inside the child so the agent is constructed with the child's
    # environment already in place -- the routing variables are read by most
    # LLM clients at construction, not at call time. Building the agent in the
    # parent and shipping it here is exactly the ordering bug that made an
    # earlier SDK's configuration channel useless.
    from .runner import (
        DEFAULT_ROLLOUT_TIMEOUT_S,
        build_agent,
        run_rollout_in_process,
    )

    # The parent always sends ``timeout_s`` (it is the episode plan's
    # ``child_s``). A payload that lost it is planned the SAME way, from the
    # rollout's own deadline -- a fallback that disagreed with the budget the
    # parent computes is how a child outlives the window the parent reserved
    # delivery time inside.
    timeout_s = payload.get("timeout_s")
    if not timeout_s:
        import time as _time

        from .runner import episode_plan

        deadline = (payload.get("rollout") or {}).get("deadline_ts")
        if not isinstance(deadline, (int, float)):
            raise SystemExit("child payload carries neither timeout_s nor a deadline")
        timeout_s = episode_plan(
            float(deadline) - _time.time(),
            rollout_timeout_s=DEFAULT_ROLLOUT_TIMEOUT_S).child_s
    timeout_s = float(timeout_s)

    resolved = resolve_entrypoint(
        build_agent(spec), input_shape=input_shape, method=payload.get("method")
    )

    out = run_rollout_in_process(
        resolved, rollout, trajectory_dir=trajectory_dir, timeout_s=timeout_s,
        # Bound HERE, in the child, because this is the process the agent runs
        # in. A callback registered in the parent sees nothing a child does.
        adapters=payload.get("adapters", "auto"),
    )
    sys.stdout.write(json.dumps(out.to_wire(), default=str) + "\n")
    sys.stdout.flush()
    return 0


def _exit_now(code: int) -> None:
    """Leave the moment the result is written, whatever the agent left running.

    A normal exit waits for every non-daemon thread, and an agent that timed
    out is still running on one (Python cannot stop a thread). The result
    line -- with the policy calls counted so far -- was already on stdout,
    but the process stayed alive until the parent killed it at its own
    timeout and discarded the output as a kill (τ³ 4c4ce5f1: the policy's
    own timeouts held out as ``llm_calls_unobserved``). The result is the
    whole of this process's job, so once it is flushed nothing remains.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)


if __name__ == "__main__":  # pragma: no cover - process entry point
    _exit_now(main())
