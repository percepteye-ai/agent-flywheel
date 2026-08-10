"""Child-process entry point: run exactly one rollout and print the result.

``python -m percepteye_agent_flywheel.child`` reads one JSON payload on stdin and
writes one JSON line on stdout. Nothing else is written to stdout, so the
parent can always parse the last line; the agent's own printing goes to
stderr's side of the pipe and is captured for diagnostics.

This module is why per-tool-call capture works under the default isolation
mode at all. Binding, context variables and in-memory ledgers do not cross a
process boundary, so a design that binds only in the parent would record
nothing while appearing to work -- and an empty tool-call list reads
downstream as "this agent used no tools", not as "we failed to look". The
child re-establishes the rollout scope on its own side, and the ledger it
writes is a file, which the parent merges.
"""
from __future__ import annotations

import json
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
    ledger_dir: str = payload["ledger_dir"]
    timeout_s: float = float(payload.get("timeout_s") or 900.0)
    input_shape = payload.get("input_shape") or "text"

    # Import inside the child so the agent is constructed with the child's
    # environment already in place -- the routing variables are read by most
    # LLM clients at construction, not at call time. Building the agent in the
    # parent and shipping it here is exactly the ordering bug that made an
    # earlier SDK's configuration channel useless.
    from .runner import build_agent, run_rollout_in_process

    resolved = resolve_entrypoint(
        build_agent(spec), input_shape=input_shape, method=payload.get("method")
    )

    out = run_rollout_in_process(
        resolved, rollout, ledger_dir=ledger_dir, timeout_s=timeout_s,
        # Bound HERE, in the child, because this is the process the agent runs
        # in. A callback registered in the parent sees nothing a child does.
        adapters=payload.get("adapters", "auto"),
    )
    sys.stdout.write(json.dumps(out.to_wire(), default=str) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
