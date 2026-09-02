#!/usr/bin/env python3
"""Record one tool call's outcome into the flywheel trajectory.

Reads a hook event as JSON on stdin and appends one line to
``$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`` -- the wire format documented in
``packages/percepteye-agent-flywheel/src/trajectory.js``. That format is a FILE,
not a function call, precisely so a host we have no SDK for can feed the
flywheel by appending a line.

Two host vocabularies, one recorder:

  goose (Open Plugins hooks)   PostToolUse         -> ok
                               PostToolUseFailure  -> failed
  Gemini CLI                   AfterTool           -> failed iff tool_response.error

goose picks between its two events with its OWN check --
``Ok(r) if r.is_error != Some(true)`` in agents/agent.rs -- so the outcome here
is goose's observed verdict, not our reconstruction of one. Gemini reports the
error inline instead, so the branch differs while the meaning does not.

THREE PROPERTIES THIS FILE MUST KEEP:

1. Inert when ``PERCEPTEYE_TRAJECTORY_DIR`` is unset. That is the whole reason
   the same install is safe to leave in place permanently: no rollout, no file,
   no behaviour. Absence is the normal production case, not an error.
2. Never fails the agent. Every path exits 0. A telemetry writer that can break
   someone's tool call is worse than no telemetry.
3. Writes nothing to stdout. It subscribes only to POST-tool events, which are
   fire-and-forget on both hosts (goose emits them via `emit`, not
   `emit_blocking_with_outcome`). Observing cannot change what is observed --
   a property of which events we register for, not a promise about this code.

Standard library only. This runs inside someone else's agent.
"""
from __future__ import annotations

import json
import os
import sys

TRAJECTORY_DIR_ENV = "PERCEPTEYE_TRAJECTORY_DIR"
TOOL_CALLS_FILENAME = "tool_calls.jsonl"

#: Never rounded up to "ok". A host event we do not recognise means we did not
#: observe the outcome, and `unknown` is the supported, expected value for that
#: -- verifiers abstain on it rather than scoring a guess.
_GOOSE_OUTCOME = {"PostToolUse": "ok", "PostToolUseFailure": "failed"}

_MAX_OUTPUT_CHARS = 4000
_MAX_ERROR_CHARS = 1000

#: Per-argument-value cap. A file-writing tool puts the ENTIRE file in
#: `tool_input`, so an uncapped copy makes one call worth 200KB of trajectory
#: (measured). This record is evidence about the call, not a copy of its
#: payload -- the payload is already on the agent's disk.
_MAX_ARG_CHARS = 1000

#: Whole-line cap, deliberately under the 8 KiB stdio buffer. `os.write` below
#: then issues ONE syscall, which is what makes the O_APPEND interleaving
#: guarantee real: goose runs tool calls concurrently, so two hook processes
#: can append at once, and a record split across several write() calls lets the
#: other process interleave and corrupt both lines.
_MAX_RECORD_BYTES = 7000


def _text(value: object, limit: int) -> str | None:
    """A displayable string for a result body, or None if there is none."""
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value)
        except (TypeError, ValueError):
            value = str(value)
    value = value.strip()
    return value[:limit] if value else None


def _clip(value: object, limit: int) -> object:
    """A bounded stand-in for one argument value.

    Truncation is always VISIBLE. A clipped value keeps its key and says how
    much was dropped, so a reader can tell "the agent passed a huge argument"
    from "the agent passed nothing" -- silently shortening would make those two
    indistinguishable in the evidence.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + f"…[truncated, {len(text)} chars total]"


def _fit(record: dict) -> str:
    """Serialise one record, clipping arguments until the line fits the cap."""
    line = json.dumps(record)
    if len(line.encode("utf-8")) <= _MAX_RECORD_BYTES:
        return line
    # Values were already clipped once; a line still over cap means many keys
    # or many medium ones, so clip harder before giving up on the values.
    for limit in (200, 40):
        record["arguments"] = {k: _clip(v, limit)
                               for k, v in record["arguments"].items()}
        line = json.dumps(record)
        if len(line.encode("utf-8")) <= _MAX_RECORD_BYTES:
            return line
    # An enormous KEY set rather than enormous values. Say so rather than
    # emitting a line that breaks the append guarantee.
    dropped = len(record["arguments"])
    record["arguments"] = {"_truncated": f"{dropped} arguments dropped"}
    return json.dumps(record)


def _record(event: dict) -> dict | None:
    """One trajectory record from one hook event, or None to record nothing."""
    name = event.get("tool_name")
    if not isinstance(name, str) or not name:
        # A call we cannot name is a call we cannot attribute. Dropping it
        # loses one outcome; inventing a name would corrupt every join
        # downstream, because retry detection keys on name + arguments.
        return None

    kind = event.get("event") or event.get("hook_event_name") or ""
    response = event.get("tool_response")
    error = None

    if kind in _GOOSE_OUTCOME:
        outcome = _GOOSE_OUTCOME[kind]
    elif kind == "AfterTool" and isinstance(response, dict):
        # Gemini reports the failure inline. `error` absent means the call
        # returned normally -- an observation, so `ok` is honest here.
        err = response.get("error")
        outcome = "failed" if err else "ok"
        error = _text(err, _MAX_ERROR_CHARS)
    else:
        outcome = "unknown"

    args = event.get("tool_input")
    if not isinstance(args, dict):
        args = {} if args is None else {"_raw": _clip(args, _MAX_ARG_CHARS)}
    else:
        args = {k: _clip(v, _MAX_ARG_CHARS) for k, v in args.items()}

    record = {
        "name": name,
        "arguments": args,
        "outcome": outcome,
        # No HTTP status exists for a local tool call. `null` is the honest
        # value and the contract requires the key, not a number -- inventing
        # a 200 here is exactly the failure the tri-state exists to prevent.
        "status_code": None,
    }

    # goose's post-tool context carries no result body (it sends tool_name,
    # tool_input, tool_call_id and working_dir), so `output` is simply absent
    # there rather than empty. Gemini sends one, so it is recorded.
    if isinstance(response, dict):
        output = _text(response.get("llmContent") or response.get("returnDisplay"),
                       _MAX_OUTPUT_CHARS)
        if output:
            record["output"] = output
    if error:
        record["error"] = error

    call_id = event.get("tool_call_id")
    if isinstance(call_id, str) and call_id:
        record["tool_call_id"] = call_id

    return record


def main() -> int:
    base = os.environ.get(TRAJECTORY_DIR_ENV)
    if not base:
        return 0  # No rollout in progress. Do nothing at all.

    try:
        event = json.loads(sys.stdin.read() or "{}")
        if not isinstance(event, dict):
            return 0
        record = _record(event)
        if record is None:
            return 0
        os.makedirs(base, exist_ok=True)
        data = (_fit(record) + "\n").encode("utf-8")
        # One unbuffered O_APPEND write, not a buffered file object. Under
        # O_APPEND the kernel makes the seek-and-write atomic per syscall, so
        # concurrent hook processes interleave whole lines instead of shredding
        # each other -- but only while a record is ONE write, which is what
        # _MAX_RECORD_BYTES guarantees. There is no flush to lose, so a rollout
        # killed on its deadline keeps every call up to the kill.
        fd = os.open(os.path.join(base, TOOL_CALLS_FILENAME),
                     os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except Exception:
        # Never fail the agent over telemetry, and never print -- the agent's
        # own output is the thing being measured.
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
