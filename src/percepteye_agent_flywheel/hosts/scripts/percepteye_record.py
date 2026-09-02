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

#: TURNS GO IN A DIFFERENT FILE, and this is not tidiness.
#:
#: ``outcomes.Trajectory.read()`` builds each row with REQUIRED subscripts --
#: ``d["name"]`` and ``d["outcome"]`` -- and catches the KeyError by counting the
#: line as ``malformed``. That count becomes ``tool_calls_omitted_count``, whose
#: own comment reads "non-zero forbids training grade", and the control plane
#: turns it into a rollout exclusion. So a task or answer record written into
#: ``tool_calls.jsonl`` would not be politely ignored by older readers: it would
#: read as CORRUPTION and silently demote the rollout out of training.
TURNS_FILENAME = "turns.jsonl"

#: Where capture lands when no rollout named a directory. Until now the recorder
#: was inert without ``PERCEPTEYE_TRAJECTORY_DIR``, and the only thing that ever
#: set it was an external worker process -- so the plugin's own day-one promise
#: ("your agent's real tool failure rate", no account, no network) was
#: unreachable unless the customer exported the variable by hand, which is
#: exactly what ``percepteye_report.py``'s help text tells them to do.
#:
#: Default ON with a kill switch, because a capture default that surprises
#: nobody is worth more than one that records nothing.
HOME_ENV = "PERCEPTEYE_HOME"
CAPTURE_ENV = "PERCEPTEYE_CAPTURE"
_OFF = {"0", "false", "no", "off"}

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


#: Events that describe the TURN rather than a tool call. goose populates both
#: fields already and the recorder was reading neither, so the trajectory had
#: tool calls with no task and no answer -- nothing that can be rendered as a
#: trainable turn, and nothing a human report can title.
_TURN_EVENTS = {
    # The task, as the user (or the control plane) posed it.
    "UserPromptSubmit": ("task", "message"),
    # The agent's answer. Fires PER TURN, when the turn ends
    # (``ops_stop_hook`` returns not_applicable unless ``ends_turn``).
    "Stop": ("answer", "last_assistant_message"),
}


def _turn_record(event: dict, kind: str) -> dict | None:
    """One turn record, or None when the host gave us nothing to record."""
    field = _TURN_EVENTS[kind][1]
    text = event.get(field)
    if not isinstance(text, str) or not text.strip():
        return None
    record = {
        "_file": TURNS_FILENAME,
        "kind": _TURN_EVENTS[kind][0],
        # Bounded like every other body here: a pasted stack trace or a large
        # final answer must not make one line exceed the single-syscall cap that
        # keeps concurrent appends from shredding each other.
        "text": text.strip()[:_MAX_OUTPUT_CHARS],
    }
    for src, dst in (("session_id", "session_id"), ("working_dir", "working_dir")):
        value = event.get(src)
        if isinstance(value, str) and value:
            record[dst] = value
    return record


def _record(event: dict) -> dict | None:
    """One trajectory record from one hook event, or None to record nothing."""
    kind = event.get("event") or event.get("hook_event_name") or ""
    if kind in _TURN_EVENTS:
        return _turn_record(event, kind)

    name = event.get("tool_name")
    if not isinstance(name, str) or not name:
        # A call we cannot name is a call we cannot attribute. Dropping it
        # loses one outcome; inventing a name would corrupt every join
        # downstream, because retry detection keys on name + arguments.
        return None

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


def _capture_dir(event):
    """Where this event's records belong, or None to record nothing.

    A rollout names its own directory and that always wins -- it is what ties
    the evidence to the rollout. Otherwise the session gets one of its own,
    which is what makes the plugin useful to somebody who has never heard of a
    control plane.
    """
    if str(os.environ.get(CAPTURE_ENV, "")).strip().lower() in _OFF:
        return None
    base = os.environ.get(TRAJECTORY_DIR_ENV)
    if base:
        return base
    # ``session_id`` is on the wire for every goose hook event and was being
    # discarded. Without it two concurrent sessions would interleave into one
    # file and neither would be a turn.
    session = event.get("session_id")
    if not isinstance(session, str) or not session.strip():
        return None
    safe = "".join(c for c in session if c.isalnum() or c in "-_")[:120]
    if not safe:
        return None
    home = os.environ.get(HOME_ENV) or os.path.join(
        os.path.expanduser("~"), ".percepteye")
    return os.path.join(home, "turns", safe)


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
        if not isinstance(event, dict):
            return 0
    except Exception:
        return 0

    base = _capture_dir(event)
    if not base:
        return 0  # Capture is off, or there is nothing to key this on.

    try:
        record = _record(event)
        if record is None:
            return 0
        filename = record.pop("_file", TOOL_CALLS_FILENAME)
        os.makedirs(base, exist_ok=True)
        data = (_fit(record) + "\n").encode("utf-8")
        # One unbuffered O_APPEND write, not a buffered file object. Under
        # O_APPEND the kernel makes the seek-and-write atomic per syscall, so
        # concurrent hook processes interleave whole lines instead of shredding
        # each other -- but only while a record is ONE write, which is what
        # _MAX_RECORD_BYTES guarantees. There is no flush to lose, so a rollout
        # killed on its deadline keeps every call up to the kill.
        fd = os.open(os.path.join(base, filename),
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
