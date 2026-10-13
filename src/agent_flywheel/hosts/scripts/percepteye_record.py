#!/usr/bin/env python3
"""Record one tool call's outcome into the flywheel trajectory.

Reads a hook event as JSON on stdin and appends one line to
``$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`` -- the wire format documented in
``trajectory.js`` in the Node SDK (percepteye-ai/agent-flywheel-node). That format is a FILE,
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

1. Local by default, and silent about it. Capture no longer requires
   ``PERCEPTEYE_TRAJECTORY_DIR``: without one it writes under
   ``$PERCEPTEYE_HOME/turns/<session>``, which is what makes the plugin's
   day-one promise -- your agent's real tool failure rate, no account, no
   network -- reachable for a customer who never signs up. ``PERCEPTEYE_CAPTURE=0``
   turns it off entirely. Nothing leaves the machine without an API key.
2. Never fails the agent. Every path exits 0, and every branch that can touch
   the network or the filesystem is wrapped. A telemetry writer that can break
   someone's tool call is worse than no telemetry.
3. Writes nothing to stdout. This is now a property of THIS SCRIPT, not of the
   events it subscribes to: ``Stop`` is delivered through ``emit_blocking`` and
   its chain CAN deny a turn, so the guarantee rests on ``classify_output``
   producing a denial only for exit code 2 or a JSON verdict on stdout --
   neither of which this file ever emits. Both lines above are load-bearing for
   a reason they were not originally written for.

Standard library only ON THE PER-TOOL-CALL PATH, which is the path that runs
hundreds of times and blocks the agent each time. The three SESSION branches --
reconcile, claim, report -- import the SDK, once per session, and say so at
their own definitions. This runs inside someone else's agent.
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

#: A TURN's text (the task, the answer) is cut at this. A tool call's output
#: and error are NOT: they take the SDK's one body bound, below.
_MAX_OUTPUT_CHARS = 4000

#: The SDK's body bound, ``agent_flywheel/_body_bound.py``, located from this
#: file's own place in the package. ONE definition: the event-stream parser,
#: the adapters and this recorder shorten an output or an error the same way,
#: or ``tool_calls[].error`` has a shape per producer, as it did when the
#: parser kept both ends and this file still cut an error to its first 1000
#: chars, deleting a traceback's exception line. It is loaded BY PATH, never
#: imported: ``import agent_flywheel`` is the per-tool-call cost this file
#: must not pay, and the module it loads imports nothing.
_BODY_BOUND_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "_body_bound.py")

#: When one line holding both bodies at the full bound will not fit one write,
#: each is re-bounded at these limits in turn, output before error at every
#: step: the error is the payload a model learns to correct from.
_BODY_STEPS = (2000, 1000, 500, 250)

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


def _text(value: object) -> str | None:
    """A displayable string for a result body, or None if there is none.

    Whole, not bounded: ``_fit`` bounds a body, because only it can see
    whether the line it is building fits one write.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            value = json.dumps(value)
        except (TypeError, ValueError):
            value = str(value)
    value = value.strip()
    return value or None


def _body_bound():
    """The SDK's body-bound module, loaded from its file (see above)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_percepteye_body_bound", _BODY_BOUND_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def _fits(line: str) -> bool:
    return len(line.encode("utf-8")) <= _MAX_RECORD_BYTES


def _fit(record: dict) -> str:
    """Serialise one record within the single-write cap.

    A tool call's bodies arrive whole and are bounded HERE, because the cap is
    on the LINE: an output and an error can each be within the SDK's body
    bound and still not fit one write together. Every re-bound starts from the
    body the tool returned, so its marker counts what was removed from that,
    never from a body this file had already shortened. Arguments are clipped
    only after the bodies have given way, because retry detection keys on
    name + arguments.
    """
    raw = {k: record[k] for k in ("output", "error") if k in record}
    if raw:
        bound = _body_bound().bounded
        for key, text in raw.items():
            record[key] = bound(text)
    line = json.dumps(record)
    if _fits(line):
        return line
    for limit in _BODY_STEPS if raw else ():
        for key in ("output", "error"):
            if key in raw:
                record[key] = bound(raw[key], limit)
                line = json.dumps(record)
                if _fits(line):
                    return line
    # Values were already clipped once; a line still over cap means many keys
    # or many medium ones, so clip harder before giving up on the values.
    for limit in (200, 40):
        record["arguments"] = {k: _clip(v, limit)
                               for k, v in record["arguments"].items()}
        line = json.dumps(record)
        if _fits(line):
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
        error = _text(err)
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
        output = _text(response.get("llmContent") or response.get("returnDisplay"))
        if output:
            record["output"] = output
    if error:
        record["error"] = error

    call_id = event.get("tool_call_id")
    if isinstance(call_id, str) and call_id:
        record["tool_call_id"] = call_id

    return record


def turns_root():
    """Where per-session captures live when nothing names a directory.

    ONE definition, because there are two consumers now: this writer, and
    ``agent-flywheel report``, which had nothing to point at. A second copy of this
    join in the CLI would be a second answer to "where did my captures go" --
    and the reader is the half that gets it wrong silently, by reporting an
    empty directory rather than the one with the data in it.
    """
    home = os.environ.get(HOME_ENV) or os.path.join(
        os.path.expanduser("~"), ".percepteye")
    return os.path.join(home, "turns")


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
    return os.path.join(turns_root(), safe)


#: The claim's own record, written beside the captured files and keyed by the
#: same session. SessionStart writes it; Stop consumes and removes it.
CLAIM_FILENAME = "claim.json"

#: Where a claimed task is delivered. A hook cannot inject a prompt — the host
#: emits ``UserPromptSubmit`` observe-only — so the working directory is the
#: channel, and the hook's cwd IS the agent's.
TASK_FILENAME = "PERCEPTEYE_TASK.md"


def _contributing():
    """The control-plane client, or ``None`` when this install only records.

    IMPORTED HERE AND NOWHERE ELSE. This file runs once per TOOL CALL and blocks
    the agent each time; importing the SDK costs 46 ms against a 16 ms
    interpreter floor, so the per-tool-call path must never reach it. Inside
    these two session branches it runs once per SESSION, where it is nothing.

    Returns ``None`` for the ordinary case, not an error: a customer with no key
    is not misconfigured, they are recording locally, which is the plugin's
    day-one promise. A marketplace install with no SDK at all lands in the same
    branch through the ImportError.
    """
    key = os.environ.get("PERCEPTEYE_API_KEY")
    if not key:
        return None
    try:
        from agent_flywheel._http import ControlPlaneClient
        from agent_flywheel.control_plane import resolve_control_plane_url
        from agent_flywheel.transport import AttachTransport
    except Exception:
        return None
    # THE LIBRARY'S RULE, not a second reading of the variable. This used to
    # read an un-prefixed ``CONTROL_PLANE_ENV_URL`` FIRST -- a name no library
    # module reads, and which the Node SDK ranked LAST -- and then required
    # PERCEPTEYE_CONTROL_PLANE_URL while the SDKs default it, so the recorder
    # and the SDK in one process could disagree about where the control plane
    # is. Imported here, on the once-per-session path, never per tool call.
    url = resolve_control_plane_url()
    agent = os.environ.get("PERCEPTEYE_AGENT_ID") or "goose"
    client = ControlPlaneClient(base_url=url, api_key=key)
    return AttachTransport(client, agent_id=agent), client


def _reconcile():
    """Make the host's endpoint match ``PERCEPTEYE_AGENT_MODE``.

    DELIBERATELY NOT GATED ON THE API KEY, unlike everything else that talks to
    the control plane. The state this exists to repair is precisely the one
    where the key is gone: a customer who flips to production AND unsets
    ``PERCEPTEYE_API_KEY`` would otherwise keep a config pointing at our gateway
    with nothing left to authenticate it -- every completion failing, and the
    kill switch they just used appearing to be the cause.

    Takes effect on the NEXT session: goose resolves its endpoint at process
    start (measured -- see ``endpoint.py``). Running it every SessionStart is
    what makes that self-correcting instead of a repair someone must remember.
    """
    from agent_flywheel.install_hooks import reconcile_endpoint

    reconcile_endpoint()


def _claim(event, base):
    """Claim ONE rollout and put its task where the agent will find it.

    One, not a batch: a batch is a worker's idea, so a spawning parent can run
    several children at once. This session is one agent doing one task, and
    claiming more would lease work nobody executes until the lease lapses.

    An empty queue is the NORMAL case — the customer started their agent and the
    control plane had nothing for it.
    """
    got = _contributing()
    if got is None:
        return
    transport, client = got
    try:
        rollouts, _ = transport.claim(1)
        if not rollouts:
            return
        r = rollouts[0]
        work = event.get("working_dir")
        if isinstance(work, str) and work:
            # The task lands BEFORE the claim record, so a crash between them
            # leaves a task with no claim (harmless — the lease lapses) rather
            # than a claim whose task never arrived, which would report an
            # empty turn as though the agent had tried and failed.
            with open(os.path.join(work, TASK_FILENAME), "w",
                      encoding="utf-8") as fh:
                fh.write(r.turn_input)
        os.makedirs(base, exist_ok=True)
        with open(os.path.join(base, CLAIM_FILENAME), "w",
                  encoding="utf-8") as fh:
            json.dump({"rollout_id": r.rollout_id, "task": r.turn_input,
                       "lease_token": r.lease_token,
                       "deadline_ts": r.deadline_ts, "llm_url": r.llm_url}, fh)
    finally:
        client.close()


def _report(event, base):
    """Report the captured turn against the rollout this session claimed."""
    path = os.path.join(base, CLAIM_FILENAME)
    if not os.path.isfile(path):
        return                      # nothing was claimed for this session
    got = _contributing()
    if got is None:
        return
    transport, client = got
    try:
        with open(path, encoding="utf-8") as fh:
            state = json.load(fh)
        from agent_flywheel.contract import (
            RolloutOutput,
            RolloutRequest,
        )
        from agent_flywheel.outcomes import Trajectory

        traj = Trajectory(base)
        transport.report(
            RolloutRequest(
                rollout_id=state["rollout_id"],
                turn_input=state.get("task") or "",
                llm_url=state.get("llm_url") or "",
                api_key="",          # the report authenticates with OUR key
                artifacts_dir="",
                deadline_ts=float(state.get("deadline_ts") or 0.0),
                lease_token=state.get("lease_token"),
            ),
            RolloutOutput(
                final_text=str(event.get("last_assistant_message") or ""),
                tool_calls=traj.read(),
                # Reported, never swallowed: non-zero forbids the training
                # grade, so a partly unreadable trajectory must not be graded
                # as a complete one.
                tool_calls_omitted_count=traj.malformed,
            ),
        )
        os.unlink(path)
    finally:
        client.close()


def main() -> int:
    try:
        event = json.loads(sys.stdin.read() or "{}")
        if not isinstance(event, dict):
            return 0
    except Exception:
        return 0

    kind = event.get("event") or event.get("hook_event_name") or ""

    # BEFORE the capture check, not after. `_capture_dir` returns None when the
    # customer has set PERCEPTEYE_CAPTURE=0 -- and a customer who has turned
    # capture off is exactly the one whose endpoint most needs handing back.
    # Reconciling below the early return would strand them routed at us with
    # capture disabled: the worst of both, and silent.
    if kind == "SessionStart":
        try:
            _reconcile()
        except Exception:
            pass

    base = _capture_dir(event)
    if not base:
        return 0  # Capture is off, or there is nothing to key this on.

    # ── the two SESSION events, which talk to the control plane ────────────
    # Everything else on this path is local and stdlib-only. These two run once
    # per session and are the whole difference between recording locally and
    # contributing a graded rollout. Both are wrapped: a control plane that is
    # unreachable, or absent entirely, must not fail the customer's turn.
    if kind == "SessionStart":
        try:
            _claim(event, base)
        except Exception:
            pass
        return 0
    if kind in ("Stop", "SessionEnd"):
        try:
            _report(event, base)
        except Exception:
            pass
        # falls through: Stop also records the ANSWER locally

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
