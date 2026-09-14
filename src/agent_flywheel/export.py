"""Render captured production turns into the shape the control plane ingests.

AN OFFLINE PROJECTION, AND NOT THE UPLOAD PATH. Read that first, because the
two are easy to confuse and only one of them ships anything anywhere.

``POST /turns`` sends :meth:`Attachment._turn_wire`, which is TURN-NATIVE: the
user's message, the answer and the calls exactly as they were observed, with
the projection into any training-row format happening on the far side, where it
can change without a customer redeploying. Nothing in this package imports this
module, and none of its names is re-exported from ``agent_flywheel``
-- so an ``attach()`` that is uploading turns is not running a line of it.

This module is for the OTHER caller: somebody who wants their captured turns as
a LOCAL CORPUS, without a control plane in the loop at all. It is reached by
importing it, which is supported and is the whole interface::

    from agent_flywheel.export import write_records

    write_records("/var/lib/percepteye", "turns.jsonl")

TWO RENDERINGS OF ONE TURN, DELIBERATELY. They answer different questions --
this one is corpus-shaped and stamps ``step_index`` so a conversation's calls
attach to the right assistant turn; ``_turn_wire`` is turn-native and has no
step concept because the far side does that chunking. What they must never
disagree on is the EVIDENCE: the same turn id, the same text, the same calls,
and above all the same tri-state, which neither may round ``unknown`` up into a
success. ``tests/test_export_matches_the_upload_path.py`` is what holds them to
that; without it the drift would be silent, because no shipped code path runs
both.

WHY THE PIPELINE ACCEPTS THIS SHAPE. The control plane has a post-hoc pipeline
that ingests recorded trajectories, compiles a reward suite against them, scores
them WITHOUT an environment to re-execute against, and emits training rows. The
record built here is what that pipeline reads, which is why a corpus produced
offline is worth anything at all.

WHY NOTHING NEW IS INVENTED HERE. The ingest side already accepts a "unified
messages log": one record per rollout carrying an OpenAI-style ``messages`` list
and an optional ``tool_calls`` list, which it chunks into per-step transitions
keyed by assistant turns. A production turn IS that shape. So this module is a
projection, not a format -- and deliberately so, because a second format would
be a second thing to keep in sync with a parser we do not own.

``messages`` is OpenAI-style rather than ours for the same reason: it is the
interoperable shape, not a private one, so a customer can feed these records to
anything that reads chat logs and we have not locked them in.

WHAT IS DELIBERATELY ABSENT. No reward and no success. A production turn has no
verdict -- nobody graded it -- and inventing one here would be the exact defect
the emitter's ``success is None`` rule exists to prevent: an ungraded turn
cloned as a demonstration of good work. Grading happens on the control-plane
side, against a compiled suite, and this module must not pre-empt it.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .outcomes import (
    Trajectory,
    _read_execution_fingerprint,
    read_conversation_id,
    read_turn_text,
)

#: Every public name here, including ``conversation_to_record`` -- it is the
#: one that carries a turn's observed next state, and leaving it out made the
#: headline capability of this module invisible to ``from ... import *`` and
#: to every tool that reads ``__all__`` as the supported surface.
__all__ = [
    "conversation_to_record",
    "iter_records",
    "turn_to_record",
    "write_records",
]


def turn_to_record(
    turn_dir: str | os.PathLike[str],
    *,
    turn_id: str | None = None,
) -> dict[str, Any] | None:
    """One captured turn as an ingestible record, or ``None``.

    ``None`` means the turn recorded nothing that can be trained on. A turn with
    tool calls but NO text is deliberately included -- its tool sequence is real
    evidence and several verifier kinds score exactly that -- while a turn with
    neither is not a record of anything.
    """
    path = Path(turn_dir)
    text = read_turn_text(path) or {}
    traj = Trajectory(path)
    calls = traj.read()

    messages: list[dict[str, Any]] = []
    if text.get("task"):
        messages.append({"role": "user", "content": text["task"]})
    if text.get("answer"):
        messages.append({"role": "assistant", "content": text["answer"]})
    if not messages and not calls:
        return None

    record: dict[str, Any] = {
        "messages": messages,
        # The turn id is the join key for anything that arrives later, and it
        # is what the ingest side's identity ladder prefers over hashing the
        # first user message -- so two differently-worded turns stay distinct
        # rather than colliding on a content hash.
        "rollout_id": turn_id or path.name,
        "task_id": turn_id or path.name,
    }
    if calls:
        record["tool_calls"] = [_tool_call(c) for c in calls]
    # REPORTED, NOT SWALLOWED. A partly unreadable trajectory must not be
    # graded as a complete one, and the count is the only thing that can say so.
    if traj.malformed:
        record["tool_calls_omitted_count"] = traj.malformed
    conv = _conversation_id(path)
    if conv:
        record["conversation_id"] = conv
    fingerprint = _read_execution_fingerprint(path)
    if fingerprint is not None:
        record["agent_fingerprint"] = fingerprint
    return record


def _tool_call(call: Any) -> dict[str, Any]:
    """One recorded outcome as an OpenAI-ish tool call.

    THE TRI-STATE SURVIVES, under our own key. The chat-log shape has no way to
    say "this call's result was never observed", and collapsing ``unknown`` into
    a success or a failure is the single most damaging thing this projection
    could do -- every downstream verifier that scores tool outcomes would be
    scoring a verdict nobody reached.
    """
    out: dict[str, Any] = {
        "name": call.name,
        "arguments": call.arguments,
        # Verbatim, so nothing downstream has to invert the mapping below.
        "percepteye_outcome": call.outcome,
    }
    # AND under the key the ingest side actually reads. It maps this through
    # `_maybe_bool` onto a `bool | None` field whose three states are exactly
    # ours -- and its own comment names this path: "an external trajectory that
    # does not state an outcome is unobserved ... this is the ingest path a
    # Flywheel capture would arrive on."
    #
    # `unknown` is therefore OMITTED rather than sent as False. Absent means
    # unobserved, which is the truth; False would assert a failure nobody saw.
    if call.outcome == "ok":
        out["success"] = True
    elif call.outcome == "failed":
        out["success"] = False
    for src, dst in (
        ("status_code", "status_code"), ("output", "output"),
        ("error", "error"), ("tool_call_id", "id"),
        ("latency_ms", "latency_ms"),
    ):
        value = getattr(call, src, None)
        if value is not None:
            out[dst] = value
    return out


#: Delegated to the canonical reader in ``outcomes``, which the upload path
#: also uses. Kept as a module-local name so the call sites below read the
#: same as before.
_conversation_id = read_conversation_id


def conversation_to_record(
    turn_dirs: list[Path],
    *,
    conversation_id: str | None = None,
) -> dict[str, Any] | None:
    """Several turns of ONE conversation as a single record.

    WHY A CONVERSATION AND NOT A TURN, for anything beyond plain cloning: the
    ingest side chunks a messages list into one transition per assistant turn,
    and the messages that follow an assistant turn are that turn's OBSERVED
    NEXT STATE. A lone turn exported by itself has nothing after it, so it can
    never carry one -- and that observation is the only privileged reference
    obtainable without a verifier, a passing sibling or a grader.

    So a conversation of N turns yields N-1 references and one final turn with
    none, which is the honest shape: the last thing an agent said has not been
    responded to yet.

    Turn order is the caller's, not inferred. Guessing which turn preceded
    which is the same class of guess this SDK refuses everywhere else, and
    getting it wrong would attach one turn's outcome to another's answer.
    """
    messages: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    omitted = 0
    first_id: str | None = None
    fingerprints: list[dict[str, str] | None] = []
    # WHICH ASSISTANT TURN A CALL BELONGS TO. The ingest side chunks the
    # messages list into one transition per assistant message, and matches the
    # top-level tool_calls list by ``step_index`` -- treating a MISSING
    # ``step_index`` as "belongs to every step", because the single-turn shape
    # has exactly one step and omitting it there is correct.
    #
    # So a multi-turn record MUST stamp it. Emitting none made every call match
    # every turn: MEASURED on a 2-turn, 2-call conversation, 4 attributions,
    # turn 0 carrying turn 1's failure and turn 1 carrying turn 0's success.
    # Both rows then score against outcomes their own turn never produced,
    # which is the one thing this docstring says the caller's ordering exists
    # to prevent.
    assistant_index = 0
    for d in turn_dirs:
        text = read_turn_text(d) or {}
        traj = Trajectory(d)
        calls = traj.read()
        if text or calls:
            fingerprints.append(_read_execution_fingerprint(d))
        omitted += traj.malformed
        if text.get("task"):
            messages.append({"role": "user", "content": text["task"]})
        answered = bool(text.get("answer"))
        if answered:
            messages.append({"role": "assistant", "content": text["answer"]})
        if calls:
            if answered:
                for c in calls:
                    tool_calls.append(
                        {**_tool_call(c), "step_index": assistant_index})
            else:
                # This turn made calls and recorded no answer, so it produced no
                # assistant message and there is no step for these calls to
                # belong to. They are COUNTED, not emitted: attaching them to a
                # neighbouring turn's answer is precisely the misattribution
                # above, and emitting them unindexed re-creates the match-every-
                # step bug for exactly these rows. ``record_answer`` is
                # documented as callable after the scope closes, so an
                # answerless turn is an ordinary state, not a corruption.
                omitted += len(calls)
        if answered:
            assistant_index += 1
        first_id = first_id or d.name
    if not messages and not tool_calls:
        return None

    cid = conversation_id or (_conversation_id(turn_dirs[0]) if turn_dirs else None)
    record: dict[str, Any] = {
        "messages": messages,
        "rollout_id": cid or first_id,
        "task_id": cid or first_id,
    }
    if tool_calls:
        record["tool_calls"] = tool_calls
    if omitted:
        # Two causes, one honest count: calls this SDK could not parse, and
        # calls whose own turn recorded no answer and which therefore have no
        # assistant turn to attach to. Both mean the same thing to a grader --
        # this record does not carry every call the capture saw -- and that is
        # what stops a partly-carried trajectory being scored as a complete one.
        record["tool_calls_omitted_count"] = omitted
    if cid:
        record["conversation_id"] = cid
    if (
        fingerprints
        and fingerprints[0] is not None
        and all(value == fingerprints[0] for value in fingerprints)
    ):
        record["agent_fingerprint"] = fingerprints[0]
    return record


def iter_records(
    trajectory_root: str | os.PathLike[str],
    *,
    group_conversations: bool = True,
) -> Iterator[dict[str, Any]]:
    """Every capturable turn under a trajectory root, oldest first.

    Ordered by modification time so a conversation's turns arrive in the order
    they happened -- the ingest side chunks by assistant turn, and an
    out-of-order stream would attach one turn's outcome to another's answer.

    ``group_conversations`` (the default) emits ONE record per
    ``conversation_id``, which is what lets each non-final turn carry the
    observed next state that follows it. Turns with no conversation id are
    emitted individually, because a turn that never declared which conversation
    it belonged to cannot be placed in one -- and guessing is how two users'
    turns end up in a single record.
    """
    root = Path(trajectory_root)
    try:
        dirs = [d for d in root.iterdir() if d.is_dir()]
    except OSError:
        return
    ordered = sorted(dirs, key=lambda p: _mtime(p))

    if not group_conversations:
        for d in ordered:
            record = turn_to_record(d, turn_id=d.name)
            if record is not None:
                yield record
        return

    grouped: dict[str, list[Path]] = {}
    for d in ordered:
        cid = _conversation_id(d)
        if cid is None:
            record = turn_to_record(d, turn_id=d.name)
            if record is not None:
                yield record
            continue
        grouped.setdefault(cid, []).append(d)
    for cid, turn_dirs in grouped.items():
        record = conversation_to_record(turn_dirs, conversation_id=cid)
        if record is not None:
            yield record


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def write_records(
    trajectory_root: str | os.PathLike[str],
    out_path: str | os.PathLike[str],
) -> int:
    """Write every turn as JSONL. Returns how many records were written.

    JSONL because that is what the ingest side's loader reads, and because one
    record per line means a truncated write costs that record and nothing else --
    the same property the trajectory files themselves keep.
    """
    written = 0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for record in iter_records(trajectory_root):
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written
