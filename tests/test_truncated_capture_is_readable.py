"""A turn cut mid-character by a SIGKILL costs its last record, not the turn.

The readers decoded strictly, so a write cut inside a multi-byte character
raised ``UnicodeDecodeError``: the turn failed to render on every flush,
forever. A ``turns.jsonl`` line that parsed to something other than an object
raised too. And a capture warning raised out of a customer's tool call under
``-W error``.
"""
from __future__ import annotations

import json
import warnings

from agent_flywheel import outcomes
from agent_flywheel.genai_otlp import OBSERVATION_FILENAME
from agent_flywheel.outcomes import (
    TOOL_CALLS_FILENAME,
    TURNS_FILENAME,
    Trajectory,
    read_model_boundary,
    read_turn_text,
)

CUT = "café".encode()[:-1]          # the é cut in half


def test_a_turn_text_cut_mid_character_keeps_what_was_whole(tmp_path):
    (tmp_path / TURNS_FILENAME).write_bytes(
        json.dumps({"kind": "task", "text": "refund"}).encode() + b"\n"
        + b"7\n" + b'{"kind": "answer", "text": "' + CUT)
    assert read_turn_text(tmp_path) == {"task": "refund"}


def test_tool_calls_cut_mid_character_keep_what_was_whole(tmp_path):
    whole = json.dumps({"name": "lookup", "arguments": {}, "outcome": "ok",
                        "status_code": 200})
    (tmp_path / TOOL_CALLS_FILENAME).write_bytes(
        whole.encode() + b"\n" + b'{"name": "' + CUT)
    traj = Trajectory(tmp_path)
    assert [c.name for c in traj.read()] == ["lookup"]
    assert traj.malformed == 1


def test_a_model_boundary_cut_mid_character_does_not_raise(tmp_path):
    (tmp_path / OBSERVATION_FILENAME).write_bytes(b'{"record": "x' + CUT)
    read_model_boundary(tmp_path)


def test_a_capture_warning_never_raises_into_the_tool_call(
        tmp_path, monkeypatch):
    blocked = tmp_path / "file"
    blocked.write_text("not a directory")
    sink = outcomes._Sink(blocked / "sub" / TOOL_CALLS_FILENAME)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert sink.write("{}") is False
    monkeypatch.setattr(outcomes, "_warned_no_destination", False)
    monkeypatch.delenv(outcomes.TRAJECTORY_DIR_ENV, raising=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert outcomes._trajectory_path("r1") is None
