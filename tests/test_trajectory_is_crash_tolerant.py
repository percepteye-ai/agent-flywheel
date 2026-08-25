"""A truncated write costs that record and nothing else.

The trajectory is append-only and flushed per call, so a rollout killed on its
deadline leaves the file with a partial final line. Before this guard,
``Trajectory.read`` raised ``JSONDecodeError`` out of ``_attach_trajectory``,
out of ``run_one``, and a COMPLETED rollout was reported as ``flywheel_crash``
with every one of its tool calls lost — including the forty that parsed fine.

The asymmetry that gave it away: ``_attach_stream_tool_calls`` is wrapped in a
bare ``except`` with the comment "never fail a rollout over telemetry", and the
trajectory path next to it was not.

What replaces the raise is NOT silence. An unreadable line is a call we know
executed and cannot report, so it lands in ``tool_calls_omitted_count``, whose
wire meaning is already "executed calls deliberately not reported; non-zero
forbids training grade". Dropping it quietly would be the truncated-list
failure the contract forbids agents from committing by hand.
"""
from __future__ import annotations

import json

from percepteye_agent_flywheel.contract import RolloutOutput
from percepteye_agent_flywheel.outcomes import Trajectory
from percepteye_agent_flywheel.runner import _attach_trajectory

OK = {"name": "read_file", "arguments": {"p": "/etc/hosts"},
      "outcome": "ok", "status_code": 200}


def _write(tmp_path, *lines: str):
    (tmp_path / "tool_calls.jsonl").write_text("".join(lines), encoding="utf-8")
    return tmp_path


def test_a_truncated_final_line_does_not_cost_the_whole_rollout(tmp_path):
    """The SIGKILL case, verbatim: two clean records then a half-written one."""
    _write(tmp_path,
           json.dumps(OK) + "\n",
           json.dumps({**OK, "name": "write_file"}) + "\n",
           '{"name": "deploy", "arguments": {}, "outc')      # killed mid-write

    calls = Trajectory(tmp_path).read()

    assert calls is not None
    assert [c.name for c in calls] == ["read_file", "write_file"]


def test_the_lost_record_is_counted_not_hidden(tmp_path):
    _write(tmp_path, json.dumps(OK) + "\n", '{"name": "deploy", "argum')
    t = Trajectory(tmp_path)
    t.read()
    assert t.malformed == 1


def test_the_count_reaches_the_wire_as_an_omission(tmp_path):
    """Non-zero forbids the training grade, which is the correct consequence:
    the list is genuinely incomplete and must not be scored as complete."""
    _write(tmp_path, json.dumps(OK) + "\n", "{oh no", "\n")
    out = RolloutOutput(final_text="done")

    _attach_trajectory(out, str(tmp_path))

    assert [c.name for c in out.tool_calls or []] == ["read_file"]
    assert out.tool_calls_omitted_count == 1
    assert out.to_wire()["tool_calls_omitted_count"] == 1


def test_a_record_the_contract_refuses_is_skipped_not_raised(tmp_path):
    """ContractViolation is not a ValueError, so it needs catching by name.
    An outcome outside the tri-state is exactly the shape the contract exists
    to reject — rejecting it must not also destroy the records beside it."""
    _write(tmp_path,
           json.dumps(OK) + "\n",
           json.dumps({**OK, "name": "bad", "outcome": "probably_fine"}) + "\n",
           json.dumps({**OK, "name": "after"}) + "\n")

    t = Trajectory(tmp_path)
    calls = t.read()

    assert [c.name for c in calls or []] == ["read_file", "after"]
    assert t.malformed == 1


def test_a_line_missing_a_required_key_is_skipped(tmp_path):
    _write(tmp_path, json.dumps({"arguments": {}, "outcome": "ok"}) + "\n",
           json.dumps(OK) + "\n")
    t = Trajectory(tmp_path)
    assert [c.name for c in t.read() or []] == ["read_file"]
    assert t.malformed == 1


def test_a_line_that_is_not_an_object_is_skipped(tmp_path):
    """Valid JSON, wrong shape — `d["name"]` on a list raises TypeError."""
    _write(tmp_path, "[1, 2, 3]\n", json.dumps(OK) + "\n")
    t = Trajectory(tmp_path)
    assert [c.name for c in t.read() or []] == ["read_file"]
    assert t.malformed == 1


def test_an_entirely_unreadable_file_reads_as_empty_with_the_count(tmp_path):
    """NOT None. None means "the agent never reported"; this file exists, so
    something WAS reported and we could not read it. `[]` plus a non-zero
    omission count says exactly that, and forbids the training grade."""
    _write(tmp_path, "{broken\n", "{also broken\n")
    t = Trajectory(tmp_path)
    calls = t.read()
    assert calls == []
    assert t.malformed == 2


def test_no_file_at_all_still_reads_as_none(tmp_path):
    """The absence≠zero rule the guard must not disturb."""
    t = Trajectory(tmp_path)
    assert t.read() is None
    assert t.malformed == 0


def test_a_clean_file_reports_no_omissions(tmp_path):
    _write(tmp_path, json.dumps(OK) + "\n", json.dumps({**OK, "name": "b"}) + "\n")
    out = RolloutOutput(final_text="done")
    _attach_trajectory(out, str(tmp_path))
    assert out.tool_calls_omitted_count == 0


def test_the_count_resets_between_reads(tmp_path):
    """A reader reused across rollouts must not accumulate someone else's."""
    _write(tmp_path, "{broken\n")
    t = Trajectory(tmp_path)
    t.read()
    assert t.malformed == 1
    _write(tmp_path, json.dumps(OK) + "\n")
    t.read()
    assert t.malformed == 0
