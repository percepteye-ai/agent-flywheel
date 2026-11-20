"""An ``unknown`` call carries no recoverable status, and a tool call and a
reward leave in the shape the schema publishes.

THE SCRUB. A string VALUE inside a dict was only searched for ``HTTP 200``, so
``{"body": "{\\"status_code\\": 200}"}`` kept its status; ``HTTP/1.1 200`` and
``HTTP/2 200`` were not matched; and ``Status`` / ``STATUS_CODE`` were not
status keys. A scorer could recover each and read an unobserved call as a
landed one.

THE WIRE. ``entity_ids`` was checked case-insensitively and sent as spelled,
so ``{"ID": ...}`` passed here and failed the schema's lowercase key enum; a
``reward`` of ``True``, NaN or infinity went out as sent.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from agent_flywheel.contract import RolloutOutput, ToolCallOutcome
from agent_flywheel.errors import ContractViolation


def _unknown(output):
    return ToolCallOutcome(name="POST /issue", arguments={}, outcome="unknown",
                           status_code=None, output=output)


def test_a_json_body_inside_a_dict_value_is_scrubbed():
    tc = _unknown({"body": json.dumps({"status_code": 200, "id": "PROJ-1"})})
    body = json.loads(tc.output["body"])
    assert body == {"id": "PROJ-1"}


def test_a_json_body_inside_a_list_inside_a_dict_is_scrubbed():
    tc = _unknown({"pages": [json.dumps([{"status": "201", "k": "v"}])]})
    assert json.loads(tc.output["pages"][0]) == [{"k": "v"}]


@pytest.mark.parametrize("line", [
    "HTTP/1.1 200 OK", "HTTP/2 201 Created", "http/1.0 204", "HTTP 200 OK",
])
def test_every_status_line_spelling_is_scrubbed(line):
    for tc in (_unknown(f"returned {line}"),
               _unknown({"raw": f"returned {line}"})):
        text = tc.output if isinstance(tc.output, str) else tc.output["raw"]
        assert not any(code in text for code in ("200", "201", "204")), text
        assert "returned" in text


@pytest.mark.parametrize("key", [
    "Status", "STATUS_CODE", "statusCode", "StatusCode", "HTTP_Status", "Code",
])
def test_status_keys_are_found_whatever_their_case(key):
    tc = _unknown({key: 200, "name": "ok"})
    assert tc.output == {"name": "ok"}


def test_a_scalar_json_string_is_text_not_a_body():
    assert _unknown("200").output == "200"
    assert _unknown('"HTTP 200 done"').output == '" done"'


def test_entity_ids_leave_under_the_schemas_lowercase_names():
    tc = ToolCallOutcome(name="POST /issue", arguments={}, outcome="ok",
                         status_code=201, entity_ids={"ID": "1", " Key ": "P-1"})
    wire = tc.to_wire()
    assert wire["entity_ids"] == {"id": "1", "key": "P-1"}
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json")
        .read_text(encoding="utf-8"))
    Draft202012Validator({
        "$schema": schema["$schema"], "$defs": schema["$defs"],
        "$ref": "#/$defs/ToolCallRecord",
    }).validate(wire)


def test_two_spellings_of_one_identity_field_are_refused():
    with pytest.raises(ContractViolation, match="'id' twice"):
        ToolCallOutcome(name="POST /issue", arguments={}, outcome="ok",
                        status_code=201, entity_ids={"id": "1", "ID": "2"})


@pytest.mark.parametrize("reward", [True, False, math.nan, math.inf,
                                    -math.inf, "1", 10**400])
def test_a_reward_the_wire_cannot_carry_is_refused(reward):
    with pytest.raises(ContractViolation, match="reward"):
        RolloutOutput(final_text="", reward=reward).to_wire()


@pytest.mark.parametrize("reward", [0, 1, 0.5, -2.0])
def test_a_finite_reward_is_sent_as_given(reward):
    assert RolloutOutput(final_text="", reward=reward).to_wire()["reward"] == reward
