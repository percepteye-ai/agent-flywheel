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


# ── one rule in both SDKs ─────────────────────────────────────────────────

#: Bodies an unknown call may carry, each scrubbed by both SDKs. Strings,
#: booleans, nulls and integers only: a float's spelling is the one thing
#: JavaScript's parser loses (``contract._scrub_status`` names it).
SCRUB_TABLE = [
    "got HTTP 404 back",
    "HTTP/1.1 200 OK and http/2 201 and HTTP 3000 and HTTP\t502",
    "HTTP\u0662\u0660\u0660 is not a status",          # Arabic-Indic digits
    "xHTTP 200 and HTTP 200x stay",
    '"HTTP 200 inside a JSON string"',
    "42",
    '{"status": 200, "body": "{\\"Status_Code\\": 201, \\"id\\": \\"PROJ-1\\"}"}',
    '[{"STATUS": "201", "k": "vé\u007f"}, "HTTP/2 204"]',
    '{"a": NaN, "code": 1, "b": -Infinity}',
    '{"broken": ',
    {"status": 200, "statusCode": 201, "ok": "HTTP 200 OK",
     "nested": [{"code": 7, "note": "n"}]},
    {"body": '{"http_status": 500, "msg": "fine ☃"}', "HTTP_STATUS": 1},
    ["HTTP/1.0 200", {"Code": 3, "keep": True, "none": None}],
    7,
]


def test_both_sdks_scrub_status_byte_for_byte():
    """The Node SDK's ``scrubStatus`` and this ``_scrub_status`` return the
    same bytes for every body above. Mutation: either side back to its
    pre-PY-H13 rule (case-sensitive keys, ``HTTP nnn`` only, dict strings
    only regex-scrubbed and stripped) -> red."""
    from agent_flywheel.contract import _scrub_status

    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import { scrubStatus } from './src/wire.js';\n"
        "let input = '';\n"
        "process.stdin.on('data', (c) => { input += c; });\n"
        "process.stdin.on('end', () => {\n"
        "  console.log(JSON.stringify(JSON.parse(input).map(scrubStatus)));\n"
        "});\n"
    )
    node_out = json.loads(run_node(sdk, node, script, stdin=json.dumps(SCRUB_TABLE)))
    python_out = [_scrub_status(body) for body in SCRUB_TABLE]
    for body, theirs, ours in zip(SCRUB_TABLE, node_out, python_out, strict=True):
        assert theirs == ours, f"{body!r}: Node {theirs!r}, Python {ours!r}"
    # Spot checks that the rule is the one stated, not merely shared.
    assert python_out[2] == SCRUB_TABLE[2], "a Unicode digit is no status"
    assert python_out[8] == '{"a": null, "b": null}'
    assert python_out[9] == SCRUB_TABLE[9]


def test_both_sdks_take_the_same_identity_fields():
    """``entity_ids`` keys are refused or lowercased against one set."""
    from agent_flywheel.evidence import IDENTITY_FIELDS

    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import { IDENTITY_FIELDS } from './src/evidence.js';\n"
        "console.log(JSON.stringify([...IDENTITY_FIELDS].sort()));\n"
    )
    assert json.loads(run_node(sdk, node, script)) == sorted(IDENTITY_FIELDS)


#: Ids for the two id rules both SDKs apply: the URL path segment and the
#: task id. Each row is (id, a note); the answers are compared, not listed.
ID_TABLE = [
    "rl_1", "", ".", "..", "a/b", "a\\b", "a\x00b", "a\x7fb", "a\x85b",
    "it's (a)*!?#%", "café", "x" * 256, "x" * 257,
    "\U0001F600" * 129, "\U0001F600" * 256, "\U0001F600" * 257,
    " lead", "trail ", "\tlead", "a\u200bb", "a\u2028b", "a\u00a0b",
    "a\u00adb", "\ufeffa", "a\ue000b", "\u0378", "a b", "x" * 255,
]


def test_both_sdks_apply_one_path_segment_rule_and_one_task_id_rule():
    """``_path_segment`` / ``urlSegment`` and ``_task_identifier`` /
    ``productionTaskIdentifier`` give the same answer for every id above:
    the same refusals, the same percent-encoding. Mutation: Node's segment
    length back to UTF-16 units -> the 129-emoji row differs, red."""
    from agent_flywheel.contract import _path_segment, _task_identifier

    from .node_sdk import require_node_sdk, run_node

    def python(value):
        try:
            segment = _path_segment(value, "id")
        except ContractViolation:
            segment = None
        return [segment, _task_identifier(value)]

    sdk, node = require_node_sdk()
    script = (
        "import { urlSegment } from './src/transport.js';\n"
        "import { productionTaskIdentifier } from './src/wire.js';\n"
        "let input = '';\n"
        "process.stdin.on('data', (c) => { input += c; });\n"
        "process.stdin.on('end', () => {\n"
        "  console.log(JSON.stringify(JSON.parse(input).map((v) => {\n"
        "    let segment = null;\n"
        "    try { segment = urlSegment(v, 'id'); } catch { segment = null; }\n"
        "    return [segment, productionTaskIdentifier(v)];\n"
        "  })));\n"
        "});\n"
    )
    theirs = json.loads(run_node(sdk, node, script, stdin=json.dumps(ID_TABLE)))
    for value, node_answer in zip(ID_TABLE, theirs, strict=True):
        assert node_answer == python(value), (
            f"{value!r}: Node {node_answer!r}, Python {python(value)!r}")
