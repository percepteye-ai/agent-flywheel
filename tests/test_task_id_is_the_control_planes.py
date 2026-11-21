"""A ``task_id`` is valid exactly when the control plane's intake takes it, and
an identifier pattern answers the same in Python and in ECMA-262.

THE TASK ID. Mission Control's ``is_canonical_task_id`` requires every
character to be ``str.isprintable()``; the SDK checked only the shared
identifier grammar (no edge whitespace, no C0/C1 control). A task id carrying
U+200B, U+2028, U+00AD, U+FEFF, U+00A0, an unassigned or a private-use code
point passed here and was refused there -- with a 422 for the WHOLE turn batch
it rode in. The SDK now applies the control plane's predicate, omits such an
id at capture (as it always omitted an invalid one), and the transport
refuses one a direct caller passes, an explicit ``None`` included.

THE PATTERNS. ``.`` does not match U+2028/U+2029 in ECMA-262 and does in
Python, and ``\\s`` includes U+FEFF in ECMA-262 and not in Python, so the
published identifier patterns gave different answers in the two languages.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from agent_flywheel.contract import (
    _PRODUCTION_IDENTIFIER_PATTERN,
    _PRODUCTION_TURN_IDENTIFIER_PATTERN,
    _TASK_IDENTIFIER_PATTERN,
    _production_identifier,
    _production_turn_identifier,
    _task_identifier,
)
from agent_flywheel.errors import ContractViolation
from agent_flywheel.outcomes import _normalize_task_id
from agent_flywheel.transport import AttachTransport

from .node_sdk import require_node_sdk, run_node

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json")
    .read_text(encoding="utf-8"))

#: Each a character ``str.isprintable()`` refuses, so the control plane
#: refuses a task id carrying it -- and the shared grammar did not.
NOT_PRINTABLE = {
    "zero width space": "\u200b",
    "line separator": "\u2028",
    "paragraph separator": "\u2029",
    "soft hyphen": "\u00ad",
    "byte order mark": "\ufeff",
    "no-break space": "\u00a0",
    "ideographic space": "\u3000",
    "unassigned U+0378": "\u0378",
    "private use U+E000": "\ue000",
}


class _Client:
    def __init__(self):
        self.calls: list[dict] = []

    def request(self, method, path, **kw):
        self.calls.append({"method": method, "path": path, **kw})
        return {}


def _turn(**extra):
    return {"turn_id": "t1", "input_text": "hi", "final_text": "ok",
            "tool_calls": None, "tool_calls_omitted_count": 0, **extra}


@pytest.mark.parametrize("name", sorted(NOT_PRINTABLE))
def test_a_task_id_the_intake_refuses_is_omitted_at_capture(name):
    task_id = f"case{NOT_PRINTABLE[name]}1"
    assert _production_identifier(task_id) == task_id, (
        "the case is one the shared grammar alone let through")
    assert _task_identifier(task_id) is None
    assert _normalize_task_id(task_id) is None


@pytest.mark.parametrize("name", sorted(NOT_PRINTABLE))
def test_a_task_id_the_intake_refuses_is_never_sent(name):
    c = _Client()
    with pytest.raises(ContractViolation, match="invalid task_id"):
        AttachTransport(c, "agent-1").report_turns(
            "agent-1", [_turn(task_id=f"case{NOT_PRINTABLE[name]}1")])
    assert c.calls == []


def test_an_explicit_null_task_id_is_never_sent():
    """The wire says "no task id" by omitting the key; the intake refuses a
    null, so a batch carrying one was a 422 for every turn in it."""
    c = _Client()
    with pytest.raises(ContractViolation, match="invalid task_id"):
        AttachTransport(c, "agent-1").report_turns(
            "agent-1", [_turn(task_id=None)])
    assert c.calls == []


@pytest.mark.parametrize("task_id", [
    "case-1", "a b", "caf\u00e9 \u2014 #42", "\U0001f4e6 parcel", "x" * 255,
])
def test_a_printable_task_id_is_kept_and_sent(task_id):
    assert _normalize_task_id(task_id) == task_id
    c = _Client()
    AttachTransport(c, "agent-1").report_turns(
        "agent-1", [_turn(task_id=task_id)])
    assert c.calls[0]["json_body"]["turns"][0]["task_id"] == task_id


@pytest.mark.parametrize("task_id", [
    None, "", " lead", "trail ", "x" * 256, 7, "\ud800",
])
def test_the_rules_the_shared_grammar_already_had_still_hold(task_id):
    assert _task_identifier(task_id) is None


def _task_schema():
    return Draft202012Validator(
        SCHEMA["$defs"]["ProductionTurn"]["properties"]["task_id"])


def test_the_schema_publishes_the_task_pattern():
    prop = SCHEMA["$defs"]["ProductionTurn"]["properties"]["task_id"]
    assert prop["pattern"] == _TASK_IDENTIFIER_PATTERN
    assert "isprintable" in prop["description"]


@pytest.mark.parametrize("name", sorted(NOT_PRINTABLE))
def test_the_schema_refuses_each_case(name):
    assert list(_task_schema().iter_errors(f"case{NOT_PRINTABLE[name]}1"))


def test_the_schema_never_refuses_a_task_id_the_sdk_sends():
    """Every BMP character the predicate takes, the pattern takes: the
    pattern may say less than the rule, never something else."""
    pattern = re.compile(_TASK_IDENTIFIER_PATTERN)
    refused = [
        hex(cp) for cp in range(0x10000)
        if not 0xD800 <= cp <= 0xDFFF
        and _task_identifier(f"a{chr(cp)}b") is not None
        and pattern.fullmatch(f"a{chr(cp)}b") is None
    ]
    assert refused == []


#: Ids on which ``.`` / ``\s`` made Python and ECMA-262 disagree, and
#: controls that agree either way.
PATTERN_CASES = [
    "a\u2028 ", "a\u2029 ", "a\u2028/b", "a\u2029\\b", "\ufeffa", "a\ufeff",
    "a\u00a0", "\u3000a", "a b", "a/b", "ok", ".", "..", "a\u2028b",
]


def _python_answers() -> dict[str, dict[str, bool]]:
    patterns = {"identifier": _PRODUCTION_IDENTIFIER_PATTERN,
                "turn": _PRODUCTION_TURN_IDENTIFIER_PATTERN,
                "task": _TASK_IDENTIFIER_PATTERN}
    return {name: {case: re.fullmatch(p, case) is not None
                   for case in PATTERN_CASES}
            for name, p in patterns.items()}


def test_python_refuses_what_ecma_used_to_let_through():
    answers = _python_answers()
    for case in ("a\u2028 ", "a\u2029 ", "\ufeffa", "a\ufeff"):
        assert not answers["identifier"][case], case
        assert _production_identifier(case) is None, case
    assert not answers["turn"]["a\u2028/b"]
    assert _production_turn_identifier("a\u2028/b") is None
    assert answers["identifier"]["a\u2028b"]


def test_ecma_262_answers_every_pattern_as_python_does():
    """The schema's patterns run in V8 with the ``u`` flag (how a JSON
    Schema validator in JavaScript runs them) and must answer each case as
    Python's ``re`` does."""
    sdk, node = require_node_sdk()
    script = (
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const {patterns, cases} = JSON.parse(raw);\n"
        "const out = {};\n"
        "for (const [name, p] of Object.entries(patterns)) {\n"
        "  const re = new RegExp(p, 'u'); out[name] = {};\n"
        "  for (const c of cases) out[name][c] = re.test(c);\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    payload = {"patterns": {"identifier": _PRODUCTION_IDENTIFIER_PATTERN,
                            "turn": _PRODUCTION_TURN_IDENTIFIER_PATTERN,
                            "task": _TASK_IDENTIFIER_PATTERN},
               "cases": PATTERN_CASES}
    ecma = json.loads(run_node(sdk, node, script, stdin=json.dumps(payload)))
    assert ecma == _python_answers()
