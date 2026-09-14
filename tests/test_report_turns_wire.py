"""The turn upload's actual wire call, which nothing tested.

WHY THIS FILE EXISTS. Every production-turn test in this package monkeypatches
``agent_flywheel.transport.AttachTransport`` with a duck-typed fake
that has its own ``report_turns``. That is the right shape for testing the
Attachment -- but it means the REAL ``report_turns`` was never executed by
anything: not the route it posts to, not the body it builds, not the
idempotency key it derives. A method every captured turn passes through, with
zero coverage, in a package whose whole job is getting that data across.

The route and body are also half of a cross-SDK contract. The Node package
sends the same batch to the same route and derives the same idempotency key,
and ``test_both_sdks_derive_the_same_idempotency_key`` below proves the two
derivations AGREE rather than asserting it in a comment.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from agent_flywheel.contract import (
    _PRODUCTION_AGENT_IDENTIFIER_PATTERN,
    _PRODUCTION_IDENTIFIER_MAX_CHARS,
    _PRODUCTION_IDENTIFIER_PATTERN,
    _PRODUCTION_TURN_BATCH_MAX_ITEMS,
    _PRODUCTION_TURN_IDENTIFIER_PATTERN,
)
from agent_flywheel.errors import ContractViolation
from agent_flywheel.transport import AttachTransport

from .node_sdk import ABI_CHECK, node_sdk, require_node_sdk, run_node


class _Client:
    """Records requests instead of making them."""

    def __init__(self, reply=None):
        self.calls: list[dict] = []
        self._reply = reply or {}

    def request(self, method, path, **kw):
        self.calls.append({"method": method, "path": path, **kw})
        return self._reply


#: DELIBERATELY AWKWARD. Non-ASCII text, a nested object, an explicit null and
#: an empty list -- the four shapes where `JSON.stringify` and `json.dumps`
#: diverge. A fixture of plain ASCII scalars would have agreed by accident and
#: certified a parity that did not exist.
TURNS = [
    {"turn_id": "t1", "input_text": "\u00bfd\u00f3nde est\u00e1 mi pedido?",
     "task_id": "case-1",
     "occurred_at": "2026-09-10T17:30:00Z",
     "final_text": "Ships ma\u00f1ana \u2014 gracias! \U0001f4e6",
     "tool_calls": None, "tool_calls_omitted_count": 0},
    {"turn_id": "t2", "input_text": "again", "final_text": None,
     "tool_calls": [], "tool_calls_omitted_count": 0},
    {"turn_id": "t3", "input_text": "nested",
     "final_text": "ok", "tool_calls": [
         {"name": "lookup", "arguments": {"z": 1, "a": {"deep": "\u00e9"}},
          "outcome": "unknown", "status_code": None}],
     "tool_calls_omitted_count": 2},
]


def _sent(reply=None):
    c = _Client(reply)
    AttachTransport(c, "agent-1").report_turns("agent-1", TURNS)
    return c.calls[0]


def test_it_posts_to_the_route_the_intake_serves():
    call = _sent()
    assert call["method"] == "POST"
    assert call["path"] == "/turns"


def test_the_body_is_what_the_intake_model_declares():
    """``TurnBatch`` is ``{agent_id, turns}`` and nothing else."""
    body = _sent()["json_body"]
    assert set(body) == {"agent_id", "turns"}
    assert body["agent_id"] == "agent-1"
    assert body["turns"] == TURNS


def test_the_published_schema_declares_the_exact_turn_upload_shape():
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json")
        .read_text(encoding="utf-8")
    )
    turn = schema["$defs"]["ProductionTurn"]
    batch = schema["$defs"]["ProductionTurnBatch"]
    assert turn["additionalProperties"] is False
    assert turn["properties"]["task_id"]["maxLength"] == (
        _PRODUCTION_IDENTIFIER_MAX_CHARS
    )
    assert turn["properties"]["turn_id"]["maxLength"] == (
        _PRODUCTION_IDENTIFIER_MAX_CHARS
    )
    assert turn["properties"]["conversation_id"]["maxLength"] == (
        _PRODUCTION_IDENTIFIER_MAX_CHARS
    )
    assert turn["properties"]["turn_id"]["pattern"] == (
        _PRODUCTION_TURN_IDENTIFIER_PATTERN
    )
    assert turn["properties"]["task_id"]["pattern"] == (
        _PRODUCTION_IDENTIFIER_PATTERN
    )
    assert turn["properties"]["conversation_id"]["pattern"] == (
        _PRODUCTION_IDENTIFIER_PATTERN
    )
    assert batch["properties"]["agent_id"]["pattern"] == (
        _PRODUCTION_AGENT_IDENTIFIER_PATTERN
    )
    assert "task_id" not in turn["required"]
    assert batch["properties"]["turns"]["items"]["$ref"] == (
        "#/$defs/ProductionTurn"
    )
    assert batch["properties"]["turns"]["maxItems"] == (
        _PRODUCTION_TURN_BATCH_MAX_ITEMS
    )
    closed_batch = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/ProductionTurnBatch",
    }
    Draft202012Validator(
        closed_batch, format_checker=FormatChecker(),
    ).validate(_sent()["json_body"])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_id", " padded"),
        ("task_id", "padded "),
        ("task_id", "line\nbreak"),
        ("task_id", "tab\tinside"),
        ("task_id", "x" * 256),
        ("turn_id", "../escape"),
        ("turn_id", r"..\escape"),
        ("turn_id", "."),
        ("turn_id", ".."),
        ("turn_id", " padded"),
        ("conversation_id", " padded"),
        ("conversation_id", "line\nbreak"),
        ("conversation_id", "x" * 256),
    ],
)
def test_the_published_schema_rejects_identifiers_the_sdk_refuses_or_omits(
    field, value,
):
    schema = json.loads(
        (Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json")
        .read_text(encoding="utf-8")
    )
    body = {
        **TURNS[0],
        field: value,
    }
    validator = Draft202012Validator({
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/ProductionTurn",
    })
    assert list(validator.iter_errors(body))


def test_the_batch_carries_an_idempotency_key():
    """A network-level replay of the same bytes must collapse BEFORE the
    per-turn rule is reached."""
    key = _sent()["idempotency_key"]
    assert re.fullmatch(r"turns-agent-1-[0-9a-f]{32}", key), key


@pytest.mark.parametrize(
    ("agent_id", "turns", "problem"),
    [
        ("a" * 256, TURNS, "agent_id"),
        ("customer/team", TURNS, "agent_id"),
        ("..", TURNS, "agent_id"),
        ("agent-1", [{**TURNS[0], "turn_id": "t" * 256}], "turn_id"),
        ("agent-1", [{**TURNS[0], "turn_id": "../escape"}], "turn_id"),
        ("agent-1", [{**TURNS[0], "turn_id": r"..\escape"}], "turn_id"),
        ("agent-1", [{**TURNS[0], "turn_id": "."}], "turn_id"),
        ("agent-1", [{**TURNS[0], "turn_id": " padded"}], "turn_id"),
        (
            "agent-1",
            [{**TURNS[0], "conversation_id": "c" * 256}],
            "conversation_id",
        ),
        (
            "agent-1",
            [{**TURNS[0], "conversation_id": "line\nbreak"}],
            "conversation_id",
        ),
        ("agent-1", [TURNS[0]] * 65, "between 1 and 64"),
    ],
)
def test_transport_refuses_identifier_or_batch_shapes_its_schema_forbids(
    agent_id, turns, problem,
):
    client = _Client()
    with pytest.raises(ContractViolation, match=problem):
        AttachTransport(client, agent_id).report_turns(agent_id, turns)
    assert client.calls == []


def test_opaque_agent_ids_are_escaped_at_url_and_header_transport_seams():
    agent_id = "customer team-über"
    client = _Client()
    transport = AttachTransport(client, agent_id)

    transport.capture_current(agent_id)
    transport.report_turns(agent_id, TURNS)

    assert client.calls[0]["path"] == (
        "/agents/customer%20team-%C3%BCber/capture"
    )
    assert client.calls[1]["idempotency_key"].startswith(
        "turns-customer%20team-%C3%BCber-"
    )


def test_the_key_is_stable_for_the_same_turns_and_differs_otherwise():
    same = _sent()["idempotency_key"]
    assert _sent()["idempotency_key"] == same

    c = _Client()
    AttachTransport(c, "agent-1").report_turns(
        "agent-1", [{**TURNS[0], "final_text": "a different answer"}])
    assert c.calls[0]["idempotency_key"] != same


def test_ambiguous_retries_stay_enabled():
    """The OPPOSITE of ``claim``, deliberately.

    The server keys each turn on (org, agent, turn_id) and answers an identical
    resubmission as a duplicate, so a replay costs nothing. A replayed CLAIM
    would lease a second piece of work, which is why that one -- and only that
    one -- turns retries off.
    """
    assert _sent().get("retry_if_ambiguous") is not False


# ── the cross-SDK half ────────────────────────────────────────────────────

#: WHERE THE NODE SDK IS, and whether it may be skipped, live in
#: ``tests/node_sdk.py`` -- one definition, now that two suites drive the Node
#: package. That module asks ``scripts/cross_language_abi_check.sh``, which is
#: THE resolver, because the script needs the rule too.


def test_the_node_sdk_is_resolved_where_it_was_NAMED(tmp_path, monkeypatch):
    """The regression that made the parity test below a no-op in CI.

    The resolver has to find an SDK that is neither a fixed sibling name nor
    two directories up -- which describes every CI checkout. Asserted against
    a directory that is in neither place.
    """
    fake = tmp_path / "agent-flywheel-node"
    (fake / "src").mkdir(parents=True)
    monkeypatch.setenv("PERCEPTEYE_NODE_SDK", str(fake))
    assert node_sdk() == (fake, True)


def test_a_NAMED_node_sdk_that_is_not_there_fails_the_check(tmp_path):
    """The ABI check's skip is for a developer who has only this repo. Once
    somebody names a path, an absent SDK is a broken environment: exiting 0
    would turn the only cross-language contract check into a green no-op the
    day the Node repo moves its modules or CI lands the checkout elsewhere.
    """
    missing = str(tmp_path / "not-checked-out")
    res = subprocess.run(
        [str(ABI_CHECK)], capture_output=True, text=True,
        env={**os.environ, "PERCEPTEYE_NODE_SDK": missing})
    both = res.stdout + res.stderr
    assert res.returncode == 1, f"exited {res.returncode}:\n{both}"
    assert missing in both, both
    assert "SKIPPED" not in both, both


def test_both_sdks_derive_the_same_idempotency_key():
    """A customer runs one package or the other, and may MIGRATE between them
    with turns still on disk. If the two hash the same batch differently, each
    defeats the other's batch de-duplication at exactly that moment.

    Python hashes ``json.dumps(sort_keys=True)``; JavaScript's ``JSON.stringify``
    preserves insertion order instead, so the Node side sorts keys at every
    depth to match. This asserts the two agree on a real payload rather than
    trusting that they do.
    """
    sdk, node = require_node_sdk()

    script = (
        "import {AttachTransport} from './src/transport.js';\n"
        "const calls=[];\n"
        "const tx=new AttachTransport({request:(m,p,o)=>{calls.push(o);}},"
        "{agentId:'agent-1'});\n"
        f"tx.reportTurns('agent-1', {json.dumps(TURNS)});\n"
        "console.log(calls[0].idempotencyKey);\n"
    )
    node_key = run_node(sdk, node, script).strip()
    assert node_key == _sent()["idempotency_key"], (
        f"the two SDKs derive different idempotency keys for one batch:\n"
        f"  python: {_sent()['idempotency_key']}\n"
        f"  node  : {node_key}"
    )


# ── the verdict read ──────────────────────────────────────────────────────


def test_the_verdict_is_a_GET_on_the_agent():
    c = _Client({"agent_id": "agent-1", "capture": {"turns": True}})
    out = AttachTransport(c, "agent-1").capture_current("agent-1")
    assert c.calls[0]["method"] == "GET"
    assert c.calls[0]["path"] == "/agents/agent-1/capture"
    assert out["capture"]["turns"] is True


def test_the_verdict_read_carries_no_idempotency_key_and_no_body():
    """It is a READ. Sending a body or a key would make it look like the write
    it exists to replace."""
    c = _Client()
    AttachTransport(c, "agent-1").capture_current("agent-1")
    assert "json_body" not in c.calls[0]
    assert "idempotency_key" not in c.calls[0]


def test_an_unreachable_control_plane_FAILS_CLOSED():
    """The opposite of ``policy_current``, whose safe direction is the other
    way: there, keeping the current policy is right, so it fails OPEN. Here a
    blip read as permission would send somebody's end users' messages on the
    strength of a network error.

    Asserted against the REAL method. A fake transport that raises exercises
    the CALLER's except, not this one, which is how a mutation making this
    return ``{"capture": {"turns": True}}`` survived.
    """
    class _Broken:
        def request(self, *a, **kw):
            raise RuntimeError("unreachable")

    out = AttachTransport(_Broken(), "agent-1").capture_current("agent-1")
    assert out == {}, out
    assert (out.get("capture") or {}).get("turns") is not True
