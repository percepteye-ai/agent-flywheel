"""The turn upload's actual wire call, which nothing tested.

WHY THIS FILE EXISTS. Every production-turn test in this package monkeypatches
``percepteye_agent_flywheel.transport.AttachTransport`` with a duck-typed fake
that has its own ``report_turns``. That is the right shape for testing the
Attachment -- but it means the REAL ``report_turns`` was never executed by
anything: not the route it posts to, not the body it builds, not the
idempotency key it derives. A method every captured turn passes through, with
zero coverage, in a package whose whole job is getting that data across.

The route and body are also half of a cross-SDK contract. The Node package
sends the same batch to the same route and derives the same idempotency key,
and the last test here proves the two derivations AGREE rather than asserting
it in a comment.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from percepteye_agent_flywheel.transport import AttachTransport


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


def test_the_batch_carries_an_idempotency_key():
    """A network-level replay of the same bytes must collapse BEFORE the
    per-turn rule is reached."""
    key = _sent()["idempotency_key"]
    assert re.fullmatch(r"turns-agent-1-[0-9a-f]{32}", key), key


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

_NODE = Path(__file__).resolve().parents[2] / "percepteye-agent-flywheel-node"


@pytest.mark.skipif(not _NODE.is_dir(), reason="the Node package is not checked out here")
def test_both_sdks_derive_the_same_idempotency_key():
    """A customer runs one package or the other, and may MIGRATE between them
    with turns still on disk. If the two hash the same batch differently, each
    defeats the other's batch de-duplication at exactly that moment.

    Python hashes ``json.dumps(sort_keys=True)``; JavaScript's ``JSON.stringify``
    preserves insertion order instead, so the Node side sorts keys at every
    depth to match. This asserts the two agree on a real payload rather than
    trusting that they do.
    """
    node = None
    for candidate in ("/opt/homebrew/opt/node@24/bin/node", "node"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
            node = candidate
            break
        except (OSError, subprocess.CalledProcessError):
            continue
    if node is None:
        pytest.skip("no usable node binary")

    script = (
        "import {AttachTransport} from './src/transport.js';\n"
        "const calls=[];\n"
        "const tx=new AttachTransport({request:(m,p,o)=>{calls.push(o);}},"
        "{agentId:'agent-1'});\n"
        f"tx.reportTurns('agent-1', {json.dumps(TURNS)});\n"
        "console.log(calls[0].idempotencyKey);\n"
    )
    res = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=_NODE, capture_output=True, text=True,
    )
    assert res.returncode == 0, res.stderr
    node_key = res.stdout.strip()
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
