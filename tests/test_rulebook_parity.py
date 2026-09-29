"""ONE rulebook declaration, in two languages, proved to be the same request.

A customer runs one package or the other, and may move between them. The
control plane answers rules equal to the ones in force ``unchanged``, so the
same rules must reach it as the same request from either SDK: the same route
(the agent id encoded the same way), the same JSON body, and the same
idempotency key -- the turn batch's derivation over the body, which the Node
side reproduces with ``stableStringify``. And what each tells its customer
about the answer must be the same sentence.

The body is compared as JSON, not as bytes. Python's bytes are httpx's
encoder's, and the floor this package declares (``httpx>=0.27``) spaces and
escapes them differently from 0.28 -- the control plane reads the JSON, and
the key is this package's own digest of the body, so neither depends on it.
The ANSWER each side is handed is the same bytes, so the sentences they build
from it can be compared exactly.

Both halves are driven through their REAL transport and REAL HTTP client:
Python's ``ControlPlaneClient`` over ``httpx.MockTransport``, Node's over an
injected ``fetch``. The fixture is DELIBERATELY AWKWARD -- non-ASCII text, an
astral-plane character, a quote, a backslash, a line separator and an agent id
with spaces and parentheses -- because those are where ``json.dumps`` and
``JSON.stringify`` and the two URL encoders part by default. Plain ASCII would
agree by accident and certify a parity that does not exist.

The schema's two copies are compared byte for byte here too: a comment
saying the Node copy is vendored is not a mechanism. The skip rule is
``tests/node_sdk.py``'s.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import httpx
import pytest

from agent_flywheel._http import ControlPlaneClient
from agent_flywheel.transport import AttachTransport

from .node_sdk import require_node_sdk, run_node

BASE = "https://cp.example.invalid/api/flywheel/v1"
AGENT = "desk (eu)-ü"
RULES = [
    "Frag nach der Farbe, bevor du bestellst — immer höflich "
    "\U0001f4e6.",
    'When a caller says "cancel", restate the renewal date first\\then '
    "wait.\u2028Confirm it.",
]
RECEIPT = {
    "status": "registered", "product_id": "prod-widget-desk",
    "version_id": "v-2", "version": 2, "content_sha256": "b" * 64,
    "parent_sha256": "a" * 64, "rule_count": 2,
}
#: `(name, rules, status, answer body, what the refusal says)`. The
#: refusals carry the control plane's structured detail; the 409's is longer
#: than both clients keep, so the cut is compared as well. The keyless and
#: array bodies are what a proxy or an older control plane may answer; the 502
#: may come after the rules landed, the 503 says they did not.
CASES = [
    ("registered", RULES, 200, RECEIPT, None),
    ("retracted", [], 200, {**RECEIPT, "status": "retracted",
                            "rule_count": 0, "version": 3}, None),
    # JSON Schema's integer and JavaScript's number: a receipt on both sides.
    ("integral_floats", RULES, 200, {**RECEIPT, "version": 2.0,
                                     "rule_count": 2.0}, None),
    ("unattributed", RULES, 409, {"detail": {
        "error": "rulebook_unattributed", "status": "ambiguous",
        "detail": ("agent 'desk (eu)-ü' is attributed to no Product "
                   "(ambiguous: exactly one active Product must declare the "
                   "agent in its Flywheel spec), so its rules would belong to "
                   "no Product and nothing was registered. Declare the agent "
                   "on exactly one Product.")}}, "rulebook_unattributed"),
    ("invalid", RULES, 422, {"detail": {
        "error": "rulebook_invalid", "cause": "rule_too_short", "index": 1,
        "detail": "la règle 1 est trop courte"}}, "rulebook_invalid"),
    ("keyless", RULES, 409, {"code": "gateway_busy", "hint": None},
     '{"code":"gateway_busy","hint":null}'),
    ("array", RULES, 403, ["x", 1], '["x",1]'),
    ("store_unavailable", RULES, 503, {"detail": {
        "error": "product_rulebook_store_failed",
        "detail": "the artifact store did not answer"}},
     "this agent's rules were not registered"),
    ("answer_lost", RULES, 502, {"detail": "upstream closed the connection"},
     "whether this agent's rules were registered is unknown"),
]


def _answer(body) -> str:
    """The bytes both fakes answer with -- ``JSON.stringify``'s spelling, so
    a floating ``2.0`` stays ``2.0`` on the wire for both readers."""
    return json.dumps(body, ensure_ascii=False, separators=(",", ":"))


def _python(rules, status, body) -> dict:
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append({
            "method": request.method,
            "path": request.url.raw_path.decode("ascii").removeprefix(
                "/api/flywheel/v1"),
            "key": request.headers["Idempotency-Key"],
            "body": request.content.decode("utf-8"),
        })
        return httpx.Response(
            status, content=_answer(body).encode("utf-8"),
            headers={"content-type": "application/json"})

    client = ControlPlaneClient(BASE, "pefw_test", max_attempts=1, client=(
        httpx.Client(transport=httpx.MockTransport(handler))))
    receipt, refusal = AttachTransport(client, AGENT).register_rulebook(rules)
    return {
        "seen": seen, "refusal": refusal,
        "receipt": dataclasses.asdict(receipt) if receipt else None,
    }


_NODE = r"""
import { AttachTransport } from './src/transport.js';
import { ControlPlaneClient } from './src/http.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const { base, agent, cases } = JSON.parse(raw);
const out = {};
for (const [name, rules, status, answer] of cases) {
  const seen = [];
  const client = new ControlPlaneClient(base, 'pefw_test', {
    maxAttempts: 1,
    fetchImpl: async (url, init) => {
      seen.push({ method: init.method, path: url.slice(base.length),
                  key: init.headers['Idempotency-Key'], body: init.body });
      return new Response(answer, {
        status, headers: { 'content-type': 'application/json' } });
    },
  });
  // NEVER REJECTS: a refused declaration resolves, so an await that throws
  // here fails the run -- which is the assertion.
  const { receipt, refusal } = await new AttachTransport(
    client, { agentId: agent }).registerRulebook(rules);
  out[name] = { seen, refusal, receipt };
}
process.stdout.write(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node() -> dict:
    sdk, binary = require_node_sdk()
    stdin = json.dumps({"base": BASE, "agent": AGENT, "cases": [
        [name, rules, status, _answer(body)]
        for name, rules, status, body, _ in CASES]})
    return json.loads(run_node(sdk, binary, _NODE, stdin=stdin))


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_both_sdks_send_one_request_for_one_declaration(node, case):
    """Mutations (Python side): key over the rules list instead of the body,
    ``ensure_ascii=False`` in the digest, or ``quote(..., safe="()")`` in the
    route -> red against the Node side's derivation."""
    name, rules, status, body, _ = case
    python = _python(rules, status, body)
    (mine,), (theirs,) = python["seen"], node[name]["seen"]
    assert {**theirs, "body": json.loads(theirs["body"])} == {
        **mine, "body": json.loads(mine["body"])}, (
        f"{name}:\n  python: {mine}\n  node  : {theirs}")
    assert mine["path"] == "/agents/desk%20%28eu%29-%C3%BC/rulebook"
    assert json.loads(mine["body"]) == {"rules": rules}


@pytest.mark.parametrize("case", CASES, ids=[c[0] for c in CASES])
def test_both_sdks_tell_the_customer_the_same_thing(node, case):
    """The same answer is the same receipt, or the same refusal sentence --
    the control plane's own cause in the same spelling, never a Python repr
    on one side and ``[object Object]`` on the other."""
    name, rules, status, body, says = case
    python = _python(rules, status, body)
    assert node[name]["refusal"] == python["refusal"], name
    assert node[name]["receipt"] == python["receipt"], name
    if status >= 400:
        assert python["receipt"] is None
        assert says in python["refusal"], (says, python["refusal"])
    else:
        assert python["refusal"] is None
        assert python["receipt"]["status"] == body["status"]


def test_the_two_schema_copies_are_the_same_bytes():
    """The Node package vendors ``schema/flywheel-1.json`` because it must not
    depend on this one; vendored bytes that differ are two contracts."""
    sdk, _ = require_node_sdk()
    here = Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json"
    there = sdk / "schema" / "flywheel-1.json"
    assert there.read_bytes() == here.read_bytes(), (
        f"{there} differs from {here}; copy this repository's file there")
