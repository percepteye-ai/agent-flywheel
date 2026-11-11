"""ONE customer-contract surface, in two languages, proved to agree.

Both SDKs declared customer contract 1.31.0 while Mission Control published
1.50.0, and nothing compared the two halves with each other: each suite pinned
its own copy. This runs the Node SDK's ``src/customer-api.js`` beside
``agent_flywheel.customer_api`` and requires the same declared contract, the
same refusal-code catalogue group by group, the same reading of a refusal's own
evidence (``detail``), and the same verdict on an eligibility answer for a
training location (1.50.0): returned when it is for the location asked,
refused ``CustomerContractTooOld`` when it is not, as an older server's is.

The skip rule is ``tests/node_sdk.py``'s.
"""
from __future__ import annotations

import json

import httpx
import pytest

from agent_flywheel import customer_api
from agent_flywheel.customer_api import CustomerApiClient, CustomerRefusal

from .node_sdk import require_node_sdk, run_node

#: Python name -> Node name of every exported refusal-code group.
GROUPS = {
    "RUN_LIST_REFUSAL_CODES": "RUN_LIST_REFUSAL_CODES",
    "PROMPT_RUN_REFUSAL_CODES": "PROMPT_RUN_REFUSAL_CODES",
    "ATTACK_REFUSAL_CODES": "ATTACK_REFUSAL_CODES",
    "CYBER_REFUSAL_CODES": "CYBER_REFUSAL_CODES",
    "TRAINING_REFUSAL_CODES": "TRAINING_REFUSAL_CODES",
}

#: ``name -> (asked location, the server's 200 body)``.
ELIGIBILITY_CASES: dict[str, tuple[str | None, object]] = {
    "answered for the asked location": (
        "platform_gpu", {"training_locations": {"answered_for": "platform_gpu"}}),
    "a pre-1.50.0 body, location asked": ("platform_gpu", {"prompt_optimization": {}}),
    "answered for the org's own setting": (
        "platform_gpu", {"training_locations": {"answered_for": "platform_pool"}}),
    "training_locations null": ("platform_pool", {"training_locations": None}),
    "a body that is not an object": ("platform_pool", ["x"]),
    "a pre-1.50.0 body, no location asked": (None, {"prompt_optimization": {}}),
}

#: ``name -> refusal body``.
REFUSAL_CASES: dict[str, object] = {
    "evidence": {"detail": {"code": "training_unhealthy", "message": "m",
                            "detail": {"cause": "non_finite", "evidence": [1]}}},
    "no evidence": {"detail": {"code": "run_page_too_deep", "message": "m"}},
    "evidence not an object": {"detail": {"code": "x", "detail": ["a"]}},
    "a string refusal": {"detail": "Forbidden"},
}


def _python_verdict(location: str | None, body: object) -> str:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(body).encode())
    client = CustomerApiClient(
        "mck_test_key_value_123456", base_url="https://x.test/api/v2", environ={},
        client=httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        client.training_eligibility(training_location=location)
    except customer_api.CustomerContractTooOld:
        return "too_old"
    return "answer"


@pytest.fixture(scope="module")
def node_answers() -> dict[str, object]:
    sdk, node = require_node_sdk()
    script = (
        "import * as api from './src/customer-api.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const input = JSON.parse(raw);\n"
        "const out = {version: api.CUSTOMER_CONTRACT_VERSION, groups: {}, "
        "eligibility: {}, detail: {}};\n"
        "for (const name of input.groups) out.groups[name] = api[name] ? [...api[name]] : null;\n"
        "for (const [name, [location, body]] of Object.entries(input.eligibility)) {\n"
        "  const fetchImpl = async () => ({status: 200, text: async () => JSON.stringify(body)});\n"
        "  const c = new api.CustomerApiClient({apiKey: 'mck_test_key_value_123456', "
        "baseUrl: 'https://x.test/api/v2', env: {}, fetchImpl});\n"
        "  try { await c.trainingEligibility({trainingLocation: location ?? undefined}); "
        "out.eligibility[name] = 'answer'; }\n"
        "  catch (e) { out.eligibility[name] = e instanceof api.CustomerContractTooOld "
        "? 'too_old' : String(e); }\n"
        "}\n"
        "for (const [name, body] of Object.entries(input.refusals)) "
        "out.detail[name] = new api.CustomerRefusal(422, body).detail;\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    payload = {"groups": list(GROUPS.values()), "eligibility": ELIGIBILITY_CASES,
               "refusals": REFUSAL_CASES}
    return json.loads(run_node(sdk, node, script, stdin=json.dumps(payload)))


def test_both_sdks_declare_one_customer_contract(node_answers):
    assert node_answers["version"] == customer_api.CUSTOMER_CONTRACT_VERSION


@pytest.mark.parametrize("python_name", sorted(GROUPS))
def test_both_sdks_know_the_same_refusal_codes(node_answers, python_name):
    python = list(getattr(customer_api, python_name))
    assert node_answers["groups"][GROUPS[python_name]] == python, python_name


def test_every_exported_group_is_compared():
    """A sixth group added to one SDK must be added here, or this fails."""
    exported = {n for n in customer_api.__all__ if n.endswith("_REFUSAL_CODES")}
    assert exported == set(GROUPS)


@pytest.mark.parametrize("name", sorted(ELIGIBILITY_CASES))
def test_both_sdks_judge_a_location_answer_alike(node_answers, name):
    location, body = ELIGIBILITY_CASES[name]
    assert node_answers["eligibility"][name] == _python_verdict(location, body), name


def test_the_verdicts_are_not_all_one_answer(node_answers):
    """The table could agree by both answering the same thing every time."""
    verdicts = node_answers["eligibility"]
    assert verdicts["answered for the asked location"] == "answer"
    assert verdicts["a pre-1.50.0 body, no location asked"] == "answer"
    assert verdicts["a pre-1.50.0 body, location asked"] == "too_old"
    assert verdicts["answered for the org's own setting"] == "too_old"


@pytest.mark.parametrize("name", sorted(REFUSAL_CASES))
def test_both_sdks_read_a_refusals_evidence_alike(node_answers, name):
    assert node_answers["detail"][name] == CustomerRefusal(422, REFUSAL_CASES[name]).detail
