"""The customer API reads (contract 1.30.0), case for case with the Node
SDK's test/customer-api.test.js."""
from __future__ import annotations

import json

import httpx
import pytest

from agent_flywheel.customer_api import (
    CUSTOMER_CONTRACT_VERSION, PROMPT_RUN_REFUSAL_CODES, RUN_LIST_REFUSAL_CODES,
    CustomerApiClient, CustomerRefusal, customer_api_base,
)
from agent_flywheel.errors import ConfigurationError

KEY = "mck_test_key_value_123456"
BASE = "https://x.test/api/v2"


def _client(status, body, calls):
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, content=b"" if body is None else json.dumps(body).encode())
    return CustomerApiClient(
        KEY, base_url=BASE, environ={},
        client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_pins_contract_and_new_codes():
    assert CUSTOMER_CONTRACT_VERSION == "1.30.0"
    assert RUN_LIST_REFUSAL_CODES == ("launch_uri_outside_org", "run_page_too_deep")
    assert PROMPT_RUN_REFUSAL_CODES == (
        "prompt_run_balance_too_small", "recorded_prompt_absent",
        "recorded_decisions_too_few", "recorded_prompt_mismatch")


def test_base_beside_flywheel_api(monkeypatch):
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    assert customer_api_base("https://x.test/api/flywheel/v1/") == BASE
    assert customer_api_base("https://x.test/p/api/flywheel/v1") == "https://x.test/p/api/v2"
    assert customer_api_base("https://x.test/other") == BASE
    assert customer_api_base(None) == "https://launch.percepteye.ai/api/v2"
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "http://localhost:8000/api/flywheel/v1")
    assert customer_api_base(" ") == "http://localhost:8000/api/v2"


def test_refuses_no_key_and_agent_key():
    with pytest.raises(ConfigurationError):
        CustomerApiClient(environ={})
    with pytest.raises(ConfigurationError, match=r"agent key \(pefw_\)"):
        CustomerApiClient("pefw_abc", environ={})
    c = CustomerApiClient(environ={"PERCEPTEYE_MC_API_KEY": KEY})
    assert c._headers["Authorization"] == f"Bearer {KEY}"


def test_list_runs_parameters():
    calls: list = []
    c = _client(200, [{"run_id": "r1", "prompt_evolution": {"id": "e1"}}], calls)
    rows = c.list_runs(product_id="p1", include_prompt_evolutions=True, limit=10,
                       started_before="2026-11-01T00:00:00Z")
    assert rows[0]["prompt_evolution"]["id"] == "e1"
    url = calls[0].url
    assert url.path == "/api/v2/runs"
    assert dict(url.params) == {
        "limit": "10", "started_before": "2026-11-01T00:00:00Z",
        "product_id": "p1", "include_prompt_evolutions": "true"}
    assert calls[0].headers["Authorization"] == f"Bearer {KEY}"


def test_eligibility_and_eval_sets():
    calls: list = []
    c = _client(200, {"prompt_optimization": {
        "enabled": True, "mode": "offline", "needs": "prepared_corpus", "algorithms": [],
        "model_under_test": {"model": "m", "label": "l", "customer_model": False,
                             "reason": "r"}}}, calls)
    e = c.training_eligibility(agent_id="a1")
    assert e["prompt_optimization"]["needs"] == "prepared_corpus"
    assert e["prompt_optimization"]["model_under_test"]["customer_model"] is False
    assert calls[0].url.query == b"agent_id=a1"
    c.list_eval_sets(product_id="p1", limit=5, offset=10)
    assert calls[1].url.path == "/api/v2/eval-sets"
    assert calls[1].url.query == b"product_id=p1&limit=5&offset=10"


def test_refusal_code_and_plain_words():
    calls: list = []
    c = _client(422, {"detail": {
        "code": "recorded_prompt_absent", "message": "no system message in 40 traces",
        "customer_message": "These traces hold no system message.",
        "customer_words": "Supply your prompt.", "fault_owner": "customer",
        "retryable": False, "remediation": "supply the prompt as a twin"}}, calls)
    with pytest.raises(CustomerRefusal) as info:
        c.training_eligibility(product_id="p")
    err = info.value
    assert err.status_code == 422
    assert err.code == "recorded_prompt_absent"
    assert err.message == "no system message in 40 traces"
    assert err.customer_message == "These traces hold no system message."
    assert err.plain_words == "These traces hold no system message."
    assert err.fault_owner == "customer"
    assert err.retryable is False
    no_msg = CustomerRefusal(422, {"detail": {
        "code": "run_page_too_deep", "message": "m", "customer_words": "Use an earlier date."}})
    assert no_msg.plain_words == "Use an earlier date."
    bare = CustomerRefusal(403, {"detail": "Forbidden"})
    assert bare.code is None
    assert bare.plain_words == "Forbidden"


def test_supports_contract():
    calls: list = []
    assert _client(200, {"customer_contract_versions_supported": ["1"]}, calls).supports_contract()
    assert calls[0].url.path == "/api/v2/health"
    assert not _client(200, {"customer_contract_versions_supported": ["2"]}, []).supports_contract()
