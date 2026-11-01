"""The customer API reads (contract 1.30.0), case for case with the Node
SDK's test/customer-api.test.js."""
from __future__ import annotations

import json

import httpx
import pytest

from agent_flywheel.customer_api import (
    ATTACK_REFUSAL_CODES, CUSTOMER_CONTRACT_VERSION, PROMPT_RUN_REFUSAL_CODES, RUN_LIST_REFUSAL_CODES,
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
    assert ATTACK_REFUSAL_CODES == ("attack_set_is_sealed",)
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


def test_hillclimb_security_reads_the_runs_record():
    calls: list = []
    c = _client(200, {"measurement": {"state": "not_measured"}}, calls)
    assert c.hillclimb_security("r/1")["measurement"]["state"] == "not_measured"
    assert calls[0].url.raw_path == b"/api/v2/hillclimbs/r%2F1/security"
    c = _client(404, {"detail": {"code": "hillclimb_not_found", "message": "no"}}, calls)
    with pytest.raises(CustomerRefusal) as info:
        c.hillclimb_security("r2")
    assert info.value.code == "hillclimb_not_found"


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


def test_supports_contract_is_false_for_a_body_that_is_not_an_object():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>ok</html>")
    c = CustomerApiClient(KEY, base_url=BASE, environ={},
                          client=httpx.Client(transport=httpx.MockTransport(handler)))
    assert c.supports_contract() is False
    assert _client(200, ["1"], []).supports_contract() is False


def test_the_injected_environ_chooses_the_control_plane(monkeypatch):
    """``environ=`` is the whole environment the client reads: the URL as
    well as the key (the Node SDK's ``env`` does the same)."""
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://ambient.test/api/flywheel/v1")
    env = {"PERCEPTEYE_MC_API_KEY": KEY,
           "PERCEPTEYE_CONTROL_PLANE_URL": "http://localhost:8000/api/flywheel/v1"}
    assert CustomerApiClient(environ=env).base_url == "http://localhost:8000/api/v2"
    assert customer_api_base(None, env) == "http://localhost:8000/api/v2"
    assert (CustomerApiClient(environ={"PERCEPTEYE_MC_API_KEY": KEY}).base_url
            == "https://launch.percepteye.ai/api/v2")


def test_refusal_codes_are_the_control_planes():
    """The codes are copies of the control plane's; when
    ``PERCEPTEYE_CONTROL_PLANE_CHECKOUT`` names its checkout (a release,
    ``docs/RELEASING.md``), each must be one its published contract lists
    (``x-refusal-codes``) with the status this SDK documents. A variable that
    is set is a promise: a missing document fails."""
    import os
    from pathlib import Path

    checkout = os.environ.get("PERCEPTEYE_CONTROL_PLANE_CHECKOUT")
    if not checkout:
        return
    doc = json.loads((Path(checkout) / "docs" / "openapi.json").read_text())
    published = {e["code"]: e["http_status"] for e in doc["x-refusal-codes"]}
    expected = {"launch_uri_outside_org": 403, "run_page_too_deep": 422,
                "prompt_run_balance_too_small": 402, "recorded_prompt_absent": 422,
                "recorded_decisions_too_few": 422, "recorded_prompt_mismatch": 422,
                "attack_set_is_sealed": 409}
    assert set(RUN_LIST_REFUSAL_CODES + PROMPT_RUN_REFUSAL_CODES
               + ATTACK_REFUSAL_CODES) == set(expected)
    for code, status in expected.items():
        assert published.get(code) == status, code
    assert doc["x-customer-contract-version"] == CUSTOMER_CONTRACT_VERSION
