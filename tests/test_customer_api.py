"""The customer API reads (contract 1.54.0), case for case with the Node
SDK's test/customer-api.test.js."""
from __future__ import annotations

import json

import httpx
import pytest

from agent_flywheel.customer_api import (
    AGENT_REFUSAL_CODES,
    ATTACK_REFUSAL_CODES,
    CUSTOMER_CONTRACT_VERSION,
    CYBER_REFUSAL_CODES,
    PROMPT_RUN_REFUSAL_CODES,
    RUN_LIST_REFUSAL_CODES,
    TRAINING_REFUSAL_CODES,
    CustomerApiClient,
    CustomerContractTooOld,
    CustomerRefusal,
    customer_api_base,
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
    assert CUSTOMER_CONTRACT_VERSION == "1.54.0"
    assert ATTACK_REFUSAL_CODES == (
        "attack_set_is_sealed",
        "attack_resistance_measuring",
        "attack_regression_not_accepted",
        "attack_approval_stale",
        "attack_regression_not_stated",
        "measurement_deadline_passed",
        "prompt_changed",
        "hillclimb_not_found",
        "round_not_found",
        "event_kind_is_the_platforms",
        "attack_unmeasured_undisclosed",
    )
    assert CYBER_REFUSAL_CODES == (
        "cyber_disabled", "cyber_training_disabled", "cyber_needs_training",
        "cyber_no_training_entry", "cyber_lane_has_no_attack_reward",
        "hillclimb_security_unknown", "cyber_attack_reward_not_comparable")
    assert TRAINING_REFUSAL_CODES == (
        "opsd_prompts_exceed_host", "opsd_recipe_tokenizer_hub_unavailable",
        "opsd_recipe_tokenizer_refused", "opsd_recipe_tokenizer_platform_fault",
        "opsd_teacher_preflight_no_evidence_decisions", "training_diverged_at_lr",
        "training_unhealthy", "training_pool_cannot_train_objective",
        "opsd_chosen_model_not_trainable", "hillclimb_base_model_invalid",
        "base_model_unresolvable", "destination_algorithm_unsupported",
        "provider_checkpoint_not_owned", "hillclimb_start_invalid",
        "base_merge_failed", "authority_ruling_requires_org_admin")
    assert RUN_LIST_REFUSAL_CODES == (
        "launch_uri_outside_org", "run_page_too_deep",
        "launch_input_requires_platform_operator",
        "code_sandbox_not_configured", "code_sandbox_unreachable")
    assert PROMPT_RUN_REFUSAL_CODES == (
        "prompt_run_balance_too_small", "recorded_prompt_absent",
        "recorded_decisions_too_few", "recorded_prompt_mismatch",
        "prompt_objective_unprepared", "prompt_evolution_refused",
        "prompt_model_unreachable")
    assert AGENT_REFUSAL_CODES == (
        "practice_rollouts_not_enabled", "practice_needs_a_training_process",
        "model_content_capture_not_enabled", "flywheel_no_task_source",
        "flywheel_product_undeclared", "invalid_flywheel_spec")


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


def test_eligibility_for_a_training_location():
    """1.50.0: the answer is for the location asked, and says so."""
    calls: list = []
    c = _client(200, {"training_locations": {
        "answered_for": "platform_gpu", "default": "platform_pool", "options": [
            {"location": "platform_gpu", "available": True, "refusal": None,
             "gpu_hosts": 2, "gpu_hosts_free": 1}]}}, calls)
    e = c.training_eligibility(agent_id="a1", training_location="platform_gpu")
    assert e["training_locations"]["options"][0]["gpu_hosts_free"] == 1
    assert calls[0].url.query == b"agent_id=a1&training_location=platform_gpu"


@pytest.mark.parametrize("body", [
    {"prompt_optimization": {}},                                   # before 1.50.0
    {"training_locations": None},
    {"training_locations": {"answered_for": "platform_pool"}},     # the org's own
    ["not", "an", "object"],
])
def test_an_answer_not_for_the_asked_location_is_refused(body):
    """A server before 1.50.0 ignores ``training_location`` and answers for
    the organization's own setting; that is never returned as the answer
    asked for."""
    c = _client(200, body, [])
    with pytest.raises(CustomerContractTooOld, match=r"customer contract 1\.50\.0"):
        c.training_eligibility(training_location="platform_gpu")


def test_without_a_location_an_older_answer_is_returned_unchanged():
    """No location asked, nothing to check: a 1.30.0 server's answer stands."""
    calls: list = []
    c = _client(200, {"prompt_optimization": {"enabled": False}}, calls)
    assert c.training_eligibility(product_id="p") == {"prompt_optimization": {"enabled": False}}
    assert calls[0].url.query == b"product_id=p"


def test_hillclimb_security_reads_the_runs_record():
    calls: list = []
    c = _client(200, {"measurement": {"state": "not_measured"}}, calls)
    assert c.hillclimb_security("r?1#2")["measurement"]["state"] == "not_measured"
    assert calls[0].url.raw_path == b"/api/v2/hillclimbs/r%3F1%232/security"
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
    assert err.detail is None
    diverged = CustomerRefusal(422, {"detail": {
        "code": "training_diverged_at_lr", "message": "m",
        "detail": {"learning_rate": 2e-5, "retry_learning_rate": 1e-5}}})
    assert diverged.detail == {"learning_rate": 2e-5, "retry_learning_rate": 1e-5}
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
                "attack_set_is_sealed": 409,
                "attack_resistance_measuring": 409,
                "attack_regression_not_accepted": 409,
                "attack_approval_stale": 409,
                "attack_regression_not_stated": 409,
                "measurement_deadline_passed": 409,
                "prompt_changed": 409,
                "hillclimb_not_found": 404,
                "round_not_found": 404,
                "event_kind_is_the_platforms": 403,
                # 1.32.0 through 1.49.0: every code those bumps published but
                # 1.40.0's three prompt_box_* codes (see PROMPT_RUN_REFUSAL_CODES).
                "opsd_prompts_exceed_host": 422,
                "opsd_recipe_tokenizer_hub_unavailable": 503,
                "opsd_recipe_tokenizer_refused": 422,
                "opsd_recipe_tokenizer_platform_fault": 503,
                "opsd_teacher_preflight_no_evidence_decisions": 409,
                "attack_unmeasured_undisclosed": 409,
                "training_diverged_at_lr": 422,
                "training_unhealthy": 422,
                "prompt_objective_unprepared": 500,
                "prompt_evolution_refused": 409,
                "prompt_model_unreachable": 422,
                "cyber_disabled": 503,
                "cyber_training_disabled": 503,
                "cyber_needs_training": 422,
                "cyber_no_training_entry": 422,
                "cyber_lane_has_no_attack_reward": 422,
                "hillclimb_security_unknown": 422,
                "training_pool_cannot_train_objective": 503,
                "cyber_attack_reward_not_comparable": 422,
                "opsd_chosen_model_not_trainable": 422,
                "hillclimb_base_model_invalid": 422,
                # 1.51.0: the hillclimb door's refusals of an agent's switch,
                # raised before 1.51.0 published them.
                "practice_rollouts_not_enabled": 422,
                "practice_needs_a_training_process": 422,
                "model_content_capture_not_enabled": 422,
                "flywheel_no_task_source": 422,
                "flywheel_product_undeclared": 422,
                "invalid_flywheel_spec": 422,
                "base_model_unresolvable": 422,
                "destination_algorithm_unsupported": 422,
                "provider_checkpoint_not_owned": 403,
                # 1.51.0: a hill climb request naming an input only a
                # platform operator may set. The other fourteen codes 1.51.0
                # added or published are answered by operations this SDK
                # neither calls nor documents (a Decision's answer, eval-set
                # comparisons and saves, observability sources, serving an
                # artifact, a champion's rollback).
                "launch_input_requires_platform_operator": 403,
                # 1.51.0: a coding agent's launch on a deployment with no
                # isolated sandbox for its code, or one not answering.
                "code_sandbox_not_configured": 503,
                "code_sandbox_unreachable": 503,
                # 1.52.0: a hill climb started from a model the organization
                # already trained.
                "hillclimb_start_invalid": 422,
                "base_merge_failed": 502,
                # 1.54.0: a ruling on the reward plan's Decision by anyone
                # but an org admin.
                "authority_ruling_requires_org_admin": 403,
                }
    known = (RUN_LIST_REFUSAL_CODES + PROMPT_RUN_REFUSAL_CODES
             + ATTACK_REFUSAL_CODES + CYBER_REFUSAL_CODES + TRAINING_REFUSAL_CODES
             + AGENT_REFUSAL_CODES)
    assert len(known) == len(set(known)), "a code is in two groups"
    assert set(known) == set(expected)
    for code, status in expected.items():
        assert published.get(code) == status, code
    assert doc["x-customer-contract-version"] == CUSTOMER_CONTRACT_VERSION


def test_the_declared_contract_is_the_control_planes():
    """``CUSTOMER_CONTRACT_VERSION`` is the one the control plane's source
    declares (its package's ``api/customer_contract.py``). Read from the
    source, not ``docs/openapi.json``, which is generated and can lag it.
    When ``PERCEPTEYE_CONTROL_PLANE_CHECKOUT`` names its checkout (a
    release, ``docs/RELEASING.md``) the two must be equal; a variable that is
    set is a promise, so a missing file fails. Unnamed (every CI run, which
    cannot name a private checkout) only the version's shape is checked --
    never a skip, which the zero-skip gate refuses."""
    import os
    import re
    from pathlib import Path

    assert re.fullmatch(r"1\.[0-9]+\.[0-9]+", CUSTOMER_CONTRACT_VERSION)
    checkout = os.environ.get("PERCEPTEYE_CONTROL_PLANE_CHECKOUT")
    if not checkout:
        return
    # Found by its place under the package, which this public repository
    # does not name.
    found = sorted(Path(checkout).glob("*/api/customer_contract.py"))
    assert len(found) == 1, f"no single customer contract under {checkout}"
    source = found[0].read_text(encoding="utf-8")
    declared = re.search(
        r'^CUSTOMER_CONTRACT_VERSION(?:: *Final)? *= *"([^"]+)"', source, re.M)
    assert declared is not None, "the control plane declares no version"
    assert declared.group(1) == CUSTOMER_CONTRACT_VERSION
