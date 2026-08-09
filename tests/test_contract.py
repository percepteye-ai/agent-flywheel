"""The contract must make the dangerous shape unconstructible."""
from __future__ import annotations

import json

import pytest

from percepteye_harness import RolloutOutput, RolloutRequest, ToolCallOutcome
from percepteye_harness.errors import ContractViolation


def _tc(**kw):
    base = dict(name="POST /issue", arguments={}, outcome="ok", status_code=201)
    base.update(kw)
    return ToolCallOutcome(**base)


# ── outcome and status_code are both required ───────────────────────────────

def test_outcome_has_no_default():
    with pytest.raises(TypeError):
        ToolCallOutcome(name="x", arguments={}, status_code=200)  # type: ignore[call-arg]


def test_status_code_has_no_default():
    """You cannot record half an outcome. A partially populated call is the
    dangerous shape, so forgetting the keyword must not silently produce one."""
    with pytest.raises(TypeError):
        ToolCallOutcome(name="x", arguments={}, outcome="ok")  # type: ignore[call-arg]


def test_outcome_rejects_invented_values():
    with pytest.raises(ContractViolation):
        _tc(outcome="probably")


# ── the fabrication shape cannot be built ───────────────────────────────────

def test_unknown_outcome_discards_an_explicit_status():
    """THE regression this SDK exists to prevent.

    outcome='unknown' beside status_code=200 is what an honest capture emits
    for a tool that swallowed its own exception. Sent as-is, a scorer that
    reads the status treats a failed call as a landed mutation. The type
    refuses to carry it."""
    tc = _tc(outcome="unknown", status_code=200)
    assert tc.status_code is None


def test_unknown_outcome_scrubs_status_from_a_json_body():
    """Withholding the field is not enough -- a scorer that parses the result
    envelope finds the status there instead."""
    tc = _tc(outcome="unknown", status_code=200,
             output=json.dumps({"status_code": 200, "id": "PROJ-1"}))
    body = json.loads(tc.output)
    assert "status_code" not in body
    assert body["id"] == "PROJ-1", "human-readable content must survive"


def test_unknown_outcome_scrubs_status_from_prose():
    tc = _tc(outcome="unknown", status_code=None, output="Request returned HTTP 200 OK")
    assert "HTTP 200" not in tc.output


def test_unknown_outcome_scrubs_nested_bodies():
    tc = _tc(outcome="unknown", status_code=None,
             output={"result": {"statusCode": 201, "name": "ok"}})
    assert "statusCode" not in tc.output["result"]
    assert tc.output["result"]["name"] == "ok"


def test_observed_outcomes_keep_their_status():
    """Abstention must be evidence-driven, not blanket."""
    assert _tc(outcome="ok", status_code=201).status_code == 201
    assert _tc(outcome="failed", status_code=200).status_code == 200


def test_the_app_level_refusal_is_expressible():
    """An application-level permission denial returned as HTTP 200. This exact
    pair is why status alone is not an outcome."""
    tc = _tc(outcome="failed", status_code=200, error="app_auth_error")
    assert (tc.outcome, tc.status_code) == ("failed", 200)


# ── null vs empty list ──────────────────────────────────────────────────────

def test_tool_calls_defaults_to_none_not_empty_list():
    """[] asserts 'I observed zero tool calls'. None says 'I did not report'.
    A harness that reports nothing must not accidentally assert zero."""
    assert RolloutOutput(final_text="hi").tool_calls is None
    assert RolloutOutput(final_text="hi").to_wire()["tool_calls"] is None


def test_observed_zero_is_distinguishable_from_not_reported():
    assert RolloutOutput(final_text="hi", tool_calls=[]).to_wire()["tool_calls"] == []


# ── server-authored fields are unrepresentable ──────────────────────────────

def test_output_emits_only_declared_contract_keys():
    """A harness cannot assert a field it has no way to know. The dataclass is
    closed, so anything derived from the inference wire or the rollout record is
    absent by construction rather than by filtering."""
    import json
    import pathlib

    schema = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "schema" / "harness-1.json").read_text()
    )
    declared = set(schema["$defs"]["RolloutOutput"]["properties"])
    out = RolloutOutput(
        final_text="x", tool_calls=[_tc()], reward=1.0, success=True,
        exclusion_reason="capacity", harness_fingerprint={"sdk": "0.1.0"},
        llm_call_count=3,
    )
    assert set(out.to_wire()) <= declared, "code emits a key the schema does not declare"


# ── the removed configuration channel ───────────────────────────────────────

def test_rollout_request_rejects_an_environment_key():
    """The channel that wrote server-supplied values into a running process's
    os.environ is gone, and its reappearance is an error rather than a
    surprise."""
    with pytest.raises(ContractViolation, match="environment"):
        RolloutRequest.from_wire({
            "rollout_id": "rl_1", "turn_input": "go", "llm_url": "https://x",
            "api_key": "k", "artifacts_dir": "/tmp", "deadline_ts": 1.0,
            "environment": {"OPENAI_API_KEY": "sk-leak"},
        })


def test_rollout_request_names_every_missing_field():
    with pytest.raises(ContractViolation, match="turn_input"):
        RolloutRequest.from_wire({"rollout_id": "rl_1"})


def test_deadline_comes_from_the_server():
    r = RolloutRequest(
        rollout_id="rl_1", turn_input="go", llm_url="https://x", api_key="k",
        artifacts_dir="/tmp", deadline_ts=1_000.0,
    )
    assert r.seconds_remaining(now=900.0) == 100.0
    assert r.seconds_remaining(now=2_000.0) == 0.0
