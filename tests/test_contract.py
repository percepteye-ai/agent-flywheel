"""The contract must make the dangerous shape unconstructible."""
from __future__ import annotations

import json

import pytest

from agent_flywheel import RolloutOutput, RolloutRequest, ToolCallOutcome
from agent_flywheel.errors import ContractViolation


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
    An agent that reports nothing must not accidentally assert zero."""
    assert RolloutOutput(final_text="hi").tool_calls is None
    assert RolloutOutput(final_text="hi").to_wire()["tool_calls"] is None


def test_observed_zero_is_distinguishable_from_not_reported():
    assert RolloutOutput(final_text="hi", tool_calls=[]).to_wire()["tool_calls"] == []


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "1"])
def test_llm_call_count_must_be_a_nonnegative_integer(value):
    with pytest.raises(ContractViolation, match="nonnegative integer"):
        RolloutOutput(final_text="hi", llm_call_count=value).to_wire()


# ── server-authored fields are unrepresentable ──────────────────────────────

def test_output_emits_only_declared_contract_keys():
    """An agent cannot assert a field it has no way to know. The dataclass is
    closed, so anything derived from the inference wire or the rollout record is
    absent by construction rather than by filtering."""
    import json
    import pathlib

    schema = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json").read_text()
    )
    declared = set(schema["$defs"]["RolloutOutput"]["properties"])
    out = RolloutOutput(
        final_text="x", tool_calls=[_tc()], reward=1.0, success=True,
        exclusion_reason="capacity", agent_fingerprint={"sdk": "0.1.0"},
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


def test_every_exclusion_reason_the_code_knows_is_declared_by_the_schema():
    """The schema is the authority, so a reason the code can emit and the
    schema does not declare is a report a conforming server must reject.

    This is not hypothetical. ``agent_exit_nonzero`` was added to
    ``ExclusionReason``, emitted by the runner and asserted by four test
    modules, while the published enum still listed seven reasons -- and the
    only thing that compared the two was a person reading both files. The
    equality is exact, and deliberately so: an extra reason in the schema is
    vocabulary the control plane owns, but the ORDER carries the same
    editorial meaning in both files, so a silent divergence in either
    direction is the drift worth catching.
    """
    import json
    import pathlib
    import typing

    from agent_flywheel.contract import ExclusionReason

    schema = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json").read_text()
    )
    declared = schema["$defs"]["Exclusion"]["properties"]["reason"]["enum"]
    assert list(typing.get_args(ExclusionReason)) == declared, (
        "ExclusionReason and schema/flywheel-1.json disagree; the schema is the "
        "authority, so update it in the same commit as the code"
    )


def test_rollout_request_and_the_schema_declare_the_same_keys():
    """The parity check whose absence let `lease_expires_ts` drift.

    `RolloutOutput` has `test_output_emits_only_declared_contract_keys` and the
    Exclusion enum has its own agreement test; `RolloutRequest` had neither. So
    the client parsed a field, PACED HEARTBEATS against it (runner._beat_
    interval_for), and the published schema declared `additionalProperties:
    false` without it -- meaning a conforming server could not send the field
    the client needs, and a server that sent it violated the schema this repo
    calls authoritative. Both directions are asserted, because only checking
    one is how the drift survived.
    """
    import dataclasses
    import json
    import pathlib as _pl

    schema = json.loads(
        (_pl.Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json")
        .read_text()
    )
    declared = set(schema["$defs"]["RolloutRequest"]["properties"])
    fields = {f.name for f in dataclasses.fields(RolloutRequest)}

    assert not (fields - declared), (
        "the dataclass carries fields the schema does not declare, and the "
        f"schema forbids additionalProperties: {sorted(fields - declared)}"
    )
    assert not (declared - fields), (
        f"the schema declares fields the dataclass drops: {sorted(declared - fields)}"
    )


def test_a_granted_lease_survives_the_wire():
    """`lease_expires_ts` is only useful if `from_wire` actually keeps it."""
    r = RolloutRequest.from_wire({
        "rollout_id": "rl_1", "turn_input": "t", "llm_url": "u",
        "api_key": "k", "artifacts_dir": "/tmp/a", "deadline_ts": 1.0,
        "lease_expires_ts": 1234.5,
    })
    assert r.lease_expires_ts == 1234.5
