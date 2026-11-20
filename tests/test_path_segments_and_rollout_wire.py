"""Every id this package puts in a request path is one segment, and a claimed
rollout is read strictly.

PATH SEGMENTS. ``report`` and ``abandon`` put ``rollout_id`` in the path
unquoted, ``register_rulebook`` quoted ``agent_id`` without checking it, and
``hillclimb_security`` quoted ``..`` to ``..`` -- ``/hillclimbs/../security``.
One gate (``contract._path_segment``) now refuses what cannot be one segment
and percent-encodes the rest, at every one of them and where a rollout id
arrives (``RolloutRequest.from_wire``), so a crafted id never reaches a path
on the wire or on disk.

THE ROLLOUT WIRE. ``from_wire`` turned ``None`` into ``"None"``, took ``true``
and NaN as a deadline, and raised ``TypeError``/``ValueError`` for a wrong
type; every refusal is now a ``ContractViolation`` naming the field.
"""
from __future__ import annotations

import math

import httpx
import pytest

from agent_flywheel.contract import RolloutOutput, RolloutRequest, _path_segment
from agent_flywheel.customer_api import CustomerApiClient
from agent_flywheel.errors import ContractViolation
from agent_flywheel.transport import AttachTransport

#: ``id -> the cause the refusal names``.
UNSAFE = {
    "": "empty",
    ".": "dot segment",
    "..": "dot segment",
    "../etc": "path separator",
    "a/b": "path separator",
    "a\\b": "path separator",
    "/abs": "path separator",
    "a\nb": "control character",
    "a\x00b": "control character",
    "a\x85b": "control character",
    "a\ud800": "well-formed",
    "x" * 257: "at most 256",
}


@pytest.mark.parametrize("value", sorted(UNSAFE))
def test_the_gate_refuses_what_cannot_be_one_segment(value):
    with pytest.raises(ContractViolation, match=UNSAFE[value]):
        _path_segment(value, "id")


@pytest.mark.parametrize("value", [None, 7, ["a"]])
def test_the_gate_refuses_a_non_string(value):
    with pytest.raises(ContractViolation, match="must be a string"):
        _path_segment(value, "id")


@pytest.mark.parametrize(("value", "segment"), [
    ("rl_abc234", "rl_abc234"),
    ("a.b-c~d", "a.b-c~d"),
    ("a?b#c%d", "a%3Fb%23c%25d"),
    ("caf\u00e9 1", "caf%C3%A9%201"),
    ("...", "..."),
    ("x" * 256, "x" * 256),
])
def test_the_gate_keeps_and_quotes_everything_else(value, segment):
    assert _path_segment(value, "id") == segment


class _Client:
    def __init__(self, reply=None):
        self.calls: list[dict] = []
        self._reply = reply if reply is not None else {}

    def request(self, method, path, **kw):
        self.calls.append({"method": method, "path": path, **kw})
        return self._reply


def _rollout(rollout_id: str) -> RolloutRequest:
    return RolloutRequest(
        rollout_id=rollout_id, turn_input="", llm_url="http://x",
        api_key="k", artifacts_dir="d", deadline_ts=1.0, lease_token="t")


def test_report_and_abandon_quote_the_rollout_id():
    c = _Client()
    t = AttachTransport(c, "agent-1")
    t.report(_rollout("rl_1?x"), RolloutOutput(final_text=""))
    t.abandon(_rollout("rl_1?x"), reason="cancelled")
    assert [call["path"] for call in c.calls] == [
        "/rollouts/rl_1%3Fx/result", "/rollouts/rl_1%3Fx/abandon"]


@pytest.mark.parametrize("rollout_id", ["..", "../../agents", "a/b"])
def test_report_and_abandon_refuse_a_rollout_id_that_is_not_a_segment(
        rollout_id):
    c = _Client()
    t = AttachTransport(c, "agent-1")
    with pytest.raises(ContractViolation, match="rollout_id"):
        t.report(_rollout(rollout_id), RolloutOutput(final_text=""))
    with pytest.raises(ContractViolation, match="rollout_id"):
        t.abandon(_rollout(rollout_id), reason="cancelled")
    assert c.calls == []


def test_the_rulebook_names_an_agent_id_that_is_not_a_segment():
    c = _Client()
    upload = AttachTransport(c, "..").register_rulebook(["rule"])
    assert c.calls == []
    assert upload.receipt is None
    assert "not registered" in upload.refusal
    assert "agent_id '..' is a dot segment" in upload.refusal


def test_the_rulebook_path_and_key_are_unchanged_for_an_ordinary_id():
    c = _Client()
    AttachTransport(c, "agent.1~x").register_rulebook(["rule"])
    assert c.calls[0]["path"] == "/agents/agent.1~x/rulebook"
    assert c.calls[0]["idempotency_key"].startswith("rulebook-agent.1~x-")


@pytest.mark.parametrize("run_id", ["..", ".", "a/b", "a\\b", None, ""])
def test_hillclimb_security_refuses_a_run_id_that_is_not_a_segment(run_id):
    sent: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    c = CustomerApiClient(
        "mck_test_key_value_123456", base_url="https://x.test/api/v2",
        environ={}, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(ContractViolation, match="run_id"):
        c.hillclimb_security(run_id)
    assert sent == []


WIRE = {
    "rollout_id": "rl_abc", "turn_input": "do it", "llm_url": "http://x/v1",
    "api_key": "k", "artifacts_dir": "rollouts/rl_abc", "deadline_ts": 10.5,
    "lease_expires_ts": 20, "lease_token": "lt", "metadata": {"a": 1},
}


def test_a_well_formed_rollout_is_read_as_sent():
    r = RolloutRequest.from_wire(WIRE)
    assert (r.rollout_id, r.turn_input, r.deadline_ts, r.lease_expires_ts,
            r.lease_token, r.metadata) == (
        "rl_abc", "do it", 10.5, 20.0, "lt", {"a": 1})
    bare = RolloutRequest.from_wire({
        **WIRE, "lease_expires_ts": None, "lease_token": None,
        "metadata": None})
    assert (bare.lease_expires_ts, bare.lease_token, bare.metadata) == (
        None, None, {})


@pytest.mark.parametrize(("field", "value"), [
    ("rollout_id", None), ("rollout_id", 7), ("rollout_id", ".."),
    ("rollout_id", "../../etc"), ("rollout_id", "/abs"),
    ("turn_input", None), ("llm_url", None), ("api_key", None),
    ("artifacts_dir", None), ("api_key", 5),
    ("deadline_ts", None), ("deadline_ts", True), ("deadline_ts", math.nan),
    ("deadline_ts", math.inf), ("deadline_ts", "10"), ("deadline_ts", 10**400),
    ("lease_expires_ts", False), ("lease_expires_ts", -math.inf),
    ("lease_expires_ts", "x"),
    ("lease_token", 5), ("metadata", ["a"]), ("metadata", "x"),
])
def test_a_malformed_field_is_a_contract_violation_naming_it(field, value):
    with pytest.raises(ContractViolation, match=field):
        RolloutRequest.from_wire({**WIRE, field: value})


@pytest.mark.parametrize("wire", [None, [], "rollout"])
def test_a_rollout_that_is_not_an_object_is_a_contract_violation(wire):
    with pytest.raises(ContractViolation, match="JSON object"):
        RolloutRequest.from_wire(wire)
