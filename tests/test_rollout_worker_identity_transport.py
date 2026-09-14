"""The SDK carries its exact registered execution through claim and report."""
from __future__ import annotations

from agent_flywheel.contract import RolloutOutput, RolloutRequest
from agent_flywheel.transport import AttachTransport

EXPECTED = "a" * 64


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict]] = []

    def request(self, method: str, path: str, **kwargs):
        self.calls.append((method, path, kwargs))
        if path == "/rollouts/claim":
            return {"rollouts": [], "poll_after_ms": 5_000}
        return {"accepted": True}


def _rollout(*, lease_token: str | None = "lease-1") -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_one",
        turn_input="do it",
        llm_url="https://gateway.invalid/v1",
        api_key="rollout-key",
        artifacts_dir="rollouts/rl_one",
        deadline_ts=9999999999.0,
        lease_token=lease_token,
    )


def test_successful_exact_registration_binds_later_claims() -> None:
    client = _Client()
    transport = AttachTransport(client, "customer-agent", wait_ms=0)
    transport.register({
        "agent_id": "customer-agent",
        "mode": "rollout",
        "agent_execution_sha256": EXPECTED,
    })
    transport.claim(1)

    claim = client.calls[-1]
    assert claim[:2] == ("POST", "/rollouts/claim")
    assert claim[2]["json_body"]["agent_execution_sha256"] == EXPECTED


def test_unknown_or_legacy_registration_does_not_claim_an_identity() -> None:
    client = _Client()
    transport = AttachTransport(client, "customer-agent", wait_ms=0)
    transport.register({"agent_id": "customer-agent", "mode": "rollout"})
    transport.claim(1)

    assert "agent_execution_sha256" not in client.calls[-1][2]["json_body"]


def test_failed_registration_cannot_activate_the_claim_identity() -> None:
    class _Failing(_Client):
        def request(self, method: str, path: str, **kwargs):
            if path == "/agents":
                raise RuntimeError("registration refused")
            return super().request(method, path, **kwargs)

    transport = AttachTransport(_Failing(), "customer-agent", wait_ms=0)
    try:
        transport.register({
            "agent_id": "customer-agent",
            "mode": "rollout",
            "agent_execution_sha256": EXPECTED,
        })
    except RuntimeError:
        pass
    else:  # pragma: no cover - the fake above always raises
        raise AssertionError("registration unexpectedly succeeded")
    assert transport.agent_execution_sha256 is None


def test_report_carries_the_current_lease_outside_agent_authored_payload() -> None:
    client = _Client()
    transport = AttachTransport(client, "customer-agent")
    output = RolloutOutput(
        final_text="done",
        agent_fingerprint={"execution_sha256": EXPECTED},
    )
    transport.report(_rollout(), output)

    report = client.calls[-1]
    assert report[:2] == ("POST", "/rollouts/rl_one/result")
    assert report[2]["headers"] == {
        "X-Percepteye-Lease-Token": "lease-1"
    }
    assert "lease_token" not in report[2]["json_body"]


def test_ordinary_report_without_a_server_lease_remains_compatible() -> None:
    client = _Client()
    transport = AttachTransport(client, "customer-agent")
    transport.report(_rollout(lease_token=None), RolloutOutput(final_text="done"))
    assert client.calls[-1][2]["headers"] is None
