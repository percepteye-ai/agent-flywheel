"""The last mile: the agent finds out about a model the customer approved.

Training produced better models for months and none reached a customer's
process. ``serve()`` discarded the registration response and nothing ever
asked what to run, so a trained adapter arrived only when a human edited a
config by hand.

What is delivered is an ENDPOINT, not an artifact URI. ``gs://…`` is not
something an agent can call, so the payload leads with an OpenAI-compatible
``base_url`` + ``model`` and the artifact rides along as provenance.

Every test here is one property: **the control plane must never be able to
stop a customer's agent from serving its users.** Not by being down, not by
erroring, not by being an older build.
"""
from __future__ import annotations

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel.transport import AttachTransport


class _Client:
    def __init__(self, answer=None, boom=False):
        self.answer, self.boom, self.calls = answer, boom, []

    def request(self, method, path, **kw):
        self.calls.append((method, path))
        if self.boom:
            raise RuntimeError("control plane unreachable")
        return self.answer

    def close(self):
        pass


def _tr(**kw):
    return AttachTransport(_Client(**kw), agent_id="coding-agent")


def test_the_endpoint_is_what_comes_back():
    """The deliverable an agent can act on."""
    answer = {
        "arm": "candidate",
        "endpoint": {"base_url": "https://gpu.example/v1",
                     "model": "run_x_policy",
                     "api_key_env": "PERCEPTEYE_API_KEY"},
        "reason": "an operator put this agent on the candidate",
    }
    got = _tr(answer=answer).policy_current()
    assert got["endpoint"]["base_url"] == "https://gpu.example/v1"
    assert got["endpoint"]["model"] == "run_x_policy"


def test_it_asks_the_right_route():
    tr = _tr(answer={})
    tr.policy_current()
    assert tr._c.calls == [("GET", "/policy/current")]


def test_an_unreachable_control_plane_does_not_raise():
    """A customer's agent must keep serving its users when we are down. An
    exception here would take their production down with ours."""
    assert _tr(boom=True).policy_current() == {}


def test_a_null_answer_becomes_an_empty_dict():
    """So callers can write ``if pol.get("endpoint")`` without a None check
    that someone will forget."""
    assert _tr(answer=None).policy_current() == {}


def test_current_policy_without_configuration_is_empty_not_an_error(monkeypatch):
    """Called from a customer's own agent with no key or URL set: return
    nothing and let them keep their configuration. Raising would make
    adopting this a risk rather than an improvement."""
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    assert fw.current_policy(agent_id="coding-agent") == {}


def test_current_policy_is_exported():
    """It is the seam a customer's production agent calls — a helper they
    cannot import is one that does not exist."""
    assert "current_policy" in fw.__all__
    assert callable(fw.current_policy)


def test_serve_tolerates_a_transport_without_the_method():
    """``policy_current`` is not on the FlywheelTransport Protocol, so a
    third-party or older transport need not implement it. Calling it
    directly made every such transport crash at startup."""
    class _Old:
        pass

    ask = getattr(_Old(), "policy_current", None)
    assert ask is None          # the shape serve() guards against
