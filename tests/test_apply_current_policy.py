"""The last mile: turning the policy answer into something a client can use.

current_policy already returned the right answer; every caller had to unpack it
the same way, and a caller who unpacks it differently is a caller sampling from
something other than what was trained.
"""
from __future__ import annotations

import percepteye_agent_flywheel as fw


def _patch(monkeypatch, policy):
    monkeypatch.setattr(fw, "current_policy", lambda **kw: policy)


_EP = {"base_url": "https://gpu:8001/v1", "model": "policy_acme",
       "api_key_env": "PERCEPTEYE_API_KEY"}


def test_it_returns_client_kwargs(monkeypatch):
    _patch(monkeypatch, {"arm": "candidate", "endpoint": _EP})
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "k")
    out = fw.apply_current_policy(agent_id="a")
    assert out["base_url"] == "https://gpu:8001/v1"
    assert out["model"] == "policy_acme"
    assert out["api_key"] == "k"


def test_nothing_deployed_yields_nothing_to_apply(monkeypatch):
    """Not a guessed default. 'Keep your current configuration' is the only
    safe answer when the control plane has nothing to hand you."""
    _patch(monkeypatch, {"arm": "base"})
    assert fw.apply_current_policy(agent_id="a") == {}


def test_an_unreachable_control_plane_is_fail_open(monkeypatch):
    _patch(monkeypatch, {})
    assert fw.apply_current_policy(agent_id="a") == {}


def test_an_endpoint_we_cannot_authenticate_to_is_not_returned(monkeypatch):
    """Returning it would produce 401s at the caller's first request rather
    than a legible 'nothing to apply' here."""
    _patch(monkeypatch, {"endpoint": _EP})
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    assert fw.apply_current_policy(agent_id="a") == {}


def test_env_export_is_opt_in(monkeypatch):
    """Mutating the process environment is a side effect a caller asks for."""
    _patch(monkeypatch, {"endpoint": _EP})
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "k")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    fw.apply_current_policy(agent_id="a")
    assert "OPENAI_BASE_URL" not in __import__("os").environ
    fw.apply_current_policy(agent_id="a", set_env=True)
    assert __import__("os").environ["OPENAI_BASE_URL"] == "https://gpu:8001/v1"


def test_serve_still_only_REPORTS_the_policy():
    """A serve rollout samples through the per-rollout gateway url — that is
    what attributes and captures the completion. Applying the policy endpoint
    there would bypass capture and produce a run with nothing to train on."""
    import inspect
    src = inspect.getsource(fw.serve)
    assert "INFORMATIONAL ONLY" in src
    # It may NAME the helper in a comment; it must not APPLY the endpoint.
    assert "OPENAI_BASE_URL" not in src
    assert "= apply_current_policy(" not in src
