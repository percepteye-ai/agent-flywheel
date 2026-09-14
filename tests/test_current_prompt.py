"""The prompt half of the last mile.

``current_policy`` answers "which model". Training also produces a system
PROMPT, co-optimized with that model and gated by the same human approval — and
until this existed an agent could learn only THAT the prompt had moved, never
what it moved to. ``PolicySnapshot``'s own comment says it: the generation
changes for a prompt change, "which this snapshot does not carry and a client
rebuild would not apply".

Two properties carry the weight here and neither is "it returns the text":

* ``{}`` means KEEP WHAT YOU HAVE. Never "use no prompt". A control plane that
  cannot answer must not blank a running agent's instructions.
* It does NOT apply anything. Swapping a model endpoint is invisible to an
  agent's behaviour; swapping its system prompt changes what it does, which is
  a decision for whoever owns the agent.
"""
from __future__ import annotations

import agent_flywheel as fw
from agent_flywheel.transport import AttachTransport

SERVED = {
    "name": "support-triage", "version": 3, "sha256": "c" * 64,
    "text": "You triage carefully.", "candidate_kind": "replacement",
    "status": "approved", "model_prompt_checksum": "c" * 64,
    "stale_baseline": False,
}


class _Client:
    """Stands in for ControlPlaneClient. Records the path it was asked for."""

    def __init__(self, reply=None, raises=False):
        self.reply, self.raises, self.paths, self.closed = reply, raises, [], False

    def request(self, method, path, **kw):
        self.paths.append((method, path))
        if self.raises:
            raise RuntimeError("control plane is down")
        return self.reply

    def close(self):
        self.closed = True


def _tr(**kw):
    return AttachTransport(_Client(**kw), agent_id="a")


def test_the_text_reaches_the_agent():
    assert _tr(reply=SERVED).prompt_current()["text"] == "You triage carefully."


def test_it_asks_the_prompt_route_not_the_policy_route():
    """A separate call on purpose: /policy/current is POLLED and
    identity-sized, and a system prompt is unbounded text."""
    c = _Client(reply=SERVED)
    AttachTransport(c, agent_id="a").prompt_current()
    assert c.paths == [("GET", "/prompt/current")]


def test_an_unreachable_control_plane_returns_empty_and_does_not_raise():
    """The whole fail-open contract. An exception here would take down a
    customer's agent over a deployment that has nothing to say to it."""
    assert _tr(raises=True).prompt_current() == {}


def test_a_null_body_is_empty_not_None():
    """`{}` is the one 'keep what you have' value; None would make every
    caller write a second check."""
    assert _tr(reply=None).prompt_current() == {}


def test_current_prompt_without_credentials_is_empty(monkeypatch):
    """No URL or key means nothing to ask, not a crash at import time."""
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    assert fw.current_prompt(agent_id="a") == {}


def test_it_is_exported_because_a_customer_has_to_call_it():
    assert "current_prompt" in fw.__all__


def test_it_does_NOT_apply_the_prompt_anywhere():
    """apply_current_policy exists for the model because an endpoint swap is
    invisible to behaviour. There is deliberately no apply_current_prompt: a
    prompt swap changes what the agent DOES, and that is the customer's call.
    """
    assert not hasattr(fw, "apply_current_prompt")


def test_the_pair_identity_survives_to_the_caller():
    """The contract is a PAIR. A prompt applied over a different checkpoint
    than the one it was certified against is not what was approved, and the
    caller can only notice if the field arrives."""
    out = _tr(reply=SERVED).prompt_current()
    assert out["model_prompt_checksum"] == "c" * 64
    assert out["stale_baseline"] is False
