"""``attach()`` with no entrypoint registers "not described", not "opted out".

The control plane reads a registration three ways: a description, ``null``
(introspection could not describe the agent -> ``unreadable``), and an ABSENT
key (the customer opted out -> ``disabled``). ``disabled`` is the destructive
state: it is written onto the agent's row and stops the description a
``serve()`` registration stored from driving workflow generation.

``attach()`` without ``execution_entrypoint`` has no agent object to describe,
and it omitted the key -- so every production process that attached silently
turned off workflow generation for its agent. It now sends ``null``, as the
Node plugin does; only the customer's real opt-out, PERCEPTEYE_INTROSPECT=0,
still omits it.
"""
from __future__ import annotations

from tests.test_production_turn_upload import _attached, _Transport


def test_an_attach_without_an_entrypoint_says_not_described(monkeypatch, tmp_path):
    monkeypatch.delenv("PERCEPTEYE_INTROSPECT", raising=False)
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr, adapters=[]):
        pass
    assert "discovered_agent" in tr.registered
    assert tr.registered["discovered_agent"] is None


def test_the_customers_opt_out_still_omits_it(monkeypatch, tmp_path):
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr, adapters=[]):
        pass
    assert "discovered_agent" not in tr.registered


def test_the_opt_out_holds_with_an_entrypoint_too(monkeypatch, tmp_path):
    """With ``execution_entrypoint`` the agent used to be introspected, and its
    description -- system prompt included -- registered, under
    PERCEPTEYE_INTROSPECT=0. Consent is read once and governs all of it: no
    introspection, no description, and no identity derived from one."""
    from tests.test_production_turn_upload import _exact_description, _IdentityAgent

    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    looked: list = []
    monkeypatch.setattr(
        "agent_flywheel.introspection.introspect",
        lambda fn: looked.append(fn) or _exact_description(),
    )
    tr = _Transport()
    with _attached(monkeypatch, tmp_path, tr, adapters=[],
                   execution_entrypoint=_IdentityAgent(),
                   execution_input_shape="messages",
                   attested_execution_components={"customer.release": "a" * 64}) as px:
        assert px.agent_fingerprint is None
    assert looked == []
    assert "discovered_agent" not in tr.registered
