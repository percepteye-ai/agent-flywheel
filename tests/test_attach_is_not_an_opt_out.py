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
