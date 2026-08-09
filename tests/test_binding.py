"""Binding: two outcomes, and the refusal is the point."""
from __future__ import annotations

import pytest

from percepteye_harness.binding import (
    Adapter,
    AdapterCapabilities,
    bind_all,
    probe,
)
from percepteye_harness.errors import RoutingNotInstallable


class _Fake:
    """An adapter over this package itself, so the probe has real metadata."""

    name = "fake"
    distribution = "percepteye-harness"
    supported = ""
    required_symbols = ("percepteye_harness.contract:ToolCallOutcome",)

    def __init__(self, **over):
        self.__dict__.update(over)
        self.bound_with: dict | None = None

    def capabilities(self):
        return AdapterCapabilities(routes_llm_calls=True, reports_outcomes=True)

    def bind(self, *, llm_url, api_key):
        self.bound_with = {"llm_url": llm_url, "api_key": api_key}
        return lambda: None


def test_an_adapter_over_a_real_distribution_probes_clean():
    r = probe(_Fake())
    assert r.ok and r.status == "detected" and r.version


def test_an_absent_distribution_is_typed_not_a_bare_false():
    r = probe(_Fake(distribution="definitely-not-installed-xyz"))
    assert r.status == "absent"
    assert "definitely-not-installed-xyz" in r.reason


def test_a_missing_symbol_fails_detection():
    """THE structural fix. The old probe checked a symbol that existed in no
    released version, so 'detected' meant nothing and the patch silently did
    nothing. Detection cannot pass while binding is broken because both read
    the same tuple."""
    r = probe(_Fake(required_symbols=("percepteye_harness.contract:NoSuchThing",)))
    assert r.status == "symbol_missing"
    assert "NoSuchThing" in r.evidence


def test_a_missing_module_is_absent_not_symbol_missing():
    r = probe(_Fake(required_symbols=("no_such_module_xyz:Thing",)))
    assert r.status == "absent"


def test_an_out_of_range_version_refuses():
    pytest.importorskip("packaging")
    r = probe(_Fake(supported=">=99.0"))
    assert r.status == "unsupported_version"
    assert r.version


# ── bind_all ────────────────────────────────────────────────────────────────


def test_binding_installs_routing_and_reports_capabilities():
    a = _Fake()
    report, unbinds = bind_all([a], llm_url="https://gw/x", api_key="perk_1")
    assert report.bound == ["fake"]
    assert a.bound_with == {"llm_url": "https://gw/x", "api_key": "perk_1"}
    assert report.capabilities.routes_llm_calls is True
    assert len(unbinds) == 1


def test_an_adapter_that_cannot_bind_is_a_refusal_not_a_downgrade():
    """A customer who supplies an adapter is asserting that the adapter is how
    their traffic gets routed. There is no degraded-routing state, because the
    failure destination is a live third party and the rollout still
    completes."""
    with pytest.raises(RoutingNotInstallable) as exc:
        bind_all([_Fake(required_symbols=("percepteye_harness.contract:Nope",))],
                 llm_url="u", api_key="k")
    assert exc.value.reason == "SymbolMissing"
    assert "Nope" in exc.value.evidence
    assert exc.value.remedy, "a refusal must say what to do about it"


def test_the_remedy_never_suggests_pinning_around_a_missing_symbol():
    with pytest.raises(RoutingNotInstallable) as exc:
        bind_all([_Fake(required_symbols=("percepteye_harness.contract:Nope",))],
                 llm_url="u", api_key="k")
    assert "do not pin around it" in exc.value.remedy


def test_no_adapters_is_the_default_tier_and_is_recorded():
    """Routing by child environment needs no adapter. The absence is recorded
    rather than assumed, so a report always says what happened."""
    report, unbinds = bind_all([], llm_url="u", api_key="k")
    assert report.bound == []
    assert unbinds == []
    assert report.skipped and report.skipped[0].reason.startswith("no adapters")


def test_capabilities_are_the_union_across_adapters():
    class RoutesOnly(_Fake):
        name = "routes"

        def capabilities(self):
            return AdapterCapabilities(routes_llm_calls=True, reports_outcomes=False)

    report, _ = bind_all([RoutesOnly()], llm_url="u", api_key="k")
    assert report.capabilities.routes_llm_calls is True
    assert report.capabilities.reports_outcomes is False


def test_the_protocol_is_structural():
    assert isinstance(_Fake(), Adapter)
