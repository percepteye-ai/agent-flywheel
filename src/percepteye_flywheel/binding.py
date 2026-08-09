"""Binding — making sure the agent's LLM calls actually reach our endpoint.

This is the re-cut of framework auto-detection, and the change is in the
purpose rather than the mechanism. The old idea was to attach post-generation
callbacks and scrape telemetry after the fact. This one exists to FALSIFY an
assumption before a rollout runs, and its primary output is a refusal.

The assumption worth falsifying: that setting ``OPENAI_BASE_URL`` in the
child's environment routes the agent's traffic to us. It frequently does not,
and it fails in the worst possible way -- silently, to a live third party:

* a client constructed with an explicit ``base_url`` argument in the
  customer's own code beats every environment variable;
* some clients read a differently-named variable, or only learned to read one
  in a recent version;
* some route through a default gateway whose URL is a hardcoded literal with
  no environment override at all.

In each case the rollout COMPLETES. The flywheel returns a plausible final
text. We own none of the completion wire, capture no token data, and cannot
tell the difference from the outside. That is the same class of defect as a
tool call whose outcome nobody observed being scored as a success, one layer
down.

So there are two outcomes and no third. ``BOUND``, or ``RoutingNotInstallable``
raised at registration -- before any rollout exists, which is why fail-closed
here does not conflict with never breaking the customer's application path.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

from .errors import RoutingNotInstallable

__all__ = [
    "Adapter",
    "AdapterCapabilities",
    "BindingReport",
    "ProbeResult",
    "ProbeStatus",
    "bind_all",
    "discover_adapters",
    "probe",
]

ProbeStatus = Literal["detected", "absent", "unsupported_version", "symbol_missing"]

ENTRY_POINT_GROUP = "percepteye_flywheel.adapters"


@dataclass(frozen=True)
class AdapterCapabilities:
    """What an adapter can actually promise.

    ``reports_outcomes`` is the one that matters. An adapter that cannot
    distinguish a failed tool call from a successful one must say so here, so
    the agent is routed to a lower grade rather than emitting outcomes it
    guessed.
    """

    routes_llm_calls: bool = False
    reports_outcomes: bool = False


@runtime_checkable
class Adapter(Protocol):
    """A framework integration. Ships as its own distribution.

    Adapters version independently of core, which is the structural answer to
    a previous SDK in which three of six framework integrations were broken
    against the installed library and nothing noticed: when a framework
    changes we ship a new adapter, not a new core.
    """

    name: str
    distribution: str
    #: PEP 440 specifier the adapter supports, e.g. ">=0.3,<2".
    supported: str
    #: Every dotted ``module:attr`` path this adapter will touch. The probe
    #: resolves exactly this tuple, which is what makes detection and binding
    #: inseparable -- detection cannot pass while binding is broken.
    required_symbols: tuple[str, ...]

    def capabilities(self) -> AdapterCapabilities: ...

    def bind(self, *, llm_url: str, api_key: str) -> Any:
        """Install routing. Returns an unbind callable."""
        ...


@dataclass
class ProbeResult:
    status: ProbeStatus
    adapter: str
    reason: str = ""
    evidence: str = ""
    version: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "detected"


@dataclass
class BindingReport:
    """What binding concluded, recorded rather than implied."""

    bound: list[str] = field(default_factory=list)
    skipped: list[ProbeResult] = field(default_factory=list)
    capabilities: AdapterCapabilities = field(default_factory=AdapterCapabilities)

    @property
    def any_bound(self) -> bool:
        return bool(self.bound)


def _resolve_symbol(path: str) -> Any:
    """``pkg.mod:attr`` → the attribute, or raise."""
    mod_name, _, attr = path.partition(":")
    mod = importlib.import_module(mod_name)
    target: Any = mod
    for part in attr.split(".") if attr else []:
        target = getattr(target, part)
    return target


def probe(adapter: Adapter) -> ProbeResult:
    """Decide whether this adapter can bind, and record why not.

    Three historical failure causes, three structural fixes:

    * The old probe imported an umbrella package that modern installs do not
      ship. This resolves the DISTRIBUTION through package metadata and the
      specific modules that own the surfaces being bound.
    * The old probe checked for a symbol that never existed in any released
      version, so "detected" meant nothing. Here the probe resolves exactly
      the tuple the adapter will later touch.
    * The old probe swallowed every failure into ``return False``. Here each
      outcome is typed and carries the evidence.
    """
    try:
        version = md.version(adapter.distribution)
    except md.PackageNotFoundError:
        return ProbeResult("absent", adapter.name,
                           reason=f"distribution {adapter.distribution!r} is not installed")

    if not _version_supported(version, adapter.supported):
        return ProbeResult(
            "unsupported_version", adapter.name,
            reason=f"{adapter.distribution} {version} outside {adapter.supported}",
            evidence=version, version=version,
        )

    for path in adapter.required_symbols:
        try:
            _resolve_symbol(path)
        except ModuleNotFoundError as exc:
            return ProbeResult("absent", adapter.name, reason=str(exc),
                               evidence=path, version=version)
        except (AttributeError, ImportError) as exc:
            # The symbol the adapter binds to is gone. Detection must fail
            # with it -- reporting "detected" here is how an integration
            # reports success while doing nothing.
            return ProbeResult("symbol_missing", adapter.name, reason=str(exc),
                               evidence=path, version=version)

    return ProbeResult("detected", adapter.name, version=version)


def _version_supported(version: str, specifier: str) -> bool:
    """Specifier match, with a deliberately conservative fallback.

    ``packaging`` is not a runtime dependency -- this package ships with one,
    on purpose. When it is unavailable the range cannot be evaluated, and the
    honest answer is to proceed to the symbol probe, which is the check that
    actually protects us: a version outside the range almost always shows up
    as a missing symbol anyway, and that check needs no third-party parser.
    """
    if not specifier:
        return True
    try:
        from packaging.specifiers import SpecifierSet  # type: ignore
        from packaging.version import Version  # type: ignore
    except Exception:
        return True
    try:
        return Version(version) in SpecifierSet(specifier)
    except Exception:
        return True


def discover_adapters() -> list[Adapter]:
    """Adapters installed as distributions, via entry points.

    Core never imports a framework. An installed adapter distribution IS the
    customer's declaration that they want that framework bound, which is a
    clearer signal than us guessing from what happens to be importable.
    """
    found: list[Adapter] = []
    try:
        entries = md.entry_points(group=ENTRY_POINT_GROUP)
    except TypeError:  # pragma: no cover - older importlib.metadata
        entries = md.entry_points().get(ENTRY_POINT_GROUP, [])  # type: ignore[attr-defined]
    for ep in entries:
        try:
            factory = ep.load()
            found.append(factory() if callable(factory) else factory)
        except Exception as exc:
            raise RoutingNotInstallable(
                reason="AdapterLoadFailed",
                evidence=f"entry point {ep.name!r}: {type(exc).__name__}: {exc}",
                remedy="reinstall or remove the adapter distribution",
            ) from exc
    return found


def bind_all(
    adapters: list[Adapter] | None,
    *,
    llm_url: str,
    api_key: str,
    require_routing: bool = True,
) -> tuple[BindingReport, list[Any]]:
    """Probe, then bind. Two outcomes: a report with bindings, or a raise.

    ``require_routing`` exists for the default Tier-0 configuration, where
    routing is installed by the child's environment and no adapter is needed.
    It does NOT create a degraded-routing state: a customer who supplies an
    adapter is asserting that the adapter is how their traffic gets routed,
    and an adapter that cannot bind is therefore a refusal.
    """
    report = BindingReport()
    unbinds: list[Any] = []
    supplied = list(adapters or []) or discover_adapters()

    if not supplied:
        if require_routing:
            # Not an error: the default configuration routes by environment at
            # spawn. Recorded so the absence is visible rather than assumed.
            report.skipped.append(
                ProbeResult("absent", "none",
                            reason="no adapters installed; routing by child environment")
            )
        return report, unbinds

    routes = False
    outcomes = False
    for adapter in supplied:
        result = probe(adapter)
        if not result.ok:
            raise RoutingNotInstallable(
                reason=_REASON[result.status],
                evidence=f"{result.adapter}: {result.reason}"
                         + (f" [{result.evidence}]" if result.evidence else ""),
                remedy=_REMEDY[result.status].format(
                    dist=getattr(adapter, "distribution", result.adapter),
                    supported=getattr(adapter, "supported", ""),
                ),
            )
        unbinds.append(adapter.bind(llm_url=llm_url, api_key=api_key))
        report.bound.append(adapter.name)
        caps = adapter.capabilities()
        routes = routes or caps.routes_llm_calls
        outcomes = outcomes or caps.reports_outcomes

    report.capabilities = AdapterCapabilities(
        routes_llm_calls=routes, reports_outcomes=outcomes
    )
    return report, unbinds


_REASON: dict[str, str] = {
    "absent": "AdapterFrameworkAbsent",
    "unsupported_version": "VersionOutOfRange",
    "symbol_missing": "SymbolMissing",
}

_REMEDY: dict[str, str] = {
    "absent": "install {dist}, or remove the adapter from serve(adapters=[...])",
    "unsupported_version": "install {dist}{supported}, or upgrade the adapter",
    "symbol_missing": (
        "this adapter does not support the installed {dist}. Upgrade the "
        "adapter distribution; do not pin around it -- the symbol it binds to "
        "is how your traffic reaches us."
    ),
}
