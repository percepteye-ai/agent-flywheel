"""Built-in framework adapters.

These live here rather than in separate distributions. The earlier argument
for splitting them out -- that it was the structural answer to an SDK whose
framework integrations broke silently -- does not survive inspection: the
thing that actually prevents that is the probe resolving exactly the symbols
an adapter will touch, which lives in ``binding`` and works the same either
way. Splitting would have bought a release cadence we do not need and
introduced version skew between core and adapter, which is the same class of
failure the design exists to remove.

No adapter imports its framework at module scope, so none of them is a
dependency. An adapter whose framework is absent simply fails its probe.

Third parties can still ship their own through the
``percepteye_harness.adapters`` entry-point group; that mechanism is
independent of where ours live.

**An adapter is acceleration, never a requirement.** Most harnesses need none:
routing arrives through the child's environment at spawn, and
``record_tool_call`` reports outcomes from any framework or from none. Writing
one is worth it only when a framework's own configuration beats the
environment -- so the default silently loses -- or when its outcome interface
saves the customer real instrumentation.
"""
from __future__ import annotations

from typing import Any

__all__ = ["BUILTIN_ADAPTERS", "load_builtin"]

#: name -> import path of a zero-argument factory.
BUILTIN_ADAPTERS: dict[str, str] = {
    "langchain": "percepteye_harness.adapters.langchain:LangChainAdapter",
}


def load_builtin(name: str, **kwargs: Any) -> Any:
    """Construct a built-in adapter by name."""
    import importlib

    if name not in BUILTIN_ADAPTERS:
        raise KeyError(
            f"unknown built-in adapter {name!r}; available: "
            f"{sorted(BUILTIN_ADAPTERS)}"
        )
    mod_path, _, attr = BUILTIN_ADAPTERS[name].partition(":")
    return getattr(importlib.import_module(mod_path), attr)(**kwargs)
