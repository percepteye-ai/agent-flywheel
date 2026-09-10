"""Deprecated compatibility import for the optional Prime Agent adapter.

Prime-specific execution plumbing lives in :mod:`.adapters.prime_agent` so
importing the SDK core does not load or depend on it.  This module remains for
customers that imported the original preview location directly.
"""
from __future__ import annotations

import warnings

from .adapters.prime_agent import PrimeHarnessChild, PrimeHarnessSnapshot

warnings.warn(
    "percepteye_agent_flywheel.harness is deprecated; import the optional "
    "Prime adapter from percepteye_agent_flywheel.adapters.prime_agent",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["PrimeHarnessChild", "PrimeHarnessSnapshot"]
