"""Canonical, framework-neutral identity for one agent execution descriptor.

The digest produced here names facts the SDK reconciled at registration.  It
is deliberately not a hash of a framework object or of ``repr()`` output:
those routinely contain process addresses and make equal executions look
different.  Adapter-owned immutable inputs arrive only as opaque SHA-256
components, so core does not need to understand any particular harness,
runtime, or customer agent.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Mapping
from typing import Any, Protocol

from .errors import ConfigurationError

__all__ = [
    "ExecutionChild",
    "ExecutionSnapshot",
    "canonical_observation_sha256",
    "execution_descriptor_sha256",
    "is_lower_sha256",
    "normalize_attested_components",
]

_OBSERVATION_DOMAIN = b"percepteye-agent-observation-v1\0"
_EXECUTION_DOMAIN = b"percepteye-agent-execution-v1\0"
_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}")


class ExecutionChild(Protocol):
    """One adapter-prepared child view used for exactly one execution.

    The SDK overlays ``environment`` only into an isolated child process. An
    in-process Python entrypoint can use an execution snapshot for lifecycle
    attestation, but its prepared child must expose an empty mapping because
    mutating process-global environment would cross-contaminate rollouts.
    """

    @property
    def environment(self) -> Mapping[str, str]: ...

    @property
    def discovery_environment(self) -> Mapping[str, str]:
        """Non-secret layout values a command warm-up may receive."""
        ...


class ExecutionSnapshot(Protocol):
    """Framework-neutral lifecycle and exact evidence supplied by an adapter.

    Core treats component names and digests as opaque. The adapter owns the
    meaning of its execution state, prepares a private child view when needed,
    and attests that state after use. ``execution_identity_complete`` asserts
    that the component set covers implementation/state that discovery cannot
    observe; an incomplete snapshot never yields an OPSD-eligible identity.
    """

    @property
    def adapter_name(self) -> str: ...

    @property
    def execution_components(self) -> Mapping[str, str]: ...

    @property
    def execution_identity_complete(self) -> bool: ...

    @property
    def diagnostic_fingerprint(self) -> Mapping[str, str]: ...

    def prepare_child(self, label: str = "rollout") -> ExecutionChild: ...

    def attest_child(self, child: ExecutionChild) -> None: ...

    def release_child(self, child: ExecutionChild) -> None: ...


_MAX_SAFE_INTEGER = (1 << 53) - 1


def _unicode_string(value: str, *, path: str) -> str:
    for char in value:
        if 0xD800 <= ord(char) <= 0xDFFF:
            raise ConfigurationError(
                f"execution identity contains invalid Unicode at {path}"
            )
    return value


def _f64_hex(value: int | float, *, path: str) -> str:
    if isinstance(value, int) and abs(value) > _MAX_SAFE_INTEGER:
        raise ConfigurationError(
            f"execution identity integer is outside the exact JSON range at {path}"
        )
    try:
        number = float(value)
    except OverflowError as exc:
        raise ConfigurationError(
            f"execution identity number is outside the exact JSON range at {path}"
        ) from exc
    if not math.isfinite(number):
        raise ConfigurationError(
            f"execution identity contains a non-finite number at {path}"
        )
    if number.is_integer() and abs(number) > _MAX_SAFE_INTEGER:
        raise ConfigurationError(
            f"execution identity integer is outside the exact JSON range at {path}"
        )
    # JSON has one zero. Normalizing signed zero avoids a Python/JavaScript
    # digest split while preserving every other IEEE-754 value exactly.
    if number == 0:
        number = 0.0
    return struct.pack(">d", number).hex()


def _json_value(value: Any, *, path: str = "$") -> Any:
    """Return one typed, cross-language canonical value."""
    if value is None:
        return ["null"]
    if isinstance(value, str):
        return ["string", _unicode_string(value, path=path)]
    if isinstance(value, bool):
        return ["boolean", value]
    if isinstance(value, int):
        return ["number_f64", _f64_hex(value, path=path)]
    if isinstance(value, float):
        return ["number_f64", _f64_hex(value, path=path)]
    if isinstance(value, (list, tuple)):
        return [
            "array",
            [
                _json_value(item, path=f"{path}[{index}]")
                for index, item in enumerate(value)
            ],
        ]
    if isinstance(value, Mapping):
        out: list[list[Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigurationError(
                    f"execution identity contains a non-string object key at {path}"
                )
            _unicode_string(key, path=f"{path} key")
            out.append([key, _json_value(item, path=f"{path}.{key}")])
        out.sort(key=lambda pair: pair[0])
        return ["object", out]
    raise ConfigurationError(
        f"execution identity contains unsupported {type(value).__name__} at {path}"
    )


def _canonical_bytes(value: Any) -> bytes:
    normalized = _json_value(value)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(domain: bytes, value: Any) -> str:
    h = hashlib.sha256()
    h.update(domain)
    h.update(_canonical_bytes(value))
    return h.hexdigest()


def canonical_observation_sha256(value: Any) -> str:
    """Hash a full, pre-projection discovery observation.

    Hosts use this before truncating a long prompt for the registration wire.
    Two prompts that share the first 20,000 characters therefore cannot acquire
    the same affirmative execution identity.
    """
    return _digest(_OBSERVATION_DOMAIN, value)


def normalize_attested_components(
    components: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate adapter-owned opaque evidence without interpreting its layout."""
    if components is None:
        return {}
    if not isinstance(components, Mapping):
        raise ConfigurationError("attested execution components must be a mapping")
    out: dict[str, str] = {}
    for name, digest in components.items():
        if not isinstance(name, str) or not name.strip():
            raise ConfigurationError("attested execution component names must be non-empty strings")
        if name == "execution_sha256":
            raise ConfigurationError(
                "attested execution components must not author execution_sha256"
            )
        _unicode_string(name, path="attested component name")
        if not isinstance(digest, str) or _LOWER_SHA256.fullmatch(digest) is None:
            raise ConfigurationError(
                f"attested execution component {name!r} must be lowercase 64-hex SHA-256"
            )
        out[name] = digest
    return dict(sorted(out.items()))


def is_lower_sha256(value: Any) -> bool:
    """Whether ``value`` is the one accepted canonical digest spelling."""
    return isinstance(value, str) and _LOWER_SHA256.fullmatch(value) is not None


def execution_descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    """Return the canonical lowercase SHA-256 for a reconciled descriptor."""
    return _digest(_EXECUTION_DOMAIN, descriptor)
