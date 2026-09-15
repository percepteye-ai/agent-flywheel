"""Canonical, framework-neutral identity for one agent execution descriptor.

The digest produced here names facts the SDK reconciled at registration.  It
is deliberately not a hash of a framework object or of ``repr()`` output:
those routinely contain process addresses and make equal executions look
different.  Adapter-owned immutable inputs arrive only as opaque SHA-256
components, so core does not need to understand any particular harness,
runtime, or customer agent.

A command entrypoint's argv is the one place where a per-process value can
still reach the digest through an identity fact: an agent launched with
``--daemon-socket /tmp/agent-<pid>.sock`` is the same agent on every restart,
but its argv is not the same argv.  The host therefore DECLARES which option
values are transport, by name, and :func:`canonical_command_argv` replaces
exactly those values with a marker naming the option they belonged to.  The
option token itself survives, so "runs with a socket" and "runs without one"
stay different executions.  Nothing here guesses: an undeclared temp path,
port or PID is identity, because the alternative is a heuristic that erases
real identity in the cases it gets wrong.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, Protocol

from .errors import ConfigurationError

__all__ = [
    "ExecutionChild",
    "ExecutionSnapshot",
    "canonical_command_argv",
    "canonical_observation_sha256",
    "execution_descriptor_sha256",
    "is_lower_sha256",
    "normalize_attested_components",
    "normalize_volatile_command_values",
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


#: Distinguishes "no sentinel was supplied" from any value a caller could pass,
#: so an argv item can never accidentally compare equal to the absent sentinel.
_NO_SENTINEL = object()


def normalize_volatile_command_values(
    options: Iterable[str] | None,
) -> tuple[str, ...]:
    """Validate a declaration of which command option VALUES are not identity.

    Only the shape of the declaration is checked here.  Whether each option is
    actually present in a command is checked by :func:`canonical_command_argv`,
    which is the only place the command is known.
    """
    if options is None:
        return ()
    if isinstance(options, (str, bytes, Mapping)):
        raise ConfigurationError(
            "volatile command values must be a sequence of option names, e.g. "
            "('--daemon-socket',), not a single string or mapping"
        )
    out: list[str] = []
    for option in options:
        if not isinstance(option, str) or not option.strip():
            raise ConfigurationError(
                "volatile command value entries must be non-empty option names"
            )
        _unicode_string(option, path="volatile command option")
        if option not in out:
            out.append(option)
    return tuple(out)


def canonical_command_argv(
    command: Sequence[Any],
    *,
    volatile_value_options: Sequence[str] = (),
    task_sentinel: Any = _NO_SENTINEL,
) -> list[dict[str, Any]]:
    """Project a command entrypoint into canonical identity argv entries.

    Every token is an identity fact by default.  A declared option keeps its
    own token and loses only its value, which becomes ``{"volatile_value":
    <option>}``: the option's PRESENCE stays in the digest while the
    per-process path, port or handle it carried does not.

    ``--opt value`` and ``--opt=value`` are both recognised, because a host
    that spells it the second way would otherwise be told its option is not
    present.  The two spellings are different argv and stay different
    identities; neither is rewritten into the other.

    With nothing declared this returns exactly what it returned before the
    declaration existed, so an agent that declares nothing keeps its digest.
    """
    if not command:
        raise ConfigurationError("a command entrypoint must have at least one argument")
    declared = tuple(volatile_value_options)
    program = command[0]
    for option in declared:
        if program is not task_sentinel and option == str(program):
            raise ConfigurationError(
                f"volatile command value {option!r} is argv[0], the program itself. "
                f"Voiding it would erase the entrypoint from its own identity; "
                f"name an option whose value is transport instead."
            )

    entries: list[dict[str, Any]] = []
    voided: set[str] = set()
    index = 0
    while index < len(command):
        item = command[index]
        if item is task_sentinel:
            entries.append({"task_input": True})
            index += 1
            continue
        text = str(item)
        joined = next((o for o in declared if text.startswith(f"{o}=")), None)
        if joined is not None:
            entries.append({"volatile_value": joined})
            voided.add(joined)
            index += 1
            continue
        if text in declared:
            if index + 1 >= len(command):
                raise ConfigurationError(
                    f"volatile command value {text!r} is the last argument and "
                    f"carries no value to void"
                )
            if command[index + 1] is task_sentinel:
                raise ConfigurationError(
                    f"volatile command value {text!r} is followed by the task "
                    f"input sentinel; voiding it would erase where each "
                    f"rollout's task is substituted"
                )
            entries.append({"literal": text})
            entries.append({"volatile_value": text})
            voided.add(text)
            index += 2
            continue
        entries.append({"literal": text})
        index += 1

    missing = [option for option in declared if option not in voided]
    if missing:
        raise ConfigurationError(
            "volatile command values name "
            + ", ".join(repr(option) for option in missing)
            + ", which this command entrypoint does not contain. Declare each "
            "option exactly as the command spells it ('--opt value' or "
            "'--opt=value'); a declaration that matches nothing would silently "
            "stop covering the value it was added for."
        )
    return entries


def is_lower_sha256(value: Any) -> bool:
    """Whether ``value`` is the one accepted canonical digest spelling."""
    return isinstance(value, str) and _LOWER_SHA256.fullmatch(value) is not None


def execution_descriptor_sha256(descriptor: Mapping[str, Any]) -> str:
    """Return the canonical lowercase SHA-256 for a reconciled descriptor."""
    return _digest(_EXECUTION_DOMAIN, descriptor)
