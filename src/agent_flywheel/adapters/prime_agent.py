"""Optional Prime Agent execution-snapshot adapter.

Prime Agent's continual harness is global by default: every process reads and
may refine ``~/.prime/agent/harness``.  That is useful interactively and is a
confound in an on-policy comparison group.  Two replicas of one prompt must
not see different memories, skills, or prompt addenda merely because another
replica finished first.

``PrimeHarnessSnapshot`` takes one exact snapshot at flywheel startup, gives
each child its own private copy, and attests the copy after the child exits.
The child never receives the master snapshot's path. A private-harness write
therefore invalidates only that rollout, while a write to shared attested
runtime state poisons the run and prevents a later child from silently seeing
different bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import struct
import subprocess
import tempfile
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..errors import ConfigurationError
from ..event_stream import (
    CommandOutputProjection,
    StreamFormat,
    parse_tool_calls_with_omissions,
)

# The TRANSLATION LAYER, and only that. The adapter asks core how long one
# blocking call may take for an episode of a given length; core knows nothing
# about Prime, settings.json, or any harness. Delete this adapter's ceiling and
# the runner still delivers every outcome; delete the runner's arithmetic and
# no adapter's ceiling saves goose, incident-lens, or a harness not yet written.
from ..runner import harness_call_ceiling_s

__all__ = [
    "PRIME_TOOL_FORMAT",
    "PrimeAgentOutputAdapter",
    "PrimeHarnessChild",
    "PrimeHarnessSnapshot",
    "unix_socket_directory",
]

_DIGEST_DOMAIN = b"percepteye-prime-harness-v1\0"
_KERNEL_DIGEST_DOMAIN = b"percepteye-prime-kernel-v1\0"
_ARTIFACT_DIGEST_DOMAIN = b"percepteye-prime-execution-artifact-v1\0"
_SETTINGS_DIGEST_DOMAIN = b"percepteye-prime-fixed-settings-v1\0"
_ROUTE_CONFIG_DIGEST_DOMAIN = b"percepteye-prime-route-config-v1\0"
_BUFFER_BYTES = 1 << 20
#: The settings EVERY Prime child runs under -- the daemon supervisor, the
#: discovery warm-up and every rollout -- written read-only and attested, so
#: the scaffold's configuration is part of its identity rather than ambient.
#:
#: ``retry`` is a BUDGET, not a preference. Prime routes completions through
#: the bundled OpenAI/Anthropic client, whose own default request timeout is
#: 600000ms (in the shipped bundle: ``TIMEOUT = 6e5``) and whose retries
#: multiply it. So a gateway that ACCEPTS the connection and then never
#: answers parks one completion for ten minutes, and a single retry carries it
#: past any episode budget this platform mints -- the agent is not slow, it is
#: stopped, and nothing in Prime's flag surface says otherwise.
#:
#: Measured 2026-09-15 against a loopback endpoint that accepted and never
#: replied: the rollout child emitted its session line, produced nothing more,
#: spent its ENTIRE budget in that one request, and was killed by the runner as
#: ``deadline_exceeded`` with no completion finished -- a wedged agent
#: indistinguishable from an idle one. With the ceiling below the same child
#: returns in one timeout and still has most of the episode left to report what
#: happened. That is a mechanism this deployment can reach; it is NOT a reading
#: of the three production rows of the same day, which are ``status=abandoned``
#: (reaper-written, so no report landed) and have no session line at all. See
#: the ``on_event`` comment in examples/prime_agent/serve_prime.py.
#:
#: ``--autonomous-timeout-ms`` is NOT this bound and must not be mistaken for
#: it. It is evaluated between turns, so it never fires while a turn is
#: blocked in the provider call: measured at 20000ms against the same endpoint,
#: the child still ran to the full budget. This setting is the only ceiling
#: that applies to the request itself.
#:
#: ONE generous attempt, and no multiplication. The ceiling is DERIVED, not
#: typed: ``runner.harness_call_ceiling_s`` divides the reference episode
#: budget the way the runner divides every real one and returns the share of
#: the child's window one blocking call may take. So this adapter holds a
#: reference budget -- a fact about this deployment -- and no arithmetic, and
#: raising the runner's delivery reserve tightens this ceiling instead of
#: silently invalidating it. Retries are OFF at both levels deliberately:
#: retrying a call that
#: already spent the whole ceiling re-hangs on the same wedged gateway and
#: multiplies straight through the episode, and a rollout that cannot reach
#: its provider is not made better by spending more of the episode finding
#: that out again. What was MEASURED at an 8000ms ceiling is that the child
#: ends its run cleanly on the first timeout and the runner reports it with a
#: complete Prime lifecycle -- an excluded rollout that SAYS WHY, in a tenth
#: of the budget, rather than a SIGKILL that says nothing in all of it.
#:
#: THE CEILING IS INERT BELOW A 894s EPISODE, and this deployment has minted
#: shorter ones -- run 08e40b36's rollouts had a 300s budget (see
#: examples/prime_agent/serve_prime.py). Below that floor the runner's own
#: child window is short enough that one wedged call can still consume the
#: episode. The floor is re-derived from the runner's arithmetic in
#: tests/test_a_wedged_provider_call_cannot_eat_the_episode.py, so it cannot
#: drift silently when either side moves -- and, since 2026-09-16, the runner
#: also SAYS SO AT RUNTIME: ``worst_case_call_s`` below is read by
#: ``RolloutRunner._warn_if_call_ceiling_is_inert``, which compares it against
#: the episode actually minted and warns on rollout #1. A floor in a comment
#: was only ever a promise about episodes nobody was checking.
#:
#: It CANNOT be made per-rollout, and that is not an omission. These bytes are
#: a frozen constant: their digest is ``prime.fixed_settings`` in
#: ``execution_components``, which the control plane freezes as the agent's
#: ``agent_execution_sha256`` before dispatch and matches on every claim, and
#: ``_attest_fixed_settings`` re-reads them byte-for-byte around every rollout.
#: A ceiling that varied with the budget would change the identity per rollout
#: and make the agent unclaimable. DERIVED-AT-IMPORT and per-rollout are not
#: the same thing: this is still one frozen constant, computed once from the
#: runner's constants rather than copied out of them by hand.
#:
#: What protects a sub-floor episode is not this setting at all. It is the
#: runner: the deadline path reports the child's stderr, and every outcome is
#: spooled and redelivered, so the rollout is still lost but it SAYS WHY on the
#: row instead of being swept with no trace.
#:
#: The ceiling is deliberately generous for a HEALTHY call, because it bounds
#: the whole streamed response and not time-to-first-byte (measured: a 25s
#: response is cut by a 10s ceiling). A value near the real generation time
#: would throw away finished work, which is the worse failure -- 4096
#: ``max_tokens`` at a pessimistic 20 tok/s is ~205s, and the derived value
#: must stay well above it (asserted in the wedged-call test).
#:
#: Deliberately spelled ``provider.maxRetryDelayMs`` and never the legacy
#: ``retry.maxDelayMs``: Prime migrates the latter into the former on load,
#: and a settings file this adapter seals must not be one Prime rewrites.
#:
#: The episode budget these SEALED bytes are sized against. OBSERVED, on the
#: OPSD queue on 2026-09-15: ``budget_at_creation_s = 894.0``. It lives here
#: and not in core because it is a fact about what this deployment mints, and
#: core must never carry one deployment's numbers.
PRIME_REFERENCE_EPISODE_BUDGET_S = 894.0

#: What one blocking provider call may cost, in milliseconds, for an episode
#: that long -- asked of the runner rather than decided here.
PRIME_PROVIDER_TIMEOUT_MS = int(
    harness_call_ceiling_s(PRIME_REFERENCE_EPISODE_BUDGET_S)
) * 1000

_FIXED_SETTINGS = (
    b'{"autoRefine":{"enabled":false,"compact":false},'
    b'"telemetry":{"enabled":false,"noticeShown":true},'
    b'"retry":{"enabled":false,"maxRetries":0,"provider":'
    b'{"timeoutMs":' + str(PRIME_PROVIDER_TIMEOUT_MS).encode("ascii")
    + b',"maxRetries":0,"maxRetryDelayMs":60000}}}\n'
)
_FIXED_SETTINGS_SHA256 = hashlib.sha256(
    _SETTINGS_DIGEST_DOMAIN + _FIXED_SETTINGS
).hexdigest()
_COMPONENT_NAME = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
_ROUTE_ENV = {
    "model_id": "PRIME_AGENT_FLYWHEEL_MODEL_ID",
    "context_window": "PRIME_AGENT_FLYWHEEL_CONTEXT_WINDOW",
    "max_tokens": "PRIME_AGENT_FLYWHEEL_MAX_TOKENS",
}

# A continual-harness state file is normally kilobytes.  These ceilings are
# deliberately generous, but finite: an operator accidentally pointing the
# source at HOME must fail before the SDK copies a disk into every rollout.
MAX_HARNESS_ENTRIES = 10_000
MAX_HARNESS_BYTES = 64 * 1024 * 1024
MAX_KERNEL_ENTRIES = 100_000
MAX_KERNEL_BYTES = 4 * 1024 * 1024 * 1024

#: The sealed kernel reaches Prime as PRIME_AGENT_KERNEL_PYTHON, the variable
#: prime-agent documents for using an existing Python environment instead of
#: auto-bootstrapping one (README, 0.7.2). Given it, Prime only CHECKS the
#: interpreter when a session first starts its kernel -- it imports
#: ``ipykernel``, runs a runtime check that begins ``import rlm`` (the package
#: prime-agent-runtime installs) and imports its default packages -- and
#: refuses one that fails, writing nothing. It reads no ``.bootstrap-version``
#: marker and never deletes or rebuilds anything. PRIME_AGENT_KERNEL_VENV, the
#: variable this adapter used to hand over, is the opposite: Prime ``rm -rf``s
#: and rebuilds a venv there that fails its private readiness check, including
#: one a different prime-agent version provisioned, and the seal could report
#: that only as a mutated kernel on a rollout already lost.
#:
#: An interpreter that cannot import ``ipykernel`` and ``rlm`` is therefore no
#: longer rebuilt, but every rollout's kernel would still be refused, so the
#: snapshot probes these imports once, before sealing, and refuses it at
#: construction for that reason. These are the two packages Prime's docs name
#: for that variable (``ipykernel``, and prime-agent-runtime, which provides
#: ``rlm``). The rest of its check -- the runtime's API and "the default
#: runtime packages", whose list is the installed release's -- the example's
#: runtime preflight asks that prime-agent for, instead of restating it here.
_KERNEL_IMPORT_PROBE = "import ipykernel, rlm"
#: A cold import of ipykernel is a second or two; this only bounds a hung one.
_KERNEL_PROBE_TIMEOUT_S = 60.0


#: The longest path a Unix-domain socket can bind: ``sun_path`` holds 104
#: bytes on macOS and the BSDs and 108 on Linux, NUL included. The smaller
#: limit is used everywhere, so a path that passes here binds on any of them.
_UNIX_SOCKET_PATH_MAX = 103

#: The longest socket Prime binds below its TMPDIR, by shape: a session
#: worker's, at ``prime-agent-<uid>/worker-<12 hex>-<12 hex>.sock``
#: (prime-agent 0.7.2). Its forkserver's ``control.sock`` sits shallower.
_PRIME_WORKER_SOCKET = "prime-agent-{uid}/worker-" + "0" * 12 + "-" + "0" * 12 + ".sock"


def unix_socket_directory(relative_path: str) -> str:
    """A temp directory below which ``relative_path`` can still bind as a socket.

    The temp directory itself when the joined path fits ``sun_path``, else
    ``/tmp``. A long TMPDIR -- a sandbox's, a CI runner's, a per-job scratch
    area -- puts every socket below it past the limit, and ``listen`` then
    fails with EINVAL: Prime's daemon exits at startup, or each session
    worker dies before the supervisor can reach it. ``/tmp`` is the POSIX
    directory every user can create in and already the default on Linux;
    Prime refuses a socket directory there that another user owns. Windows
    has no ``sun_path`` for this to fit.
    """
    preferred = tempfile.gettempdir()
    joined = os.path.join(preferred, relative_path)
    if os.name == "nt" or len(os.fsencode(joined)) <= _UNIX_SOCKET_PATH_MAX:
        return preferred
    return "/tmp"


def _refuse_unprovisioned_kernel(
    venv: Path, *, env: Mapping[str, str], cwd: Path,
) -> None:
    """Refuse a kernel venv whose interpreter Prime would refuse, naming why.

    Runs ``<venv>/bin/python -c "import ipykernel, rlm"`` -- the two packages
    Prime's docs name for a PRIME_AGENT_KERNEL_PYTHON -- under the scrubbed
    daemon environment, from the empty run-owned working
    directory so a module beside the serving process cannot answer for the
    venv. ``PYTHONDONTWRITEBYTECODE`` is part of that environment, so the
    probe writes nothing into the venv it is about to seal.
    """
    def _refusal(reason: str) -> ConfigurationError:
        return ConfigurationError(
            f"PRIME_AGENT_KERNEL_VENV is not a provisioned Prime kernel: {reason}. "
            "Prime refuses that interpreter when a session starts its kernel, "
            "so every rollout's ipython call would fail; provision the venv "
            "with this prime-agent's own kernel bootstrap before serving"
        )

    python = venv / "bin" / "python"
    if not python.is_file():
        raise _refusal("it has no interpreter at bin/python")
    try:
        probe = subprocess.run(
            [str(python), "-c", _KERNEL_IMPORT_PROBE],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            cwd=cwd,
            env=dict(env),
            timeout=_KERNEL_PROBE_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise _refusal(
            f"bin/python did not import ipykernel and rlm within "
            f"{_KERNEL_PROBE_TIMEOUT_S:.0f}s"
        ) from exc
    except OSError as exc:
        raise _refusal(
            f"bin/python cannot start ({exc.strerror or type(exc).__name__})"
        ) from exc
    if probe.returncode != 0:
        lines = probe.stderr.decode("utf-8", "replace").strip().splitlines()
        detail = lines[-1].strip()[:200] if lines else f"exit {probe.returncode}"
        raise _refusal(
            "bin/python cannot import ipykernel and rlm "
            f"(prime-agent-runtime): {detail}"
        )


class _HarnessDrift(ConfigurationError):
    """A snapshot or child no longer has the bytes it was created with."""


def _record(h: Any, kind: bytes, rel: str, mode: int) -> None:
    raw = rel.encode("utf-8", "surrogateescape")
    h.update(kind)
    h.update(struct.pack(">I", len(raw)))
    h.update(raw)
    h.update(struct.pack(">I", mode & 0o777))


def _open_regular_nofollow(path: Path) -> int:
    flags = os.O_RDONLY
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    # The lstat checks in the callers are the portable guard.  O_NOFOLLOW
    # closes the remaining swap-to-symlink race on platforms that expose it.
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(path, flags)


@dataclass(frozen=True)
class _ManifestEntry:
    kind: str
    rel: str
    mode: int
    body: bytes | None = None


@dataclass(frozen=True)
class _MetadataEntry:
    """Creation-time identity used to catch mutate-then-restore attempts.

    These fields are deliberately *not* part of the canonical content
    digests: inode and timestamps are properties of one materialization, not
    of its bytes. They are a second, local tamper-evident seal. A child can
    chmod a 0444 file because it runs as the same OS user, but it cannot restore
    ctime or the original inode after doing so.
    """

    kind: str
    rel: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int
    link_target: str | None = None
    resolved_target: str | None = None


def _metadata_entry(
    kind: str,
    rel: str,
    st: os.stat_result,
    *,
    link_target: str | None = None,
    resolved_target: str | None = None,
) -> _MetadataEntry:
    return _MetadataEntry(
        kind=kind,
        rel=rel,
        device=st.st_dev,
        inode=st.st_ino,
        mode=st.st_mode,
        size=st.st_size,
        mtime_ns=st.st_mtime_ns,
        ctime_ns=st.st_ctime_ns,
        link_target=link_target,
        resolved_target=resolved_target,
    )


def _metadata_seal(root: Path) -> tuple[_MetadataEntry, ...]:
    """Seal one already-stable materialization without changing its digest."""
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise _HarnessDrift("Prime harness child is missing or unreadable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise _HarnessDrift("Prime harness child root is not a real directory")

    entries = [_metadata_entry("D", ".", root_stat)]

    def _walk_error(exc: OSError) -> None:
        raise _HarnessDrift("Prime harness child changed while being sealed") from exc

    for current, dirnames, filenames in os.walk(
        root, topdown=True, onerror=_walk_error, followlinks=False,
    ):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for name in dirnames:
            path = current_path / name
            try:
                st = path.lstat()
            except OSError as exc:
                raise _HarnessDrift("Prime harness child changed while being sealed") from exc
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise _HarnessDrift("Prime harness child contains a link or special entry")
            entries.append(_metadata_entry("D", path.relative_to(root).as_posix(), st))
        for name in filenames:
            path = current_path / name
            try:
                st = path.lstat()
            except OSError as exc:
                raise _HarnessDrift("Prime harness child changed while being sealed") from exc
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
                raise _HarnessDrift("Prime harness child contains a link or special file")
            entries.append(_metadata_entry("F", path.relative_to(root).as_posix(), st))
    return tuple(sorted(entries, key=lambda entry: (entry.rel, entry.kind)))


@dataclass(frozen=True)
class _KernelEntry:
    kind: str
    rel: str
    path: Path
    metadata: _MetadataEntry
    target_path: Path | None = None
    target_metadata: _MetadataEntry | None = None


def _same_metadata(expected: _MetadataEntry, actual: os.stat_result) -> bool:
    return (
        expected.device == actual.st_dev
        and expected.inode == actual.st_ino
        and expected.mode == actual.st_mode
        and expected.size == actual.st_size
        and expected.mtime_ns == actual.st_mtime_ns
        and expected.ctime_ns == actual.st_ctime_ns
    )


def _kernel_entries(root: Path) -> tuple[_KernelEntry, ...]:
    """Describe a Prime kernel venv, including direct symlink targets."""
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise _HarnessDrift("Prime kernel venv is missing or unreadable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise _HarnessDrift("Prime kernel venv root must be a real directory")

    entries = [
        _KernelEntry("D", ".", root, _metadata_entry("D", ".", root_stat))
    ]
    entry_count = 1
    total_bytes = 0

    def _account(size: int = 0) -> None:
        nonlocal entry_count, total_bytes
        entry_count += 1
        total_bytes += size
        if entry_count > MAX_KERNEL_ENTRIES or total_bytes > MAX_KERNEL_BYTES:
            raise ConfigurationError(
                "Prime kernel venv exceeds its 100,000-entry or 4 GiB safety limit"
            )

    def _link_entry(path: Path, rel: str, before: os.stat_result) -> _KernelEntry:
        try:
            link_target = os.readlink(path)
            after = path.lstat()
            resolved = path.resolve(strict=True)
            target_stat = resolved.lstat()
        except OSError as exc:
            raise _HarnessDrift("Prime kernel symlink is broken or changed") from exc
        if not _same_metadata(_metadata_entry("L", rel, before), after):
            raise _HarnessDrift("Prime kernel symlink changed while being attested")
        if stat.S_ISLNK(target_stat.st_mode):
            raise ConfigurationError("Prime kernel venv has an unresolved symlink")
        if stat.S_ISDIR(target_stat.st_mode):
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise ConfigurationError(
                    "Prime kernel directory symlinks must resolve inside the venv"
                ) from exc
            # Common Linux venv shape: lib64 -> lib. os.walk does not follow
            # this alias; the real in-root directory is traversed exactly once.
            _account()
        elif stat.S_ISREG(target_stat.st_mode):
            _account(target_stat.st_size)
        else:
            raise ConfigurationError(
                "Prime kernel venv symlinks must resolve to regular files or "
                "in-venv directories"
            )
        resolved_text = str(resolved)
        link_metadata = _metadata_entry(
            "L",
            rel,
            after,
            link_target=link_target,
            resolved_target=resolved_text,
        )
        target_metadata = _metadata_entry(
            "T",
            rel,
            target_stat,
            link_target=link_target,
            resolved_target=resolved_text,
        )
        return _KernelEntry(
            "L",
            rel,
            path,
            link_metadata,
            target_path=resolved,
            target_metadata=target_metadata,
        )

    def _walk_error(exc: OSError) -> None:
        raise _HarnessDrift("Prime kernel venv changed while being attested") from exc

    for current, dirnames, filenames in os.walk(
        root, topdown=True, onerror=_walk_error, followlinks=False,
    ):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for name in list(dirnames):
            path = current_path / name
            rel = path.relative_to(root).as_posix()
            try:
                before = path.lstat()
            except OSError as exc:
                raise _HarnessDrift("Prime kernel venv changed while being attested") from exc
            if stat.S_ISLNK(before.st_mode):
                # os.walk lists a directory symlink in dirnames even when it
                # will not follow it. Treat it as a leaf. _link_entry accepts
                # the common in-root ``lib64 -> lib`` venv alias but never
                # traverses it a second time.
                dirnames.remove(name)
                entries.append(_link_entry(path, rel, before))
                continue
            if not stat.S_ISDIR(before.st_mode):
                raise ConfigurationError("Prime kernel venv contains a special entry")
            _account()
            entries.append(
                _KernelEntry("D", rel, path, _metadata_entry("D", rel, before))
            )
        for name in filenames:
            path = current_path / name
            rel = path.relative_to(root).as_posix()
            try:
                before = path.lstat()
            except OSError as exc:
                raise _HarnessDrift("Prime kernel venv changed while being attested") from exc
            if stat.S_ISLNK(before.st_mode):
                entries.append(_link_entry(path, rel, before))
                continue
            if not stat.S_ISREG(before.st_mode):
                raise ConfigurationError("Prime kernel venv contains a special file")
            _account(before.st_size)
            entries.append(
                _KernelEntry("F", rel, path, _metadata_entry("F", rel, before))
            )
    return tuple(sorted(entries, key=lambda entry: (entry.rel, entry.kind)))


def _kernel_seal(entries: tuple[_KernelEntry, ...]) -> tuple[_MetadataEntry, ...]:
    seal: list[_MetadataEntry] = []
    for entry in entries:
        seal.append(entry.metadata)
        if entry.target_metadata is not None:
            seal.append(entry.target_metadata)
    return tuple(seal)


def _kernel_metadata_seal(root: Path) -> tuple[_MetadataEntry, ...]:
    first = _kernel_seal(_kernel_entries(root))
    second = _kernel_seal(_kernel_entries(root))
    if first != second:
        raise _HarnessDrift("Prime kernel venv changed while being attested")
    return second


def _hash_stable_file(h: Any, path: Path, expected: _MetadataEntry) -> None:
    try:
        fd = _open_regular_nofollow(path)
    except OSError as exc:
        raise _HarnessDrift("Prime kernel file changed while being attested") from exc
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode) or not _same_metadata(expected, opened):
            raise _HarnessDrift("Prime kernel file changed while being attested")
        remaining = expected.size
        while remaining:
            chunk = os.read(fd, min(_BUFFER_BYTES, remaining))
            if not chunk:
                raise _HarnessDrift("Prime kernel file changed while being attested")
            h.update(chunk)
            remaining -= len(chunk)
        if os.read(fd, 1):
            raise _HarnessDrift("Prime kernel file changed while being attested")
        if not _same_metadata(expected, os.fstat(fd)):
            raise _HarnessDrift("Prime kernel file changed while being attested")
    finally:
        os.close(fd)


def _kernel_tree_state(root: Path) -> tuple[str, tuple[_MetadataEntry, ...]]:
    """Hash a bounded kernel venv and return its persistent mutation seal."""
    entries = _kernel_entries(root)
    h = hashlib.sha256()
    h.update(_KERNEL_DIGEST_DOMAIN)
    for entry in entries:
        mode = stat.S_IMODE(entry.metadata.mode)
        _record(h, entry.kind.encode("ascii"), entry.rel, mode)
        if entry.kind == "F":
            h.update(struct.pack(">Q", entry.metadata.size))
            _hash_stable_file(h, entry.path, entry.metadata)
        elif entry.kind == "L":
            assert entry.target_path is not None and entry.target_metadata is not None
            for text_value in (
                entry.metadata.link_target or "",
                entry.metadata.resolved_target or "",
            ):
                raw = text_value.encode("utf-8", "surrogateescape")
                h.update(struct.pack(">I", len(raw)))
                h.update(raw)
            h.update(struct.pack(">I", stat.S_IMODE(entry.target_metadata.mode)))
            if stat.S_ISDIR(entry.target_metadata.mode):
                h.update(b"D")
            else:
                h.update(b"F")
                h.update(struct.pack(">Q", entry.target_metadata.size))
                _hash_stable_file(h, entry.target_path, entry.target_metadata)

    seal = _kernel_seal(entries)
    if _kernel_metadata_seal(root) != seal:
        raise _HarnessDrift("Prime kernel venv changed while being snapshotted")
    return h.hexdigest(), seal


def _read_manifest(root: Path) -> tuple[_ManifestEntry, ...]:
    """Read one bounded, link-free tree into parent-process memory."""
    try:
        root_stat = root.lstat()
    except OSError as exc:
        raise _HarnessDrift("harness root is missing or unreadable") from exc
    if stat.S_ISLNK(root_stat.st_mode) or not stat.S_ISDIR(root_stat.st_mode):
        raise _HarnessDrift("harness root is not a real directory")

    entries = [_ManifestEntry("D", ".", 0o555)]
    entry_count = 1  # the root itself
    total = 0
    observed_dirs: list[tuple[Path, int, int, int]] = [
        (root, root_stat.st_dev, root_stat.st_ino, root_stat.st_mtime_ns)
    ]

    def _walk_error(exc: OSError) -> None:
        raise _HarnessDrift("harness directory changed while being attested") from exc

    def _account(size: int = 0) -> None:
        nonlocal entry_count, total
        entry_count += 1
        total += size
        if entry_count > MAX_HARNESS_ENTRIES or total > MAX_HARNESS_BYTES:
            raise ConfigurationError(
                "Prime harness snapshot exceeds its 10,000-entry or 64 MiB safety limit"
            )

    # ``os.walk(..., followlinks=False)`` still lists a symlinked directory in
    # dirnames.  Reject every entry from lstat before either traversal or read.
    for current, dirnames, filenames in os.walk(
        root, topdown=True, onerror=_walk_error, followlinks=False,
    ):
        current_path = Path(current)
        dirnames.sort()
        filenames.sort()
        for name in list(dirnames):
            path = current_path / name
            try:
                st = path.lstat()
            except OSError as exc:
                raise _HarnessDrift("harness entry changed while being attested") from exc
            if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
                raise _HarnessDrift("harness contains a link or special directory entry")
            _account()
            rel = path.relative_to(root).as_posix()
            entries.append(_ManifestEntry("D", rel, 0o555))
            observed_dirs.append((path, st.st_dev, st.st_ino, st.st_mtime_ns))

        for name in filenames:
            path = current_path / name
            try:
                before = path.lstat()
            except OSError as exc:
                raise _HarnessDrift("harness entry changed while being attested") from exc
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
                raise _HarnessDrift("harness contains a link or special file")
            _account(before.st_size)
            rel = path.relative_to(root).as_posix()

            try:
                fd = _open_regular_nofollow(path)
            except OSError as exc:
                raise _HarnessDrift("harness file changed while being attested") from exc
            try:
                opened = os.fstat(fd)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or opened.st_dev != before.st_dev
                    or opened.st_ino != before.st_ino
                ):
                    raise _HarnessDrift("harness file changed while being attested")
                remaining = before.st_size
                chunks: list[bytes] = []
                while remaining:
                    chunk = os.read(fd, min(_BUFFER_BYTES, remaining))
                    if not chunk:
                        raise _HarnessDrift("harness file changed while being attested")
                    chunks.append(chunk)
                    remaining -= len(chunk)
                if os.read(fd, 1):
                    raise _HarnessDrift("harness file changed while being attested")
                after = os.fstat(fd)
                if (
                    after.st_size != before.st_size
                    or after.st_mtime_ns != before.st_mtime_ns
                    or after.st_ctime_ns != before.st_ctime_ns
                    or after.st_mode != before.st_mode
                ):
                    raise _HarnessDrift("harness file changed while being attested")
            finally:
                os.close(fd)
            mode = 0o555 if stat.S_IMODE(before.st_mode) & 0o111 else 0o444
            entries.append(_ManifestEntry("F", rel, mode, b"".join(chunks)))

    # A concurrent create/delete can otherwise land just after os.walk scanned
    # a directory and disappear from the digest. Directory mtimes are not part
    # of identity, but they are a stability check while the identity is read.
    for path, device, inode, mtime_ns in observed_dirs:
        try:
            after = path.lstat()
        except OSError as exc:
            raise _HarnessDrift("harness directory changed while being attested") from exc
        if (
            not stat.S_ISDIR(after.st_mode)
            or after.st_dev != device
            or after.st_ino != inode
            or after.st_mtime_ns != mtime_ns
        ):
            raise _HarnessDrift("harness directory changed while being attested")
    return tuple(entries)


def _manifest_sha256(
    entries: tuple[_ManifestEntry, ...], *, domain: bytes = _DIGEST_DOMAIN,
) -> str:
    h = hashlib.sha256()
    h.update(domain)
    for entry in entries:
        _record(h, entry.kind.encode("ascii"), entry.rel, entry.mode)
        if entry.kind == "F":
            body = entry.body or b""
            h.update(struct.pack(">Q", len(body)))
            h.update(body)
    return h.hexdigest()


def _tree_sha256(root: Path) -> str:
    return _manifest_sha256(_read_manifest(root))


@dataclass(frozen=True)
class _ExecutionArtifact:
    """One adapter-owned file/tree folded opaquely into execution identity."""

    name: str
    path: Path
    digest: str
    seal: tuple[_MetadataEntry, ...]


def _execution_artifact_state(name: str, value: str | os.PathLike[str]) -> _ExecutionArtifact:
    path = Path(value).expanduser()
    try:
        root_stat = path.lstat()
    except OSError as exc:
        raise ConfigurationError(f"Prime execution artifact {name!r} does not exist") from exc
    if stat.S_ISLNK(root_stat.st_mode):
        raise ConfigurationError(f"Prime execution artifact {name!r} must not be a symlink")
    path = path.resolve()

    h = hashlib.sha256()
    h.update(_ARTIFACT_DIGEST_DOMAIN)
    if stat.S_ISDIR(root_stat.st_mode):
        manifest = _read_manifest(path)
        digest = _manifest_sha256(manifest, domain=_ARTIFACT_DIGEST_DOMAIN)
        seal = _metadata_seal(path)
        if _metadata_seal(path) != seal:
            raise _HarnessDrift(f"Prime execution artifact {name!r} changed while snapshotted")
        return _ExecutionArtifact(name, path, digest, seal)
    if not stat.S_ISREG(root_stat.st_mode):
        raise ConfigurationError(f"Prime execution artifact {name!r} must be a file or directory")
    if root_stat.st_size > MAX_HARNESS_BYTES:
        raise ConfigurationError(
            f"Prime execution artifact {name!r} exceeds the 64 MiB file safety limit"
        )
    metadata = _metadata_entry("F", ".", root_stat)
    _record(h, b"F", ".", stat.S_IMODE(root_stat.st_mode))
    h.update(struct.pack(">Q", root_stat.st_size))
    _hash_stable_file(h, path, metadata)
    seal = (metadata,)
    try:
        after = path.lstat()
    except OSError as exc:
        raise _HarnessDrift(f"Prime execution artifact {name!r} changed while snapshotted") from exc
    if not _same_metadata(metadata, after):
        raise _HarnessDrift(f"Prime execution artifact {name!r} changed while snapshotted")
    return _ExecutionArtifact(name, path, h.hexdigest(), seal)


def _attest_execution_artifact(artifact: _ExecutionArtifact) -> None:
    try:
        st = artifact.path.lstat()
    except OSError as exc:
        raise _HarnessDrift(
            f"Prime execution artifact {artifact.name!r} is missing or unreadable"
        ) from exc
    if stat.S_ISDIR(st.st_mode):
        actual = _metadata_seal(artifact.path)
    elif stat.S_ISREG(st.st_mode):
        actual = (_metadata_entry("F", ".", st),)
    else:
        raise _HarnessDrift(f"Prime execution artifact {artifact.name!r} changed type")
    if actual != artifact.seal:
        raise _HarnessDrift(f"Prime execution artifact {artifact.name!r} was mutated")


def _write_manifest(entries: tuple[_ManifestEntry, ...], target: Path) -> None:
    """Materialize one private read-only copy from immutable parent memory."""
    target.mkdir(mode=0o700)
    directories = [entry for entry in entries if entry.kind == "D" and entry.rel != "."]
    for entry in directories:
        (target / entry.rel).mkdir(mode=0o700)
    for entry in entries:
        if entry.kind != "F":
            continue
        dst = target / entry.rel
        with dst.open("xb") as output:
            output.write(entry.body or b"")
        dst.chmod(entry.mode)
    for entry in reversed(directories):
        (target / entry.rel).chmod(entry.mode)
    target.chmod(0o555)


def _write_fixed_settings(agent_dir: Path) -> Path:
    path = agent_dir / "settings.json"
    with path.open("xb") as output:
        output.write(_FIXED_SETTINGS)
    path.chmod(0o444)
    return path


def _attest_fixed_settings(path: Path) -> _MetadataEntry:
    try:
        before = path.lstat()
    except OSError as exc:
        raise _HarnessDrift("Prime fixed settings are missing or unreadable") from exc
    if (
        stat.S_ISLNK(before.st_mode)
        or not stat.S_ISREG(before.st_mode)
        or stat.S_IMODE(before.st_mode) != 0o444
        or before.st_size != len(_FIXED_SETTINGS)
    ):
        raise _HarnessDrift("Prime fixed settings changed during the rollout")
    try:
        fd = _open_regular_nofollow(path)
    except OSError as exc:
        raise _HarnessDrift("Prime fixed settings changed during the rollout") from exc
    try:
        opened = os.fstat(fd)
        if opened.st_dev != before.st_dev or opened.st_ino != before.st_ino:
            raise _HarnessDrift("Prime fixed settings changed during the rollout")
        body = os.read(fd, len(_FIXED_SETTINGS) + 1)
        after = os.fstat(fd)
        if (
            body != _FIXED_SETTINGS
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
            or after.st_ctime_ns != before.st_ctime_ns
            or after.st_mode != before.st_mode
        ):
            raise _HarnessDrift("Prime fixed settings changed during the rollout")
    finally:
        os.close(fd)
    return _metadata_entry("F", "settings.json", after)


def _make_writable(path: str) -> None:
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass


def _remove_tree(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        path.unlink()
        return
    # rmtree cannot unlink an entry from a 0555 directory even after making
    # the entry itself writable. Thaw directories top-down first; never chmod
    # a symlink, because chmod would follow it on platforms without lchmod.
    for current, dirnames, filenames in os.walk(path, topdown=True, followlinks=False):
        _make_writable(current)
        current_path = Path(current)
        for name in [*dirnames, *filenames]:
            candidate = current_path / name
            try:
                if not candidate.is_symlink():
                    _make_writable(str(candidate))
            except OSError:
                pass

    def _onerror(fn: Any, name: str, _exc: Any) -> None:
        _make_writable(str(Path(name).parent))
        _make_writable(name)
        fn(name)

    shutil.rmtree(path, onerror=_onerror)


@dataclass(frozen=True)
class PrimeHarnessChild:
    """One private child view created from a :class:`PrimeHarnessSnapshot`."""

    home_dir: Path
    agent_dir: Path
    harness_dir: Path
    settings_path: Path
    environment: dict[str, str]
    harness_sha256: str
    _root: Path
    _owner_token: object
    _harness_seal: tuple[_MetadataEntry, ...]
    _settings_seal: _MetadataEntry

    @property
    def discovery_environment(self) -> dict[str, str]:
        """Prime layout only; contains no provider or customer credentials."""
        return dict(self.environment)


class PrimeHarnessSnapshot:
    """One immutable Prime harness identity shared by isolated child copies.

    ``source_dir`` names the *harness directory itself* (the directory that
    contains ``harness_state.json``), not ``~/.prime/agent``.  An empty
    directory is a valid empty harness.  Symlinks and special files are
    rejected so the digest always names owned bytes rather than ambient data.

    ``kernel_venv`` is optional at the SDK level.  The Prime example requires
    it because changing HOME would otherwise make every concurrent child try
    to bootstrap a separate runtime. It is runtime state, not continual
    harness state, so it receives its own ``kernel_sha256`` identity and a
    metadata seal checked before and after every child. A write poisons the
    snapshot rather than silently changing later replicas. Prime receives it
    as ``PRIME_AGENT_KERNEL_PYTHON=<venv>/bin/python``, which it uses as it
    is and never rebuilds, so construction refuses a venv whose
    ``bin/python`` cannot import ``ipykernel`` and ``rlm``: Prime would refuse
    that interpreter in every rollout.

    ``execution_artifacts`` is an adapter-owned map of stable names to exact
    files or link-free trees, such as Prime's bundled package and the explicit
    routing extension. Core execution reconciliation sees only their opaque
    digests; it never needs Prime-specific path or package knowledge. This
    adapter declares complete example coverage only when a kernel and the
    ``prime_distribution``, ``prime_package_manifest``, and ``route_extension``
    artifacts are all present.
    """

    def __init__(
        self,
        source_dir: str | os.PathLike[str],
        *,
        workspace_dir: str | os.PathLike[str],
        kernel_venv: str | os.PathLike[str] | None = None,
        execution_artifacts: Mapping[str, str | os.PathLike[str]] | None = None,
        route_configuration: Mapping[str, str] | None = None,
    ) -> None:
        source = Path(source_dir).expanduser()
        try:
            source_stat = source.lstat()
        except OSError as exc:
            raise ConfigurationError("Prime harness source directory does not exist") from exc
        if stat.S_ISLNK(source_stat.st_mode) or not stat.S_ISDIR(source_stat.st_mode):
            raise ConfigurationError("Prime harness source must be a real directory, not a link")
        source = source.resolve()

        # Validate caller-owned configuration before allocating the run root.
        # A malformed optional adapter configuration must not leak a temporary
        # directory merely because construction failed before the main cleanup
        # guard was entered.
        try:
            supplied_route = dict(route_configuration or {})
        except (TypeError, ValueError) as exc:
            raise ConfigurationError("Prime route_configuration must be a mapping") from exc
        if supplied_route and set(supplied_route) != set(_ROUTE_ENV):
            raise ConfigurationError(
                "Prime route_configuration must contain model_id, "
                "context_window, and max_tokens"
            )
        if any(
            not isinstance(value, str) or not value.strip()
            for value in supplied_route.values()
        ):
            raise ConfigurationError(
                "Prime route_configuration values must be non-empty strings"
            )
        normalized_route = {
            name: supplied_route[name] for name in sorted(supplied_route)
        }
        route_configuration_sha256: str | None = None
        if normalized_route:
            route_blob = json.dumps(
                normalized_route,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
            route_configuration_sha256 = hashlib.sha256(
                _ROUTE_CONFIG_DIGEST_DOMAIN + route_blob
            ).hexdigest()

        workspace = Path(workspace_dir).expanduser()
        workspace.mkdir(parents=True, exist_ok=True)
        workspace = workspace.resolve()
        self._root = Path(tempfile.mkdtemp(prefix="prime-harness-", dir=workspace))
        self._root.chmod(0o700)
        if self._root == source or self._root.is_relative_to(source):
            _remove_tree(self._root)
            raise ConfigurationError("Prime harness workspace must be outside its source directory")

        self._children = self._root / "children"
        self._children.mkdir(mode=0o700)
        self._daemon_home = self._root / "daemon-home"
        self._daemon_agent = self._daemon_home / ".prime" / "agent"
        self._daemon_agent.mkdir(parents=True, mode=0o700)
        self._daemon_cwd = self._root / "daemon-workspace"
        self._daemon_cwd.mkdir(mode=0o700)
        self._daemon_settings = _write_fixed_settings(self._daemon_agent)
        self._daemon_settings_seal = _attest_fixed_settings(self._daemon_settings)
        self._owner_token = object()
        self._lock = threading.RLock()
        self._active: set[Path] = set()
        self._closed = False
        self._execution_artifacts: tuple[_ExecutionArtifact, ...] = ()
        self._route_configuration = normalized_route
        self._route_configuration_sha256 = route_configuration_sha256

        if kernel_venv is None:
            self._kernel_venv: Path | None = None
            self._kernel_sha256: str | None = None
            self._kernel_seal: tuple[_MetadataEntry, ...] | None = None
        else:
            candidate = Path(kernel_venv).expanduser()
            if not candidate.is_dir():
                _remove_tree(self._root)
                raise ConfigurationError("PRIME_AGENT_KERNEL_VENV must name an existing directory")
            self._kernel_venv = candidate.resolve()
            self._kernel_sha256 = None
            self._kernel_seal = None

        try:
            # The fixed master lives as immutable bytes in this parent
            # process, not as another same-uid directory a command child could
            # discover and rewrite. Two reads refuse a moving source instead
            # of quietly constructing a cross-version snapshot.
            manifest = _read_manifest(source)
            if manifest != _read_manifest(source):
                raise ConfigurationError("Prime harness source changed while being snapshotted")
            self._manifest = manifest
            self._sha256 = _manifest_sha256(manifest)
            if self._kernel_venv is not None:
                _refuse_unprovisioned_kernel(
                    self._kernel_venv,
                    env=self._scrubbed_daemon_environment(),
                    cwd=self._daemon_cwd,
                )
                self._kernel_sha256, self._kernel_seal = _kernel_tree_state(
                    self._kernel_venv
                )
            artifacts: list[_ExecutionArtifact] = []
            for name, path in (execution_artifacts or {}).items():
                if not isinstance(name, str) or _COMPONENT_NAME.fullmatch(name) is None:
                    raise ConfigurationError(
                        "Prime execution artifact names must match "
                        "[a-z][a-z0-9_.-]{0,63}"
                    )
                artifacts.append(_execution_artifact_state(name, path))
            self._execution_artifacts = tuple(sorted(artifacts, key=lambda item: item.name))
        except Exception:
            _remove_tree(self._root)
            raise

    @property
    def adapter_name(self) -> str:
        """Stable diagnostic name used by generic SDK lifecycle errors."""
        return "Prime"

    @property
    def worst_case_call_s(self) -> float:
        """The longest ONE blocking model call can hold a Prime child.

        DERIVED FROM THE SEALED BYTES rather than restated, so a ceiling that
        was edited without this moving is not expressible. Read (optionally,
        via ``getattr``) by ``RolloutRunner``, which compares it against the
        episode actually minted and warns when the ceiling cannot fire inside
        it. Retries are counted from ``maxRetries`` whatever ``enabled`` says:
        turning them back on without re-deriving must not pass by accident.
        """
        retry = json.loads(_FIXED_SETTINGS)["retry"]
        provider = retry["provider"]
        attempts = (provider["maxRetries"] + 1) * (retry["maxRetries"] + 1)
        backoff_ms = 2000 * retry["maxRetries"]  # Prime's documented 2s base delay
        return (provider["timeoutMs"] * attempts + backoff_ms) / 1000.0

    @property
    def harness_sha256(self) -> str:
        """Canonical lowercase SHA-256 stamped on every configured result."""
        return self._sha256

    @property
    def kernel_sha256(self) -> str | None:
        """Canonical kernel-venv identity, separate from continual harness."""
        return self._kernel_sha256

    @property
    def execution_components(self) -> dict[str, str]:
        """Opaque adapter evidence folded into the generic execution digest."""
        components = {
            "prime.continual_harness": self._sha256,
            "prime.fixed_settings": _FIXED_SETTINGS_SHA256,
        }
        if self._route_configuration_sha256 is not None:
            components["prime.route_configuration"] = self._route_configuration_sha256
        if self._kernel_sha256 is not None:
            components["prime.kernel_venv"] = self._kernel_sha256
        components.update({
            f"prime.artifact.{artifact.name}": artifact.digest
            for artifact in self._execution_artifacts
        })
        return components

    @property
    def execution_identity_complete(self) -> bool:
        """Whether this Prime adapter covers the example's unobserved code inputs."""
        names = {artifact.name for artifact in self._execution_artifacts}
        required = {
            "prime_distribution",
            "prime_package_manifest",
            "route_extension",
        }
        return (
            self._route_configuration_sha256 is not None
            and self._kernel_sha256 is not None
            and required.issubset(names)
        )

    @property
    def diagnostic_fingerprint(self) -> dict[str, str]:
        """Prime-specific evidence retained for diagnosis, not OPSD binding."""
        diagnostics = {"harness_sha256": self._sha256}
        if self._kernel_sha256 is not None:
            diagnostics["kernel_sha256"] = self._kernel_sha256
        return diagnostics

    @property
    def daemon_working_directory(self) -> Path:
        """Empty run-owned cwd, isolated from ambient project settings."""
        return self._daemon_cwd

    def _attest_master(self) -> None:
        if _manifest_sha256(self._manifest) != self._sha256:
            raise _HarnessDrift("run-scoped Prime harness master drifted")
        if _attest_fixed_settings(self._daemon_settings) != self._daemon_settings_seal:
            raise _HarnessDrift("run-scoped Prime fixed settings were mutated")
        if self._kernel_venv is not None:
            assert self._kernel_seal is not None
            if _kernel_metadata_seal(self._kernel_venv) != self._kernel_seal:
                raise _HarnessDrift("run-scoped Prime kernel venv was mutated")
        for artifact in self._execution_artifacts:
            _attest_execution_artifact(artifact)

    def _environment_for(self, home: Path, agent: Path) -> dict[str, str]:
        env = {
            "HOME": str(home),
            "USERPROFILE": str(home),
            "XDG_CACHE_HOME": str(home / ".cache"),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local" / "share"),
            "XDG_STATE_HOME": str(home / ".local" / "state"),
            "PRIME_AGENT_CODING_AGENT_DIR": str(agent),
            "PRIME_AGENT_SESSION_DIR": str(agent / "sessions"),
            "PRIME_AGENT_CODING_AGENT_SESSION_DIR": str(agent / "sessions"),
            # Prime binds its worker and forkserver sockets below TMPDIR. The
            # supervisor and every child get the same one, with room for the
            # longest of those sockets, or no session ever starts.
            "TMPDIR": unix_socket_directory(_PRIME_WORKER_SOCKET.format(
                uid=os.getuid() if hasattr(os, "getuid") else "user")),
            # Always set, so an interpreter from the serving shell's
            # configuration can never stand in for the attested venv: Prime
            # prefers this variable over any venv. Without a kernel it is
            # empty, which Prime treats as unset.
            "PRIME_AGENT_KERNEL_PYTHON": "",
            # Import caches are runtime writes, not policy behavior. Suppress
            # them so a clean Python import cannot poison the shared kernel
            # seal; an explicit package/runtime mutation still fails closed.
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        env.update({
            _ROUTE_ENV[name]: value
            for name, value in self._route_configuration.items()
        })
        if self._kernel_venv is not None:
            # Prime's documented way to use an existing environment without
            # auto-bootstrap (see _KERNEL_IMPORT_PROBE): the sealed venv's own
            # interpreter, which Prime checks and then uses as it is. This is
            # the exact path Prime would derive from PRIME_AGENT_KERNEL_VENV,
            # so the kernel process is the same one; what goes is Prime's
            # private readiness marker and the delete-and-rebuild behind it.
            env["PRIME_AGENT_KERNEL_PYTHON"] = str(self._kernel_venv / "bin" / "python")
            # Prime reads this only to find a venv to manage, which it never
            # does once the interpreter is given. It stays set to the same
            # sealed tree so an ambient value cannot name a different one to
            # anything that does read it.
            env["PRIME_AGENT_KERNEL_VENV"] = str(self._kernel_venv)
        return env

    def prepare_child(self, label: str = "rollout") -> PrimeHarnessChild:
        """Create one isolated HOME/agent directory from the fixed master."""
        del label  # diagnostic only; never put an untrusted rollout id in a path
        with self._lock:
            if self._closed:
                raise ConfigurationError("Prime harness snapshot is closed")
            self._attest_master()
            child_root = Path(tempfile.mkdtemp(prefix="child-", dir=self._children))
            child_root.chmod(0o700)
            home = child_root / "home"
            agent = home / ".prime" / "agent"
            agent.mkdir(parents=True, mode=0o700)
            settings = _write_fixed_settings(agent)
            harness = agent / "harness"
            try:
                _write_manifest(self._manifest, harness)
                if _tree_sha256(harness) != self._sha256:
                    raise _HarnessDrift("Prime harness child did not match its master")
                harness_seal = _metadata_seal(harness)
                settings_seal = _attest_fixed_settings(settings)
            except Exception:
                _remove_tree(child_root)
                raise
            self._active.add(child_root)

        env = self._environment_for(home, agent)
        return PrimeHarnessChild(
            home_dir=home,
            agent_dir=agent,
            harness_dir=harness,
            settings_path=settings,
            environment=env,
            harness_sha256=self._sha256,
            _root=child_root,
            _owner_token=self._owner_token,
            _harness_seal=harness_seal,
            _settings_seal=settings_seal,
        )

    def attest_child(self, child: PrimeHarnessChild) -> None:
        """Fail when either the private child or the shared master drifted."""
        with self._lock:
            if child._owner_token is not self._owner_token or child._root not in self._active:
                raise ConfigurationError("Prime harness child does not belong to this snapshot")
            self._attest_master()
            if _attest_fixed_settings(child.settings_path) != child._settings_seal:
                raise _HarnessDrift("Prime fixed settings were mutated during the rollout")
            actual = _tree_sha256(child.harness_dir)
            if actual != self._sha256:
                raise _HarnessDrift("Prime harness child mutated during the rollout")
            if _metadata_seal(child.harness_dir) != child._harness_seal:
                raise _HarnessDrift("Prime harness child was mutated during the rollout")

    def daemon_environment(self) -> dict[str, str]:
        """A scrubbed, stable environment for the run's Prime supervisor.

        The supervisor persists descriptors beside its agent directory, so it
        cannot live under a private rollout HOME that is deleted after exit.
        It receives no model endpoint or credential; each client-owned session
        worker gets the private child's full launch environment from Prime's
        daemon protocol.
        """
        with self._lock:
            if self._closed:
                raise ConfigurationError("Prime harness snapshot is closed")
            self._attest_master()
            return self._scrubbed_daemon_environment()

    def _scrubbed_daemon_environment(self) -> dict[str, str]:
        """``daemon_environment`` without the attestation, for construction.

        The kernel probe runs before there is a seal to attest against, and
        must see exactly the environment the daemon will get.
        """
        from ..isolation import ROUTING_VARS, build_child_env

        extra = self._environment_for(self._daemon_home, self._daemon_agent)
        env = build_child_env(
            rollout_id="percepteye-daemon-not-a-rollout",
            llm_url="http://127.0.0.1:1/v1",
            api_key="percepteye-daemon-no-sampling",
            trajectory_dir=str(self._root / "daemon-trajectory-unused"),
            # The supervisor does not execute the customer's task. It
            # needs only enough process environment to start; session
            # workers receive their own child environments over Prime's
            # client-owned launch protocol. An arbitrary Jira/AWS/etc.
            # credential has no reason to enter this long-lived process.
            allowlist=(),
            extra=extra,
        )
        for name in [key for key in env if key.startswith("PERCEPTEYE_")]:
            env.pop(name, None)
        for name in ROUTING_VARS:
            env.pop(name, None)
        return env

    def release_child(self, child: PrimeHarnessChild) -> None:
        """Remove a child copy after attestation."""
        with self._lock:
            if child._owner_token is not self._owner_token:
                raise ConfigurationError("Prime harness child does not belong to this snapshot")
            self._active.discard(child._root)
        _remove_tree(child._root)

    def close(self) -> None:
        """Delete the run snapshot after all children have finished."""
        with self._lock:
            if self._active:
                raise ConfigurationError("cannot close Prime harness snapshot with active children")
            if self._closed:
                return
            self._closed = True
        _remove_tree(self._root)


PRIME_TOOL_FORMAT = StreamFormat(
    name="prime-agent",
    start_type="tool_execution_start",
    end_type="tool_execution_end",
    result_status_path=("details", "status"),
    failed_status_values=("error", "aborted"),
)
_PRIME_EVENT_TYPES = frozenset({
    "session",
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "message_start",
    "message_update",
    "message_end",
    "tool_execution_start",
    "tool_execution_update",
    "tool_execution_end",
    "compaction_start",
    "compaction_end",
    "refine_complete",
    "refine_failed",
})
_POLICY_EXTERNAL_EVENTS = frozenset({
    "compaction_start",
    "compaction_end",
    "refine_complete",
    "refine_failed",
})


def _assistant_text(message: Any) -> str | None:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return None
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        block["text"]
        for block in content
        if isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    )


class PrimeAgentOutputAdapter:
    """Interpret Prime Agent JSON mode without teaching core its vocabulary."""

    @property
    def adapter_name(self) -> str:
        return "Prime Agent JSONL"

    def project(self, stdout: str) -> CommandOutputProjection:
        events: list[dict[str, Any]] = []
        malformed_structured_line = False
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                value = json.loads(line)
            except (TypeError, ValueError):
                malformed_structured_line = True
                continue
            if isinstance(value, dict):
                events.append(value)
        recognized = any(event.get("type") in _PRIME_EVENT_TYPES for event in events)
        if not recognized:
            return CommandOutputProjection(recognized=False)

        final_text: str | None = None
        llm_call_count = 0
        control_events: list[str] = []
        terminal: dict[str, Any] | None = None
        agent_start_indices: list[int] = []
        agent_end_indices: list[int] = []

        for index, event in enumerate(events):
            kind = event.get("type")
            if kind == "agent_start":
                agent_start_indices.append(index)
            if kind in _POLICY_EXTERNAL_EVENTS and kind not in control_events:
                control_events.append(str(kind))
            if kind == "message_end":
                message = event.get("message")
                text = _assistant_text(message)
                if text is not None:
                    llm_call_count += 1
                    final_text = text
                    if not agent_end_indices and isinstance(message, dict):
                        terminal = message
            elif kind == "agent_end" and isinstance(event.get("messages"), list):
                agent_end_indices.append(index)
                final_text = ""
                terminal = None
                for message in reversed(event["messages"]):
                    text = _assistant_text(message)
                    if text is not None:
                        final_text = text
                        terminal = message
                        break

        terminal_failure = None
        if terminal is not None:
            reason = terminal.get("stopReason")
            if not isinstance(reason, str):
                reason = terminal.get("stop_reason")
            normalized = reason.strip().lower() if isinstance(reason, str) else ""
            if normalized in {"error", "aborted"}:
                terminal_failure = normalized

        invalid: list[str] = []
        lifecycle_complete = (
            not malformed_structured_line
            and len(agent_start_indices) == 1
            and len(agent_end_indices) == 1
            and agent_start_indices[0] < agent_end_indices[0]
        )
        if not lifecycle_complete:
            invalid.append("incomplete or ambiguous Prime agent lifecycle")
        if control_events:
            invalid.append(
                "policy-external control event(s) " + ", ".join(control_events)
            )
        if terminal_failure:
            invalid.append(
                "terminal assistant stopReason="
                + terminal_failure
                + " (diagnostic redacted)"
            )

        parsed = parse_tool_calls_with_omissions(
            stdout, formats=(PRIME_TOOL_FORMAT,)
        )
        return CommandOutputProjection(
            recognized=True,
            final_text=final_text or "",
            # One assistant message_end is Prime's provider-call terminal
            # witness. Assert the count only when the enclosing command
            # lifecycle is complete and the structured stream was readable;
            # a killed/truncated stream may have dropped an arbitrary suffix.
            llm_call_count=llm_call_count if lifecycle_complete else None,
            tool_calls=parsed.calls,
            tool_calls_omitted_count=parsed.omitted_count,
            exclusion_reason="cancelled" if invalid else None,
            exclusion_detail=(
                "untrainable command event stream: observed " + "; ".join(invalid)
                if invalid
                else None
            ),
        )
