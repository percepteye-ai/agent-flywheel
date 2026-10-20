"""The keys ``agent-flywheel login`` stores, and the ONE rule that finds a key.

A ``pefw_`` key is bound to one agent on one control plane, so it is stored
per ``(control-plane URL, agent id)`` in ``$PERCEPTEYE_HOME/credentials.json``
(``~/.percepteye/credentials.json`` when ``PERCEPTEYE_HOME`` is unset), a file
only its owner can read (0600) and that is replaced, never edited in place.
The Node SDK reads and writes the same file in the same format
(``src/credentials.js``), so a key stored by either is found by both;
``tests/test_credentials_parity.py`` holds the two to one table.

WHICH KEY, in order (:func:`resolve_api_key`):

1. the key passed in code (``api_key=``);
2. ``PERCEPTEYE_API_KEY``;
3. the key stored for THIS agent on THIS control plane.

The file is read only when neither of the first two is set, so nothing about
an existing deployment changes.

A STORED KEY IS CONSENT, like the variable (owner ruling 2026-10-20, R2): it
exists only because someone ran ``agent-flywheel login`` (or said yes when
``serve()`` offered it) and clicked Approve for that named agent in a browser.
It is never silent: the first time a process uses one it says so, naming the
file, the agent, the control plane and how to stop
(:func:`announce_stored_key`). It applies to its own agent id and no other,
and uploading production turns still needs the per-agent consent the control
plane holds, exactly as with the variable.

A ROLLOUT CHILD CAN READ THIS FILE. ``isolation.build_child_env`` strips
every ``PERCEPTEYE_`` variable from a rollout child, but ``HOME`` passes
through, so model-authored code with a shell could ``cat`` it -- as it could
``~/.aws/credentials``. The owner accepted that (R2b) on one condition, which
``_http.ControlPlaneClient`` meets: no request body this SDK sends carries a
Flywheel key, the one it authenticates with or any other.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .errors import ConfigurationError

__all__ = [
    "CREDENTIALS_FILENAME",
    "ENV_HOME",
    "FORMAT_VERSION",
    "ResolvedKey",
    "StoredCredential",
    "announce_stored_key",
    "credentials_path",
    "find_stored",
    "load_stored",
    "login_hint",
    "remove_stored",
    "resolve_api_key",
    "save_stored",
    "stored_key_refusal",
]

ENV_HOME = "PERCEPTEYE_HOME"
CREDENTIALS_FILENAME = "credentials.json"
#: The file's own format version. A reader refuses a version it does not know
#: rather than guess, and a writer refuses to replace one: a newer SDK's file
#: is never rewritten by an older one into a shape that loses its entries.
FORMAT_VERSION = 1

#: The fields every stored entry carries, in the order both SDKs write them.
_FIELDS = (
    "control_plane_url", "agent_id", "api_key", "org_id", "org_name",
    "key_id", "created_at", "expires_at", "client",
)


@dataclass(frozen=True)
class StoredCredential:
    """One stored key: what it opens and how long for."""

    control_plane_url: str
    agent_id: str
    api_key: str
    org_id: str | None = None
    org_name: str | None = None
    key_id: str | None = None
    #: ISO 8601, UTC.
    created_at: str | None = None
    #: ISO 8601, UTC; ``None`` for a key that never expires.
    expires_at: str | None = None
    #: Which SDK stored it, e.g. ``agent-flywheel-python/0.3.0``.
    client: str | None = None

    def expired(self, now: datetime | None = None) -> bool:
        when = _parse_time(self.expires_at)
        if when is None:
            return False
        return when <= (now or datetime.now(timezone.utc))


@dataclass(frozen=True)
class ResolvedKey:
    """The key to use, and where it came from."""

    value: str
    source: Literal["argument", "environment", "stored"]
    #: The file it was read from, for a stored key.
    path: Path | None = None
    stored: StoredCredential | None = None


def credentials_path(environ: Mapping[str, str] | None = None) -> Path:
    """``$PERCEPTEYE_HOME/credentials.json``, or ``~/.percepteye/...``.

    The same home the goose recorder (``turns_root``) and the Node SDK
    (``percepteyeHome``) use, by the same rule: a non-empty
    ``PERCEPTEYE_HOME`` wins.
    """
    env = os.environ if environ is None else environ
    home = env.get(ENV_HOME)
    root = Path(home) if home else Path.home() / ".percepteye"
    return root / CREDENTIALS_FILENAME


def _normalise_url(url: str) -> str:
    return url.strip().rstrip("/")


def _parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # ``Z`` is what JavaScript's ``toISOString`` writes; Python before
        # 3.11 parses only the offset spelling.
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _read(path: Path) -> list[dict[str, Any]]:
    """Every entry in ``path``; ``[]`` when there is no file.

    Raises :class:`ConfigurationError` naming the file when it exists and
    cannot be used: unreadable, not JSON, or a version this SDK does not
    know.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        raise ConfigurationError(
            f"{path} is not valid JSON ({exc}); move it aside and run "
            "`agent-flywheel login` again"
        ) from exc
    if not isinstance(doc, dict) or doc.get("version") != FORMAT_VERSION:
        version = doc.get("version") if isinstance(doc, dict) else None
        raise ConfigurationError(
            f"{path} is credentials format version {version!r}; this SDK "
            f"reads version {FORMAT_VERSION}. Upgrade agent-flywheel."
        )
    entries = doc.get("credentials")
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


def _entry(raw: Mapping[str, Any]) -> StoredCredential | None:
    url, agent, key = raw.get("control_plane_url"), raw.get("agent_id"), raw.get("api_key")
    if not (isinstance(url, str) and isinstance(agent, str) and isinstance(key, str)):
        return None
    if not (url and agent and key):
        return None
    extra = {
        f: raw.get(f) if isinstance(raw.get(f), str) else None
        for f in _FIELDS[3:]
    }
    return StoredCredential(
        control_plane_url=url, agent_id=agent, api_key=key, **extra,
    )


def load_stored(environ: Mapping[str, str] | None = None) -> list[StoredCredential]:
    """Every well-formed entry in the file. Raises on an unusable file."""
    entries = (_entry(e) for e in _read(credentials_path(environ)))
    return [e for e in entries if e is not None]


def _match(cred: StoredCredential, url: str, agent_id: str) -> bool:
    return (
        _normalise_url(cred.control_plane_url) == _normalise_url(url)
        and cred.agent_id == agent_id
    )


def _lookup(
    control_plane_url: str, agent_id: str, environ: Mapping[str, str] | None,
) -> StoredCredential | None:
    for cred in load_stored(environ):
        if _match(cred, control_plane_url, agent_id):
            return cred
    return None


def find_stored(
    control_plane_url: str,
    agent_id: str,
    environ: Mapping[str, str] | None = None,
) -> StoredCredential | None:
    """The UNEXPIRED key stored for this agent on this control plane.

    An unusable file is said once, on stderr, and reads as no key: a broken
    file must not stop an agent that has no other way to start from saying
    what it needs, and the sentence names the file to fix.
    """
    try:
        cred = _lookup(control_plane_url, agent_id, environ)
    except ConfigurationError as exc:
        _say_once(("unusable", str(exc)), f"[percepteye] ignoring stored keys: {exc}")
        return None
    if cred is None or cred.expired():
        return None
    return cred


def stored_key_refusal(
    control_plane_url: str,
    agent_id: str,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """Why the key stored for this agent cannot be used, or ``None``.

    ``None`` both when one can be used and when none is stored; this names
    only a stored key that EXISTS and is refused (expired), so a message can
    say "it expired" rather than "there is none".
    """
    try:
        cred = _lookup(control_plane_url, agent_id, environ)
    except ConfigurationError as exc:
        return str(exc)
    if cred is not None and cred.expired():
        return (
            f"the key stored for agent {agent_id!r} expired on "
            f"{cred.expires_at}"
        )
    return None


def login_hint(agent_id: str | None) -> str:
    """The command that gets this agent a key, as a sentence fragment."""
    target = f" --agent-id {agent_id}" if agent_id else ""
    return f"run `agent-flywheel login{target}` to create one"


def save_stored(
    cred: StoredCredential, environ: Mapping[str, str] | None = None,
) -> Path:
    """Store ``cred``, replacing any key stored for the same agent and control
    plane. Returns the file written.

    ATOMIC: the new file is written beside the old one, created 0600 so the
    key is never readable by anyone else even for a moment, flushed, and
    renamed over it. A directory this creates is 0700.
    """
    path = credentials_path(environ)
    existing = _read(path)  # raises rather than lose entries it cannot read
    kept = [
        e for e in existing
        if not ((c := _entry(e)) is not None
                and _match(c, cred.control_plane_url, cred.agent_id))
    ]
    kept.append({f: v for f, v in asdict(cred).items()})
    _write(path, kept)
    return path


def remove_stored(
    control_plane_url: str,
    agent_id: str,
    environ: Mapping[str, str] | None = None,
) -> bool:
    """Forget the key stored for this agent on this control plane.

    True if one was stored. Local only: the key itself stays live until it is
    revoked in Mission Control (Settings), which is what stops it everywhere.
    """
    path = credentials_path(environ)
    existing = _read(path)
    kept = [
        e for e in existing
        if not ((c := _entry(e)) is not None
                and _match(c, control_plane_url, agent_id))
    ]
    if len(kept) == len(existing):
        return False
    _write(path, kept)
    return True


def _write(path: Path, entries: list[dict[str, Any]]) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    body = json.dumps(
        {"version": FORMAT_VERSION, "credentials": entries}, indent=2,
    ) + "\n"
    fd, tmp = tempfile.mkstemp(
        prefix=".credentials.", suffix=".tmp", dir=str(path.parent),
    )
    try:
        # mkstemp creates 0600 already; said again for platforms that differ.
        os.chmod(tmp, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


#: What this process has already said, so each sentence is said once.
_SAID: set[tuple[str, ...]] = set()


def _say_once(key: tuple[str, ...], line: str) -> None:
    if key in _SAID:
        return
    _SAID.add(key)
    print(line, file=sys.stderr, flush=True)


def announce_stored_key(resolved: ResolvedKey, control_plane_url: str) -> None:
    """Say, once per process and agent, that a stored key is being used.

    On stderr, so a CLI piping this process's stdout is not interrupted, and
    once, so a production server asking for its policy on a timer is not.
    """
    cred = resolved.stored
    if resolved.source != "stored" or cred is None:
        return
    _say_once(
        ("stored", str(resolved.path), cred.agent_id, _normalise_url(control_plane_url)),
        f"[percepteye] using the key stored for agent {cred.agent_id!r} in "
        f"{resolved.path} (control plane {control_plane_url}). Run "
        f"`agent-flywheel logout --agent-id {cred.agent_id}` to stop.",
    )


def resolve_api_key(
    explicit: str | None,
    *,
    agent_id: str | None,
    control_plane_url: str,
    environ: Mapping[str, str] | None = None,
    announce: bool = True,
) -> ResolvedKey | None:
    """The key to use, by the three-step rule in this module's docstring.

    ``None`` when there is none. ``agent_id=None`` (a caller that does not
    know which agent it speaks for) never reads the file: a stored key is
    bound to one agent and must not be lent to another.
    """
    from . import ENV_API_KEY

    env = os.environ if environ is None else environ
    if explicit:
        return ResolvedKey(explicit, "argument")
    from_env = env.get(ENV_API_KEY)
    if from_env:
        return ResolvedKey(from_env, "environment")
    if not agent_id:
        return None
    cred = find_stored(control_plane_url, agent_id, environ)
    if cred is None:
        return None
    resolved = ResolvedKey(
        cred.api_key, "stored", path=credentials_path(environ), stored=cred,
    )
    if announce:
        announce_stored_key(resolved, control_plane_url)
    return resolved
