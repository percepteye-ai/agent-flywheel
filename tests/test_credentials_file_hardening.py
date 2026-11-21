"""The credentials file survives concurrent writers and corruption.

What each test pins, and the defect it exists to catch (PY-H11, PY-M3): a
concurrent save never loses a key; the directory holding keys is 0700; the
temp fd is closed when chmod fails; an ``expires_at`` nobody can read is
expired, and any RFC 3339 fraction reads on every Python; a ``credentials``
that is not a list is refused, not wiped; a blank key is no key; hints quote
the agent id.
"""
from __future__ import annotations

import json
import os
import stat
import tempfile
import threading

import pytest

from agent_flywheel import credentials as creds
from agent_flywheel.errors import ConfigurationError

CP = "https://cp.test/api/flywheel/v1"
AGENT = "my-agent"
KEY = "pefw_" + "K" * 43


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "pe-home"
    monkeypatch.setenv("PERCEPTEYE_HOME", str(root))
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.setattr(creds, "_SAID", set())
    return root


# ── PY-H11 / PY-M3: the file ─────────────────────────────────────────


def _cred(agent, key="pefw_" + "Z" * 43, **over):
    return creds.StoredCredential(control_plane_url=CP, agent_id=agent, api_key=key, **over)


def test_a_concurrent_save_never_loses_the_other_key(home, monkeypatch):
    """A reads the file, B saves while A holds what it read, A writes: without
    the lock A's write silently drops B's key."""
    creds.save_stored(_cred("seed"))
    real_read = creds._read
    a_read, b_done = threading.Event(), threading.Event()

    def read(path):
        entries = real_read(path)
        if threading.current_thread().name == "A":
            a_read.set()
            b_done.wait(1.5)  # B, if the lock lets it, writes now
        return entries

    monkeypatch.setattr(creds, "_read", read)

    def save_b():
        a_read.wait(5)
        creds.save_stored(_cred("b"))
        b_done.set()

    a = threading.Thread(target=lambda: creds.save_stored(_cred("a")), name="A")
    b = threading.Thread(target=save_b, name="B")
    a.start()
    b.start()
    a.join(10)
    b.join(10)
    agents = sorted(c.agent_id for c in creds.load_stored())
    assert agents == ["a", "b", "seed"], agents


def _stale_lock(home):
    """A lock a dead writer left, older than the takeover age."""
    creds._private_dir(home)
    lock = home / creds.LOCK_FILENAME
    lock.write_text("99999 0 deadbeef\n")
    old = os.path.getmtime(lock) - creds.LOCK_STALE_S - 60
    os.utime(lock, (old, old))
    return lock


def test_a_stale_takeover_never_removes_a_fresh_lock_taken_meanwhile(
    home, monkeypatch,
):
    """Between judging a lock stale and renaming it aside, another writer
    takes the stale lock over and creates a FRESH one. The rename then moves
    the fresh lock: it must be put back, and this writer must keep waiting,
    not take the lock while the other writer believes it holds it. Mutation:
    remove whatever was moved aside unchecked -> this writer writes, red."""
    lock = _stale_lock(home)
    real_rename = os.rename
    fresh = "4242 1 c0ffee\n"

    def rename(src, dst, *a, **kw):
        if str(src) == str(lock) and lock.read_text() != fresh:
            # Another writer wins the takeover first and holds a fresh lock.
            lock.unlink()
            lock.write_text(fresh)
        return real_rename(src, dst, *a, **kw)

    monkeypatch.setattr(os, "rename", rename)
    monkeypatch.setattr(creds, "LOCK_WAIT_S", 0.3)
    with pytest.raises(ConfigurationError, match="another process"):
        creds.save_stored(_cred("a"))
    assert lock.read_text() == fresh, "the other writer's fresh lock was removed"
    assert not list(home.glob(creds.LOCK_FILENAME + ".stale-*")), "left aside"


def test_a_writer_releases_only_a_lock_that_still_holds_its_nonce(
    home, monkeypatch,
):
    """A writer whose lock was taken over (it outlived the stale age) must not
    remove the next holder's lock on its way out. Mutation: unlink the lock
    unchecked at release -> the next holder's lock is gone, red."""
    lock = home / creds.LOCK_FILENAME
    theirs = "4242 1 c0ffee\n"
    real_read = creds._read

    def read(path):
        # While this writer holds the lock, another one takes it over.
        lock.unlink()
        lock.write_text(theirs)
        return real_read(path)

    monkeypatch.setattr(creds, "_read", read)
    creds.save_stored(_cred("a"))
    assert lock.read_text() == theirs


def test_a_stale_lock_is_still_taken_over(home):
    lock = _stale_lock(home)
    creds.save_stored(_cred("a"))
    assert [c.agent_id for c in creds.load_stored()] == ["a"]
    assert not lock.exists()
    assert not list(home.glob(creds.LOCK_FILENAME + ".stale-*")), "left aside"


def test_the_directory_holding_keys_is_tightened_to_0700(home):
    home.mkdir(parents=True, mode=0o755)
    os.chmod(home, 0o755)  # as the goose recorder used to leave it
    creds.save_stored(_cred(AGENT))
    assert stat.S_IMODE(os.stat(home).st_mode) == 0o700


def test_a_failed_chmod_closes_the_temp_file(home, monkeypatch):
    fds: list[int] = []
    real_mkstemp = tempfile.mkstemp

    def mkstemp(*a, **kw):
        fd, name = real_mkstemp(*a, **kw)
        fds.append(fd)
        return fd, name

    real_chmod = os.chmod

    def chmod(path, mode, *a, **kw):
        if str(path).endswith(".tmp"):
            raise PermissionError("no")
        return real_chmod(path, mode, *a, **kw)

    monkeypatch.setattr(creds.tempfile, "mkstemp", mkstemp)
    monkeypatch.setattr(creds.os, "chmod", chmod)
    with pytest.raises(PermissionError):
        creds.save_stored(_cred(AGENT))
    (fd,) = fds
    with pytest.raises(OSError):
        os.fstat(fd)
    assert not list(home.glob(".credentials.*"))


@pytest.mark.parametrize(("expires_at", "expired"), [
    ("not a time", True),
    ("2999-13-45T99:00:00Z", True),
    ("2999-01-01T00:00:00.12345Z", False),   # 5 fractional digits (3.10)
    ("2999-01-01T00:00:00.123456789Z", False),
    ("2999-01-01T00:00:00.1+02:00", False),
    ("2001-01-01T00:00:00.5Z", True),
    (None, False),
])
def test_an_unreadable_expiry_is_expired_and_any_rfc3339_fraction_reads(expires_at, expired):
    assert _cred(AGENT, expires_at=expires_at).expired() is expired


def test_a_credentials_field_that_is_not_a_list_is_refused_not_wiped(home):
    home.mkdir(parents=True)
    doc = {"version": 1, "credentials": {"alpha": {"api_key": "pefw_keep"}}}
    (home / "credentials.json").write_text(json.dumps(doc))
    with pytest.raises(ConfigurationError, match="not a list"):
        creds.save_stored(_cred(AGENT))
    assert json.loads((home / "credentials.json").read_text()) == doc


def test_a_blank_key_is_no_key(home, monkeypatch):
    with pytest.raises(ConfigurationError, match="blank key"):
        creds.save_stored(_cred(AGENT, key="   "))
    home.mkdir(parents=True, exist_ok=True)
    (home / "credentials.json").write_text(json.dumps({"version": 1, "credentials": [
        {"control_plane_url": CP, "agent_id": AGENT, "api_key": "  \t"}]}))
    assert creds.find_stored(CP, AGENT) is None
    creds.save_stored(_cred(AGENT, key=KEY))
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "   ")
    got = creds.resolve_api_key(None, agent_id=AGENT, control_plane_url=CP, announce=False)
    assert (got.value, got.source) == (KEY, "stored")


def test_shell_hints_quote_the_agent_id(home, capsys):
    hostile = "a b;touch /tmp/x"
    assert "--agent-id 'a b;touch /tmp/x'" in creds.login_hint(hostile)
    creds.save_stored(_cred(hostile))
    creds.resolve_api_key(None, agent_id=hostile, control_plane_url=CP)
    assert "logout --agent-id 'a b;touch /tmp/x'" in capsys.readouterr().err
