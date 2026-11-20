"""Internal planning notes and a coding agent's settings are not in the
public tree, and cannot come back by a ``git add -A`` (PY-M20).

``docs/plan-*.md`` (one named a competitor) and ``.claude/launch.json`` were
tracked in the public repository. They are removed and gitignored; what is in
history stays there until an owner purges it.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="needs this repository's git checkout",
)


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def _internal(path: str) -> bool:
    return path.startswith(".claude/") or (
        path.startswith("docs/plan-") and path.endswith(".md"))


def test_no_internal_note_is_tracked():
    tracked = _git("ls-files").stdout.splitlines()
    assert [p for p in tracked if _internal(p)] == []


def test_a_new_one_is_ignored():
    for path in ("docs/plan-next-quarter.md", ".claude/launch.json",
                 ".claude/settings.local.json"):
        assert _git("check-ignore", "-q", "--no-index", path).returncode == 0, path
