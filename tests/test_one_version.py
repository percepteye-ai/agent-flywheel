"""The package states one version, and everything that reports it agrees.

The version is written in ``agent_flywheel/_version.py``. Three other places
state or report it: ``pyproject.toml`` (what the build stamps, and what the
release workflow checks the tag against), the goose plugin manifest the wheel
ships, and the User-Agent every control-plane request carries.

The User-Agent was a hand-copied ``"0.1.0"`` of its own, so the 0.1.1 bump
would have shipped a release announcing itself as 0.1.0 -- and the control
plane files bugs by that header. It is built from ``__version__`` now; these
tests hold the remaining copies to the same number, so a bump that misses one
fails here instead of in a customer's logs. The lockfile records it as well,
and ``CHANGELOG.md`` must open with its entry.
"""
from __future__ import annotations

import json
import pathlib
import re

import pytest

import agent_flywheel
from agent_flywheel import _http
from agent_flywheel.hosts import GOOSE_MANIFEST

ROOT = pathlib.Path(__file__).resolve().parents[1]


def test_pyproject_and_the_goose_manifest_state_the_package_version():
    # Read with a regex rather than tomllib: CI still runs Python 3.10.
    m = re.search(r'^version = "([^"]+)"$',
                  (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    assert m, "pyproject.toml no longer states a static version"
    manifest = json.loads(GOOSE_MANIFEST.read_text(encoding="utf-8"))
    assert {m.group(1), manifest["version"]} == {agent_flywheel.__version__}, (
        f"pyproject says {m.group(1)}, the goose manifest {manifest['version']}, "
        f"agent_flywheel/_version.py {agent_flywheel.__version__}: bump them together")


def test_the_user_agent_reports_the_package_version():
    assert _http._SDK_UA == f"agent-flywheel-python/{agent_flywheel.__version__}"


def test_the_lockfile_records_the_package_version():
    """``uv lock`` after the bump is a release step, and a lockfile still
    naming the previous version is the step that was missed: ``uv sync
    --locked`` then refuses the checkout. Not packaged, so an unpacked sdist
    has nothing to be wrong about."""
    lock = ROOT / "uv.lock"
    if not lock.is_file():
        pytest.skip("uv.lock is not packaged")
    m = re.search(r'^name = "agent-flywheel"\nversion = "([^"]+)"$',
                  lock.read_text(encoding="utf-8"), re.M)
    assert m, "uv.lock no longer records this package's version"
    assert m.group(1) == agent_flywheel.__version__, (
        f"uv.lock records {m.group(1)}, agent_flywheel/_version.py "
        f"{agent_flywheel.__version__}: run `uv lock`")


def test_the_changelog_says_what_this_version_changed():
    """A version with no entry is a release nobody can read the changes of.
    Not packaged, so an unpacked sdist skips."""
    changelog = ROOT / "CHANGELOG.md"
    if not changelog.is_file():
        pytest.skip("CHANGELOG.md is not packaged")
    headings = re.findall(r"^## (\S+)", changelog.read_text(encoding="utf-8"), re.M)
    assert headings and headings[0] == agent_flywheel.__version__, (
        f"CHANGELOG.md's newest entry is {headings[:1]}, the package is "
        f"{agent_flywheel.__version__}: add its entry at the top")
