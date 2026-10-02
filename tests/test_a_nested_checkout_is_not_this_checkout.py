"""The source guards read this checkout, and only this checkout.

Every guard that walks the source tree walks it through ``tests/repo_files.py``,
which prunes any directory below the root that carries a ``.git`` entry. These
tests hold that from both sides, because each side alone is satisfied by a
broken walker:

* NOTHING NESTED IS READ. A worktree of another branch under
  ``.claude/worktrees/`` (where a coding agent puts them), a stale one beside
  it, a clone nested inside the package itself and a virtualenv nobody
  gitignored, each holding an offender for every guard, leave every guard
  green. In a sibling repo exactly this made two seam guards fail on a clean
  merged main, over another branch's copy of the code.
* NOTHING OWN IS LOST. The same guards, given the same offenders in this
  checkout's own tree -- in files no git index has seen, as a new file is
  before anyone stages it -- go red, each naming its own planted file. A
  walker that pruned too much would pass the first half by reading nothing.

The guards run in a subprocess against a COPY whose ``src`` is the package
they import, laid out both ways the published-files scan can meet it: a tree
git has never seen, which it walks (as it walks an unpacked sdist), and a git
checkout, which it asks ``git ls-files`` about and narrows to what the walker
yields -- with every planted file untracked, as a new one is before anyone
stages it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from .repo_files import NOT_SOURCE, ROOT, repo_files
from .test_one_control_plane_spelling import _FOREIGN
from .test_oss_boundary import INTERNAL_DETAILS, INTERNAL_NAMES

#: A developer path, assembled so this file does not publish one itself.
_DEV_PATH = "/".join(("", "Users", "someone", "Projects", "app", ""))

#: Every guard that walks the source tree, and the offender that must turn it
#: red when it sits in this checkout's own tree. The path is relative to the
#: checkout root; a nested checkout gets the same file renamed ``nested_*``.
GUARDS = (
    ("tests/test_oss_boundary.py::test_no_internal_system_names_in_the_package",
     "src/agent_flywheel/planted_name.py", f"# {INTERNAL_NAMES[0]}\n"),
    ("tests/test_oss_boundary.py::test_no_internal_invariants_disclosed",
     "planted_detail.md", f"{INTERNAL_DETAILS[0]}\n"),
    ("tests/test_oss_boundary.py::test_no_internal_system_names_anywhere_published",
     "examples/planted_name.md", f"{INTERNAL_NAMES[1]}\n"),
    ("tests/test_oss_boundary.py::test_no_local_developer_paths_published",
     "examples/planted_path.sh", f"cd {_DEV_PATH}\n"),
    ("tests/test_one_control_plane_spelling.py::"
     "test_no_library_module_reads_a_second_spelling",
     "src/agent_flywheel/planted_env.py",
     f"import os\nos.environ.get({_FOREIGN[0]!r})\n"),
    ("tests/test_the_delivery_mechanism_cannot_become_the_outage.py::"
     "test_the_rollout_timeout_is_declared_in_exactly_one_place",
     "src/agent_flywheel/planted_timeout.py", "TIMEOUT_S = 900.0\n"),
)

#: What the guard modules need beside themselves to import.
_SUPPORT = ("__init__.py", "conftest.py", "repo_files.py",
            "test_an_outcome_is_never_silently_dropped.py")

_PACKAGE = "src/agent_flywheel/"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _nested(rel: str) -> str:
    return rel.replace("planted_", "nested_")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid",
         "-c", "commit.gpgsign=false", *args],
        cwd=repo, check=True, capture_output=True,
    )


def _copy_of_this_checkout(tmp_path: Path, git: bool) -> Path:
    """The package and the guard modules, committed to a fresh repository
    when ``git`` is set and in a directory git has never seen otherwise."""
    repo = tmp_path / "repo"
    for src in repo_files(ROOT / "src"):
        dest = repo / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    modules = {node.split("::")[0] for node, _, _ in GUARDS}
    for name in (*_SUPPORT, *(Path(m).name for m in modules)):
        _write(repo / "tests" / name, (ROOT / "tests" / name).read_text(encoding="utf-8"))
    if git:
        _git(repo, "init", "-q")
        _git(repo, "add", "src", "tests")
        _git(repo, "commit", "-q", "-m", "base")
    return repo


def _plant_nested_checkouts(repo: Path, git: bool) -> None:
    """Every offender, inside trees that are not this checkout.

    A linked worktree at the root (its ``.git`` is a FILE), and a clone nested
    inside the package (its ``.git`` is a directory), so the guards rooted at
    the package meet one too. In a git checkout both are REAL.

    Beside them, in both layouts, a stale worktree whose ``.git`` names a
    gitdir that no longer exists: git cannot resolve it, so ``git ls-files
    --others`` lists every file under it, and a guard that took git's word for
    where a checkout starts would read them all. And a virtualenv nobody
    gitignored, known only by its ``pyvenv.cfg``:
    Python 3.10-3.12's ``venv`` writes no ``.gitignore`` of its own, so
    ``python -m venv env`` leaves every installed package for ``git ls-files
    --others`` to list, and pip's own vendored sources match a name on the
    boundary scan's list.
    """
    worktree = repo / ".claude" / "worktrees" / "other-branch"
    stale = repo / ".claude" / "worktrees" / "stale-branch"
    venv = repo / ".fresh"
    clone = repo / _PACKAGE / "nested_clone"
    if git:
        _git(repo, "worktree", "add", "-q", "--detach", str(worktree))
        _git(repo, "init", "-q", str(clone))
    else:
        _write(worktree / ".git", "gitdir: /elsewhere/.git/worktrees/other-branch\n")
        _write(clone / ".git" / "HEAD", "ref: refs/heads/main\n")
    _write(stale / ".git", f"gitdir: {repo / '.git' / 'worktrees' / 'stale-branch'}\n")
    _write(venv / "pyvenv.cfg", f"home = /usr/bin\ncommand = python -m venv {venv}\n")
    for _, rel, text in GUARDS:
        for tree in (worktree, stale, venv):
            _write(tree / _nested(rel), text)
        if rel.startswith(_PACKAGE):
            _write(clone / Path(_nested(rel)).name, text)


def _run_guards(repo: Path, tmp_path: Path) -> dict[str, str | None]:
    """Each guard's outcome: None for a pass, else ``"<failure|error|skipped>: <text>"``."""
    report = tmp_path / "report.xml"
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env["PYTHONPATH"] = str(repo / "src")  # the copy's package, not this one
    env["GIT_CEILING_DIRECTORIES"] = str(repo.parent)  # never an enclosing repo
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
         f"--basetemp={tmp_path / 'basetemp'}", f"--junitxml={report}",
         *(node for node, _, _ in GUARDS)],
        cwd=repo, env=env, capture_output=True, text=True, timeout=300,
    )
    assert report.is_file(), f"pytest wrote no report:\n{proc.stdout}{proc.stderr}"
    outcomes: dict[str, str | None] = {}
    for case in ET.parse(report).iter("testcase"):
        node = f"{case.get('classname', '').replace('.', '/')}.py::{case.get('name')}"
        outcomes[node] = None
        for tag in ("failure", "error", "skipped"):
            found = case.find(tag)
            if found is not None:
                outcomes[node] = f"{tag}: {found.get('message')}\n{found.text or ''}"
                break
    assert set(outcomes) == {node for node, _, _ in GUARDS}, (
        f"the guards that ran are not the guards named:\n{proc.stdout}{proc.stderr}"
    )
    return outcomes


def test_the_walker_prunes_every_tree_that_is_not_this_checkout(tmp_path):
    """Nested checkouts by their ``.git`` entry, virtualenvs by their marker,
    caches and build output by name -- and nothing else.

    The root is a linked worktree, as this checkout often is, so its own
    ``.git`` is a file that must not be yielded as source. Nothing here is
    tracked: the walker never asks git, so a file nobody has staged yet is
    found like any other.
    """
    root = tmp_path / "checkout"
    _write(root / ".git", "gitdir: /elsewhere/.git/worktrees/checkout\n")
    _write(root / "src/pkg/mod.py", "")
    _write(root / "src/pkg/new.py", "")
    _write(root / ".claude/settings.json", "{}")
    _write(root / ".claude/worktrees/other/.git", "gitdir: /elsewhere\n")
    _write(root / ".claude/worktrees/other/src/pkg/mod.py", "")
    _write(root / ".worktrees/old/.git", "gitdir: /elsewhere\n")
    _write(root / ".worktrees/old/src/pkg/mod.py", "")
    _write(root / "src/pkg/vendored/.git/HEAD", "ref: refs/heads/main\n")
    _write(root / "src/pkg/vendored/mod.py", "")
    _write(root / ".fresh/pyvenv.cfg", "home = /usr/bin\n")
    _write(root / ".fresh/lib/site.py", "")
    for name in NOT_SOURCE - {".git"}:  # a .git here would prune src itself
        _write(root / "src" / name / "x.py", "")

    def rel(paths, top):
        return [p.relative_to(top).as_posix() for p in paths]

    assert rel(repo_files(root), root) == [
        ".claude/settings.json", "src/pkg/mod.py", "src/pkg/new.py",
    ]
    pkg = root / "src" / "pkg"
    assert rel(repo_files(pkg, suffix=".py"), pkg) == ["mod.py", "new.py"]


def test_the_walker_keeps_every_file_this_checkout_tracks():
    """The other edge of the prune: a name in ``NOT_SOURCE`` or a stray
    ``.git`` that swallowed a directory of real source would pass the test
    above and silently shrink every guard's scope."""
    proc = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.skip("not a git checkout (an unpacked sdist): nothing is tracked")
    tracked = {rel for rel in proc.stdout.split("\0")
               if rel and (ROOT / rel).is_file()}
    assert tracked, "git ls-files returned nothing -- the check would pass vacuously"
    walked = {p.relative_to(ROOT).as_posix() for p in repo_files()}
    assert tracked <= walked, f"the walker drops tracked files: {sorted(tracked - walked)}"


_LAYOUTS = pytest.mark.parametrize("git", [False, True],
                                   ids=["unpacked-tree", "git-checkout"])


@_LAYOUTS
def test_no_guard_reads_a_nested_checkout(tmp_path, git):
    repo = _copy_of_this_checkout(tmp_path, git)
    _plant_nested_checkouts(repo, git)
    failed = {node: text for node, text in _run_guards(repo, tmp_path).items() if text}
    assert not failed, "a guard read a checkout that is not this one:\n" + "\n".join(
        f"{node}\n{text}" for node, text in failed.items()
    )


@_LAYOUTS
def test_every_guard_still_catches_an_offender_in_this_checkout(tmp_path, git):
    repo = _copy_of_this_checkout(tmp_path, git)
    _plant_nested_checkouts(repo, git)
    for _, rel, text in GUARDS:
        _write(repo / rel, text)
    outcomes = _run_guards(repo, tmp_path)
    for node, rel, _ in GUARDS:
        text = outcomes[node] or "passed"
        assert text.startswith("failure: "), f"{node} did not fail on {rel}: {text}"
        assert Path(rel).name in text, f"{node} failed, but not on {rel}:\n{text}"
        assert "nested_" not in text, f"{node} named a nested checkout's file:\n{text}"
