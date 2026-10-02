"""This checkout's own files, for every test that reads the source tree. ONE definition.

The guards that read source -- the boundary scan, the one-spelling check, the
one-declaration check for the rollout timeout -- each walked the tree their own
way: three ``rglob`` calls with no exclusions at all, and one fallback with a
list of directory names plus a virtualenv probe of its own. Each answer was
right only for the trees its author happened to have.

A NESTED CHECKOUT IS NOT THIS CHECKOUT. ``git worktree add`` inside the working
tree is ordinary -- ``.worktrees/<name>`` is the convention this repo's own
``.gitignore`` names, and a coding agent puts its worktrees under
``.claude/worktrees/<name>`` -- and each one is a complete copy of another
branch below the root. A guard that walks into it reports that branch's code as
this one's: it fails on a defect this checkout does not have, or a fixed defect
stays red until somebody deletes a directory that has nothing to do with the
change. The same shape broke two seam guards in a sibling repo on a clean
merged main. So any directory below ``top`` that carries a ``.git`` entry -- a
FILE for a linked worktree, a directory for a nested clone -- is pruned with
everything under it. That is the entry git itself looks for to find where a
repository starts, not a name somebody has to remember to add.

A VIRTUALENV IS FOUND BY ITS OWN MARKER, ``pyvenv.cfg``, as well as by the
conventional names. The name is a convention; the marker is the definition.
An auditor who unpacked the sdist and built the environment beside the suite
as ``.fresh`` had every installed third-party package scanned as though this
project published it: three boundary tests failed, naming Google ADK and two
dependency licence files as leaks.

UNTRACKED FILES ARE KEPT. This is a walk of the tree, not ``git ls-files``: the
new module nobody has ``git add``-ed yet is exactly where a new offender lives,
and it has to be caught on the run before the commit, not after. (Inside a
git checkout the published-files scan also asks ``git ls-files`` -- tracked and
untracked files alike, minus what ``.gitignore`` excludes -- and keeps only
what this walker yields, so this module is what decides "this checkout's
source" there too: git lists a virtualenv nobody gitignored, and every file of
a nested worktree whose ``.git`` it cannot resolve.)
"""
from __future__ import annotations

import os
from pathlib import Path

#: The root of this checkout.
ROOT = Path(__file__).resolve().parents[1]

#: Directory names that are never this project's source, at any depth: git's
#: own store, virtualenvs, installed dependencies, build output and tool caches.
NOT_SOURCE = frozenset({
    ".git", ".venv", "venv", "node_modules", "__pycache__", "build", "dist",
    ".eggs", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
})


def _another_tree(directory: Path) -> bool:
    """Whether ``directory`` holds something other than this checkout's source."""
    return (
        directory.name in NOT_SOURCE
        or os.path.lexists(directory / ".git")
        or (directory / "pyvenv.cfg").is_file()
    )


def repo_files(top: Path = ROOT, *, suffix: str = "") -> list[Path]:
    """Every file of this checkout under ``top`` whose name ends in ``suffix``, sorted.

    ``top`` itself is never pruned -- the root of a checkout carries ``.git``
    by definition -- but every directory below it is asked :func:`_another_tree`
    before the walk descends. Symlinked directories are not followed.
    """
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(top):
        here = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not _another_tree(here / d)]
        found.extend(
            here / name for name in filenames
            if name.endswith(suffix) and name != ".git"
        )
    return sorted(found)
