"""The numbers this project publishes about itself are re-derived, not copied.

The suite size appears in four places -- the README's ``tests`` badge, the
README's Development section, ``website/docs/python.md`` and the docs-site
landing stats -- and nothing kept them equal. They drifted to three DIFFERENT
values at once (749, 695 and 509), and the commit that noticed corrected the
badge, left the prose fifty-four short, and added a comment above the badge
saying *"that badge is hardcoded and nothing checks it"*.

A comment is not a mechanism. This is: the count is collected from the suite
itself, with the same recipe the README records, and every published copy has
to match it.

WHY IT IS WORTH A TEST AT ALL. The number is the one claim on the page that a
reader can check, and the argument the page makes is that this project does not
round anything up. Two numbers fifty-four apart in one document spends exactly
the credibility the number was there to buy -- and the CI comment records what a
wrong anchor costs: seventy-two tests once went silently absent from a green
build.

Only files that are PRESENT are checked. ``website/`` is not packaged, so a run
from inside an unpacked sdist checks the README and says so.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Where a suite size is published, and how to find it. Each pattern's first
#: group is the number. Add a row when a new surface starts quoting it --
#: NOT a comment asking the next author to remember.
PUBLISHED = (
    ("README.md", re.compile(r"tests-(\d+)_·_0_skips")),
    ("README.md", re.compile(r"^(\d+) tests, and CI permits", re.M)),
    ("website/docs/python.md", re.compile(r"the (\d+)-test suite")),
    ("website/docs/index.md",
     re.compile(r'fw-stat__n">(\d+)</div><div class="fw-stat__l">tests,')),
)


def _collected() -> int:
    """The suite's own size, by the recipe the README documents."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-p", "no:warnings"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"collection failed:\n{proc.stdout}{proc.stderr}"
    total = sum(
        int(m.group(1))
        for m in re.finditer(r"^tests/.*: (\d+)$", proc.stdout, re.M)
    )
    assert total > 0, f"collected nothing:\n{proc.stdout}"
    return total


@pytest.fixture(scope="module")
def collected() -> int:
    return _collected()


def test_every_published_test_count_matches_the_suite(collected):
    wrong = []
    checked = 0
    for rel, pattern in PUBLISHED:
        path = ROOT / rel
        if not path.is_file():
            continue          # not packaged; nothing to be wrong about here
        text = path.read_text(encoding="utf-8")
        m = pattern.search(text)
        assert m, (
            f"{rel} no longer publishes a count in the shape this checks. "
            f"Either the wording moved -- update the pattern -- or the claim "
            f"was dropped, and then so should the row be.")
        checked += 1
        if int(m.group(1)) != collected:
            wrong.append(f"{rel}: says {m.group(1)}, suite has {collected}")
    assert checked, "no published count was reachable from this tree"
    assert not wrong, (
        "published test counts have drifted from the suite:\n  "
        + "\n  ".join(wrong)
        + "\nRe-derive with: pytest tests/ --collect-only -q -p no:warnings"
          " | awk -F': ' '/^tests\\/.*: [0-9]+$/{n+=$2} END{print n}'")


def test_the_readme_and_the_docs_site_agree_with_each_other(collected):
    """Cheap, and it is the failure that actually shipped.

    The badge was corrected and the prose sixty lines below it was not, so one
    page carried two counts fifty-four apart.
    """
    found = {}
    for rel, pattern in PUBLISHED:
        path = ROOT / rel
        if not path.is_file():
            continue
        m = pattern.search(path.read_text(encoding="utf-8"))
        if m:
            found.setdefault(int(m.group(1)), []).append(rel)
    assert len(found) == 1, f"the published counts disagree with each other: {found}"
