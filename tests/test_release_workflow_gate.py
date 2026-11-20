"""Nothing reaches PyPI that CI has not passed on the same commit (XR-3).

The release workflow ran only ``python -m build`` and ``twine check`` before
publishing -- no test, no lint -- and a manual dispatch with ``target=pypi``
skipped the tag-matches-version check, so any branch could publish an untagged
build, irreversibly. Every action was referenced by a movable tag.

Read as text, not parsed: the bare CI job installs no YAML library, and these
are properties of a few lines.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"
CI = (WORKFLOWS / "ci.yml").read_text(encoding="utf-8")
RELEASE = (WORKFLOWS / "workflow.yml").read_text(encoding="utf-8")


def _job(text: str, name: str) -> str:
    """The body of one top-level job: from ``  name:`` to the next job."""
    m = re.search(rf"^  {re.escape(name)}:\n(.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
                  text, re.M | re.S)
    assert m, f"no job {name!r}"
    return m.group(1)


def test_the_release_runs_the_whole_ci_suite_before_it_builds():
    ci_job = _job(RELEASE, "ci")
    assert re.search(r"^    uses: \./\.github/workflows/ci\.yml$", ci_job, re.M)
    assert re.search(r"^  workflow_call:", CI, re.M), "ci.yml cannot be called"
    assert re.search(r"^    needs: ci$", _job(RELEASE, "build"), re.M)
    for publish in ("pypi", "testpypi"):
        assert re.search(r"^    needs: build$", _job(RELEASE, publish), re.M)


def test_a_manual_pypi_publish_passes_the_tag_check_too():
    step = re.search(r"- name: the tag must match the version in pyproject\n(.*?)\n      - ",
                     _job(RELEASE, "build"), re.S)
    assert step, "the tag check is gone"
    body = step.group(1)
    cond = re.search(r"^        if: (.*)$", body, re.M).group(1)
    assert "workflow_dispatch" in cond and "inputs.target == 'pypi'" in cond, cond
    # ...and a dispatch on a branch is refused, not compared against nothing.
    assert 'ref.startswith("refs/tags/v")' in body


def test_tokens_are_read_only_by_default():
    for text in (CI, RELEASE):
        assert re.search(r"^permissions:\n  contents: read$", text, re.M)


def test_every_third_party_action_is_pinned_to_a_commit():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        for line in path.read_text(encoding="utf-8").splitlines():
            m = re.search(r"uses: (\S+)(.*)$", line)
            if not m or m.group(1).startswith("./"):
                continue
            ref = m.group(1).split("@", 1)[1] if "@" in m.group(1) else ""
            assert re.fullmatch(r"[0-9a-f]{40}", ref), f"{path.name}: {line.strip()}"
            assert re.match(r"\s+# v\d", m.group(2)), (
                f"{path.name}: name the pinned release in a trailing comment: {line.strip()}")
