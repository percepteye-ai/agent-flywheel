"""The customer pages both SDKs share say the same thing in both repositories.

The README, the quickstart, the production guide, the OpenTelemetry push page,
the FAQ, the troubleshooting page and the dashboard click paths exist in this
repository and in the Node SDK's, and a reader moving between the two must not
meet two answers. A divergence between them is worse than a stale line: it is
two stale lines, one of them wrong in a way nobody is looking for.

So the rule is mechanical. Everything outside a ``<!-- lang -->`` ...
``<!-- /lang -->`` region is byte-identical in both copies, and both copies
have the same number of regions, in the same places; only what sits inside a
region is the language's own (its code, its API names). The README is compared
after its links to the docs pages are written the same way in both
(``website/docs/`` here, ``docs/`` there).

The Node SDK's own suite holds the same rule from its side
(``test/shared-docs.test.js``), so an edit to either copy alone turns one of
the two red. Finding the Node checkout follows the one rule every
cross-language test here uses (``tests/node_sdk.py``): auto-discovery may skip,
a named ``PERCEPTEYE_NODE_SDK`` may not.
"""
from __future__ import annotations

import difflib
import re
from pathlib import Path

import pytest

from .node_sdk import node_sdk

ROOT = Path(__file__).resolve().parents[1]

#: This repository's copy -> the Node repository's copy.
SHARED = {
    "README.md": "README.md",
    "website/docs/quickstart.md": "docs/quickstart.md",
    "website/docs/production.md": "docs/production.md",
    "website/docs/otlp.md": "docs/otlp.md",
    "website/docs/faq.md": "docs/faq.md",
    "website/docs/troubleshooting.md": "docs/troubleshooting.md",
    "website/docs/dashboard.md": "docs/dashboard.md",
}

LANG = re.compile(r"<!-- lang -->.*?<!-- /lang -->", re.S)


def shared_parts(text: str, *, readme: bool) -> list[str]:
    """The text outside the language regions, in order."""
    if readme:
        text = text.replace("](website/docs/", "](docs/")
    return LANG.split(text)


def first_difference(ours: list[str], theirs: list[str]) -> str:
    for i, (a, b) in enumerate(zip(ours, theirs, strict=False)):
        if a != b:
            diff = difflib.unified_diff(
                a.splitlines(), b.splitlines(), "python", "node", lineterm="", n=1)
            return f"shared part {i} differs:\n" + "\n".join(diff)
    return (f"{len(ours) - 1} language region(s) here, "
            f"{len(theirs) - 1} in the Node copy")


@pytest.mark.parametrize("ours,theirs", sorted(SHARED.items()))
def test_a_shared_page_matches_the_node_sdks_copy(ours, theirs):
    here = ROOT / ours
    if not here.exists():
        pytest.skip(f"{ours} is not in this tree (website/ is not packaged)")
    sdk, _named = node_sdk()
    if sdk is None:
        pytest.skip("the Node package is not checked out here")
    there = sdk / theirs
    assert there.exists(), f"the Node SDK has no {theirs} to match {ours}"
    readme = ours == "README.md"
    a = shared_parts(here.read_text(encoding="utf-8"), readme=readme)
    b = shared_parts(there.read_text(encoding="utf-8"), readme=readme)
    assert a == b, f"{ours} and the Node SDK's {theirs}: " + first_difference(a, b)


def test_every_language_region_is_closed():
    """An unclosed marker would swallow the rest of the page into one region,
    and the comparison above would pass on nothing."""
    for ours in SHARED:
        path = ROOT / ours
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        opened = text.count("<!-- lang -->")
        closed = text.count("<!-- /lang -->")
        assert opened == closed == len(LANG.findall(text)), (
            f"{ours}: {opened} opened, {closed} closed")
        assert LANG.split(text)[0].strip(), f"{ours} has no shared text before its first region"
