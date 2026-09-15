"""The README's own figures and in-repo links, checked against a PRIVATE repo.

WHY THIS FILE EXISTS. The figures have now broken TWICE the same way. Both
times an edit rewrote every relative ``docs/assets/*.gif`` src to an absolute
``https://raw.githubusercontent.com/...`` URL, reasoning correctly that
``README.md`` is copied verbatim into ``PKG-INFO`` and that relative paths do
not resolve on PyPI. The reasoning is sound and the conclusion is wrong here,
because of a fact neither edit checked: **this repository is private.**

``raw.githubusercontent.com`` is fetched by the browser anonymously and does
not carry a github.com session, so for a private repo it answers 404 to
everybody -- including the owner, viewing their own README while signed in.
The first time, that shipped as "I don't see any GIFs in the readmes today".
A relative src, by contrast, is resolved by GitHub against the repo you are
already authenticated to, so it renders.

The trade-off is settled and it is not close: GitHub is the target surface for
these figures. The package is on PyPI, whose page is this README, but an
absolute URL would not rescue the figures there either: while the repo is
private, ``raw.githubusercontent.com`` answers PyPI's visitors with the same
404. And ``docs/assets/`` is deliberately excluded from the wheel and the
sdist -- ``test_packaging.py`` builds the real artifacts and asserts it -- so
no shipped artifact carries the files. Absolute URLs become correct only once
the repo is PUBLIC, and on that day this test is what tells you it is now safe
to flip.

A comment saying "keep these relative" is what was there the first time. This
is a mechanism: it fails the suite instead of the page.

The last two tests are the bill for choosing relative. A relative path can be
typo'd or outlived by the file it names and GitHub renders that as a dead link
with no error anywhere, so every one of them is resolved against the tree, and
every in-page anchor against the headings actually present -- which is also
what keeps a section move from silently orphaning the nav bar at the top.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: This repository. The sibling Node SDK is a DIFFERENT repo, so absolute links
#: to it are correct and must keep working -- every pattern below is anchored so
#: that ``agent-flywheel-node`` can never match ``agent-flywheel``.
SLUG = "percepteye-ai/agent-flywheel"


def _readable(text: str) -> str:
    """The README as a reader's browser sees it.

    HTML comments and fenced code are dropped: the badge block deliberately
    parks the absolute URLs to restore ON PUBLISH, and the note beside the
    first figure has to be able to name the broken form to explain it.
    """
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"```.*?```", "", text, flags=re.S)


def _slug(heading: str) -> str:
    """GitHub's heading anchor: strip tags and punctuation, spaces to hyphens.

    Emoji are punctuation to the slugger and leave the space in front of them
    behind, which is why every anchor on this page starts with a hyphen.
    """
    text = re.sub(r"<[^>]+>", "", heading)
    text = re.sub(r"[^\w\s-]", "", text, flags=re.U)
    return text.strip("\n").lower().replace(" ", "-")


@pytest.fixture(scope="module")
def body() -> str:
    return _readable(README.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def targets(body: str) -> list[str]:
    """Every link target and image source a reader can follow."""
    return [m.group(1) for m in re.finditer(r'<img[^>]*?\ssrc="([^"]+)"', body, re.S)] + [
        m.group(1) for m in re.finditer(r"\]\(([^)\s]+)\)", body)
    ]


def test_no_figure_is_fetched_from_raw_githubusercontent(body):
    """The regression itself: an anonymous fetch of a private repo is a 404."""
    bad = re.findall(rf"raw\.githubusercontent\.com/{SLUG}/\S*", body)
    assert not bad, (
        "README figures point at raw.githubusercontent.com. This repo is "
        "PRIVATE, so that URL 404s for every unauthenticated fetch -- which is "
        "every <img> fetch, including the owner's own browser -- and the page "
        "renders with broken images:\n  " + "\n  ".join(bad)
        + "\nUse a relative path (docs/assets/...). These become correct only "
          "once the repo is public; see the note above the first figure.")


def test_no_link_into_this_repo_is_an_absolute_github_url(body):
    """Same edit, non-image half. A relative link follows the ref you are
    browsing; ``blob/main`` pins to main and buys nothing on the one surface
    these links are for."""
    bad = re.findall(rf"https://github\.com/{SLUG}/(?:blob|tree)/\S*", body)
    assert not bad, (
        "links to files IN THIS REPO are absolute; keep them relative so they "
        "resolve against whatever ref the reader is on:\n  " + "\n  ".join(bad))


def test_every_relative_target_exists(targets):
    """The cost of relative paths, paid here instead of by the reader."""
    missing = [
        t for t in targets
        if not t.startswith(("http://", "https://", "#", "mailto:"))
        and not (ROOT / t).exists()
    ]
    assert not missing, (
        "README points at paths that are not in the tree -- GitHub renders "
        "these as dead links with no error:\n  " + "\n  ".join(missing))


def test_every_internal_anchor_resolves(body, targets):
    """Guards the nav bar against a section being moved or renamed under it."""
    have = {_slug(m.group(1)) for m in re.finditer(r"^#+\s+(.*)$", body, re.M)}
    dead = [t for t in targets if t.startswith("#") and t[1:] not in have]
    assert not dead, (
        f"README anchors point at headings that are not on the page: {dead}\n"
        f"headings present: {sorted(have)}")
