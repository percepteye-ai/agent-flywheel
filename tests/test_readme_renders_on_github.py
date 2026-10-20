"""The README's figures and in-repo links, checked for the surfaces that show it.

The README is shown in two places: on github.com, and as every release's PyPI
page, because ``README.md`` is copied verbatim into ``PKG-INFO``. PyPI cannot
resolve a relative path, and ``docs/assets/`` is deliberately excluded from
the wheel and the sdist (``test_packaging.py`` builds the real artifacts and
asserts it). So a figure reaches a PyPI reader only by absolute URL, and until
2026-10-20 the PyPI page showed alt text for every one of them.

WHY THIS FILE EXISTS, AND WHY IT USED TO SAY THE OPPOSITE. While this
repository was PRIVATE the rule had to be the reverse. ``raw.githubusercontent
.com`` is fetched by the browser anonymously and does not carry a github.com
session, so for a private repository it answered 404 to everybody, the owner
included, and the figures broke on GitHub twice when an edit made them
absolute. This test then enforced relative paths, and its docstring said it was
"what tells you it is now safe to flip". The repository went public on
2026-10-20, the URLs were flipped, and this test was flipped with them.

If the repository ever goes private again, flip both back together: absolute
URLs 404 for every reader of a private repository.

A comment saying "keep these absolute" is not a mechanism. This is: it fails the
suite instead of the page. It also pays the bill for absolute URLs, which can be
typo'd or outlived by the file they name just as a relative path can: every
in-repository URL is resolved against the tree, and every in-page anchor against
the headings actually present, which also keeps a section move from silently
orphaning the nav bar at the top.
"""
from __future__ import annotations

import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"

#: This repository. The sibling Node SDK is a DIFFERENT repo, so every prefix
#: below ends in ``/`` and ``agent-flywheel-node`` can never match it.
SLUG = "percepteye-ai/agent-flywheel"
RAW = f"https://raw.githubusercontent.com/{SLUG}/main/"
BLOB_OR_TREE = re.compile(rf"https://github\.com/{re.escape(SLUG)}/(?:blob|tree)/main/(.+)")


def _readable(text: str) -> str:
    """The README as a reader's browser sees it.

    HTML comments and fenced code are dropped: a comment may name a broken
    form in order to explain it, and code is not a link.
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
def images(body: str) -> list[str]:
    return [m.group(1) for m in re.finditer(r'<img[^>]*?\ssrc="([^"]+)"', body, re.S)]


@pytest.fixture(scope="module")
def targets(body: str, images: list[str]) -> list[str]:
    """Every link target and image source a reader can follow."""
    return images + [m.group(1) for m in re.finditer(r"\]\(([^)\s]+)\)", body)]


def test_nothing_is_a_relative_path(targets):
    """PyPI renders this README and resolves no relative path at all."""
    relative = [
        t for t in targets
        if not t.startswith(("http://", "https://", "#", "mailto:"))
    ]
    assert not relative, (
        "README targets are relative. PyPI cannot resolve them, so its page "
        "shows alt text and dead links. Use "
        f"{RAW}<path> for a figure and https://github.com/{SLUG}/blob/main/<path> "
        "for a link:\n  " + "\n  ".join(relative))


def test_every_figure_of_this_repo_comes_from_raw_githubusercontent(images):
    """A github.com/blob URL is an HTML page, not an image: it renders as a
    broken image on both surfaces."""
    wrong = [
        src for src in images
        if f"/{SLUG}/" in src and not src.startswith(RAW)
    ]
    assert not wrong, (
        f"figures of this repository must be served from {RAW}:\n  "
        + "\n  ".join(wrong))


def test_every_in_repo_target_exists(targets):
    """The cost of absolute URLs, paid here instead of by the reader."""
    missing = []
    for t in targets:
        if t.startswith(RAW):
            rel = t[len(RAW):]
        else:
            m = BLOB_OR_TREE.match(t)
            if not m:
                continue
            rel = m.group(1)
        if not (ROOT / rel.split("#")[0]).exists():
            missing.append(t)
    assert not missing, (
        "README points at paths that are not in the tree -- GitHub and PyPI "
        "render these as dead links with no error:\n  " + "\n  ".join(missing))


def test_every_internal_anchor_resolves(body, targets):
    """Guards the nav bar against a section being moved or renamed under it."""
    have = {_slug(m.group(1)) for m in re.finditer(r"^#+\s+(.*)$", body, re.M)}
    dead = [t for t in targets if t.startswith("#") and t[1:] not in have]
    assert not dead, (
        f"README anchors point at headings that are not on the page: {dead}\n"
        f"headings present: {sorted(have)}")
