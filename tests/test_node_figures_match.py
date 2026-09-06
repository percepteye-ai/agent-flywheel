"""The Node README's figures are this repo's build outputs, not a copy.

``docs/assets/render.sh node`` writes ``flywheel-{overview,loop}-node.{gif,png}``
here. Those four files have no consumer in THIS repo -- its README references
only the plain-named figures -- because their real consumer is the sibling Node
package, which carries byte-identical copies under plain names and has no
renderer of its own.

Until this file existed the only thing keeping the two in step was a sentence:
"Regenerate after touching either renderer." That is not a mechanism, and the
asset README says exactly why it matters -- ``_ROWS`` in ``render_overview.py``
encodes a CLAIM about which halves each package applies, so a stale copy in the
other repo advertises the wrong capability to everyone who opens it on npm,
while looking perfectly fine.

Compared BYTE FOR BYTE, because that is the property: not "a figure exists
there" but "it is the one this renderer produced".
"""
from __future__ import annotations

import pathlib

import pytest

from .node_sdk import node_sdk

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "assets"

#: ``<what this repo renders>`` -> ``<what the Node repo ships it as>``.
PAIRS = [
    (f"flywheel-{fig}-node{ext}", f"flywheel-{fig}{ext}")
    for fig in ("overview", "loop")
    for ext in (".gif", ".png")
]


@pytest.mark.parametrize(("mine", "theirs"), PAIRS)
def test_the_node_repo_ships_the_figure_this_repo_rendered(mine, theirs):
    sdk, _named = node_sdk()
    if sdk is None:
        pytest.skip("the Node package is not checked out here")
    if not ASSETS.is_dir():
        # `docs/` is deliberately not packaged, so a run from an unpacked
        # sdist has no renderer output to compare.
        pytest.skip("docs/assets is not present in this tree")

    src = ASSETS / mine
    dst = sdk / "docs" / "assets" / theirs
    assert src.is_file(), f"{src} is missing — run ./docs/assets/render.sh"
    assert dst.is_file(), f"{dst} is missing in the Node repo"
    assert src.read_bytes() == dst.read_bytes(), (
        f"{theirs} in the Node repo is not what {mine} renders here. "
        f"Run ./docs/assets/render.sh and copy the four *-node files across; "
        f"a figure copied between the two repos is wrong even when it looks "
        f"right, because the rows encode which halves each package applies.")
