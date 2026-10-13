"""The Node README's figures are this repo's build outputs, not a copy.

Both READMEs show the same figures: the improvement loop is the same however
an agent is connected, so ``docs/assets/render.sh`` renders ONE set, and the
Node package carries the same files under the same names. It has no renderer
of its own.

They used to come in a Python and a Node flavour, each naming what its own
customer would type, and a sentence -- "regenerate after touching either
renderer" -- was all that kept the two repos in step. That is not a mechanism:
a figure rendered here and never copied leaves the other README showing the
previous loop, while looking perfectly fine.

Compared BYTE FOR BYTE, because that is the property: not "a figure exists
there" but "it is the one this renderer produced".
"""
from __future__ import annotations

import pathlib

import pytest

from .node_sdk import node_sdk

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "docs" / "assets"

#: Every figure both READMEs show, under the one name both repos use.
FIGURES = [
    f"flywheel-{fig}{ext}" for fig in ("overview", "loop") for ext in (".gif", ".png")
] + ["flywheel-spinner.gif"]


@pytest.mark.parametrize("name", FIGURES)
def test_the_node_repo_ships_the_figure_this_repo_rendered(name):
    sdk, _named = node_sdk()
    if sdk is None:
        pytest.skip("the Node package is not checked out here")
    if not ASSETS.is_dir():
        # `docs/` is deliberately not packaged, so a run from an unpacked
        # sdist has no renderer output to compare.
        pytest.skip("docs/assets is not present in this tree")

    src = ASSETS / name
    dst = sdk / "docs" / "assets" / name
    assert src.is_file(), f"{src} is missing — run ./docs/assets/render.sh"
    assert dst.is_file(), f"{dst} is missing in the Node repo"
    assert src.read_bytes() == dst.read_bytes(), (
        f"{name} in the Node repo is not the one rendered here. Run "
        f"./docs/assets/render.sh and copy it across: both READMEs show the "
        f"same figures.")


def test_no_flavoured_figure_is_left_behind():
    """One set: a ``-node`` render left here would be a second answer."""
    if not ASSETS.is_dir():
        pytest.skip("docs/assets is not present in this tree")
    assert not sorted(p.name for p in ASSETS.glob("*-node.*"))


def test_both_readmes_show_the_same_figures():
    sdk, _named = node_sdk()
    if sdk is None:
        pytest.skip("the Node package is not checked out here")
    import re

    def figures(readme: pathlib.Path) -> list[str]:
        return re.findall(r'<img src="(docs/assets/[^"]+)"',
                          readme.read_text(encoding="utf-8"))

    ours = figures(ASSETS.parents[1] / "README.md")
    assert ours, "no figure found in this README"
    assert ours == figures(sdk / "README.md")
