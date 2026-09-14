"""The architecture SVG is what its renderer draws -- in both places it ships.

``docs/assets/render_architecture.py`` writes ``flywheel-architecture.svg``, and
the docs site carries a copy of it at ``website/docs/assets/``. Nothing checked
either one against the renderer, and the gap was not theoretical: a fix to the
figure was made in the SVG instead of the renderer, so the next ``render.sh``
would have silently undone it -- while the site's copy, still the renderer's
output, went on drawing what the fix had removed.

This figure is pure Python (no Pillow, no ffmpeg) and every coordinate in it is
plain arithmetic, so it re-renders here in milliseconds and to the same bytes on
every platform. Compared BYTE FOR BYTE, because that is the property: not "a
figure exists there" but "it is the one this renderer produces".
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
RENDERER = ROOT / "docs" / "assets" / "render_architecture.py"
SVG = "flywheel-architecture.svg"

#: Every published copy. The site's is a copy by design -- see website/README.md.
COPIES = [ROOT / "docs" / "assets" / SVG, ROOT / "website" / "docs" / "assets" / SVG]


@pytest.fixture(scope="module")
def rendered(tmp_path_factory) -> bytes:
    if not RENDERER.is_file():
        # `docs/` and `website/` are deliberately not packaged, so a run from
        # an unpacked sdist has no renderer to compare against.
        pytest.skip("docs/assets is not present in this tree")
    # The renderer writes BESIDE ITSELF, so it runs from a copy: re-rendering
    # in place would overwrite the very file this test exists to check.
    work = tmp_path_factory.mktemp("architecture")
    shutil.copy(RENDERER, work)
    subprocess.run([sys.executable, str(work / RENDERER.name)],
                   check=True, capture_output=True)
    return (work / SVG).read_bytes()


@pytest.mark.parametrize("copy", COPIES, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_the_published_svg_is_what_the_renderer_draws(rendered, copy):
    assert copy.is_file(), f"{copy} is missing"
    assert copy.read_bytes() == rendered, (
        f"{copy.relative_to(ROOT)} is not what render_architecture.py draws. "
        f"Change the renderer, not the SVG: run ./docs/assets/render.sh "
        f"architecture, then cp docs/assets/{SVG} website/docs/assets/.")
