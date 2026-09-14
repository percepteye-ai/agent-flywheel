"""What actually ships, asserted against a real build.

Two properties were held here by manifest correctness alone, with nothing
checking either, and both had already broken in ways only visible to someone
who unpacked a tarball by hand.

1. **The animated figures never reach a package.** They are for the README and
   for GitHub; 7.9 MB of them sit under ``docs/assets/`` and none of it belongs
   in a wheel a customer installs into a production process. The Node repo
   writes this guarantee down as a sentence -- and a sentence is not a
   mechanism.

2. **The sdist include list means what it appears to say.** It does not, by
   default: hatchling matches those entries as *gitignore-style patterns*, so a
   bare ``README.md`` matches that basename AT EVERY DEPTH. Three files outside
   the declared roots were shipping because of it -- ``website/README.md`` and
   both ``examples/*/README.md`` -- and property 1 above was one
   ``docs/assets/README.md`` away from being false, which is a file the sibling
   Node repo actually keeps.

Built through hatchling's own builders rather than a re-implementation of its
pattern matching: a second matcher here would be a second answer to "what
ships", and it would agree with the first one right up until the day it
mattered.
"""
from __future__ import annotations

import pathlib
import tarfile
import tempfile
import zipfile

import pytest
from hatchling.builders.sdist import SdistBuilder
from hatchling.builders.wheel import WheelBuilder

from agent_flywheel import install_hooks
from agent_flywheel.hosts import HOSTS_DIR

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Extensions that are documentation assets, never package payload. Checked by
#: suffix rather than by directory so a figure moved somewhere else is still
#: refused.
_ASSET_SUFFIXES = {".gif", ".png", ".svg", ".jpg", ".jpeg", ".webp", ".mp4",
                   ".webm", ".mov", ".ico"}

#: The roots ``[tool.hatch.build.targets.sdist].include`` declares, plus the
#: files hatchling adds on its own (``.gitignore`` unconditionally, and the
#: metadata it generates).
_SDIST_ROOTS = ("src/", "schema/", "tests/", "examples/", "scripts/")
_SDIST_FILES = {"README.md", "LICENSE", "PKG-INFO", "pyproject.toml",
                ".gitignore"}


@pytest.fixture(scope="module")
def built() -> dict[str, list[str]]:
    """Member names of a real sdist and a real wheel, built from this tree."""
    with tempfile.TemporaryDirectory() as td:
        (sdist,) = SdistBuilder(str(ROOT)).build(directory=td)
        (wheel,) = WheelBuilder(str(ROOT)).build(directory=td)
        with tarfile.open(sdist) as tf:
            # Every sdist member is prefixed with `<name>-<version>/`; strip it
            # so the names read like repository paths.
            sdist_names = [
                n.split("/", 1)[1] for n in tf.getnames() if "/" in n]
        with zipfile.ZipFile(wheel) as zf:
            wheel_names = zf.namelist()
    return {"sdist": sdist_names, "wheel": wheel_names}


def test_no_documentation_asset_reaches_either_package(built):
    """The standing constraint, as a check rather than as prose."""
    offenders = {
        kind: sorted(n for n in names
                     if pathlib.PurePosixPath(n).suffix.lower() in _ASSET_SUFFIXES)
        for kind, names in built.items()
    }
    assert not any(offenders.values()), (
        f"documentation assets reached a package: {offenders}. "
        f"docs/assets/ is for the README and GitHub only.")


def test_the_sdist_ships_only_what_the_include_list_names(built):
    """The anchored patterns, asserted where they can actually be wrong.

    Without the leading slashes this fails on ``website/README.md`` -- a file
    telling the reader to ``cd website && mkdocs serve`` inside a tarball that
    contains no ``website/`` directory.
    """
    strays = sorted(
        n for n in built["sdist"]
        if n not in _SDIST_FILES and not n.startswith(_SDIST_ROOTS))
    assert not strays, (
        f"the sdist carries files outside its declared roots: {strays}. "
        f"An include entry with no leading slash matches that basename at "
        f"every depth.")


def test_every_host_manifest_in_the_wheel_has_an_installer(built):
    """A payload the customer's host can load, that no command can register,
    is a promise nothing keeps.

    ``hosts/gemini/gemini-extension.json`` shipped in every wheel describing
    itself as recording ok/failed/unknown per tool call, with no hook wiring in
    it and no ``install_gemini`` anywhere -- so a Gemini CLI user who found it
    and copied it into their extensions directory registered nothing, and got
    no error saying so.

    Asked as producer-and-consumer rather than by name: a manifest ships if and
    only if ``install_hooks`` holds a path to it.
    """
    shipped = {
        n for n in built["wheel"]
        if n.startswith("agent_flywheel/hosts/")
        and n.endswith(".json")
    }
    installable = {
        f"agent_flywheel/hosts/{value.relative_to(HOSTS_DIR).as_posix()}"
        for value in vars(install_hooks).values()
        if isinstance(value, pathlib.Path)
        and value.is_relative_to(HOSTS_DIR)
        and value.suffix == ".json"
    }
    assert shipped == installable, (
        f"host manifests in the wheel: {sorted(shipped)}\n"
        f"manifests install_hooks can place: {sorted(installable)}\n"
        f"A manifest with no installer is a dead end for whoever finds it.")
