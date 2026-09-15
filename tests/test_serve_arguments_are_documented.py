"""*Every `serve()` argument* is a completeness claim. This is what makes it one.

The docs-site table under that heading in ``website/docs/configuration.md``
enumerates every keyword ``serve()`` accepts, down to ``gate_flag`` and the
deprecated ``harness_snapshot``. Nothing checked it, and it went false the
first time a parameter was added: ``volatile_command_values`` shipped with the
execution-identity fix and the table never learned about it.

WHY THAT ONE MATTERED PARTICULARLY. The whole design of that parameter is that
the host DECLARES which command values are transport rather than the SDK
guessing -- and a declaration a customer cannot discover is not an affordance.
The published surface was the only place a customer would have read about it,
so every command agent carrying a per-process socket path or port would have
gone on re-minting its ``execution_sha256`` on each restart, invalidating a
prepared OPSD run, with no documented way to stop.

So the heading is derived from the signature instead of trusted. Adding a
keyword to ``serve()`` without a row, or leaving a row behind for one that was
removed, turns this red -- which is the only thing that keeps the word "Every"
true. A comment asking the next author to remember is not a mechanism; this is.

``website/`` is not packaged, so a run from inside an unpacked sdist has
nothing to check here and says so, exactly as the count check does.
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pytest

from agent_flywheel import serve

ROOT = pathlib.Path(__file__).resolve().parents[1]
DOC = ROOT / "website" / "docs" / "configuration.md"
HEADING = "## Every `serve()` argument"


def _documented_arguments(text: str) -> set[str]:
    """The names in the first column of the table under :data:`HEADING`.

    The section runs to the next ``##`` heading, so prose below the table
    cannot contribute a name and a row cannot hide in another section.
    """
    assert HEADING in text, (
        f"{DOC.name} no longer carries the heading {HEADING!r}. Either it was "
        f"reworded -- update this test -- or the complete-enumeration claim "
        f"was dropped, and then this check should be dropped with it.")
    section = text.split(HEADING, 1)[1].split("\n## ", 1)[0]
    return set(re.findall(r"^\|\s*`([A-Za-z_][A-Za-z0-9_]*)`", section, re.M))


def test_the_serve_argument_table_lists_every_serve_argument():
    if not DOC.is_file():
        pytest.skip("website/ is not packaged; no docs-site table to check here")
    documented = _documented_arguments(DOC.read_text(encoding="utf-8"))
    accepted = set(inspect.signature(serve).parameters)

    missing = sorted(accepted - documented)
    assert not missing, (
        "serve() accepts arguments the docs-site table does not list: "
        + ", ".join(missing)
        + f". The table is headed {HEADING!r}, so an argument absent from it "
        f"is one a customer has no way to discover. Add a row per argument.")

    stale = sorted(documented - accepted)
    assert not stale, (
        "the docs-site table lists arguments serve() does not accept: "
        + ", ".join(stale)
        + ". A row for an argument that was renamed or removed sends a "
        "customer to write a TypeError.")
