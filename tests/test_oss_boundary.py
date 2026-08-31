"""The open-source boundary, enforced.

This package is published MIT and runs inside other people's production
processes. Two properties have to hold mechanically rather than by review:

1. One runtime dependency. Every dependency we add is a version conflict we
   impose on a customer who did not ask for it.
2. Nothing internal leaks -- not a private module name, not an internal
   repository path, and in particular not in an error message, where it would
   be printed to a customer's logs and tell them something they cannot act on.
"""
from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "percepteye_agent_flywheel"

#: Names of internal systems. None of them may appear anywhere in the package.
INTERNAL_NAMES = (
    "agent_protocol", "mission_control", "missioncontrol",
    "ranger", "compass", "scout", "litesim", "litesimulation",
    "ftagent", "scoutagent", "percepteye-platform",
)

#: Internal invariants that must not be disclosed in a refusal. A customer can
#: act on none of them, so naming them only discloses.
INTERNAL_DETAILS = ("max-loras", "vllm", "offline_grpo", "bundle_materializer",
                    "scored_suite_sha", "prompt_group_key", "doctrine")


def _package_files():
    return sorted(p for p in SRC.rglob("*.py"))


def _published_files():
    """Every file git tracks, which is exactly what the public repo shows.

    ``_package_files`` scans ``src/`` only, and that hole is not theoretical:
    the e2e demo under ``examples/`` named twelve internal control-plane
    modules and two local developer paths, and shipped to a public repo with
    every boundary test green. The README, the examples and the tests are as
    public as the package, so they are scanned the same way.

    This file is excluded because it has to name the internals in order to
    forbid them.
    """
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT,
        capture_output=True, text=True, check=True,
    ).stdout
    here = pathlib.Path(__file__).resolve()
    paths = []
    for rel in filter(None, out.split("\0")):
        p = (ROOT / rel).resolve()
        if p == here or not p.is_file():
            continue
        paths.append(p)
    assert paths, "git ls-files returned nothing — the scan would pass vacuously"
    return sorted(paths)


def _text(path):
    """The text of a published file, or "" for one that has no text.

    Decoded STRICTLY, so a binary asset raises and is skipped. It used to
    decode with ``errors="ignore"``, which made the ``UnicodeDecodeError``
    branch below unreachable and scanned the GIFs as if they were text -- and
    that is worse than useless in both directions. It cannot find a real leak,
    because a name rendered into a figure is pixels rather than bytes; and it
    invents fake ones, because dropping the invalid bytes welds whatever
    survives into new words. Re-rendering ``flywheel-loop.gif`` produced a file
    whose bytes do not contain any entry of ``INTERNAL_DETAILS`` anywhere, but
    which decoded to a string carrying one of them -- its four letters were
    scattered through the compressed stream with dropped bytes between them.
    Every re-render was a fresh roll of that dice against a thirty-word list,
    and the figures are regenerated whenever their renderer changes.
    """
    try:
        return path.read_text().lower()
    except (OSError, UnicodeDecodeError):  # a binary asset carries no text
        return ""


def test_exactly_one_runtime_dependency():
    text = (ROOT / "pyproject.toml").read_text()
    block = re.search(r"^dependencies = \[(.*?)\]", text, re.S | re.M)
    assert block, "dependencies list not found"
    deps = re.findall(r'"([^"]+)"', block.group(1))
    assert deps == ["httpx>=0.27"], f"runtime dependencies drifted: {deps}"


def test_no_internal_system_names_in_the_package():
    offenders = []
    for path in _package_files():
        text = path.read_text().lower()
        for name in INTERNAL_NAMES:
            # `percepteye_agent_flywheel` is our own name; match the internal ones only
            # as standalone words.
            for m in re.finditer(rf"\b{re.escape(name)}\b", text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line} -> {name}")
    assert not offenders, "internal names leaked into the OSS package:\n" + "\n".join(offenders)


def test_no_internal_invariants_disclosed():
    """The same scope as the names check, for the same reason.

    This scanned ``src/`` alone while its sibling scanned everything git
    tracks, so ``INTERNAL_NAMES`` was enforced across the whole published repo
    and ``INTERNAL_DETAILS`` only inside the package. The earlier widening
    fixed one list and left the other, and the gap was not theoretical:
    ``schema/flywheel-1.json`` named three internal fields in a description
    and no test could see it. A guard with two scopes is a guard you have to
    remember the shape of; both now scan what the public repo shows.
    """
    offenders = []
    for path in _published_files():
        text = _text(path)
        for name in INTERNAL_DETAILS:
            if name in text:
                offenders.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not offenders, (
        "internal detail disclosed in PUBLISHED files:\n" + "\n".join(offenders)
    )


def test_no_internal_system_names_anywhere_published():
    """The same rule, applied to everything the public repo shows.

    The package-only version of this test passed while an example under
    ``examples/`` imported a dozen internal control-plane modules by their real
    dotted paths, in a public repo. A reader could reconstruct the layout from
    the imports alone. The paths are deliberately not repeated here — this file
    is exempt from the scan, so anything named in it is published unchecked.
    """
    offenders = []
    for path in _published_files():
        text = _text(path)
        for name in INTERNAL_NAMES:
            for m in re.finditer(rf"\b{re.escape(name)}\b", text):
                line = text[: m.start()].count("\n") + 1
                offenders.append(f"{path.relative_to(ROOT)}:{line} -> {name}")
    assert not offenders, (
        "internal names leaked into PUBLISHED files:\n" + "\n".join(offenders)
    )


def test_no_local_developer_paths_published():
    """Absolute paths off a developer's machine disclose the username, the
    internal repo names and the working layout, and they cannot work for anyone
    else. ``/Users/<name>/Projects/MissionControl/.worktrees/...`` shipped
    publicly for exactly this reason."""
    offenders = []
    pattern = re.compile(r"(/users/[^/\s\"']+/|/home/[^/\s\"']+/|c:\\\\users\\\\)")
    for path in _published_files():
        for m in pattern.finditer(_text(path)):
            offenders.append(f"{path.relative_to(ROOT)} -> {m.group(0)}")
    assert not offenders, (
        "a local developer path is published:\n" + "\n".join(offenders)
    )


def test_package_imports_with_only_httpx_available():
    """A smoke check that no module reaches for something undeclared."""
    import importlib

    for mod in ("contract", "errors", "entrypoint", "isolation", "outcomes",
                "transport", "runner", "child", "_http"):
        importlib.import_module(f"percepteye_agent_flywheel.{mod}")


def test_license_exists_and_is_mit():
    text = (ROOT / "LICENSE").read_text()
    assert "MIT License" in text
    assert "Permission is hereby granted, free of charge" in text


def test_schema_is_present_and_parses():
    import json

    schema = json.loads((ROOT / "schema" / "flywheel-1.json").read_text())
    assert schema["x-contract-version"] == "flywheel/1"
    # The contract this repo owns must define the shapes the SDK sends.
    for name in ("ToolCallRecord", "RolloutRequest", "RolloutOutput"):
        assert name in schema["$defs"]


def test_schema_and_code_agree_on_the_outcome_enum():
    import json

    from percepteye_agent_flywheel.contract import _VALID_OUTCOMES

    schema = json.loads((ROOT / "schema" / "flywheel-1.json").read_text())
    enum = schema["$defs"]["ToolCallRecord"]["properties"]["outcome"]["enum"]
    assert tuple(enum) == _VALID_OUTCOMES


def test_schema_requires_outcome_and_status_code():
    import json

    schema = json.loads((ROOT / "schema" / "flywheel-1.json").read_text())
    required = schema["$defs"]["ToolCallRecord"]["required"]
    assert "outcome" in required and "status_code" in required


@pytest.mark.parametrize("mod", ["contract", "errors", "isolation", "outcomes"])
def test_modules_are_importable_standalone(mod):
    __import__(f"percepteye_agent_flywheel.{mod}")


def test_every_serve_kwarg_in_the_readme_actually_exists():
    """The README documented `adapters=` for weeks while serve() had no such
    parameter, so following the documentation raised TypeError. Prose drifts
    from a signature silently; this makes it a test failure."""
    import inspect
    import pathlib
    import re

    import percepteye_agent_flywheel as fw

    readme = (pathlib.Path(__file__).resolve().parent.parent / "README.md").read_text()
    params = set(inspect.signature(fw.serve).parameters)

    # Balance the parentheses rather than looking for a closing "\n)". The
    # regex form could not see a serve() call written on ONE line: with no
    # newline-paren to stop at, it ran on to the next multi-line call and
    # collected the kwargs of everything in between -- so `record_tool_call`'s
    # `outcome=` and `RolloutOutput`'s `reward=` were reported as non-existent
    # serve() parameters. A false failure trains you to ignore the check.
    def _arg_blocks(text: str) -> list[str]:
        blocks = []
        for m in re.finditer(r"(?:ph|fw)\.serve\(", text):
            depth, i = 1, m.end()
            while i < len(text) and depth:
                depth += (text[i] == "(") - (text[i] == ")")
                i += 1
            blocks.append(text[m.end(): i - 1])
        return blocks

    used: set[str] = set()
    for block in _arg_blocks(readme):
        # Only at the call's own nesting level: a kwarg inside a nested call
        # (a lambda, a dict, another constructor) is not a serve() parameter.
        depth, top = 0, []
        for ch in block:
            depth += (ch in "([{") - (ch in ")]}")
            top.append(ch if depth == 0 else " ")
        used |= set(re.findall(r"([a-z_]+)\s*=(?!=)", "".join(top)))

    assert used, "no serve() example found in the README — did the format change?"
    unknown = sorted(used - params)
    assert not unknown, f"README uses serve() kwargs that do not exist: {unknown}"
