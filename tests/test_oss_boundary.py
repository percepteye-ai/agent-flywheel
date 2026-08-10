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
    offenders = []
    for path in _package_files():
        text = path.read_text().lower()
        for name in INTERNAL_DETAILS:
            if name in text:
                offenders.append(f"{path.relative_to(ROOT)} -> {name}")
    assert not offenders, "internal detail disclosed in the OSS package:\n" + "\n".join(offenders)


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

    used: set[str] = set()
    for block in re.finditer(r"(?:ph|fw)\.serve\(\s*(.*?)\n\)", readme, re.S):
        used |= set(re.findall(r"^\s*([a-z_]+)\s*=", block.group(1), re.M))

    assert used, "no serve() example found in the README — did the format change?"
    unknown = sorted(used - params)
    assert not unknown, f"README uses serve() kwargs that do not exist: {unknown}"
