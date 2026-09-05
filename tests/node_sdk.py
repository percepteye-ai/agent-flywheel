"""Where the Node SDK is, and what to run it with. ONE definition.

Two suites now drive the Node package across the process boundary --
``test_report_turns_wire`` for the idempotency key and
``test_prompt_decision_parity`` for the prompt predicate -- and a second copy
of "where is it, and may I skip?" is exactly how the first one broke: it looked
two directories up for a fixed sibling name, which is nowhere near where CI
puts the checkout, so the parity test skipped on every CI run while looking
green on a laptop. ``test-full`` fails the build on ANY skip, so that made the
matrix red for a reason unrelated to the change under test while the parity it
exists to prove went unchecked.

THE SKIP RULE, and the two cases are NOT the same case:

* AUTO-DISCOVERY MAY SKIP. A developer who checked out only this repo has no
  Node SDK to check against, and turning that into a red suite would teach
  everyone to ignore the failure.
* AN EXPLICIT ``PERCEPTEYE_NODE_SDK`` IS A PROMISE. Somebody said the SDK is
  there. If it is not, or if there is no node to run it with, that is a broken
  environment and the test FAILS. CI always sets the variable, so CI cannot
  turn either check into a green no-op.

The path itself is not resolved here. ``scripts/cross_language_abi_check.sh
--print-node-sdk`` is the resolver, because that script needs the rule too;
asking it is what stops the two from drifting.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

#: The ONE resolver that knows where the Node SDK is. It honours
#: ``PERCEPTEYE_NODE_SDK`` and the sibling checkout names.
ABI_CHECK = Path(__file__).resolve().parents[1] / "scripts" / "cross_language_abi_check.sh"

#: Tried in order. The pinned path first because a developer's ``node`` may be
#: an older major than the SDK's engine range; ``node`` is the CI case.
_NODE_CANDIDATES = ("/opt/homebrew/opt/node@24/bin/node", "node")


def node_sdk() -> tuple[Path | None, bool]:
    """``(path, was_named)``. ``path`` is None only when nobody named one.

    A NAMED path that is not there is an error, not a skip: the script exits
    non-zero and this fails. Only auto-discovery returns None.
    """
    named = bool(os.environ.get("PERCEPTEYE_NODE_SDK"))
    res = subprocess.run(
        [str(ABI_CHECK), "--print-node-sdk"], capture_output=True, text=True)
    if res.returncode != 0:
        pytest.fail(f"resolving the Node SDK failed:\n{res.stdout}{res.stderr}")
    out = res.stdout.strip()
    return (Path(out) if out else None), named


def require_node_sdk() -> tuple[Path, str]:
    """The SDK directory and a usable node binary, or skip/fail per the rule.

    Callers get a pair they can run immediately. Deciding here rather than in
    each test is the point: the skip condition is the part that was wrong, so
    it is stated once.
    """
    sdk, named = node_sdk()
    if sdk is None:
        # Auto-discovery only: a named path that is missing already failed.
        pytest.skip("the Node package is not checked out here")

    for candidate in _NODE_CANDIDATES:
        try:
            subprocess.run([candidate, "--version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            continue
        return sdk, candidate

    if named:
        pytest.fail(
            "PERCEPTEYE_NODE_SDK names a Node SDK but there is no usable "
            "node binary to run it with")
    pytest.skip("no usable node binary")
    raise AssertionError("unreachable")  # pragma: no cover -- pytest.skip raises


def run_node(sdk: Path, node: str, script: str, *, stdin: str = "") -> str:
    """Run one ES-module snippet inside the Node SDK checkout, return stdout.

    A non-zero exit is a failure of the test, not a reason to fall back to
    anything: the whole point is that the other implementation answered.
    """
    res = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=sdk, input=stdin, capture_output=True, text=True,
    )
    assert res.returncode == 0, f"node exited {res.returncode}:\n{res.stderr}"
    return res.stdout
