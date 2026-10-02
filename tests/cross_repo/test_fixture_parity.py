"""The control plane's ``/policy/current`` answers, held BYTE FOR BYTE in
every copy.

Three repositories test against the same two answers -- the gateway delivery
and the withdrawal -- and each vendors its own copy: this SDK under
``tests/fixtures/``, the Node SDK under ``test/fixtures/``, the control plane
under ``tests/fixtures/`` in its own repository. Equal JSON in a different
order or spelling is already two fixtures a reviewer has to diff by hand, and
the day one copy gains a field the others lack, each repository is green
against an answer the other two never see. So the bytes are what is held
equal.

Three mechanisms, because the siblings are not always there:

* THE DIGESTS, always. They are the digests every copy was made from, so a copy
  edited here alone is red in this repository with no sibling in sight. The
  Node SDK pins the same two.
* THE NODE SDK'S COPY, whenever ``tests/node_sdk.py`` -- the one resolver --
  finds that checkout. Found means its copy must exist and match: a sibling
  that lacks the file is the drift this exists to catch, not a reason to skip.
* THE CONTROL PLANE'S COPY, when ``PERCEPTEYE_CONTROL_PLANE_CHECKOUT`` names its
  checkout. Its repository is not public, so nothing published here can find it
  by name; the variable is the only way in, and like ``PERCEPTEYE_NODE_SDK`` a
  variable that is set is a promise -- a checkout or a copy that is not there
  fails instead of skipping.
"""
from __future__ import annotations

import hashlib
import os
import pathlib

import pytest

from ..node_sdk import node_sdk

FIXTURES = pathlib.Path(__file__).resolve().parents[1] / "fixtures"

#: Each shared fixture and the sha256 every copy of it was made from.
SHARED = {
    "policy_current_gateway.json":
        "8eb1fa741706b268a76db597384e5e698065e1c313e88f20a474ca067e92b191",
    "policy_current_withdrawn.json":
        "2f54007e6108e99b03d5907d7a231e66a97e334445156290880814c2e2a81511",
}

#: Names the control plane's checkout. Read by these tests alone.
CONTROL_PLANE_CHECKOUT = "PERCEPTEYE_CONTROL_PLANE_CHECKOUT"


def _ours(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.mark.parametrize("name", sorted(SHARED))
def test_this_copy_is_the_shared_fixture_by_digest(name):
    assert hashlib.sha256(_ours(name)).hexdigest() == SHARED[name], (
        f"tests/fixtures/{name} is not the shared fixture. Change every copy "
        f"together -- both SDKs and the control plane -- never one alone.")


@pytest.mark.parametrize("name", sorted(SHARED))
def test_the_node_sdk_tests_against_the_same_bytes(name):
    sdk, _named = node_sdk()
    if sdk is None:
        pytest.skip("the Node package is not checked out here")
    theirs = sdk / "test" / "fixtures" / name
    assert theirs.is_file(), (
        f"the Node SDK at {sdk} has no test/fixtures/{name}: the two SDKs are "
        f"testing different answers")
    assert theirs.read_bytes() == _ours(name), (
        f"test/fixtures/{name} in the Node SDK differs from this SDK's copy")


@pytest.mark.parametrize("name", sorted(SHARED))
def test_the_control_plane_tests_against_the_same_bytes(name):
    named = os.environ.get(CONTROL_PLANE_CHECKOUT, "").strip()
    if not named:
        pytest.skip(f"{CONTROL_PLANE_CHECKOUT} does not name the control "
                    f"plane's checkout")
    root = pathlib.Path(named).expanduser()
    assert root.is_dir(), (
        f"{CONTROL_PLANE_CHECKOUT}={named} is not a directory. It was set on "
        f"purpose, so this is a broken checkout, not a reason to skip.")
    theirs = root / "tests" / "fixtures" / name
    assert theirs.is_file(), (
        f"the control plane checkout at {root} has no tests/fixtures/{name}: "
        f"its endpoint is held to an answer the SDKs never see")
    assert theirs.read_bytes() == _ours(name), (
        f"the control plane's tests/fixtures/{name} differs from this SDK's copy")
