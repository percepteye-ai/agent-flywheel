"""``scripts/cross_language_abi_check.sh``, run from the Python suite.

THE CHECK ITSELF IS THE DEFINITION; this is one more way in. Until now it ran
only from the ``node-plugins`` CI job, so a developer editing
``execution_identity.py`` here saw a green ``pytest`` while the Node half of
the same digest went unexecuted, and learned on a push -- or, if the Node SDK
was not checked out in that job, not at all.

WHY THE EXECUTION DIGEST MADE THAT MATTER. Both SDKs hash the same descriptor
under the same domain separators, and Mission Control binds the result into an
ed25519-signed training workload and refuses every episode whose attestation
does not match. The two agreed only by convention: nothing executed both. A
one-sided change to the canonical encoding, or to the command-argv projection
that drops declared transport values, makes the SAME agent attest a DIFFERENT
identity depending on which SDK registered it -- silently, on both sides.
A convention is not a mechanism. This is the mechanism.

THE SKIP RULE IS NOT RESTATED HERE. ``tests/node_sdk.py`` owns it, because a
second copy of "where is it, and may I skip?" is exactly how the first parity
test broke: auto-discovery may skip, a NAMED ``PERCEPTEYE_NODE_SDK`` that is
not there fails. The script is handed the node that rule resolved and the
interpreter running this suite, so neither has a second answer at runtime.
"""
from __future__ import annotations

import os
import subprocess
import sys

from .node_sdk import ABI_CHECK, require_node_sdk

#: What the script prints when each half of it actually ran. A script that
#: exited 0 having skipped its way past both contracts would otherwise pass
#: here, which is the failure this whole file exists to stop -- and it is not
#: hypothetical: an earlier local approximation of the capture check passed
#: while the real one failed.
_EVIDENCE = (
    "cross-language capture contract holds",
    "execution descriptor digest agrees",
    "volatile-value projection agrees",
    "both languages refuse",
    "node refuses the un-stringified argv python stringifies earlier",
)


def test_both_sdks_agree_on_the_capture_contract_and_the_execution_digest():
    sdk, node = require_node_sdk()
    result = subprocess.run(
        [str(ABI_CHECK)],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            # NAMED, so the script cannot auto-discover a different checkout
            # than the one this suite resolved.
            "PERCEPTEYE_NODE_SDK": str(sdk),
            "PERCEPTEYE_NODE_BIN": node,
            # The interpreter running this suite is by definition the one with
            # agent_flywheel importable; letting the script search again could
            # answer with a different install than the code under test.
            "PYTHON": sys.executable,
        },
    )
    assert result.returncode == 0, (
        f"the cross-language ABI check failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}")
    missing = [line for line in _EVIDENCE if line not in result.stdout]
    assert not missing, (
        "the ABI check exited 0 without running everything it claims to "
        f"check -- missing: {missing}\n{result.stdout}\n{result.stderr}")
