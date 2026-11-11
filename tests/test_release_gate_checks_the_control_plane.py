"""The release runbook's Mission Control gate (docs/RELEASING.md §1a) is a
check that runs, not a sentence: both SDKs declare customer contract
``CUSTOMER_CONTRACT_VERSION`` and upload to ``POST /outcomes``, and tagged
against a Mission Control without that route every reported outcome would be
refused for a week and then dropped.

The documented Python command is run here, verbatim, against a local stand-in
for Mission Control, and must pass only where the contract and the route are
both served.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from agent_flywheel.customer_api import CUSTOMER_CONTRACT_VERSION

ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = (ROOT / "docs" / "RELEASING.md").read_text(encoding="utf-8")


def _gate_section() -> str:
    start = RUNBOOK.index("## 1a. The Mission Control gate")
    return RUNBOOK[start:RUNBOOK.index("\n## 2. ", start)]


def _python_gate() -> str:
    block = re.findall(r"```bash\n(.*?)```", _gate_section(), re.S)[0]
    return block.split("<<'EOF'\n", 1)[1].rsplit("EOF", 1)[0]


def _serve(version: str, outcomes_status: int):
    class _Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _answer(self, status: int, body: bytes = b"") -> None:
            self.send_response(status)
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/api/v2/contract":
                self._answer(200, json.dumps(
                    {"info": {"version": version}}).encode())
            else:
                self._answer(404)

        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length") or 0))
            self._answer(outcomes_status
                         if self.path == "/api/flywheel/v1/outcomes" else 404)

    server = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def _bump(version: str, index: int, by: int) -> str:
    parts = [int(x) for x in version.split(".")]
    parts[index] += by
    return ".".join(str(p) for p in parts)


@pytest.mark.parametrize(("version", "status", "passes"), [
    (CUSTOMER_CONTRACT_VERSION, 401, True),
    (CUSTOMER_CONTRACT_VERSION, 422, True),
    (_bump(CUSTOMER_CONTRACT_VERSION, 1, 1), 403, True),
    (_bump(CUSTOMER_CONTRACT_VERSION, 1, -1), 401, False),
    (_bump(CUSTOMER_CONTRACT_VERSION, 0, 1), 401, False),
    (CUSTOMER_CONTRACT_VERSION, 404, False),
    (CUSTOMER_CONTRACT_VERSION, 405, False),
    (CUSTOMER_CONTRACT_VERSION, 200, False),
])
def test_the_documented_gate_passes_only_a_control_plane_that_serves_the_release(
    version, status, passes,
):
    server = _serve(version, status)
    try:
        res = subprocess.run(
            [sys.executable, "-"], input=_python_gate(), text=True,
            capture_output=True, cwd=ROOT,
            env={**os.environ,
                 "CP": f"http://127.0.0.1:{server.server_address[1]}/"},
            timeout=60,
        )
    finally:
        server.shutdown()
    assert (res.returncode == 0) is passes, res.stdout + res.stderr
    if passes:
        assert res.stdout.startswith("ok: ")


def test_the_runbook_runs_the_gate_before_tagging_in_both_repositories():
    steps = RUNBOOK[RUNBOOK.index("## 0."):RUNBOOK.index("## 1. ")]
    assert "[§1a](#1a-the-mission-control-gate)" in steps
    section = _gate_section()
    assert "/api/flywheel/v1/outcomes" in section
    assert 'from "./src/customer-api.js"' in section, "the Node command"


def _node_gate() -> str:
    block = re.findall(r"```bash\n(.*?)```", _gate_section(), re.S)[1]
    return block.split("-e '\n", 1)[1].rsplit("'", 1)[0]


@pytest.mark.parametrize(("version", "status", "passes"), [
    (CUSTOMER_CONTRACT_VERSION, 401, True),
    (_bump(CUSTOMER_CONTRACT_VERSION, 1, -1), 401, False),
    (CUSTOMER_CONTRACT_VERSION, 404, False),
])
def test_the_documented_node_gate_judges_alike(version, status, passes):
    """The Node repository's half of the gate, from its own root: it reads
    the Node SDK's declared version and must agree with the Python half."""
    from .node_sdk import require_node_sdk

    sdk, node = require_node_sdk()
    server = _serve(version, status)
    try:
        res = subprocess.run(
            [node, "--input-type=module", "-e", _node_gate()],
            text=True, capture_output=True, cwd=sdk, timeout=60,
            env={**os.environ,
                 "CP": f"http://127.0.0.1:{server.server_address[1]}"},
        )
    finally:
        server.shutdown()
    assert (res.returncode == 0) is passes, res.stdout + res.stderr
