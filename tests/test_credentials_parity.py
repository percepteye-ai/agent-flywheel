"""Both SDKs read and write ONE credentials file, by one rule.

A developer may run ``agent-flywheel login`` (Python) and serve the same agent
from OpenClaw (Node), or the reverse, so a key stored by either must be found
by both -- for the same agent on the same control plane, and for no other. This
drives the Node SDK across the process boundary over one table, both ways: a
file Python wrote is read by Node, and a file Node wrote is read by Python.
"""
from __future__ import annotations

import json
import os
import stat

from agent_flywheel import credentials as creds

from .node_sdk import require_node_sdk, run_node

CP = "https://cp.test/api/flywheel/v1"
KEY = "pefw_" + "P" * 43

#: (control plane, agent) -> what both SDKs find for it, given the store below.
QUERIES = [
    ((CP, "alpha"), KEY),
    ((CP + "/", "alpha"), KEY),
    ((CP, "beta"), None),                          # expired
    ((CP, "gamma"), None),                         # never stored
    (("https://other.test/api/flywheel/v1", "alpha"), None),
]


def _store(home):
    env = {"PERCEPTEYE_HOME": str(home)}
    creds.save_stored(creds.StoredCredential(
        control_plane_url=CP, agent_id="alpha", api_key=KEY,
        expires_at="2999-01-01T00:00:00Z"), env)
    creds.save_stored(creds.StoredCredential(
        control_plane_url=CP, agent_id="beta", api_key="pefw_" + "B" * 43,
        expires_at="2001-01-01T00:00:00Z"), env)
    return env


def test_node_finds_what_python_stored(tmp_path):
    sdk, node = require_node_sdk()
    env = _store(tmp_path / "home")
    script = (
        "import { loadStored, storedKeyFor } from './src/credentials.js';\n"
        "let input = '';\n"
        "process.stdin.on('data', (c) => { input += c; });\n"
        "process.stdin.on('end', () => {\n"
        "  const { env, queries } = JSON.parse(input);\n"
        "  const stored = loadStored(env);\n"
        "  console.log(JSON.stringify(queries.map(([url, agent]) =>\n"
        "    storedKeyFor(stored, url, agent).cred?.api_key ?? null)));\n"
        "});\n"
    )
    answers = json.loads(run_node(sdk, node, script, stdin=json.dumps(
        {"env": env, "queries": [q for q, _ in QUERIES]})))
    python = [
        (lambda c: c.api_key if c else None)(creds.find_stored(url, agent, env))
        for (url, agent), _ in QUERIES
    ]
    assert answers == python == [want for _, want in QUERIES]


def test_python_finds_what_node_stored(tmp_path):
    sdk, node = require_node_sdk()
    home = tmp_path / "home"
    env = {"PERCEPTEYE_HOME": str(home)}
    script = (
        "import { saveStored } from './src/credentials.js';\n"
        "let input = '';\n"
        "process.stdin.on('data', (c) => { input += c; });\n"
        "process.stdin.on('end', () => {\n"
        "  const { env, cred } = JSON.parse(input);\n"
        "  console.log(saveStored(cred, env));\n"
        "});\n"
    )
    path = run_node(sdk, node, script, stdin=json.dumps({"env": env, "cred": {
        "control_plane_url": CP, "agent_id": "openclaw", "api_key": KEY,
        "org_name": "Acme", "expires_at": "2999-01-01T00:00:00Z",
        "client": "agent-flywheel-node/test",
    }})).strip()
    assert path == str(creds.credentials_path(env))
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    found = creds.find_stored(CP, "openclaw", env)
    assert found is not None and (found.api_key, found.org_name) == (KEY, "Acme")
    # And Python keeps Node's entry when it stores one of its own.
    creds.save_stored(creds.StoredCredential(
        control_plane_url=CP, agent_id="alpha", api_key="pefw_" + "A" * 43), env)
    assert creds.find_stored(CP, "openclaw", env).api_key == KEY
