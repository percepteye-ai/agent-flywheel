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


# ── one lock for both SDKs ────────────────────────────────────────────────


def test_both_sdks_lock_the_file_by_one_protocol(tmp_path):
    """The same lock file, the same stale timeout, the same wait: each SDK's
    lock is the other's. Mutation: either SDK back to a lock of its own
    (Python's flock, Node's ``credentials.json.node-lock``) -> red."""
    sdk, node = require_node_sdk()
    script = (
        "import { LOCK_FILENAME, LOCK_STALE_MS, LOCK_WAIT_MS } "
        "from './src/credentials.js';\n"
        "console.log(JSON.stringify([LOCK_FILENAME, LOCK_STALE_MS, LOCK_WAIT_MS]));\n"
    )
    assert json.loads(run_node(sdk, node, script)) == [
        creds.LOCK_FILENAME, creds.LOCK_STALE_S * 1000, creds.LOCK_WAIT_S * 1000]


def test_a_python_writer_waits_on_a_node_writers_lock(tmp_path):
    """A lock Node holds (fresh, so not stale) keeps a Python write out
    until it is released; a stale one Node left is taken over."""
    import threading
    import time as _time

    sdk, node = require_node_sdk()
    home = tmp_path / "home"
    env = {"PERCEPTEYE_HOME": str(home)}
    script = (
        "import { saveStored } from './src/credentials.js';\n"
        "import { existsSync } from 'node:fs';\n"
        "let input = '';\n"
        "process.stdin.on('data', (c) => { input += c; });\n"
        "process.stdin.on('end', () => {\n"
        "  const { env } = JSON.parse(input);\n"
        "  saveStored({ control_plane_url: %r, agent_id: 'node', api_key: %r }, env);\n"
        "  console.log(existsSync(env.PERCEPTEYE_HOME + '/credentials.json.lock'));\n"
        "});\n" % (CP, KEY)
    )
    # Node's own write leaves no lock behind.
    assert run_node(sdk, node, script, stdin=json.dumps({"env": env})).strip() == "false"
    lock = home / creds.LOCK_FILENAME
    lock.write_text("99999 0\n")
    done = threading.Event()
    threading.Thread(target=lambda: (creds.save_stored(creds.StoredCredential(
        control_plane_url=CP, agent_id="py", api_key=KEY), env), done.set()),
        daemon=True).start()
    assert not done.wait(0.4), "Python wrote while the lock was held"
    lock.unlink()
    assert done.wait(5)
    assert {c.agent_id for c in creds.load_stored(env)} == {"node", "py"}
    old = _time.time() - creds.LOCK_STALE_S - 5
    lock.write_text("99999 0\n")
    os.utime(lock, (old, old))
    creds.save_stored(creds.StoredCredential(
        control_plane_url=CP, agent_id="late", api_key=KEY), env)
    assert not lock.exists()
    assert not list(home.glob(creds.LOCK_FILENAME + ".stale-*")), "left aside"


def test_concurrent_python_and_node_logins_lose_no_key(tmp_path):
    """Writers from BOTH languages at once, each storing its own agents: every
    key survives, and no lock is left behind. Mutation: Python back to an
    flock (or Node to its own lock file) -> a writer of one language reads,
    the other writes, and the first's write drops the second's key: red."""
    import subprocess
    import sys

    sdk, node = require_node_sdk()
    home = tmp_path / "home"
    env = {**os.environ, "PERCEPTEYE_HOME": str(home)}
    per_writer, writers = 15, 3
    node_script = (
        "const { saveStored } = await import('./src/credentials.js');\n"
        "const me = process.argv[1];\n"
        f"for (let i = 0; i < {per_writer}; i += 1) {{\n"
        f"  saveStored({{ control_plane_url: {json.dumps(CP)}, "
        "agent_id: `${me}-${i}`, api_key: 'pefw_x' }, process.env);\n"
        "}\n"
    )
    py_script = (
        "import sys\n"
        "from agent_flywheel import credentials as c\n"
        "me = sys.argv[1]\n"
        f"for i in range({per_writer}):\n"
        f"    c.save_stored(c.StoredCredential(control_plane_url={CP!r}, "
        "agent_id=f'{me}-{i}', api_key='pefw_x'))\n"
    )
    procs = []
    for w in range(writers):
        procs.append(subprocess.Popen(
            [node, "--input-type=module", "-e", node_script, f"node{w}"],
            cwd=sdk, env=env, stderr=subprocess.PIPE, text=True))
        procs.append(subprocess.Popen(
            [sys.executable, "-c", py_script, f"py{w}"],
            env=env, stderr=subprocess.PIPE, text=True))
    for p in procs:
        _, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    stored = {c.agent_id for c in creds.load_stored({"PERCEPTEYE_HOME": str(home)})}
    want = {f"{lang}{w}-{i}" for lang in ("node", "py")
            for w in range(writers) for i in range(per_writer)}
    assert stored == want, f"lost {len(want - stored)} keys: {sorted(want - stored)[:5]}"
    assert not (home / creds.LOCK_FILENAME).exists(), "the lock was left behind"
