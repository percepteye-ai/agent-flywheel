"""On-disk retention means the same thing in both SDKs.

Python had ``attach(retention_turns=...)`` and ``Attachment.prune()``; the
Node plugin had neither, so its production capture grew without bound and
could not be bounded at all. It now has ``retentionTurns`` (config) and
``capture.prune()``, and this holds the two to one rule from this side:

* the same values are accepted (a whole number of turns, 0 or more, or unset)
  and every other value is refused, Python by raising and Node by dropping it
  with the same sentence, since a throw would fail an OpenClaw plugin load;
* the automatic prune runs on the same cadence;
* the same tree, pruned with the same arguments, keeps the same turns.

The Node suite's own cases are ``test/retention.test.js``.
"""
from __future__ import annotations

import json
import os
import time

import pytest

import agent_flywheel as fw
from agent_flywheel.errors import ConfigurationError
from agent_flywheel.production import _PRUNE_EVERY

from .node_sdk import require_node_sdk, run_node

VALUES = [None, 0, 1, 50_000, -1, 2.5, "100", True, [], {}]


def _python_accepts(value, tmp_path) -> bool:
    try:
        px = fw.attach(agent_id="a", trajectory_root=str(tmp_path),
                       register=False, retention_turns=value)
    except ConfigurationError as exc:
        assert "retention_turns must be a whole number of turns, 0 or more" in str(exc)
        return False
    px.close()
    return True


def _node(script: str, payload) -> object:
    sdk, node = require_node_sdk()
    return json.loads(run_node(sdk, node, script, stdin=json.dumps(payload)))


_READ_STDIN = (
    "let input = '';\n"
    "process.stdin.on('data', (c) => { input += c; });\n"
    "process.stdin.on('end', () => { const data = JSON.parse(input);\n"
)


def test_both_accept_the_same_values(tmp_path):
    python = [_python_accepts(v, tmp_path / str(i)) for i, v in enumerate(VALUES)]
    node = _node(
        "import { retentionTurnsFrom } from './src/capture.js';\n" + _READ_STDIN
        + "  console.log(JSON.stringify(data.map((v) => {\n"
          "    const r = retentionTurnsFrom(v);\n"
          "    return r.refusal === null;\n"
          "  })));\n});\n",
        VALUES,
    )
    assert python == node, list(zip(VALUES, python, node, strict=False))
    assert python == [True, True, True, True, False, False, False, False, False, False]


def test_the_cadence_is_the_same():
    node = _node(
        "import { PRUNE_EVERY } from './src/capture.js';\n" + _READ_STDIN
        + "  console.log(JSON.stringify(PRUNE_EVERY));\n});\n",
        None,
    )
    assert node == _PRUNE_EVERY


def _tree(root, n: int) -> None:
    """``n`` turns, t0 the oldest, each ten seconds apart, the first two old."""
    now = time.time()
    for i in range(n):
        d = root / f"t{i}"
        d.mkdir(parents=True)
        (d / "turns.jsonl").write_text(
            json.dumps({"kind": "task", "text": f"ask {i}", "turn_id": f"t{i}"}) + "\n")
        age = 100_000 if i < 2 else (n - i) * 10
        os.utime(d, (now - age, now - age))


@pytest.mark.parametrize("keep_last,older_than_s", [
    (3, None), (0, None), (10, None), (None, 3600), (4, 3600), (None, None),
])
def test_the_same_tree_keeps_the_same_turns(tmp_path, keep_last, older_than_s):
    py_root, node_root = tmp_path / "py", tmp_path / "node"
    _tree(py_root, 6)
    _tree(node_root, 6)
    px = fw.attach(agent_id="a", trajectory_root=str(py_root), register=False)
    with px:
        removed = px.prune(keep_last=keep_last, older_than_s=older_than_s)
    node = _node(
        "import { readdirSync } from 'node:fs';\n"
        "import { pruneTurns } from './src/capture.js';\n" + _READ_STDIN
        + "  const removed = pruneTurns({ root: data.root, keepLast: data.keepLast,\n"
          "    olderThanS: data.olderThanS });\n"
          "  const kept = readdirSync(data.root).sort();\n"
          "  console.log(JSON.stringify({ removed, kept }));\n});\n",
        {"root": str(node_root), "keepLast": keep_last, "olderThanS": older_than_s},
    )
    kept = sorted(p.name for p in py_root.iterdir() if p.is_dir())
    assert node == {"removed": removed, "kept": kept}
