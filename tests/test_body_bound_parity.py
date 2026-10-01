"""ONE body bound, in two languages, proved to agree.

``agent_flywheel/_body_bound.py`` here and ``src/body-bound.js`` in the Node
package shorten a captured tool body the same way: both ends kept, a marker
between them counting what it replaced. They are two implementations of one
rule because the packages share no code, and two implementations drift unless
something runs both. They had: the Python parser kept both ends of an error
while the Node classifiers kept its first 1000 characters, so one wire field,
``tool_calls[].error``, had a shape per SDK, and the Node one had deleted the
exception line a traceback ends with.

This asserts byte-identity on the helper, including where the two languages'
string models differ (Python counts code points, a JS ``length`` counts UTF-16
units), and then through the two Node classifiers that produce the field, so a
classifier that stops calling the bound is caught too.

THE SKIP TRAP. ``test-full`` fails the build on ANY skip, so a test that always
skips turns the build red while proving nothing. The rule for when this may
skip lives in ``tests/node_sdk.py``: auto-discovery skips, a NAMED
``PERCEPTEYE_NODE_SDK`` that is not there fails.
"""
from __future__ import annotations

import json

from agent_flywheel._body_bound import bounded

from .node_sdk import require_node_sdk, run_node
from .test_error_evidence_survives_capture import _long_traceback

_TB = _long_traceback()

#: (text, limit) pairs; a None limit means the default. Each block pins an edge
#: one of the two implementations could get wrong on its own.
CASES: list[tuple[str, int | None]] = [
    # within the bound: untouched, no marker
    ("", None), ("stdout: fine\n", None), ("x" * 4000, None),
    # one over, and the real case
    ("x" * 4001, None), (_TB, None), (_TB, 1000),
    # the marker's own digits change the room it leaves
    *[("y" * size, 100) for size in range(990, 1110, 7)],
    # too small to hold the marker: the tail, and nothing at all at 0
    (_TB, 20), (_TB, 0),
    # non-ASCII inside the Basic Multilingual Plane
    ("é…中文 " * 2000, None),
    # astral characters: a UTF-16 count disagrees with Python's here
    ("😀a" * 3000, None), ("😀" * 50, 30), ("a😀" * 60, 101),
]

_HELPER = """
import { bounded } from "./src/body-bound.js";
let raw = "";
for await (const chunk of process.stdin) raw += chunk;
const cases = JSON.parse(raw);
process.stdout.write(JSON.stringify(
  cases.map(([text, limit]) => limit === null ? bounded(text) : bounded(text, limit))));
"""

_CLASSIFIERS = """
import { classify } from "./src/classify.js";
import { classifyDsh } from "./src/dsh-classify.js";
let tb = "";
for await (const chunk of process.stdin) tb += chunk;
process.stdout.write(JSON.stringify({
  openclaw: classify("exec", tb, undefined).error,
  dsh: classifyDsh("bash", { isError: true, error: { message: tb } }).error,
}));
"""


def _python(text: str, limit: int | None) -> str:
    return bounded(text) if limit is None else bounded(text, limit)


def test_node_bounds_every_case_exactly_as_python_does():
    sdk, node = require_node_sdk()
    got = json.loads(run_node(sdk, node, _HELPER, stdin=json.dumps(CASES)))
    want = [_python(text, limit) for text, limit in CASES]
    assert len(got) == len(want)
    mismatched = [
        (i, limit, len(text))
        for i, ((text, limit), g, w) in enumerate(zip(CASES, got, want, strict=True))
        if g != w
    ]
    assert not mismatched, mismatched


def test_both_node_classifiers_report_the_error_python_does():
    """The field, not just the helper: what the OpenClaw and DeepSeek Harness
    classifiers put in ``error`` for a long traceback is what every Python
    producer puts there."""
    sdk, node = require_node_sdk()
    got = json.loads(run_node(sdk, node, _CLASSIFIERS, stdin=_TB))
    assert got == {"openclaw": bounded(_TB), "dsh": bounded(_TB)}
