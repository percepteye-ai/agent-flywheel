"""ONE reading of a refused turn batch, in two languages, proved to agree.

A 403 on ``POST /agents/{id}/turns`` is the intake refusing the turns -- an
operator withdrew consent, or the key lost ``turn.report`` -- and both SDKs
must stop uploading AND stop recording model content on it, in the intake's
own words. Python's ``_capture_refusal`` has done so since the refusal was
first read; Node read every error as a blip, so it kept POSTing and kept
writing each run's system prompt, history and tool results while consent was
withdrawn. This pins the sentence each side takes from the same error, and
that a blip is a blip on both.

The Node side's stop itself is pinned by its own suite (``model-content-
capture.test.js``: "stops recording model content when the intake refuses the
turns"); the skip rule is ``tests/node_sdk.py``'s.
"""
from __future__ import annotations

import json

import pytest

from agent_flywheel._http import TransportError
from agent_flywheel.production import _capture_refusal

from .node_sdk import require_node_sdk, run_node

#: ``(status, the message each SDK's HTTP client formats for it)``.
CASES: dict[str, tuple[int, str]] = {
    "a refusal with the intake's sentence": (
        403, "POST /api/flywheel/v1/agents/a/turns -> HTTP 403: production "
             "capture is not enabled for this agent."),
    "a refusal with a padded sentence": (
        403, "POST /x -> HTTP 403:   the key lacks turn.report  "),
    "a refusal with no sentence": (403, "POST /x -> HTTP 403: "),
    "a server error": (503, "POST /x -> HTTP 503: upstream"),
    "a validation error": (422, "POST /x -> HTTP 422: bad turn"),
}


@pytest.fixture(scope="module")
def node_answers() -> dict[str, object]:
    sdk, node = require_node_sdk()
    script = (
        "import {captureRefusal} from './src/capture.js';\n"
        "import {TransportError} from './src/errors.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const out = {};\n"
        "for (const [name, [status, message]] of Object.entries(JSON.parse(raw))) "
        "out[name] = captureRefusal("
        "new TransportError(message, { statusCode: status }));\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    return json.loads(run_node(sdk, node, script, stdin=json.dumps(CASES)))


@pytest.mark.parametrize("name", sorted(CASES))
def test_both_sdks_read_the_same_refusal(node_answers, name):
    _status, message = CASES[name]
    python = _capture_refusal(TransportError(message))
    assert node_answers[name] == python, (
        f"{name}: Node read {node_answers[name]!r}, Python {python!r}")


def test_a_refusal_is_read_as_one_and_a_blip_is_not(node_answers):
    """The table above could agree by both answering null; it must not."""
    assert node_answers["a refusal with the intake's sentence"] == (
        "production capture is not enabled for this agent.")
    assert node_answers["a refusal with no sentence"]
    assert node_answers["a server error"] is None
    assert node_answers["a validation error"] is None
