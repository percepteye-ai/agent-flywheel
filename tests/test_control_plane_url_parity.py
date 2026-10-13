"""Both SDKs find the control plane by one rule, with one default.

Python had no default for ``PERCEPTEYE_CONTROL_PLANE_URL`` while the Node SDK
defaulted it to the production origin, so one customer following one
onboarding got an agent that worked from one SDK and one that refused to start
or sent nothing from the other. The SDKs ship in unison, so the rule is now
the same in both, and this drives both over one table: the value given in code
(Node: the plugin config's ``controlPlaneUrl``), then the variable, then the
default, each trimmed, a blank one falling through.
"""
from __future__ import annotations

import json

import pytest

from agent_flywheel.control_plane import (
    DEFAULT_CONTROL_PLANE_URL,
    ENV_CONTROL_PLANE_URL,
    resolve_control_plane_url,
)

from .node_sdk import require_node_sdk, run_node

#: (given in code, the environment variable) -> what both SDKs must use.
CASES = [
    (None, None, DEFAULT_CONTROL_PLANE_URL),
    (None, "https://env.example/v1", "https://env.example/v1"),
    ("https://arg.example/v1", "https://env.example/v1", "https://arg.example/v1"),
    ("https://arg.example/v1", None, "https://arg.example/v1"),
    ("  https://arg.example/v1  ", None, "https://arg.example/v1"),
    (None, " https://env.example/v1\n", "https://env.example/v1"),
    ("", "https://env.example/v1", "https://env.example/v1"),
    ("   ", None, DEFAULT_CONTROL_PLANE_URL),
    (None, "   ", DEFAULT_CONTROL_PLANE_URL),
    (None, "", DEFAULT_CONTROL_PLANE_URL),
]


def test_the_default_is_the_production_origin():
    assert DEFAULT_CONTROL_PLANE_URL == "https://launch.percepteye.ai/api/flywheel/v1"


@pytest.mark.parametrize(("given", "env", "expected"), CASES)
def test_the_python_rule(given, env, expected):
    environ = {} if env is None else {ENV_CONTROL_PLANE_URL: env}
    assert resolve_control_plane_url(given, environ) == expected


def test_it_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv(ENV_CONTROL_PLANE_URL, "https://env.example/v1")
    assert resolve_control_plane_url() == "https://env.example/v1"
    monkeypatch.delenv(ENV_CONTROL_PLANE_URL)
    assert resolve_control_plane_url() == DEFAULT_CONTROL_PLANE_URL


def test_the_node_sdk_answers_the_same_table():
    sdk, node = require_node_sdk()
    table = [
        {"config": {} if given is None else {"controlPlaneUrl": given},
         "env": {} if env is None else {ENV_CONTROL_PLANE_URL: env}}
        for given, env, _ in CASES
    ]
    script = (
        "import { resolveConfig } from './src/config.js';\n"
        "let input = '';\n"
        "process.stdin.on('data', (c) => { input += c; });\n"
        "process.stdin.on('end', () => {\n"
        "  const rows = JSON.parse(input);\n"
        "  console.log(JSON.stringify(rows.map((r) =>\n"
        "    resolveConfig(r.config, r.env).controlPlaneUrl)));\n"
        "});\n"
    )
    answers = json.loads(run_node(sdk, node, script, stdin=json.dumps(table)))
    expected = [want for _, _, want in CASES]
    assert answers == expected, list(zip(CASES, answers, strict=False))
