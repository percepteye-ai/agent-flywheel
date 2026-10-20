"""Every refusal a missing key causes says where a key comes from.

They used to name only the variable -- "an API key is required (or set
PERCEPTEYE_API_KEY)" -- which says what is missing and not where it comes
from. A developer who ran ``pip install agent-flywheel`` and never opened the
dashboard had no way to learn from it that a key is one sign-up away, and that
refusal is read at the one moment the developer needs the answer. So the answer
is in it: ``serve()``'s error, the one-shot reads' warnings, ``attach()``'s
``upload_refusal`` and both CLI subcommands all name :data:`SIGN_UP_URL`.

The Node plugin says the same, with the same URL, byte for byte; the last test
drives its module and compares.
"""
from __future__ import annotations

import json
from urllib.parse import urlsplit

import pytest

import agent_flywheel as fw
from agent_flywheel.cli import main
from agent_flywheel.control_plane import (
    DEFAULT_CONTROL_PLANE_URL,
    GET_A_KEY,
    SIGN_UP_URL,
)
from agent_flywheel.errors import ConfigurationError

from .node_sdk import require_node_sdk, run_node

KEY = "PERCEPTEYE_API_KEY"


@pytest.fixture
def keyless(monkeypatch):
    monkeypatch.delenv(KEY, raising=False)
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)


def _origin(url: str) -> tuple[str, str]:
    parts = urlsplit(url)
    return parts.scheme, parts.netloc


def test_the_link_is_on_the_default_control_planes_own_origin():
    assert _origin(SIGN_UP_URL) == _origin(DEFAULT_CONTROL_PLANE_URL)
    assert SIGN_UP_URL in GET_A_KEY


def test_serve_names_it(keyless, tmp_path):
    with pytest.raises(ConfigurationError) as exc:
        fw.serve("my_package.agent:build", agent_id="a", input_shape="text",
                 artifacts_dir=str(tmp_path))
    assert KEY in str(exc.value)
    assert SIGN_UP_URL in str(exc.value)


@pytest.mark.parametrize("call", ["current_policy", "current_prompt"])
def test_the_one_shot_reads_name_it(keyless, caplog, call):
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        assert getattr(fw, call)(agent_id="a") == {}
    assert any(SIGN_UP_URL in r.getMessage() for r in caplog.records), caplog.text


def test_attach_names_it_in_the_upload_refusal(keyless, tmp_path):
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    try:
        assert px.may_upload is False
        assert SIGN_UP_URL in (px.upload_refusal or "")
    finally:
        px.close()


def test_the_cli_names_it(keyless, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setenv("PERCEPTEYE_HOME", str(tmp_path / ".percepteye"))
    for name in ("PERCEPTEYE_AGENT_MODE", "PERCEPTEYE_LLM_BASE_URL",
                 "PERCEPTEYE_CAPTURE"):
        monkeypatch.delenv(name, raising=False)
    assert main(["install-hooks"]) == 0
    assert SIGN_UP_URL in capsys.readouterr().out
    main(["status"])
    assert SIGN_UP_URL in capsys.readouterr().out


def test_the_node_plugin_names_the_same_link():
    sdk, node = require_node_sdk()
    script = (
        "import { SIGN_UP_URL, GET_A_KEY } from './src/config.js';\n"
        "console.log(JSON.stringify({ SIGN_UP_URL, GET_A_KEY }));\n"
    )
    theirs = json.loads(run_node(sdk, node, script))
    assert theirs == {"SIGN_UP_URL": SIGN_UP_URL, "GET_A_KEY": GET_A_KEY}
