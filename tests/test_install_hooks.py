"""One artifact: the wheel ships the host payload, one command registers it.

The customer used to need TWO installs — a marketplace plugin and this package —
and that split was a packaging accident, not an architecture. Its own commit says
so: "Both marketplaces install from a repository ROOT and neither can install a
subdirectory."

Two facts decide the shape here, both measured against goose 1.48.0:

* A ``plugins:`` entry in goose's config.yaml registers NOTHING. ``filter_by_config``
  only filters what ``list_dir_children`` already found on disk, auto-inserting
  unknown paths as enabled. A real directory has to exist under the plugins dir.
* A hook runs as ``sh -c`` with the LOGIN SHELL's PATH, so a bare ``python3``
  there is not necessarily the virtualenv that has this package importable.
  Hence absolute ``sys.executable`` and an absolute recorder path, and hence no
  dependence on ``${PLUGIN_ROOT}`` at all.
"""
from __future__ import annotations

import json

import pytest

from percepteye_agent_flywheel.contract import CONTRACT_VERSION
from percepteye_agent_flywheel.hosts import RECORDER
from percepteye_agent_flywheel.install_hooks import (
    PLUGIN_NAME,
    goose_plugins_dir,
    install_goose,
    installed_state,
)

# ── the payload ships in the package ───────────────────────────────────────


def test_the_recorder_ships_with_the_wheel():
    """If this fails the customer needs a second install again."""
    assert RECORDER.is_file()
    assert RECORDER.read_text(encoding="utf-8").lstrip().startswith(
        ("#!/usr/bin/env python3", '"""', "#"))


def test_the_tool_call_path_never_imports_the_sdk():
    """The recorder runs once per TOOL CALL and BLOCKS the agent each time.

    Measured on CPython 3.14: importing this package costs 46 ms against a 16 ms
    interpreter floor, so a module-level import would nearly triple the
    per-tool-call latency of every rollout.

    The assertion moved from "this file never names the package" to "no import
    of it runs at module scope", because the two SESSION events — claim on
    SessionStart, report on Stop — legitimately need the transport and pay that
    cost once per session. What must stay true is that the tool-call path never
    reaches it.
    """
    src = RECORDER.read_text(encoding="utf-8")
    for line in src.splitlines():
        if line.startswith(("import ", "from ")):        # column 0 == module scope
            assert "percepteye_agent_flywheel" not in line, (
                f"module-level SDK import in the recorder: {line.strip()}")


def test_the_sdk_is_reached_only_from_the_session_branches():
    """Every SDK import sits inside a function the tool-call path cannot enter."""
    import ast

    tree = ast.parse(RECORDER.read_text(encoding="utf-8"))
    allowed = {"_contributing", "_claim", "_report"}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef,)):
            continue
        if node.name in allowed:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, (ast.Import, ast.ImportFrom)):
                names = getattr(sub, "module", "") or ""
                names += " ".join(a.name for a in sub.names)
                assert "percepteye_agent_flywheel" not in names, (
                    f"SDK import inside {node.name}, which the per-tool-call "
                    f"path can reach")


def test_a_missing_sdk_degrades_to_local_recording():
    """A marketplace install has the plugin and no Python package at all. The
    import failure must land in the same branch as "no key": record locally,
    exit 0, say nothing."""
    src = RECORDER.read_text(encoding="utf-8")
    assert "except Exception:\n        return None" in src or "return None" in src


# ── what gets rendered ─────────────────────────────────────────────────────


def test_install_renders_a_directory_goose_will_discover(tmp_path):
    out = install_goose(plugins_dir=tmp_path)
    root = tmp_path / PLUGIN_NAME
    # goose's HookManager::from_plugins checks exactly this path and nothing else.
    assert (root / "hooks" / "hooks.json").is_file()
    assert out["root"] == str(root)


def test_the_command_is_absolute_on_both_halves(tmp_path):
    """No PATH lookup and no ${PLUGIN_ROOT}: the hook's PATH is the login
    shell's, where ``python3`` is not necessarily the venv that has us."""
    install_goose(plugins_dir=tmp_path, python="/opt/venv/bin/python")
    doc = json.loads((tmp_path / PLUGIN_NAME / "hooks" / "hooks.json").read_text())
    cmd = doc["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
    assert cmd.startswith('"/opt/venv/bin/python"')
    assert str(RECORDER) in cmd
    assert "${PLUGIN_ROOT}" not in cmd
    assert "python3 " not in cmd


def test_paths_are_quoted_because_virtualenv_paths_contain_spaces(tmp_path):
    """The command is handed to ``sh -c``; "Application Support" is a real path
    on every macOS install."""
    install_goose(plugins_dir=tmp_path, python="/opt/Application Support/py")
    doc = json.loads((tmp_path / PLUGIN_NAME / "hooks" / "hooks.json").read_text())
    cmd = doc["hooks"]["PostToolUse"][0]["hooks"][0]["command"]
    assert cmd.startswith('"/opt/Application Support/py"')


def test_pre_tool_use_is_never_subscribed(tmp_path):
    """``PreToolUse`` exists to allow or deny a call before it runs. Nothing
    here has any business on that path."""
    install_goose(plugins_dir=tmp_path)
    doc = json.loads((tmp_path / PLUGIN_NAME / "hooks" / "hooks.json").read_text())
    assert "PreToolUse" not in doc["hooks"]
    assert "PreToolUseResult" not in doc["hooks"]


def test_the_turn_events_are_subscribed(tmp_path):
    """Without them the trajectory is tool calls with no task and no answer."""
    install_goose(plugins_dir=tmp_path)
    doc = json.loads((tmp_path / PLUGIN_NAME / "hooks" / "hooks.json").read_text())
    assert set(doc["hooks"]) == {
        "PostToolUse", "PostToolUseFailure", "AfterTool",
        "UserPromptSubmit", "Stop", "SessionStart",
    }


def test_session_start_is_subscribed_because_it_is_what_claims(tmp_path):
    """The whole difference between recording locally and contributing a graded
    rollout. It is also why no worker is needed: the host runs it before the
    agent's first turn, it blocks, and its cwd IS the working directory — so it
    can fetch a task and put it where the agent reads it."""
    install_goose(plugins_dir=tmp_path)
    doc = json.loads((tmp_path / PLUGIN_NAME / "hooks" / "hooks.json").read_text())
    assert "SessionStart" in doc["hooks"]


def test_the_recorder_can_never_deny_a_turn(tmp_path):
    """``Stop`` goes through ``emit_blocking`` and its chain CAN deny.

    goose's ``classify_output`` produces a denial only on exit code 2 or a JSON
    verdict on stdout. The recorder's invariants — every path exits 0, nothing
    written to stdout — are what make subscribing a blocking event safe, so they
    are now load-bearing for something they were not written for.
    """
    from percepteye_agent_flywheel.hosts import RECORDER

    src = RECORDER.read_text(encoding="utf-8")
    assert "sys.exit(2)" not in src and "exit(2)" not in src
    # print() to stdout would be read as a verdict document.
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("print("):
            pytest.fail(f"the recorder writes to stdout: {stripped}")


def test_both_host_vocabularies_are_covered(tmp_path):
    """goose splits ok/failed across two events; Gemini reports it inline on one."""
    install_goose(plugins_dir=tmp_path)
    doc = json.loads((tmp_path / PLUGIN_NAME / "hooks" / "hooks.json").read_text())
    assert "PostToolUseFailure" in doc["hooks"]      # goose
    assert "AfterTool" in doc["hooks"]               # Gemini CLI


# ── idempotence, and not clobbering someone else's install ─────────────────


def test_install_is_idempotent(tmp_path):
    first = install_goose(plugins_dir=tmp_path)
    second = install_goose(plugins_dir=tmp_path)
    assert first["root"] == second["root"]
    assert installed_state(tmp_path)["installed"] is True


def test_a_plugin_we_did_not_install_is_not_clobbered(tmp_path):
    """A marketplace install of the standalone plugin is the EXPECTED case.

    Replacing what the customer chose with what we chose, without being asked,
    is the kind of surprise this package refuses everywhere else.
    """
    foreign = tmp_path / PLUGIN_NAME / "hooks"
    foreign.mkdir(parents=True)
    (foreign / "hooks.json").write_text('{"hooks":{}}', encoding="utf-8")

    assert installed_state(tmp_path)["foreign"] is True
    with pytest.raises(FileExistsError, match="marketplace"):
        install_goose(plugins_dir=tmp_path)

    out = install_goose(plugins_dir=tmp_path, force=True)
    assert out["replaced_foreign"] is True
    assert installed_state(tmp_path)["installed"] is True


# ── the stamp exists so drift is visible ───────────────────────────────────


def test_a_moved_virtualenv_reports_stale_rather_than_recording_nothing(tmp_path):
    """The rendered command is a SNAPSHOT of two absolute paths. If the
    virtualenv is rebuilt or moved, goose still loads the plugin and every hook
    silently does nothing — exactly the silent capture loss this package
    front-loads its checks to prevent. It must be reportable."""
    install_goose(plugins_dir=tmp_path)
    stamp = tmp_path / PLUGIN_NAME / ".percepteye-install.json"
    data = json.loads(stamp.read_text())
    data["recorder"] = "/gone/percepteye_record.py"
    stamp.write_text(json.dumps(data), encoding="utf-8")

    st = installed_state(tmp_path)
    assert st["recorder_present"] is False
    assert st["stale"] is True


def test_a_contract_bump_reports_stale(tmp_path):
    install_goose(plugins_dir=tmp_path)
    stamp = tmp_path / PLUGIN_NAME / ".percepteye-install.json"
    data = json.loads(stamp.read_text())
    assert data["contract_version"] == CONTRACT_VERSION
    data["contract_version"] = "flywheel/0"
    stamp.write_text(json.dumps(data), encoding="utf-8")
    assert installed_state(tmp_path)["stale"] is True


def test_not_installed_is_not_stale_it_is_absent(tmp_path):
    st = installed_state(tmp_path)
    assert st["installed"] is False and st["foreign"] is False


# ── where goose actually looks ─────────────────────────────────────────────


def test_the_plugins_dir_honours_goose_path_root(monkeypatch, tmp_path):
    """goose's ``Paths::plugins_dir`` reads $GOOSE_PATH_ROOT when it is
    ABSOLUTE, and ignores it otherwise (config/paths.rs)."""
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path))
    assert goose_plugins_dir() == tmp_path / ".agents" / "plugins"

    monkeypatch.setenv("GOOSE_PATH_ROOT", "relative/path")
    assert goose_plugins_dir() == __import__("pathlib").Path.home() / ".agents" / "plugins"

    monkeypatch.delenv("GOOSE_PATH_ROOT")
    assert goose_plugins_dir() == __import__("pathlib").Path.home() / ".agents" / "plugins"
