"""The goose install reads and writes the customer's config faithfully, and
says what it would do without doing it.

What each test pins, and the defect it exists to catch:

  * YAML (PY-H6): a quoted value with a trailing comment reads as the value,
    not with its quotes; ``null``/``~`` are no value; a tab-preceded ``#`` is
    a comment; a written value is a JSON string (quotes and backslashes
    escaped), and one holding a line break is refused -- it was a way to add
    a second key.
  * ``status`` IS READ-ONLY (PY-H3): it reconciled -- wrote the config -- and a
    bad mode crashed it instead of being said in its endpoint line.
  * ONE READING OF THE MODE (PY-H7): ``plan`` treated anything but the exact
    string ``"production"`` as training; an alias or a typo routed at us.
  * THE STAMP (PY-M5): a non-dict stamp crashed; a moved interpreter and a
    hooks.json this release would not render were not stale; a corrupt stamp
    replaced with --force recorded OUR gateway as the customer's original;
    the stamp and hooks were written in place.
  * THE WRITE (PY-M8): a symlinked config was replaced by a plain file; its
    mode was reset to 0600; the bytes were not fsynced before the rename.
  * ``report`` (PY-H17): picked the session by directory mtime, and crashed on
    a session deleted while it scanned.
"""
from __future__ import annotations

import json
import os
import stat
import time

import pytest

from agent_flywheel import endpoint as endpoint_mod
from agent_flywheel import install_hooks as ih
from agent_flywheel.cli import main
from agent_flywheel.endpoint import GOOSE_ENDPOINT_KEY, plan, read_scalar, set_scalar
from agent_flywheel.errors import ConfigurationError

GATEWAY = "https://gateway.percepteye.test/v1"
THEIRS = "https://their-own-proxy.internal/v1"


@pytest.fixture
def host(tmp_path, monkeypatch):
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setenv("PERCEPTEYE_HOME", str(tmp_path / ".percepteye"))
    for var in ("OPENAI_HOST", "OPENAI_BASE_URL", "PERCEPTEYE_CAPTURE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PERCEPTEYE_AGENT_MODE", "training")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pefw_test")
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    cfg = tmp_path / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(f"GOOSE_PROVIDER: openai\n{GOOSE_ENDPOINT_KEY}: {THEIRS}\n",
                   encoding="utf-8")
    return tmp_path, cfg


def _plugins(root):
    return root / ".agents" / "plugins"


# ── PY-H6: YAML ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(("rhs", "want"), [
    (f'"{THEIRS}" # mine', THEIRS),
    (f"'{THEIRS}'   # mine", THEIRS),
    (f"{THEIRS}\t# mine", THEIRS),
    ("null", None),
    ("~", None),
    ("NULL  # unset", None),
    ('"https://x/a\\"b"', 'https://x/a"b'),
    ("'it''s'", "it's"),
    ("https://x/#frag", "https://x/#frag"),
])
def test_every_yaml_spelling_reads_as_its_value(rhs, want):
    assert read_scalar(f"{GOOSE_ENDPOINT_KEY}: {rhs}\n", GOOSE_ENDPOINT_KEY) == want


@pytest.mark.parametrize("value", [
    'https://x/"quoted"\\path',
    "https://x/ünï",
])
def test_a_written_value_reads_back_exactly(value):
    text = set_scalar("A: 1\n", GOOSE_ENDPOINT_KEY, value)
    assert read_scalar(text, GOOSE_ENDPOINT_KEY) == value
    assert [ln.split(":")[0] for ln in text.splitlines()] == ["A", GOOSE_ENDPOINT_KEY]


@pytest.mark.parametrize("value", [
    'https://x"\nGOOSE_PROVIDER: "attacker',
    "https://x\rY: 1",
    "https://x\x00",
])
def test_a_value_that_could_write_a_second_key_is_refused(value):
    with pytest.raises(ConfigurationError, match="line break or control"):
        set_scalar("A: 1\n", GOOSE_ENDPOINT_KEY, value)


# ── PY-H3: status changes nothing ───────────────────────────────────────


def test_status_reports_where_traffic_will_go_without_writing_it(host, monkeypatch, capsys):
    root, cfg = host
    ih.install_goose(plugins_dir=_plugins(root), config_path=cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == GATEWAY
    monkeypatch.setenv("PERCEPTEYE_AGENT_MODE", "production")
    before = cfg.read_bytes()
    main(["status"])
    out = capsys.readouterr().out
    assert cfg.read_bytes() == before, "status rewrote the customer's config"
    assert THEIRS in out and "still says" in out


def test_status_with_a_bad_mode_says_so_and_prints_the_rest(host, monkeypatch, capsys):
    root, cfg = host
    ih.install_goose(plugins_dir=_plugins(root), config_path=cfg)
    monkeypatch.setenv("PERCEPTEYE_AGENT_MODE", "prodction")
    main(["status"])
    out = capsys.readouterr().out
    assert "endpoint  UNKNOWN" in out and "prodction" in out
    assert "rollouts  NOT claimed" in out


# ── PY-H7: one reading of the mode ──────────────────────────────────────


@pytest.mark.parametrize("mode", ["serving", "prod", "PRODUCTION", " production "])
def test_every_spelling_of_production_hands_the_endpoint_back(mode):
    decision = plan(mode, original=THEIRS, training_url=GATEWAY, has_key=True)
    assert (decision["mode"], decision["value"], decision["routed"]) == (
        "production", THEIRS, False)


def test_a_mode_nobody_recognises_is_refused_not_routed():
    with pytest.raises(ConfigurationError, match="prodction"):
        plan("prodction", original=THEIRS, training_url=GATEWAY, has_key=True)


def test_reconcile_reads_an_explicit_mode_through_the_same_rule(host):
    root, cfg = host
    ih.install_goose(plugins_dir=_plugins(root), config_path=cfg)
    ih.reconcile_endpoint(mode="serving", plugins_dir=_plugins(root), config_path=cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == THEIRS


# ── PY-M5: the stamp, and the key ───────────────────────────────────────


@pytest.mark.parametrize("stamp", [
    "[1, 2]", "null", '"text"', '{"python": 3, "recorder": "/x"}', "{not json",
])
def test_a_corrupt_stamp_is_reported_not_a_crash(tmp_path, stamp):
    root = tmp_path / ih.PLUGIN_NAME
    root.mkdir()
    (root / ih.STAMP_FILENAME).write_text(stamp)
    state = ih.installed_state(tmp_path)
    assert state["installed"] is False and state.get("corrupt") is True
    with pytest.raises(FileExistsError, match="cannot read"):
        ih.install_goose(plugins_dir=tmp_path)


def test_a_corrupt_stamp_and_force_never_record_our_gateway_as_theirs(host, monkeypatch):
    root, cfg = host
    ih.install_goose(plugins_dir=_plugins(root), config_path=cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == GATEWAY
    (_plugins(root) / ih.PLUGIN_NAME / ih.STAMP_FILENAME).write_text("{broken")
    ih.install_goose(plugins_dir=_plugins(root), config_path=cfg, force=True)
    stamp = json.loads((_plugins(root) / ih.PLUGIN_NAME / ih.STAMP_FILENAME).read_text())
    assert stamp[ih.ORIGINAL_ENDPOINT_KEY] is None
    monkeypatch.setenv("PERCEPTEYE_AGENT_MODE", "production")
    ih.reconcile_endpoint(plugins_dir=_plugins(root), config_path=cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) != GATEWAY


def test_a_moved_interpreter_is_stale(tmp_path):
    ih.install_goose(plugins_dir=tmp_path, python=str(tmp_path / "gone" / "python"))
    state = ih.installed_state(tmp_path)
    assert state["python_present"] is False and state["stale"] is True


def test_a_hooks_file_this_release_would_not_render_is_stale(tmp_path):
    import sys

    ih.install_goose(plugins_dir=tmp_path, python=sys.executable)
    assert ih.installed_state(tmp_path)["stale"] is False
    hooks = tmp_path / ih.PLUGIN_NAME / "hooks" / "hooks.json"
    doc = json.loads(hooks.read_text())
    del doc["hooks"]["Stop"]  # an older release that did not subscribe to it
    hooks.write_text(json.dumps(doc, indent=2) + "\n")
    state = ih.installed_state(tmp_path)
    assert state["hooks_current"] is False and state["stale"] is True


def test_the_install_writes_whole_files_only(tmp_path, monkeypatch):
    written: list[str] = []
    real = endpoint_mod.write_config

    def spy(path, text):
        written.append(os.path.basename(path))
        return real(path, text)

    monkeypatch.setattr(ih, "write_config", spy)
    ih.install_goose(plugins_dir=tmp_path)
    assert {"hooks.json", ih.STAMP_FILENAME} <= set(written)


def test_a_stored_key_routes_like_the_variable(host, monkeypatch):
    from agent_flywheel import credentials as creds
    from agent_flywheel.control_plane import resolve_control_plane_url

    root, cfg = host
    monkeypatch.delenv("PERCEPTEYE_API_KEY")
    creds.save_stored(creds.StoredCredential(
        control_plane_url=resolve_control_plane_url(), agent_id="goose",
        api_key="pefw_" + "S" * 43))
    out = ih.install_goose(plugins_dir=_plugins(root), config_path=cfg)
    assert out["endpoint"]["routed"] is True
    assert ih.claim_refusal() is None
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "   ")  # blank is unset, everywhere
    assert ih.claim_refusal() is None


# ── PY-M8: the write ────────────────────────────────────────────────────


def test_a_symlinked_config_stays_a_symlink_and_keeps_its_mode(host):
    root, cfg = host
    real = root / "dotfiles" / "goose.yaml"
    real.parent.mkdir()
    real.write_text(cfg.read_text())
    os.chmod(real, 0o640)
    cfg.unlink()
    cfg.symlink_to(real)
    ih.install_goose(plugins_dir=_plugins(root), config_path=cfg)
    assert cfg.is_symlink(), "the customer's symlink was replaced by a plain file"
    assert read_scalar(real.read_text(), GOOSE_ENDPOINT_KEY) == GATEWAY
    assert stat.S_IMODE(os.stat(real).st_mode) == 0o640


def test_the_bytes_are_fsynced_before_the_rename(tmp_path, monkeypatch):
    order: list[str] = []
    monkeypatch.setattr(endpoint_mod.os, "fsync", lambda fd: order.append("fsync"))
    real_replace = os.replace
    monkeypatch.setattr(endpoint_mod.os, "replace",
                        lambda a, b: (order.append("replace"), real_replace(a, b)))
    endpoint_mod.write_config(tmp_path / "c.yaml", "A: 1\n")
    assert order[:2] == ["fsync", "replace"]


# ── PY-H17: report picks the session that recorded last ─────────────────


def test_report_picks_the_session_whose_records_are_newest(host, capsys):
    root, _ = host
    turns = root / ".percepteye" / "turns"
    for name, ok in (("busy-session", "failed"), ("touched-session", "ok")):
        (turns / name).mkdir(parents=True)
        (turns / name / "tool_calls.jsonl").write_text(
            f'{{"name": "bash", "outcome": "{ok}"}}\n', encoding="utf-8")
    old, now = time.time() - 3600, time.time()
    # The busy session appended a record just now (file mtime moves, the
    # directory's does not); the other only had an entry added long ago.
    os.utime(turns / "touched-session" / "tool_calls.jsonl", (old, old))
    os.utime(turns / "busy-session" / "tool_calls.jsonl", (now, now))
    os.utime(turns / "busy-session", (old, old))
    os.utime(turns / "touched-session", (now, now))
    assert main(["report"]) == 0
    out = capsys.readouterr().out
    assert "busy-session" in out and "touched-session" not in out, out


def test_report_skips_a_session_deleted_while_it_scans(host, monkeypatch, capsys):
    root, _ = host
    turns = root / ".percepteye" / "turns"
    for name in ("kept", "vanishing"):
        (turns / name).mkdir(parents=True)
        (turns / name / "tool_calls.jsonl").write_text(
            '{"name": "bash", "outcome": "ok"}\n', encoding="utf-8")
    real_getmtime = os.path.getmtime

    def getmtime(path):
        if "vanishing" in str(path):
            raise FileNotFoundError(path)
        return real_getmtime(path)

    monkeypatch.setattr(os.path, "getmtime", getmtime)
    assert main(["report"]) == 0
    assert "kept" in capsys.readouterr().out
