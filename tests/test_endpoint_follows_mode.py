"""The mode owns the endpoint, and production hands it back.

Routing a customer's agent through our gateway is the one thing this install
changes about the agent itself. It has to be reversible by the switch they
already have -- ``PERCEPTEYE_AGENT_MODE`` -- and reversible to what THEY had,
not to a default we picked.
"""
from __future__ import annotations

import json

import pytest

from agent_flywheel.endpoint import (
    GOOSE_ENDPOINT_KEY,
    plan,
    read_scalar,
    set_scalar,
)
from agent_flywheel.install_hooks import (
    ORIGINAL_ENDPOINT_KEY,
    PLUGIN_NAME,
    STAMP_FILENAME,
    install_goose,
    reconcile_endpoint,
)

GATEWAY = "https://gateway.percepteye.test/v1"
THEIRS = "https://their-own-proxy.internal/v1"


@pytest.fixture
def host(tmp_path, monkeypatch):
    """An isolated goose: plugins dir, config, and a clean environment."""
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path))
    for var in ("OPENAI_HOST", "OPENAI_BASE_URL", "PERCEPTEYE_AGENT_MODE",
                "PERCEPTEYE_LLM_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    # Routing needs BOTH the training mode and a key. The keyless case has its
    # own test below; every other test here is about the switch, not consent.
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pefw_test")
    cfg = tmp_path / ".config" / "goose" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        "# my own notes\n"
        "GOOSE_PROVIDER: openai\n"
        f"{GOOSE_ENDPOINT_KEY}: {THEIRS}\n"
        "GOOSE_MODEL: gpt-4o\n", encoding="utf-8")
    return tmp_path, cfg


def _plugins(root):
    return root / ".agents" / "plugins"


def _install(root, cfg):
    return install_goose(plugins_dir=_plugins(root), config_path=cfg)


def _reconcile(root, cfg):
    return reconcile_endpoint(plugins_dir=_plugins(root), config_path=cfg)


# ── the switch ─────────────────────────────────────────────────────────────


def test_training_routes_at_the_gateway(host, monkeypatch):
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == GATEWAY


def test_production_gives_back_the_customers_own_endpoint(host, monkeypatch):
    """Not a default, not blank -- the value THEY had."""
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    monkeypatch.setenv("PERCEPTEYE_AGENT_MODE", "production")
    _reconcile(root, cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == THEIRS


def test_production_removes_the_key_when_they_never_had_one(host, monkeypatch):
    """Restoring "nothing" means ABSENT, not empty. An empty key reads as ours
    to keep updating, and goose filters it out anyway -- so the file would be
    the only thing left claiming we are still involved."""
    root, cfg = host
    cfg.write_text("GOOSE_PROVIDER: openai\n", encoding="utf-8")
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == GATEWAY
    monkeypatch.setenv("PERCEPTEYE_AGENT_MODE", "production")
    _reconcile(root, cfg)
    assert GOOSE_ENDPOINT_KEY not in cfg.read_text()


def test_the_flip_round_trips(host, monkeypatch):
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    for mode, expected in (("production", THEIRS), ("training", GATEWAY),
                           ("production", THEIRS)):
        monkeypatch.setenv("PERCEPTEYE_AGENT_MODE", mode)
        _reconcile(root, cfg)
        assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == expected


# ── the trap: a re-install must not overwrite the memory of "theirs" ───────


def test_reinstalling_does_not_adopt_our_own_url_as_theirs(host, monkeypatch):
    """`pip install --upgrade` then `install-hooks` is the ORDINARY upgrade
    path, and at that moment the config already holds OUR gateway. Capturing it
    again would record the gateway as "what the customer had", and production
    would then restore the training endpoint -- a kill switch that looks like it
    worked and changes nothing."""
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    _install(root, cfg)                     # the upgrade
    stamp = json.loads(
        (_plugins(root) / PLUGIN_NAME / STAMP_FILENAME).read_text())
    assert stamp[ORIGINAL_ENDPOINT_KEY] == THEIRS
    monkeypatch.setenv("PERCEPTEYE_AGENT_MODE", "production")
    _reconcile(root, cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == THEIRS


# ── refusing to guess ──────────────────────────────────────────────────────


def test_training_with_no_known_gateway_leaves_routing_alone(host):
    """An endpoint we invent would send the customer's traffic -- and their API
    key -- somewhere they never named."""
    root, cfg = host
    out = _install(root, cfg)
    assert out["endpoint"]["routed"] is False
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == THEIRS


def test_an_uninstalled_plugin_never_writes_the_config(host):
    root, cfg = host
    before = cfg.read_text()
    assert _reconcile(root, cfg)["changed"] is False
    assert cfg.read_text() == before


# ── not fighting the customer's environment ────────────────────────────────


@pytest.mark.parametrize("var", ["OPENAI_HOST", "OPENAI_BASE_URL"])
def test_an_env_override_is_reported_not_overwritten(host, monkeypatch, var):
    """Both env ranks outrank the config in goose's resolver, so a customer who
    exports one wins over us immediately. Saying so is the difference between a
    customer debugging for an hour and reading one line."""
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    monkeypatch.setenv(var, "https://i-decide.example/v1")
    assert _reconcile(root, cfg)["shadowed_by"] == var


# ── the line editor keeps the rest of the file ─────────────────────────────


def test_the_customers_comments_and_keys_survive(host, monkeypatch):
    """A parse-and-dump would drop their comments and reorder their keys as a
    side effect of us changing one endpoint."""
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    text = cfg.read_text()
    assert "# my own notes" in text
    assert "GOOSE_MODEL: gpt-4o" in text
    assert text.index("GOOSE_PROVIDER") < text.index(GOOSE_ENDPOINT_KEY)


def test_an_indented_key_is_not_ours_to_touch():
    """An indented key belongs to somebody else's mapping."""
    text = f"extensions:\n  {GOOSE_ENDPOINT_KEY}: https://nested/v1\n"
    assert read_scalar(text, GOOSE_ENDPOINT_KEY) is None
    assert "https://nested/v1" in set_scalar(text, GOOSE_ENDPOINT_KEY, GATEWAY)


@pytest.mark.parametrize("raw,want", [
    (f'{GOOSE_ENDPOINT_KEY}: "{THEIRS}"', THEIRS),
    (f"{GOOSE_ENDPOINT_KEY}: '{THEIRS}'", THEIRS),
    (f"{GOOSE_ENDPOINT_KEY}: {THEIRS}  # inline", THEIRS),
    (f"{GOOSE_ENDPOINT_KEY}:   {THEIRS}", THEIRS),
    (f"{GOOSE_ENDPOINT_KEY}:", None),
])
def test_every_shape_a_url_is_written_in(raw, want):
    assert read_scalar(raw + "\n", GOOSE_ENDPOINT_KEY) == want


def test_production_is_a_pure_decision_that_reads_no_files():
    assert plan("production", original=THEIRS)["value"] == THEIRS
    assert plan("production", original=None)["value"] is None


# ── the hook path, run as goose runs it ────────────────────────────────────


def _session_start(root, cfg, env_extra):
    """Invoke the recorder exactly as goose does: a JSON event on stdin, in a
    subprocess, with only the environment goose passes through."""
    import os
    import subprocess
    import sys

    from agent_flywheel.hosts import RECORDER

    env = dict(os.environ)
    env.update({"GOOSE_PATH_ROOT": str(root),
                "PERCEPTEYE_LLM_BASE_URL": GATEWAY})
    for var in ("OPENAI_HOST", "OPENAI_BASE_URL", "PERCEPTEYE_AGENT_MODE",
                "PERCEPTEYE_CAPTURE"):
        env.pop(var, None)
    env["PERCEPTEYE_API_KEY"] = "pefw_test"
    # The control-plane URL defaults to production now (the Node SDK's rule),
    # and SessionStart claims a practice task with a key set: point it at a
    # port nothing listens on, so the subprocess never reaches the live
    # service. The in-process guard in conftest cannot follow it here.
    env["PERCEPTEYE_CONTROL_PLANE_URL"] = "http://127.0.0.1:9/api/flywheel/v1"
    env.update(env_extra)
    proc = subprocess.run(
        [sys.executable, str(RECORDER)],
        input=json.dumps({"event": "SessionStart", "session_id": "s1",
                          "working_dir": str(root)}),
        capture_output=True, text=True, env=env, timeout=60)
    return proc


def test_a_mode_flip_takes_effect_with_no_command_at_all(host, monkeypatch):
    """THE POINT OF THE WHOLE MECHANISM. The customer changes one environment
    variable and starts their agent; nothing else runs. goose fires SessionStart,
    the hook reconciles, and the endpoint follows."""
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == GATEWAY

    proc = _session_start(root, cfg, {"PERCEPTEYE_AGENT_MODE": "production"})
    assert proc.returncode == 0
    assert proc.stdout == "", "stdout on a blocking hook chain can deny the turn"
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == THEIRS


def test_capture_off_still_hands_the_endpoint_back(host, monkeypatch):
    """`_capture_dir` returns None for PERCEPTEYE_CAPTURE=0 and main() returns
    immediately. A customer who has turned capture OFF is exactly the one whose
    endpoint most needs handing back -- reconciling below that early return
    would strand them routed at us with capture disabled: the worst of both,
    and silent."""
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    proc = _session_start(root, cfg, {"PERCEPTEYE_AGENT_MODE": "production",
                                      "PERCEPTEYE_CAPTURE": "0"})
    assert proc.returncode == 0
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == THEIRS


def test_a_keyless_install_is_never_routed(host, monkeypatch):
    """The README's ordinary first run: pip install, install-hooks, no account.

    Routing at our gateway with no key sends every completion to an endpoint
    that refuses it -- the agent stops working, and the visible cause is goose.
    Nothing leaves the machine without a key, and that includes the traffic.
    """
    root, cfg = host
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    out = _install(root, cfg)
    assert out["endpoint"]["routed"] is False
    assert "PERCEPTEYE_API_KEY" in out["endpoint"]["reason"]
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == THEIRS


def test_removing_the_key_hands_the_endpoint_back(host, monkeypatch):
    """Consent withdrawn is consent withdrawn, without a mode flip."""
    root, cfg = host
    monkeypatch.setenv("PERCEPTEYE_LLM_BASE_URL", GATEWAY)
    _install(root, cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == GATEWAY
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    _reconcile(root, cfg)
    assert read_scalar(cfg.read_text(), GOOSE_ENDPOINT_KEY) == THEIRS
