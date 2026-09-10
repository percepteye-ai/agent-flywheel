"""Describing a command agent from the HOST, and the wiring that uses it.

A CLI agent has no Python object to reflect on, so ``introspect(None)`` returns
None and the agent registers as ``unreadable`` -- which blocks workflow
generation and every phase downstream of it. These tests cover the way out:
the host writes ``discovered_agent.json``, and ``serve()`` runs the agent once
at startup to make it do so.
"""
from __future__ import annotations

import ast
import json
import os
import stat
import sys

import pytest

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel.contract import CONTRACT_VERSION
from percepteye_agent_flywheel.host_description import (
    DISCOVERED_AGENT_FILENAME,
    parse_discovered_agent,
    read_discovered_agent,
    warm_up_command,
)

HOST_RECORD = {
    "agent_name": "triage-agent",
    "framework": "openclaw",
    "source_type": "openclaw_llm_input",
    "role": "standalone",
    "system_prompt": "You are a triage agent.",
    "tool_definitions": [
        {
            "name": "glob",
            "description": "Find files",
            "input_schema": {"type": "object", "properties": {"pattern": {"type": "string"}}},
            "original_format": "openclaw_tool",
        },
        {
            "name": "read",
            "description": "Read a file",
            "input_schema": {},
            "original_format": "openclaw_tool",
        },
    ],
    "discovered_model_config": {"provider": "anthropic", "model_name": "claude-opus-5"},
}


# ── parsing ────────────────────────────────────────────────────────────────


def test_parses_the_host_record_into_a_description():
    d = parse_discovered_agent(HOST_RECORD)
    assert d is not None
    assert d.agent_name == "triage-agent"
    # `framework` is a lookup key downstream, not a label -- the exact spelling
    # is what maps it to a known framework.
    assert d.framework == "openclaw"
    assert [t.name for t in d.tool_definitions] == ["glob", "read"]
    assert d.tool_definitions[0].input_schema["properties"] == {"pattern": {"type": "string"}}
    assert d.discovered_model_config.model_name == "claude-opus-5"


def test_the_description_survives_to_wire_with_its_tool_names():
    """The whole point is that tool NAMES reach the control plane."""
    wire = parse_discovered_agent(HOST_RECORD).to_wire()
    assert [t["name"] for t in wire["tool_definitions"]] == ["glob", "read"]
    assert wire["system_prompt"] == "You are a triage agent."
    assert len(wire["framework"]) <= 64 and len(wire["source_type"]) <= 64


def test_null_tools_means_not_observed_and_empty_means_observed_empty():
    """The absence contract, which the consumer grades on."""
    assert (
        parse_discovered_agent({**HOST_RECORD, "tool_definitions": None}).tool_definitions is None
    )
    assert parse_discovered_agent({**HOST_RECORD, "tool_definitions": []}).tool_definitions == []


def test_a_nameless_tool_is_dropped_so_our_count_is_the_surviving_count():
    d = parse_discovered_agent(
        {**HOST_RECORD, "tool_definitions": [{"name": "keep"}, {"description": "no name"}, "junk"]}
    )
    assert [t.name for t in d.tool_definitions] == ["keep"]


def test_an_unusable_record_is_an_absence_not_a_crash():
    assert parse_discovered_agent(None) is None
    assert parse_discovered_agent("not a dict") is None
    assert parse_discovered_agent({"framework": "openclaw"}) is None  # no name
    assert parse_discovered_agent({"agent_name": "a"}) is None  # no framework


def test_a_huge_prompt_is_bounded_the_same_way_reflection_bounds_one():
    from percepteye_agent_flywheel.introspection import _MAX_PROMPT_CHARS

    d = parse_discovered_agent({**HOST_RECORD, "system_prompt": "x" * 500_000})
    assert len(d.system_prompt) == _MAX_PROMPT_CHARS


def test_exact_observation_ignores_unknown_telemetry_but_keeps_full_prompt():
    first = parse_discovered_agent({
        **HOST_RECORD,
        "system_prompt": "x" * 20_000 + "A",
        "debug_timestamp": 1,
    })
    telemetry_only = parse_discovered_agent({
        **HOST_RECORD,
        "system_prompt": "x" * 20_000 + "A",
        "debug_timestamp": 999,
    })
    behavior_change = parse_discovered_agent({
        **HOST_RECORD,
        "system_prompt": "x" * 20_000 + "B",
        "debug_timestamp": 1,
    })
    assert first.exact_observation_sha256 == telemetry_only.exact_observation_sha256
    assert first.exact_observation_sha256 != behavior_change.exact_observation_sha256


# ── reading ────────────────────────────────────────────────────────────────


def test_reads_the_file_the_host_wrote(tmp_path):
    (tmp_path / DISCOVERED_AGENT_FILENAME).write_text(json.dumps(HOST_RECORD))
    assert read_discovered_agent(tmp_path).agent_name == "triage-agent"


def test_a_missing_or_corrupt_file_is_an_absence(tmp_path):
    assert read_discovered_agent(tmp_path) is None
    (tmp_path / DISCOVERED_AGENT_FILENAME).write_text("{ not json")
    assert read_discovered_agent(tmp_path) is None


# ── the warm-up ────────────────────────────────────────────────────────────


def _fake_cli(tmp_path, body: str, name: str = "fake_agent.py") -> list[str]:
    """A stand-in for a CLI agent whose host writes a description."""
    script = tmp_path / name
    script.write_text(body)
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return [sys.executable, str(script)]


# `%r` embeds HOST_RECORD as a Python literal into GENERATED SOURCE. Kept as
# percent-format on purpose: an f-string or .format() gives brace characters
# meaning inside a template whose whole job is emitting Python, so a dict or
# set literal added here later would break confusingly or silently. UP031 is a
# good rule with a bad fit for generated source.
WRITES_THEN_SLEEPS = """
import json, os, sys, time
d = os.environ["PERCEPTEYE_TRAJECTORY_DIR"]
open(os.path.join(d, "discovered_agent.json"), "w").write(json.dumps(%r))
# The real agent keeps running after its first model call; the warm-up must
# not wait for it to finish.
time.sleep(600)
""" % (HOST_RECORD,)  # noqa: UP031


def test_warm_up_reads_the_description_and_does_not_wait_for_the_agent(tmp_path):
    argv = _fake_cli(tmp_path, WRITES_THEN_SLEEPS)
    out = tmp_path / "traj"
    d = warm_up_command(argv, trajectory_dir=out, timeout_s=30)
    assert d is not None
    assert [t.name for t in d.tool_definitions] == ["glob", "read"]


def test_warm_up_passes_the_trajectory_dir_the_plugin_reads(tmp_path):
    """The env var IS the interface; without it the plugin stays inert."""
    argv = _fake_cli(
        tmp_path,
        "import os\n"
        "open(os.environ['PERCEPTEYE_TRAJECTORY_DIR'] + '/seen', 'w').write('1')\n",
    )
    out = tmp_path / "traj"
    warm_up_command(argv, trajectory_dir=out, timeout_s=30)
    assert (out / "seen").exists()


# Percent-format on purpose -- same reason as WRITES_THEN_SLEEPS above.
WRITES_THEN_ENRICHES = """
import json, os, sys, time
d = os.environ["PERCEPTEYE_TRAJECTORY_DIR"]
p = os.path.join(d, "discovered_agent.json")
rec = %r
# The first write is a LOWER BOUND: an integration may add tools the host's
# own hook could not see (MCP-bridged ones, for instance) after the agent
# exits. This is the real observed shape, not a hypothetical.
first = dict(rec); first["tool_definitions"] = rec["tool_definitions"][:1]
open(p, "w").write(json.dumps(first))
time.sleep(0.6)
open(p, "w").write(json.dumps(rec))
""" % (HOST_RECORD,)  # noqa: UP031


def test_warm_up_waits_for_the_catalogue_to_settle(tmp_path):
    """Regression: killing on first sight truncated the catalogue.

    Measured against the real agent, the first write held 38 tools and the
    finished description held 77 -- the warm-up returned the 38 and the MCP
    tools, which are the whole point of that agent, were silently absent.
    """
    argv = _fake_cli(tmp_path, WRITES_THEN_ENRICHES, name="enriching_agent.py")
    d = warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30)
    assert d is not None
    assert [t.name for t in d.tool_definitions] == ["glob", "read"], (
        "the warm-up returned the FIRST write, not the finished catalogue"
    )


def test_warm_up_returns_none_when_the_host_writes_nothing(tmp_path):
    argv = _fake_cli(tmp_path, "import sys; sys.exit(0)\n")
    assert warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=10) is None


def test_warm_up_returns_none_when_the_binary_cannot_start(tmp_path):
    assert warm_up_command(
        ["/nonexistent/agent/binary"], trajectory_dir=tmp_path / "t", timeout_s=5
    ) is None


def test_a_stale_description_is_not_reported_as_this_runs(tmp_path):
    """A previous warm-up's file would make a broken agent look healthy."""
    out = tmp_path / "traj"
    out.mkdir()
    (out / DISCOVERED_AGENT_FILENAME).write_text(json.dumps(HOST_RECORD))
    argv = _fake_cli(tmp_path, "import sys; sys.exit(1)\n")
    assert warm_up_command(argv, trajectory_dir=out, timeout_s=10) is None


def test_warm_up_kills_the_child_it_started(tmp_path):
    """A warm-up that leaks a process would leave one per serve() restart."""
    argv = _fake_cli(tmp_path, WRITES_THEN_SLEEPS)
    out = tmp_path / "traj"
    assert warm_up_command(argv, trajectory_dir=out, timeout_s=30) is not None
    # The sleeping child must be gone. Its pid is not exposed, so assert on the
    # only observable: no python process still holds this script open.
    import subprocess

    listing = subprocess.run(
        ["/bin/ps", "-eo", "command"], capture_output=True, text=True
    ).stdout
    assert str(tmp_path / "fake_agent.py") not in listing


# ── the wiring: serve() must actually use it ───────────────────────────────
#
# The recurring failure in this workspace is a correct function nothing calls.
# These go through serve() rather than calling the helper.


class _FakeTransport:
    def __init__(self, client, agent_id):
        self.agent_id = agent_id
        self.registered: dict | None = None

    def health(self):
        return {"contract_versions_supported": [CONTRACT_VERSION]}

    def register(self, spec):
        self.registered = spec
        return {"agent_id": self.agent_id, "grade": "advisory", "grade_reasons": []}

    def claim(self, max_rollouts):
        return [], 0


@pytest.fixture
def registered(monkeypatch, tmp_path):
    box: dict = {}

    def _factory(client, agent_id):
        tr = _FakeTransport(client, agent_id)
        box["tr"] = tr
        return tr

    monkeypatch.setattr(fw, "AttachTransport", _factory)
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid/v1")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "perk_test")

    def _run(entrypoint, **kw):
        kw.setdefault("agent_id", "a1")
        kw.setdefault("input_shape", "text")
        kw.setdefault("artifacts_dir", str(tmp_path / "artifacts"))
        kw.setdefault("max_rollouts", 0)
        fw.serve(entrypoint, **kw)
        return box["tr"].registered

    return _run


def test_serve_registers_a_command_agent_as_described(registered, tmp_path):
    """The end this whole module exists for.

    Before: a command entrypoint registered `discovered_agent: null`, the
    control plane recorded `unreadable`, and no workflows were ever generated.
    """
    argv = _fake_cli(tmp_path, WRITES_THEN_SLEEPS)
    spec = registered(argv)

    described = spec["discovered_agent"]
    assert described is not None, "a command agent still registers as unreadable"
    assert [t["name"] for t in described["tool_definitions"]] == ["glob", "read"]
    assert described["framework"] == "openclaw"
    assert described["system_prompt"] == "You are a triage agent."


def test_serve_still_sends_null_when_the_host_describes_nothing(registered, tmp_path):
    """`null` is load-bearing: it means we could not look, and the control
    plane distinguishes it from the key being absent entirely."""
    argv = _fake_cli(tmp_path, "import sys; sys.exit(0)\n")
    with pytest.warns(RuntimeWarning, match="allowConversationAccess"):
        spec = registered(argv)
    assert spec["discovered_agent"] is None


def test_introspect_false_still_omits_the_key_entirely(registered, tmp_path):
    """The refusal must not be bypassed by the new path."""
    argv = _fake_cli(tmp_path, WRITES_THEN_SLEEPS)
    spec = registered(argv, introspect=False)
    assert "discovered_agent" not in spec


def test_the_env_refusal_still_wins_over_the_new_path(registered, tmp_path, monkeypatch):
    monkeypatch.setenv("PERCEPTEYE_INTROSPECT", "0")
    argv = _fake_cli(tmp_path, WRITES_THEN_SLEEPS)
    spec = registered(argv)
    assert "discovered_agent" not in spec


def test_the_warm_up_does_not_run_for_a_python_agent(registered, tmp_path, monkeypatch):
    """Reflection already answered; spawning anything would be wasted work."""
    calls = []
    monkeypatch.setattr(
        fw, "warm_up_command", lambda *a, **k: calls.append(a) or None
    )
    spec = registered("tests.fixtures_agent:build", isolation="inprocess")
    assert calls == [], "warm-up ran for an agent reflection could already read"
    assert "discovered_agent" in spec


def test_the_warm_up_child_does_not_inherit_an_ambient_rollout_id(
    tmp_path, monkeypatch
):
    """The describe child is not a rollout, so it must not record as one.

    `build_child_env` scrubs the whole PERCEPTEYE_ namespace by inheritance;
    this path cannot, because the warm-up needs the ambient credentials to
    construct the agent at all. So the rollout id is dropped explicitly --
    otherwise a serving process's id reaches this child and every tool call it
    makes records under a rollout that never ran it.
    """
    monkeypatch.setenv("PERCEPTEYE_ROLLOUT_ID", "rl_from_the_parent")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_LONG_LIVED_CONTROL_PLANE_KEY")
    argv = _fake_cli(
        tmp_path,
        "import os\n"
        "open(os.environ['PERCEPTEYE_TRAJECTORY_DIR'] + '/env', 'w').write(\n"
        "    repr(sorted(k for k in os.environ if k.startswith('PERCEPTEYE_'))))\n",
    )
    out = tmp_path / "traj"
    warm_up_command(argv, trajectory_dir=out, timeout_s=30)
    seen = (out / "env").read_text()
    assert "PERCEPTEYE_ROLLOUT_ID" not in seen
    # and the long-lived control-plane key, which has no business in any child
    assert "PERCEPTEYE_API_KEY" not in seen
    # the one variable it does need survives, or the plugin stays inert
    assert "PERCEPTEYE_TRAJECTORY_DIR" in seen


def test_warm_up_scrubs_ambient_customer_route_configuration(tmp_path):
    """Core knows no customer route vocabulary; an adapter must opt values in."""
    argv = _fake_cli(
        tmp_path,
        "import os\n"
        "open(os.environ['PERCEPTEYE_TRAJECTORY_DIR'] + '/route-env', 'w').write(\n"
        "    repr({k: v for k, v in os.environ.items() if k.startswith('PERCEPTEYE_')}))\n",
    )
    out = tmp_path / "traj"
    warm_up_command(
        argv,
        trajectory_dir=out,
        timeout_s=30,
        observe_wire=False,
        env={
            "PATH": os.environ["PATH"],
            "PERCEPTEYE_MODEL_ID": "candidate",
            "PERCEPTEYE_CONTEXT_WINDOW": "32768",
            "PERCEPTEYE_MAX_TOKENS": "2048",
            "PERCEPTEYE_API_KEY": "must-not-cross",
            "PERCEPTEYE_AGENT_MODE": "training",
        },
    )
    seen = ast.literal_eval((out / "route-env").read_text())
    assert seen == {"PERCEPTEYE_TRAJECTORY_DIR": str(out)}


def test_warm_up_scrubs_control_plane_and_live_provider_credentials(tmp_path):
    """Discovery runs untrusted agent code but is not a rollout. It may see
    neither the durable control-plane key nor a route to any live model."""
    names = [
        "HOME",
        "USERPROFILE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "PRIME_AGENT_CODING_AGENT_DIR",
        "PRIME_AGENT_SESSION_DIR",
        "PRIME_AGENT_CODING_AGENT_SESSION_DIR",
        "PRIME_AGENT_KERNEL_VENV",
        "PRIME_AGENT_KERNEL_PYTHON",
        "PYTHONDONTWRITEBYTECODE",
        "CUSTOMER_AGENT_HOME",
        "CUSTOMER_AGENT_SECRET",
        "ARBITRARY_SECRET",
        "JIRA_TOKEN",
        "AWS_ACCESS_KEY_ID",
        "PERCEPTEYE_API_KEY",
        "GEMINI_API_KEY",
        "AWS_BEARER_TOKEN_BEDROCK",
        "OPENROUTER_API_KEY",
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_API_KEY",
    ]
    argv = _fake_cli(
        tmp_path,
        "import json, os\n"
        f"names = {names!r}\n"
        "open(os.environ['PERCEPTEYE_TRAJECTORY_DIR'] + '/credentials', 'w').write(\n"
        "    json.dumps({name: os.environ.get(name) for name in names}))\n",
    )
    out = tmp_path / "traj"
    warm_up_command(
        argv,
        trajectory_dir=out,
        timeout_s=30,
        observe_wire=False,
        endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"),
        env={
            "PATH": os.environ["PATH"],
            "HOME": "/isolated-root",
            "USERPROFILE": "/isolated-root",
            "XDG_CACHE_HOME": "/isolated-root/.cache",
            "XDG_CONFIG_HOME": "/isolated-root/.config",
            "XDG_DATA_HOME": "/isolated-root/.local/share",
            "XDG_STATE_HOME": "/isolated-root/.local/state",
            "PRIME_AGENT_CODING_AGENT_DIR": "/isolated-root/.prime/agent",
            "PRIME_AGENT_SESSION_DIR": "/isolated-root/.prime/agent/sessions",
            "PRIME_AGENT_CODING_AGENT_SESSION_DIR": (
                "/isolated-root/.prime/agent/sessions"
            ),
            "PRIME_AGENT_KERNEL_VENV": "/fixed/kernel",
            "PRIME_AGENT_KERNEL_PYTHON": "",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUSTOMER_AGENT_HOME": "/ambient-customer-home",
            "CUSTOMER_AGENT_SECRET": "must-not-cross",
            "ARBITRARY_SECRET": "must-not-cross",
            "JIRA_TOKEN": "must-not-cross",
            "AWS_ACCESS_KEY_ID": "must-not-cross",
            "PERCEPTEYE_API_KEY": "long-lived-control-plane-key",
            "GEMINI_API_KEY": "live-gemini-key",
            "AWS_BEARER_TOKEN_BEDROCK": "live-bedrock-key",
            "OPENROUTER_API_KEY": "live-openrouter-key",
            "OPENAI_BASE_URL": "https://live-provider.invalid/v1",
            "OPENAI_API_KEY": "live-openai-key",
            "DEEPSEEK_BASE_URL": "https://live-deepseek.invalid/v1",
            "DEEPSEEK_API_KEY": "live-deepseek-key",
        },
        isolation_environment={"CUSTOMER_AGENT_HOME": "/isolated-customer-home"},
    )
    seen = json.loads((out / "credentials").read_text())
    assert seen["HOME"] == "/isolated-root"
    assert seen["USERPROFILE"] == "/isolated-root"
    assert seen["XDG_CACHE_HOME"] == "/isolated-root/.cache"
    assert seen["XDG_CONFIG_HOME"] == "/isolated-root/.config"
    assert seen["XDG_DATA_HOME"] == "/isolated-root/.local/share"
    assert seen["XDG_STATE_HOME"] == "/isolated-root/.local/state"
    assert seen["PRIME_AGENT_CODING_AGENT_DIR"] is None
    assert seen["PRIME_AGENT_SESSION_DIR"] is None
    assert seen["PRIME_AGENT_CODING_AGENT_SESSION_DIR"] is None
    assert seen["PRIME_AGENT_KERNEL_VENV"] is None
    assert seen["PRIME_AGENT_KERNEL_PYTHON"] is None
    assert seen["PYTHONDONTWRITEBYTECODE"] == "1"
    assert seen["CUSTOMER_AGENT_HOME"] == "/isolated-customer-home"
    assert seen["CUSTOMER_AGENT_SECRET"] is None
    assert seen["ARBITRARY_SECRET"] is None
    assert seen["JIRA_TOKEN"] is None
    assert seen["AWS_ACCESS_KEY_ID"] is None
    assert seen["PERCEPTEYE_API_KEY"] is None
    assert seen["GEMINI_API_KEY"] is None
    assert seen["AWS_BEARER_TOKEN_BEDROCK"] is None
    assert seen["OPENROUTER_API_KEY"] is None
    assert seen["OPENAI_BASE_URL"] == "http://127.0.0.1:1/v1"
    assert seen["OPENAI_API_KEY"] == "percepteye-warmup-no-sampling"
    assert seen["DEEPSEEK_BASE_URL"] == "http://127.0.0.1:1/v1"
    assert seen["DEEPSEEK_API_KEY"] == "percepteye-warmup-no-sampling"
