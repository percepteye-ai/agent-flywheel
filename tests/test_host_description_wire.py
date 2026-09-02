"""Describing a CLI agent from the completion request it sends.

The file-based path (``discovered_agent.json``) needs a host event that carries
a tool catalogue, and not every host has one -- goose has twelve hook events and
none of them names a tool list, so a hook-based describer for it is unwritable
rather than merely unwritten. The wire is the fallback: the ``tools`` field of
the completion request is where every routable host puts its catalogue.

Every agent here is a THROWAWAY SCRIPT, not a mock of the observer. The thing
under test is an ABI between two processes, and an in-process fake would pass
whether or not the socket, the streaming reply, or the env vars work.
"""
from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

from percepteye_agent_flywheel.host_description import (
    DISCOVERED_AGENT_FILENAME,
    framework_from_argv,
    warm_up_command,
)

# ── agents ──────────────────────────────────────────────────────────────────
#
# Each writes a Python file that talks to $OPENAI_BASE_URL with urllib only,
# so the test has no dependency the SDK itself does not have.

_PREAMBLE = """
import json, os, sys, urllib.request

BASE = os.environ["OPENAI_BASE_URL"].rstrip("/")

def post(body):
    req = urllib.request.Request(
        BASE + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read().decode()
"""


def _agent(tmp_path: Path, name: str, body: str) -> list[str]:
    path = tmp_path / f"{name}.py"
    path.write_text(_PREAMBLE + textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(path)]


def _openai_tool(name: str, desc: str = "d") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}},
        },
    }


# ── the OpenAI dialect ──────────────────────────────────────────────────────


def test_a_catalogue_on_the_wire_describes_an_agent_with_no_plugin(tmp_path):
    """The whole point: no file is written, and the agent is still described."""
    argv = _agent(tmp_path, "oa", f"""
        post({{
            "model": "gpt-4o-mini",
            "messages": [
                {{"role": "system", "content": "You are a filing agent."}},
                {{"role": "user", "content": "hi"}},
            ],
            "tools": {json.dumps([_openai_tool("read"), _openai_tool("write")])},
        }})
    """)

    described = warm_up_command(argv, trajectory_dir=tmp_path / "t",
                                agent_name="filer", timeout_s=30)

    assert not (tmp_path / "t" / DISCOVERED_AGENT_FILENAME).exists(), (
        "this agent writes no description file; the wire is the only source")
    assert described is not None
    assert [t.name for t in described.tool_definitions] == ["read", "write"]
    assert described.system_prompt == "You are a filing agent."
    assert described.agent_name == "filer"
    assert described.source_type == "host_wire", (
        "provenance must say the wire, not sdk_runtime -- nothing inside the "
        "agent's process described it")
    assert described.discovered_model_config.model_name == "gpt-4o-mini"


def test_the_wire_cannot_see_annotations_and_says_so(tmp_path):
    """``read_only`` is three-valued and the wire carries none of it.

    An OpenAI function declaration has no room for MCP's ``readOnlyHint``, so
    every tool here must report ``None`` -- not False. False is the claim that
    keeps a tool out of the held-out pool, and asserting it from a vantage
    point that cannot see it would be a fabrication.
    """
    argv = _agent(tmp_path, "anno", f"""
        post({{"model": "m", "messages": [], "tools": {json.dumps([_openai_tool("t")])}}})
    """)
    described = warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30)
    assert described is not None
    assert all(t.read_only is None for t in described.tool_definitions)
    assert "tool_annotations" in described.incomplete


# ── the Anthropic dialect ───────────────────────────────────────────────────


def test_the_anthropic_tool_shape_is_read_too(tmp_path):
    """Which dialect an agent speaks is its business, not a requirement."""
    tools = [{"name": "search", "description": "d",
              "input_schema": {"type": "object", "properties": {"q": {"type": "string"}}}}]
    argv = _agent(tmp_path, "an", f"""
        post({{
            "model": "claude-x",
            "system": "Anthropic-shaped.",
            "messages": [{{"role": "user", "content": "hi"}}],
            "tools": {json.dumps(tools)},
        }})
    """)
    described = warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30)
    assert described is not None
    assert [t.name for t in described.tool_definitions] == ["search"]
    assert described.tool_definitions[0].original_format == "anthropic_tool"
    assert described.tool_definitions[0].input_schema["properties"] == {"q": {"type": "string"}}
    assert described.system_prompt == "Anthropic-shaped."


# ── which request the fields come from ──────────────────────────────────────


def test_an_auxiliary_completion_does_not_become_the_description(tmp_path):
    """THE regression, measured against a live goose.

    goose opens a run by asking the model to name the session -- a toolless
    completion, with its own 1.4 KB "generate a short title" system prompt and
    potentially its own fast model, dispatched BEFORE the agent's first turn.
    Taking request #1 described the session-namer: the wrong prompt, and zero
    tools for an agent that has five.
    """
    argv = _agent(tmp_path, "aux", f"""
        post({{"model": "fast-titler", "messages": [
            {{"role": "system", "content": "Generate a short title."}}]}})
        post({{"model": "the-real-policy", "messages": [
            {{"role": "system", "content": "You are the agent."}}],
            "tools": {json.dumps([_openai_tool("shell")])}}})
    """)
    described = warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30)
    assert described is not None
    assert described.system_prompt == "You are the agent."
    assert described.discovered_model_config.model_name == "the-real-policy"


def test_a_catalogue_grown_mid_run_is_kept_whole(tmp_path):
    """The first request is a LOWER BOUND, exactly as the first file write is.

    An agent may enable an extension part-way through a turn. Keeping only the
    first catalogue reports a subset with nothing logged to say so.
    """
    argv = _agent(tmp_path, "grow", f"""
        post({{"model": "m", "messages": [], "tools": {json.dumps([_openai_tool("a")])}}})
        post({{"model": "m", "messages": [],
               "tools": {json.dumps([_openai_tool("a"), _openai_tool("b")])}}})
    """)
    described = warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30)
    assert described is not None
    assert [t.name for t in described.tool_definitions] == ["a", "b"]


# ── absence, and what it must not be rounded to ─────────────────────────────


def test_an_agent_that_declares_nothing_is_not_described_as_toolless(tmp_path):
    """``None`` and "an agent with no tools" are different answers.

    From the wire we cannot tell an agent that has no tools from one that died
    before its first turn. Returning ``[]`` would assert the difference we
    cannot see, and the launch gate grades on it.
    """
    argv = _agent(tmp_path, "silent", """
        post({"model": "m", "messages": [{"role": "user", "content": "hi"}]})
    """)
    assert warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30) is None


def test_an_agent_that_never_dials_is_not_described(tmp_path):
    argv = _agent(tmp_path, "mute", """
        pass
    """)
    assert warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30) is None


# ── precedence and the safety properties ────────────────────────────────────


def test_the_host_written_file_beats_the_wire(tmp_path):
    """A host that describes itself knows things the wire drops.

    Framework, sub-agents and read-only annotations exist only in the file, so
    an agent that writes one must never be downgraded to a wire description
    just because it also sent a request.
    """
    traj = tmp_path / "t"
    payload = {
        "agent_name": "from-the-file", "framework": "openclaw",
        "tool_definitions": [{"name": "only_in_file", "read_only": True}],
    }
    argv = _agent(tmp_path, "both", f"""
        import pathlib
        d = pathlib.Path(os.environ["PERCEPTEYE_TRAJECTORY_DIR"])
        d.mkdir(parents=True, exist_ok=True)
        (d / {DISCOVERED_AGENT_FILENAME!r}).write_text({json.dumps(payload)!r})
        post({{"model": "m", "messages": [],
               "tools": {json.dumps([_openai_tool("only_on_wire")])}}})
    """)
    described = warm_up_command(argv, trajectory_dir=traj, timeout_s=30)
    assert described is not None
    assert described.framework == "openclaw"
    assert [t.name for t in described.tool_definitions] == ["only_in_file"]
    assert described.tool_definitions[0].read_only is True


def test_the_observer_never_answers_with_a_tool_call(tmp_path):
    """Observing must not be able to make the agent ACT.

    The endpoint answers -- a refused request is a catalogue we never see --
    but the reply carries text and nothing else, so the agent finishes its turn
    without touching the machine. This asserts the answer the agent actually
    receives, not our intent about it.
    """
    seen = tmp_path / "reply.json"
    argv = _agent(tmp_path, "reply", f"""
        import pathlib
        raw = post({{"model": "m", "messages": [],
                     "tools": {json.dumps([_openai_tool("shell")])}}})
        pathlib.Path({str(seen)!r}).write_text(raw)
    """)
    warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30)
    body = json.loads(seen.read_text())
    message = body["choices"][0]["message"]
    assert message.get("tool_calls") in (None, []), body
    assert body["choices"][0]["finish_reason"] == "stop"


def test_the_child_never_inherits_a_real_provider_credential(tmp_path):
    """The warm-up is not a rollout, so it must not be able to sample one.

    Ambient provider variables are overwritten, not merely supplemented: an
    inherited key here would let an unwatched child call a live provider.
    """
    out = tmp_path / "env.json"
    argv = _agent(tmp_path, "env", f"""
        import pathlib
        pathlib.Path({str(out)!r}).write_text(json.dumps({{
            k: os.environ.get(k) for k in
            ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL",
             "ANTHROPIC_BASE_URL", "PERCEPTEYE_API_KEY", "PERCEPTEYE_ROLLOUT_ID")
        }}))
    """)
    warm_up_command(
        argv, trajectory_dir=tmp_path / "t", timeout_s=30,
        env={**{"PATH": "/usr/bin:/bin"},
             "OPENAI_API_KEY": "sk-live-real",
             "ANTHROPIC_API_KEY": "sk-ant-live-real",
             "PERCEPTEYE_API_KEY": "perk_longlived",
             "PERCEPTEYE_ROLLOUT_ID": "rollout-that-never-ran"},
    )
    env = json.loads(out.read_text())
    assert env["OPENAI_API_KEY"] == "percepteye-warmup-no-sampling"
    assert env["ANTHROPIC_API_KEY"] == "percepteye-warmup-no-sampling"
    assert env["OPENAI_BASE_URL"].startswith("http://127.0.0.1:")
    assert env["ANTHROPIC_BASE_URL"] == env["OPENAI_BASE_URL"]
    assert env["PERCEPTEYE_API_KEY"] is None
    assert env["PERCEPTEYE_ROLLOUT_ID"] is None


def test_observe_wire_off_restores_the_closed_port(tmp_path):
    """The opt-out is a real one: nothing listens, and nothing is described."""
    out = tmp_path / "base.txt"
    argv = _agent(tmp_path, "off", f"""
        import pathlib
        pathlib.Path({str(out)!r}).write_text(BASE)
    """)
    described = warm_up_command(argv, trajectory_dir=tmp_path / "t",
                                timeout_s=30, observe_wire=False)
    assert described is None
    assert out.read_text() == "http://127.0.0.1:1/v1"


# ── naming ──────────────────────────────────────────────────────────────────


def test_the_framework_is_the_binary_not_a_guess():
    assert framework_from_argv(["/usr/local/bin/goose", "run"]) == "goose"
    assert framework_from_argv(["C:/x/openclaw.EXE"]) == "openclaw"
    assert framework_from_argv([]) == "unknown"


def test_the_agent_name_defaults_to_the_binary(tmp_path):
    """No ``agent_name=``: fall back to the binary rather than inventing one."""
    argv = _agent(tmp_path, "unnamed", f"""
        post({{"model": "m", "messages": [], "tools": {json.dumps([_openai_tool("t")])}}})
    """)
    described = warm_up_command(argv, trajectory_dir=tmp_path / "t", timeout_s=30)
    assert described is not None
    assert described.agent_name == framework_from_argv(argv)
    assert described.framework == framework_from_argv(argv)
