"""Reading an agent's description from the HOST, when reflection cannot reach it.

WHY THIS MODULE EXISTS
----------------------
``introspection.py`` describes an agent by reflecting on a constructed Python
object. That is the right instrument for an in-process Python agent and it is
structurally incapable of describing a CLI one: ``fw.serve()`` given a command
entrypoint spawns an OS subprocess, so there is no object to reflect on.
``runner.resolved`` is ``None``, ``introspect(None)`` returns ``None``, and the
control plane records ``introspection_state = "unreadable"``.

That state is not cosmetic. It is the first domino: no tool catalogue means no
generated workflows, so there are no evaluation suites to write against them,
so there are no graded rollouts, so there is no dataset and no training run.
Every phase of the flywheel is downstream of it.

The answer is not to reflect harder. It is to let the host describe the agent,
because the host is the only party that can: it assembled the system prompt and
it built the tool list it sent to the model. The companion plugin observes one
``llm_input`` event and writes ``discovered_agent.json``; this module reads it.
The two are one ABI, exactly as the plugin's trajectory writer and
``outcomes.py`` are the two ends of the tool-outcome ABI. A file, not a
function call, so anything that can write JSON can describe an agent.

THE BOOTSTRAP PROBLEM, AND WHY THE WARM-UP IS NOT OPTIONAL
----------------------------------------------------------
The host only emits ``llm_input`` when the agent actually runs. Under
``fw.serve()`` the CLI is spawned ONLY to execute a claimed rollout, and
rollouts exist only after a run launches. But the control plane refuses to
launch a run whose product has no API spec, docs or environment URL and whose
agent is not ``described`` -- and the refusal explains itself: it will not infer
tasks from a system prompt alone, because every workflow would name tools
nobody confirmed exist.

So the gate and the only producer of the fact it gates on are mutually
blocking::

    no launch -> no rollout -> no CLI spawn -> no llm_input
              -> never described -> no launch

A CLI agent with no API spec can never leave that loop by running rollouts.
The way out is to run the agent ONCE at ``serve()`` time, before registering,
so the description exists before anything needs it. The saving grace is that
``llm_input`` fires BEFORE the model request is dispatched: the warm-up needs
no working completion, no gateway, and no API key. We spawn, wait for the file,
and kill the child. Typically well under a second.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from .introspection import _MAX_PROMPT_CHARS, DiscoveredAgent, ModelConfig, ToolDefinition

#: The file the companion plugin writes. One end of an ABI -- do not rename
#: without renaming ``DISCOVERED_AGENT_FILENAME`` on the plugin side.
DISCOVERED_AGENT_FILENAME = "discovered_agent.json"

#: What the warm-up child is asked to do. Deliberately trivial and read-only:
#: it exists to make the host assemble a prompt and a tool list, and the child
#: is killed the moment it has. It is never graded and never trained on.
WARMUP_PROMPT = "Reply with the single word: ready"

#: How long to wait for the host to emit its first ``llm_input``. Generous
#: because a cold CLI loads plugins, skills and a large prompt from disk;
#: bounded because a hang here would block every ``serve()`` forever.
WARMUP_TIMEOUT_S = 90.0

_POLL_INTERVAL_S = 0.1

#: How long to let the child keep running after a description first appears.
#: The first write is a LOWER BOUND on the catalogue -- a host or integration
#: may still add to it -- so we wait for the child to finish rather than
#: killing on first sight, bounded so a long-running agent cannot stall
#: registration.
_SETTLE_S = 60.0

#: Where the companion plugin lives, in preference order. A host that
#: redirects its state directory -- which an isolated per-rollout run does --
#: stops discovering globally installed plugins, so the config that run uses
#: has to name a path explicitly. Both candidates are checked rather than one
#: guessed: the first is a source checkout (the editable install most
#: integrations use), the second is what the host's own installer produces.
_PLUGIN_DIRNAME = "percepteye-agent-flywheel"


def openclaw_plugin_path() -> str | None:
    """The directory of the companion plugin, or None if it is not present.

    Returned as a path rather than baked into a config file, so an integration
    never has to hardcode a developer's absolute path into a repository.
    """
    candidates = [
        # A source checkout: <repo>/src/percepteye_agent_flywheel/ -> <repo>/packages/<name>
        Path(__file__).resolve().parents[2] / "packages" / _PLUGIN_DIRNAME,
        Path.home() / ".openclaw" / "extensions" / _PLUGIN_DIRNAME,
    ]
    for c in candidates:
        if (c / "package.json").is_file():
            return str(c)
    return None


def _tool_from_wire(raw: Any) -> ToolDefinition | None:
    """One tool dict from the host, or None if it is not usable.

    A tool with no name is DROPPED rather than repaired, because the consumer
    that reads this catalogue drops it too. Keeping it here would only make our
    reported count differ from the count that survives -- the kind of quiet
    discrepancy that turns "7 tools" into "3 tools" further down the pipeline
    with nothing logged in between.
    """
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    if not name:
        return None
    schema = raw.get("input_schema")
    # THREE-VALUED, and only a real bool counts. A host that omits the key, or
    # sends null, or sends a string, has NOT declared anything -- coercing any
    # of those to False would silently assert "this tool writes", which is the
    # claim that keeps a tool out of the held-out pool. ``read_only`` is the
    # framework-neutral spelling; whoever produced it (MCP readOnlyHint, an
    # HTTP verb, a GraphQL op_kind) has already been resolved by then.
    ro = raw.get("read_only")
    return ToolDefinition(
        name=name,
        description=str(raw.get("description") or ""),
        input_schema=schema if isinstance(schema, dict) else {},
        original_format=str(raw.get("original_format") or "unknown"),
        read_only=ro if isinstance(ro, bool) else None,
    )


def parse_discovered_agent(payload: Any) -> DiscoveredAgent | None:
    """Build a ``DiscoveredAgent`` from the host's JSON, or None if unusable.

    Tolerant of missing optional fields and strict about the one distinction
    that carries meaning: ``tool_definitions`` is ``None`` when the host could
    not observe a catalogue and ``[]`` when it observed one that is empty. The
    consumer grades on that difference, so it must survive the parse rather
    than being flattened into a confident empty list.
    """
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("agent_name") or "").strip()
    framework = str(payload.get("framework") or "").strip()
    if not name or not framework:
        return None

    raw_tools = payload.get("tool_definitions")
    tools: list[ToolDefinition] | None
    if raw_tools is None:
        tools = None
    elif isinstance(raw_tools, list):
        tools = [t for t in (_tool_from_wire(r) for r in raw_tools) if t is not None]
    else:
        tools = None

    model = None
    raw_model = payload.get("discovered_model_config")
    if isinstance(raw_model, dict):
        model = ModelConfig(
            provider=str(raw_model.get("provider") or "unknown"),
            model_name=str(raw_model.get("model_name") or "unknown"),
        )

    prompt = payload.get("system_prompt")
    if isinstance(prompt, str) and prompt:
        # The same bound the reflection probes apply. A host-assembled prompt
        # can be very large -- hundreds of kilobytes, concatenated from many
        # files -- and the cap belongs in one place for both sources, or the
        # two paths would report different sizes for the same agent.
        prompt = prompt[:_MAX_PROMPT_CHARS]
    else:
        prompt = None

    incomplete = payload.get("incomplete")
    return DiscoveredAgent(
        agent_name=name,
        framework=framework,
        # Trusted from the host, not overwritten: it is a PROVENANCE claim, and
        # this description did not come from reflecting on a Python object.
        source_type=str(payload.get("source_type") or "sdk_runtime"),
        system_prompt=prompt,
        tool_definitions=tools,
        discovered_model_config=model,
        role=str(payload.get("role") or "standalone"),
        incomplete=[str(x) for x in incomplete] if isinstance(incomplete, list) else [],
    )


def read_discovered_agent(trajectory_dir: str | os.PathLike[str]) -> DiscoveredAgent | None:
    """Read ``discovered_agent.json`` from a trajectory directory.

    Absence is the normal case, not an error: a host with no plugin installed,
    or a plugin whose conversation hook the host refused, simply never writes
    one. A malformed file is also an absence -- the writer renames atomically,
    so a partial read means something else is wrong, and failing a whole
    ``serve()`` over a telemetry file would be the wrong trade.
    """
    path = Path(trajectory_dir) / DISCOVERED_AGENT_FILENAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return parse_discovered_agent(payload)


def warm_up_command(
    argv: list[str],
    *,
    trajectory_dir: str | os.PathLike[str],
    timeout_s: float = WARMUP_TIMEOUT_S,
    prompt: str = WARMUP_PROMPT,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
) -> DiscoveredAgent | None:
    """Run the agent once, only far enough for the host to describe it.

    Returns the description, or ``None`` if the child never produced one.

    The child is killed as soon as the file appears -- we want the host's
    assembly of prompt and tools, not the agent's answer. It is spawned with
    the AMBIENT environment and no gateway routing on purpose: there is no
    rollout here, so there is no per-rollout endpoint or single-use key to
    route to, and ``llm_input`` fires before the model request is dispatched
    anyway. A warm-up that cannot reach a provider still describes the agent.

    Nothing this child does is captured, graded, or trained on. It is not a
    rollout and is never reported as one.
    """
    argv = [str(a) for a in argv]
    if not argv:
        return None

    target = Path(trajectory_dir)
    target.mkdir(parents=True, exist_ok=True)
    path = target / DISCOVERED_AGENT_FILENAME
    # A description left by an earlier warm-up would be indistinguishable from
    # one this child produced, so a broken agent would keep reporting the last
    # good catalogue forever.
    try:
        path.unlink()
    except OSError:
        pass

    child_env = dict(os.environ if env is None else env)
    # The SAME namespace scrub `build_child_env` does, for the same two
    # reasons, and this path is not exempt from either.
    #
    # A rollout id inherited from the serving process would make every tool
    # call this child records land under a rollout that never ran it, against
    # the promise above that this "is not a rollout and is never reported as
    # one". And the long-lived control-plane key has no business in a child
    # process at all -- that key is what `build_child_env` keeps in the parent
    # so it never reaches code the model can influence, and a describe child
    # runs the customer's agent exactly as a rollout child does.
    #
    # An earlier version of this dropped only the rollout id, on the grounds
    # that the warm-up "needs the ambient credentials". It does not: the
    # docstring above says it needs "no working completion, no gateway, and no
    # API key", the credentials a CLI actually needs are the PROVIDER
    # variables, and those are overwritten with a loopback URL and a dummy key
    # a few lines below. The only PERCEPTEYE_ variable the child has ever read
    # is the trajectory directory, which is set immediately after this.
    for _name in [k for k in child_env if k.startswith("PERCEPTEYE_")]:
        del child_env[_name]
    child_env["PERCEPTEYE_TRAJECTORY_DIR"] = str(target)

    # A DELIBERATELY UNROUTABLE endpoint, forced rather than defaulted.
    #
    # Two things are true at once. Many agent CLIs refuse to start at all
    # without a configured provider, so leaving these unset means the warm-up
    # exits before the host ever assembles a prompt -- no description, and a
    # confusing one to debug. But inheriting the AMBIENT values would be
    # worse: this child is not a rollout, nothing captures it, and a real key
    # here would let it sample from a live provider unobserved -- the precise
    # thing the rollout path strips these variables to prevent.
    #
    # So they are overwritten with an address nothing can answer. The host
    # gets a provider it can configure, `llm_input` fires (it is emitted
    # BEFORE the request is dispatched, which is what makes this work at all),
    # and a dispatch that somehow outruns the kill fails closed against a
    # closed loopback port instead of reaching anyone. Loopback also avoids
    # tripping the private-network guards some hosts apply to LAN endpoints.
    unroutable = "http://127.0.0.1:1/v1"
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL"):
        child_env[name] = unroutable
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        child_env[name] = "percepteye-warmup-no-sampling"

    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=child_env, text=True, cwd=cwd,
            # Its own process group, for the same reason the rollout path uses
            # one: a CLI agent spawns freely, and killing only the process we
            # launched leaves children holding the stdout pipe open.
            start_new_session=(os.name != "nt"),
        )
    except (OSError, ValueError):
        # The binary could not be started. The caller reports this; it is a
        # configuration error, not a missing description.
        return None

    try:
        if proc.stdin is not None:
            try:
                proc.stdin.write(prompt)
                proc.stdin.close()
            except OSError:
                pass

        deadline = time.time() + timeout_s
        settle_deadline: float | None = None
        while time.time() < deadline:
            if proc.poll() is not None:
                # The child exited. It may have written -- or REWRITTEN -- the
                # file on its way out, so read after the exit, never before.
                return read_discovered_agent(target)
            if settle_deadline is None and path.exists():
                # THE FILE APPEARING IS NOT THE END OF THE STORY.
                #
                # Killing here was wrong, and measurably so: an integration
                # that ENRICHES the description after the agent exits -- with
                # a catalogue the host's own hook could not see -- never got
                # to run, and the warm-up returned 38 tools where the finished
                # description had 77. The first write is a lower bound.
                #
                # So once a description exists we stop being in a hurry and
                # let the child finish, but only for a bounded grace period:
                # an agent that would run for its full timeout should not hold
                # up registration, and we already have something usable.
                settle_deadline = time.time() + _SETTLE_S
            if settle_deadline is not None and time.time() >= settle_deadline:
                break
            time.sleep(_POLL_INTERVAL_S)
        return read_discovered_agent(target)
    finally:
        _kill(proc)


def _kill(proc: subprocess.Popen) -> None:
    """Kill the warm-up child and its group, and never raise doing it."""
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(os.getpgid(proc.pid), 9)
        else:
            proc.kill()
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.communicate(timeout=5)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        pass
