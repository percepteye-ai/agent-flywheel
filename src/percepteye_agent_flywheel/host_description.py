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
it built the tool list it sent to the model.

TWO SOURCES, IN PREFERENCE ORDER.

1. ``discovered_agent.json``, written by the companion plugin from one
   ``llm_input`` event. A file, not a function call, so anything that can write
   JSON can describe an agent -- one ABI, exactly as the plugin's trajectory
   writer and ``outcomes.py`` are the two ends of the tool-outcome ABI. This
   is the richer source and it always wins: only the host knows its own
   framework, its sub-agents, and its tools' read-only annotations.

2. THE WIRE, when no plugin can write that file. Some hosts have no event that
   carries a catalogue at all -- goose has twelve hook events and not one of
   them names a tool list, so a hook-based describer for goose is unwritable
   rather than merely unwritten. What every routable host does have is the
   ``tools`` field of the completion request. The warm-up already stands a
   loopback endpoint in front of the child; having it READ what arrives turns
   "we could not describe this agent" into "we could not route this agent",
   which is a far smaller set. See ``_CatalogueObserver``.

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
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .errors import ConfigurationError
from .execution_identity import canonical_observation_sha256
from .introspection import _MAX_PROMPT_CHARS, DiscoveredAgent, ModelConfig, ToolDefinition
from .isolation import build_child_env

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


def _parse_discovered_agent(
    payload: Any, depth: int,
) -> tuple[DiscoveredAgent, dict[str, Any]] | None:
    """Parse one agent and retain an exact behavior-bearing observation.

    Unknown host telemetry is intentionally not identity: adding a timestamp
    or debug field cannot change the execution digest. The normalized private
    observation does retain the full prompt (before the 20k wire projection),
    tools, model config, capabilities, and recursively parsed sub-agents.
    """
    if not isinstance(payload, dict):
        return None
    name = str(payload.get("agent_name") or "").strip()
    framework = str(payload.get("framework") or "").strip()
    if not name or not framework:
        return None

    raw_tools = payload.get("tool_definitions")
    tool_observation_incomplete = False
    tools: list[ToolDefinition] | None
    if raw_tools is None:
        tools = None
    elif isinstance(raw_tools, list):
        tools = []
        for raw_tool in raw_tools:
            tool = _tool_from_wire(raw_tool)
            if tool is None:
                tool_observation_incomplete = True
            else:
                tools.append(tool)
    else:
        tools = None

    model = None
    raw_model = payload.get("discovered_model_config")
    model_observation_incomplete = False
    if isinstance(raw_model, dict):
        raw_provider = raw_model.get("provider")
        raw_model_name = raw_model.get("model_name")
        raw_temperature = raw_model.get("temperature")
        temperature = (
            float(raw_temperature)
            if isinstance(raw_temperature, (int, float))
            and not isinstance(raw_temperature, bool)
            else None
        )
        raw_max_tokens = raw_model.get("max_tokens")
        max_tokens = (
            raw_max_tokens
            if isinstance(raw_max_tokens, int) and not isinstance(raw_max_tokens, bool)
            else None
        )
        model = ModelConfig(
            provider=str(raw_provider or "unknown"),
            model_name=str(raw_model_name or "unknown"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
        model_observation_incomplete = (
            not isinstance(raw_provider, str)
            or not raw_provider.strip()
            or not isinstance(raw_model_name, str)
            or not raw_model_name.strip()
            or (
                "temperature" in raw_model
                and not (
                    isinstance(raw_temperature, (int, float))
                    and not isinstance(raw_temperature, bool)
                )
            )
            or (
                "max_tokens" in raw_model
                and not (
                    isinstance(raw_max_tokens, int)
                    and not isinstance(raw_max_tokens, bool)
                )
            )
        )

    full_prompt = payload.get("system_prompt")
    if isinstance(full_prompt, str) and full_prompt:
        # The same bound the reflection probes apply. A host-assembled prompt
        # can be very large -- hundreds of kilobytes, concatenated from many
        # files -- and the cap belongs in one place for both sources, or the
        # two paths would report different sizes for the same agent.
        prompt = full_prompt[:_MAX_PROMPT_CHARS]
    else:
        full_prompt = None
        prompt = None

    incomplete = payload.get("incomplete")
    incomplete_values = (
        [str(x) for x in incomplete] if isinstance(incomplete, list) else []
    )
    if incomplete is not None and not isinstance(incomplete, list):
        incomplete_values.append("incomplete")
    if tool_observation_incomplete and "tool" not in incomplete_values:
        incomplete_values.append("tool")
    if (
        model_observation_incomplete
        and "discovered_model_config" not in incomplete_values
    ):
        incomplete_values.append("discovered_model_config")
    raw_capabilities = payload.get("capabilities")
    capabilities = (
        [str(value) for value in raw_capabilities]
        if isinstance(raw_capabilities, list) else []
    )
    if (
        raw_capabilities is not None
        and not isinstance(raw_capabilities, list)
        and "capabilities" not in incomplete_values
    ):
        incomplete_values.append("capabilities")
    children: list[DiscoveredAgent] = []
    exact_children: list[dict[str, Any]] = []
    raw_children = payload.get("sub_agents")
    child_observation_incomplete = False
    if isinstance(raw_children, list):
        if depth < 4:
            for raw_child in raw_children:
                parsed = _parse_discovered_agent(raw_child, depth + 1)
                if parsed is None:
                    child_observation_incomplete = True
                    continue
                child, exact_child = parsed
                children.append(child)
                exact_children.append(exact_child)
        elif raw_children:
            child_observation_incomplete = True
    elif raw_children is not None:
        child_observation_incomplete = True
    if child_observation_incomplete and "sub_agents" not in incomplete_values:
        incomplete_values.append("sub_agents")

    source_type = str(payload.get("source_type") or "sdk_runtime")
    role = str(payload.get("role") or "standalone")
    exact_projection: dict[str, Any] = {
        "agent_name": name,
        "framework": framework,
        "source_type": source_type,
        "role": role,
        "tool_definitions": (
            None if tools is None else [tool.to_wire() for tool in tools]
        ),
    }
    if full_prompt is not None:
        exact_projection["system_prompt"] = full_prompt
    if model is not None:
        exact_projection["discovered_model_config"] = model.to_wire()
    if capabilities:
        exact_projection["capabilities"] = capabilities
    if exact_children:
        exact_projection["sub_agents"] = exact_children
    if incomplete_values:
        exact_projection["incomplete"] = incomplete_values

    try:
        exact_observation = canonical_observation_sha256(exact_projection)
    except ConfigurationError:
        # Discovery remains advisory. The projected catalogue can still be
        # registered, but an ambiguous value cannot support exact identity.
        exact_observation = None
    agent = DiscoveredAgent(
        agent_name=name,
        framework=framework,
        # Trusted from the host, not overwritten: it is a PROVENANCE claim, and
        # this description did not come from reflecting on a Python object.
        source_type=source_type,
        system_prompt=prompt,
        tool_definitions=tools,
        discovered_model_config=model,
        capabilities=capabilities,
        sub_agents=children,
        role=role,
        incomplete=incomplete_values,
        exact_observation_sha256=exact_observation,
    )
    return agent, exact_projection


def parse_discovered_agent(payload: Any) -> DiscoveredAgent | None:
    """Build a ``DiscoveredAgent`` from the host's JSON, or None if unusable.

    Tolerant of missing optional fields and strict about the one distinction
    that carries meaning: ``tool_definitions`` is ``None`` when the host could
    not observe a catalogue and ``[]`` when it observed one that is empty. The
    consumer grades on that difference, so it must survive the parse rather
    than being flattened into a confident empty list.
    """
    parsed = _parse_discovered_agent(payload, 0)
    return parsed[0] if parsed is not None else None


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


# ── the second source: the catalogue on its way to the model ────────────────
#
# ``discovered_agent.json`` needs a plugin, and a plugin needs a host event
# that carries the catalogue. Not every host has one. goose, for instance, has
# twelve hook events and NONE of them carries a tool list -- its ``HookContext``
# is (event, session_id, tool_name, tool_input, tool_output, ...), so a
# hook-based describer for goose is not unwritten, it is unwritable.
#
# But every host that can be routed at all puts its catalogue in exactly one
# place: the ``tools`` field of the completion request. That is the same fact
# ``describe.js`` reads from OpenClaw's ``llm_input`` -- "the real tool list AS
# SENT TO THE MODEL" -- observed one layer lower, where being lower is what
# makes it host-agnostic. If we can route an agent, we can describe it.
#
# THIS ENDPOINT ANSWERS, WHERE THE OLD ONE REFUSED. The warm-up used to point
# the child at ``127.0.0.1:1`` so that a dispatch outrunning the kill would
# fail closed. It still cannot reach a provider -- this socket is ours, on
# loopback, and forwards nothing -- but it now replies, because a request that
# is refused is a catalogue we never see. Two properties keep that safe:
#
#   * the reply carries NO tool calls, ever, so the agent has nothing to act
#     on and finishes its turn without touching the machine; and
#   * nothing said here is captured, graded or trained on -- this is not a
#     rollout, and ``_describe_via_host`` reports it as none.

#: What the observer answers with. Deliberately the warm-up's own expected
#: reply: short, inert, and recognisable in a log as ours rather than a model's.
_OBSERVER_REPLY = "ready"

#: Stop answering after this many completions. The deadline and the kill are
#: the real bounds; this is the cheap one that keeps a confused agent from
#: spinning against us for the whole timeout.
_OBSERVER_MAX_REQUESTS = 12


def _tools_from_request(body: Any) -> list[dict[str, Any]]:
    """The tool declarations in one completion request, in wire order.

    Both dialects, because which one an agent speaks is its business and not
    a fact we get to require: OpenAI nests the declaration under ``function``,
    Anthropic puts it flat with ``input_schema``. Anything else yields nothing
    rather than a guess.
    """
    if not isinstance(body, dict):
        return []
    raw = body.get("tools")
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        fn = entry.get("function")
        if isinstance(fn, dict):                       # OpenAI
            out.append({
                "name": fn.get("name"),
                "description": fn.get("description"),
                "input_schema": fn.get("parameters"),
                "original_format": "openai_function",
            })
        elif entry.get("name"):                        # Anthropic
            out.append({
                "name": entry.get("name"),
                "description": entry.get("description"),
                "input_schema": entry.get("input_schema"),
                "original_format": "anthropic_tool",
            })
    return out


def _system_prompt_from_request(body: Any) -> str | None:
    """The assembled system prompt in one completion request, or None."""
    if not isinstance(body, dict):
        return None
    top = body.get("system")                           # Anthropic
    if isinstance(top, str) and top.strip():
        return top
    if isinstance(top, list):                          # Anthropic blocks
        joined = "\n".join(
            b.get("text", "") for b in top
            if isinstance(b, dict) and isinstance(b.get("text"), str)
        ).strip()
        if joined:
            return joined
    for msg in body.get("messages") or []:             # OpenAI
        if not isinstance(msg, dict):
            continue
        if msg.get("role") in ("system", "developer"):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content
            if isinstance(content, list):
                joined = "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and isinstance(b.get("text"), str)
                ).strip()
                if joined:
                    return joined
    return None


class _CatalogueObserver:
    """A loopback completion endpoint that records what an agent declares.

    Accumulates the UNION of every catalogue it sees, keyed by tool name and
    in first-seen order, rather than keeping the first request's. An agent may
    enable an extension part-way through a turn, and the file-based path
    already learned this lesson the expensive way: the first write is a lower
    bound, not the answer (see ``_SETTLE_S``).

    The first request is very often NOT the agent's turn. goose opens a run by
    asking the model to name the session -- a toolless auxiliary completion
    that arrives BEFORE the agent's own first request -- so a describer that
    takes request #1 and stops reports zero tools for an agent that has five.
    """

    def __init__(self) -> None:
        # LAZY, and load-bearing. ``http.server`` pulls in ``socketserver`` and
        # with it ``ssl``, and ``test_importing_the_package_does_not_import_
        # httpx`` asserts that neither reaches a process that merely imports
        # this package. A describer used once per serve() must not cost every
        # importer an ssl context.
        import http.server

        self._lock = threading.Lock()
        self._tools: dict[str, dict[str, Any]] = {}
        self._system: str | None = None
        self._model: str | None = None
        self._requests = 0
        observer = self

        class _Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                return              # never write to the serving process's stderr

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except OSError:
                    pass            # the child was killed mid-write

            def do_GET(self) -> None:
                # Model-list probes. Some CLIs refuse to start against an
                # endpoint that cannot enumerate models.
                self._send(200, b'{"object":"list","data":[]}', "application/json")

            def do_POST(self) -> None:
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                    raw = self.rfile.read(length) if length else b""
                    body = json.loads(raw or b"{}")
                except (OSError, ValueError):
                    body = {}
                over_budget = observer._record(body)
                if over_budget:
                    self._send(429, b'{"error":{"message":"percepteye describe probe"}}',
                               "application/json")
                    return
                if body.get("stream"):
                    self._send(200, _sse_reply(), "text/event-stream")
                else:
                    self._send(200, _json_reply(), "application/json")

        self._server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _record(self, body: Any) -> bool:
        """Fold one request in. Returns True once the request budget is spent.

        THE PROMPT AND THE MODEL ARE TAKEN FROM A TOOL-BEARING REQUEST ONLY.
        An auxiliary completion is still a completion: goose's session-namer
        arrives first, carries its own 1.4 KB "generate a short title" prompt,
        and may run on a different model entirely (``complete_fast``). Keying
        these to "the first request seen" describes the namer, not the agent --
        measured against goose 1.48.0: 1444 chars of the namer's prompt in
        place of the agent's own 2185. Tools are what distinguish the agent's
        turn, so tools are what gate the fields that must belong to it.
        """
        with self._lock:
            self._requests += 1
            spent = self._requests > _OBSERVER_MAX_REQUESTS
            tools = _tools_from_request(body)
            if not tools:
                return spent
            for tool in tools:
                name = str(tool.get("name") or "").strip()
                if name and name not in self._tools:
                    self._tools[name] = tool
            if self._system is None:
                self._system = _system_prompt_from_request(body)
            if self._model is None and isinstance(body, dict):
                model = body.get("model")
                if isinstance(model, str) and model.strip():
                    self._model = model
            return spent

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def described(self, *, agent_name: str, framework: str) -> DiscoveredAgent | None:
        """What was observed, or None if no request ever declared a tool.

        ``None`` and "an agent with no tools" are NOT the same answer and this
        returns None for both on purpose: an agent that declared nothing may
        genuinely have nothing, or may have died before its first turn, and
        this vantage point cannot tell those apart. Claiming ``[]`` here would
        assert the difference we cannot see, and the consumer grades on it.
        """
        with self._lock:
            if not self._tools:
                return None
            raw_tools = list(self._tools.values())
            tools = [_tool_from_wire(t) for t in raw_tools]
            prompt = self._system
            model = self._model
        normalized_tools = [tool.to_wire() for tool in tools if tool is not None]
        try:
            exact_observation = canonical_observation_sha256({
                "agent_name": agent_name,
                "framework": framework,
                "source_type": "host_wire",
                "system_prompt": prompt,
                "tool_definitions": normalized_tools,
                "discovered_model_config": {
                    "provider": "openai_compatible",
                    "model_name": model,
                } if model else None,
            })
        except ConfigurationError:
            exact_observation = None
        return DiscoveredAgent(
            agent_name=agent_name,
            framework=framework,
            #: A PROVENANCE claim, and not ``sdk_runtime``: nothing inside the
            #: agent's process described it. We read what it put on the wire.
            source_type="host_wire",
            system_prompt=prompt[:_MAX_PROMPT_CHARS] if prompt else None,
            tool_definitions=[t for t in tools if t is not None],
            discovered_model_config=(
                ModelConfig(provider="openai_compatible", model_name=model)
                if model else None),
            #: The wire carries a tool's name, description and schema -- and
            #: DROPS its annotations. MCP's ``readOnlyHint`` never survives
            #: translation into an OpenAI function, so every tool here reports
            #: ``read_only=None``: not observed. Saying so is the whole point;
            #: the consumer treats an unstated read_only as "assume it writes",
            #: which is the safe direction, and a False here would have
            #: asserted it.
            incomplete=["tool_annotations"],
            exact_observation_sha256=exact_observation,
        )

    def close(self) -> None:
        try:
            self._server.shutdown()
        except Exception:
            pass
        try:
            self._server.server_close()
        except Exception:
            pass


def _sse_reply() -> bytes:
    """One complete streaming completion carrying text and NO tool calls."""
    chunks = [
        {"id": "percepteye-describe", "object": "chat.completion.chunk",
         "created": 0, "model": "percepteye-describe",
         "choices": [{"index": 0, "finish_reason": None,
                      "delta": {"role": "assistant", "content": _OBSERVER_REPLY}}]},
        {"id": "percepteye-describe", "object": "chat.completion.chunk",
         "created": 0, "model": "percepteye-describe",
         "choices": [{"index": 0, "finish_reason": "stop", "delta": {}}],
         "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return body.encode("utf-8")


def _json_reply() -> bytes:
    """The non-streaming form of the same inert answer."""
    return json.dumps({
        "id": "percepteye-describe", "object": "chat.completion", "created": 0,
        "model": "percepteye-describe",
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": _OBSERVER_REPLY}}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }).encode("utf-8")


def framework_from_argv(argv: list[str]) -> str:
    """A framework name for a command agent: the binary it runs.

    The host's own plugin states its framework; the wire does not carry one,
    and the executable's name is the only thing here that is a fact rather
    than an inference. ``unknown`` when even that is unavailable, never a
    guess dressed as an observation.
    """
    if not argv:
        return "unknown"
    name = os.path.basename(str(argv[0])).strip()
    for suffix in (".exe", ".cmd", ".bat"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return name or "unknown"


def warm_up_command(
    argv: list[str],
    *,
    trajectory_dir: str | os.PathLike[str],
    timeout_s: float = WARMUP_TIMEOUT_S,
    prompt: str = WARMUP_PROMPT,
    env: dict[str, str] | None = None,
    cwd: str | None = None,
    agent_name: str | None = None,
    observe_wire: bool = True,
    endpoint_env: tuple[str, str] | None = None,
    isolation_environment: Mapping[str, str] | None = None,
) -> DiscoveredAgent | None:
    """Run the agent once, only far enough for the host to describe it.

    Returns the description, or ``None`` if the child never produced one.

    The child is killed as soon as the file appears -- we want the host's
    assembly of prompt and tools, not the agent's answer. It is spawned through
    the same credential scrub as a rollout child, with every supported model
    route forced to a loopback observer and an inert key. There is no rollout
    here, so there is no per-rollout credential to use, and no request is ever
    forwarded to a provider. A warm-up that cannot reach a live provider still
    describes the agent.

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

    # A DELIBERATELY UNREACHABLE PROVIDER, forced rather than defaulted.
    #
    # Two things are true at once. Many agent CLIs refuse to start at all
    # without a configured provider, so leaving these unset means the warm-up
    # exits before the host ever assembles a prompt -- no description, and a
    # confusing one to debug. But inheriting the AMBIENT values would be
    # worse: this child is not a rollout, nothing captures it, and a real key
    # here would let it sample from a live provider unobserved -- the precise
    # thing the rollout path strips these variables to prevent.
    #
    # So they are overwritten with an address that is ours. Either it is a
    # closed loopback port (``observe_wire=False``) or it is the observer
    # above, which answers on loopback and forwards nothing. In both cases no
    # completion can reach a provider; the difference is only whether we get
    # to READ the catalogue on its way past. Loopback also avoids tripping the
    # private-network guards some hosts apply to LAN endpoints.
    observer = _CatalogueObserver() if observe_wire else None
    endpoint = observer.base_url if observer is not None else "http://127.0.0.1:1/v1"
    # Use the rollout environment boundary itself, rather than maintaining a
    # nearly-identical warm-up scrub. This removes the long-lived Percepteye
    # credential and every known off-gateway provider credential, preserves
    # only explicit adapter-owned layout fields, and overwrites all supported
    # model routes with our loopback observer and inert key. The temporary
    # rollout id is removed immediately: discovery is not a rollout and must
    # never attribute its tool calls to one.
    parent_environment = dict(os.environ if env is None else env)
    parent_environment.update(isolation_environment or {})
    child_env = build_child_env(
        rollout_id="percepteye-discovery-not-a-rollout",
        llm_url=endpoint,
        api_key="percepteye-warmup-no-sampling",
        trajectory_dir=str(target),
        parent_env=parent_environment,
        # Discovery assembles a catalogue; it does not execute a customer's
        # task and has no reason to receive Jira, AWS, database, or arbitrary
        # application credentials. Keep only locations the explicit execution
        # adapter prepared. build_child_env adds its small process baseline
        # (PATH, HOME, locale, TLS roots, etc.) independently.
        allowlist={
            "USERPROFILE",
            "XDG_CACHE_HOME",
            "XDG_CONFIG_HOME",
            "XDG_DATA_HOME",
            "XDG_STATE_HOME",
            "PYTHONDONTWRITEBYTECODE",
            *(isolation_environment or {}).keys(),
        },
        endpoint_env=endpoint_env,
    )
    child_env.pop("PERCEPTEYE_ROLLOUT_ID", None)

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
        if observer is not None:
            observer.close()
        return None

    def _resolve() -> DiscoveredAgent | None:
        """The best description available, from either source.

        THE FILE WINS, always. A host that describes itself states its own
        framework, its sub-agents and its tools' read-only annotations --
        everything the wire drops on the way to the model. The observer is the
        floor under hosts that have no way to write that file, not a rival to
        the ones that do.
        """
        described = read_discovered_agent(target)
        if described is not None or observer is None:
            return described
        return observer.described(
            agent_name=agent_name or framework_from_argv(argv),
            framework=framework_from_argv(argv),
        )

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
                return _resolve()
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
        return _resolve()
    finally:
        _kill(proc)
        if observer is not None:
            observer.close()


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
