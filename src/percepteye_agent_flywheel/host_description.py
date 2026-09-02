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
            tools = [_tool_from_wire(t) for t in self._tools.values()]
            prompt = self._system
            model = self._model
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
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL"):
        child_env[name] = endpoint
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
