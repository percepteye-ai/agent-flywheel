# Node harnesses

**One npm package, two harnesses, zero dependencies.** [`@percepteye/agent-flywheel`](https://www.npmjs.com/package/@percepteye/agent-flywheel) is a plugin the harness already loads, and it registers there as `agent-flywheel`. It records what your agent's tools actually did — `ok`, `failed` or **`unknown`** — into the same trajectory file the Python SDK writes and reads.

It is its own repository — [`percepteye-ai/agent-flywheel-node`](https://github.com/percepteye-ai/agent-flywheel-node) — and that repo's README is the reference for the plugin's internals. This page is the contract it shares with the Python SDK.

```bash
openclaw plugins install @percepteye/agent-flywheel            # OpenClaw
dsh plugin --profile <name> add @percepteye/agent-flywheel     # DeepSeek Harness
```

| harness | entry module | hook | can it change the run? |
|:--|:--|:--|:--|
| **OpenClaw** | `src/index.js` (default export + `register`) | `api.on("after_tool_call", …)` | no — the hook returns `void` |
| **DeepSeek Harness** | `src/dsh.js` (`apply`, no default) | `ctx.on("tools/result", …)` | no — `@mode emit`, over a deep-frozen object |

No mutation seam is registered, in either mode. Every hook below is void-returning or observe-only, and the suite asserts the boundary rather than trusting review: no `registerAgentToolResultMiddleware`, and the OpenClaw manifest declares no `contracts`, because an observer claims no capability seam.

## The mode decides which hooks are subscribed

`PERCEPTEYE_AGENT_MODE` is the switch — `training` (the default) or `production`, resolved from the same closed set of spellings the Python SDK accepts, and `plugins.entries.agent-flywheel.config.mode` overrides the environment. A value neither can parse is not guessed: the plugin degrades to `production` and says so, because someone typing `prodction` is trying to *stop* contributing.

The split is structural. `registerRolloutDriver` is called from one place under one predicate, so in production mode the module that could claim a rollout or start a turn is never reached — "observe-only" is the absence of a subscription rather than a branch inside each handler that a later edit could invert.

| mode | hooks the OpenClaw entry subscribes | what they are for |
|:--|:--|:--|
| both | `after_tool_call` · `llm_input` | record one tool outcome; describe the agent |
| **`production`** | `message_received` · `agent_end` · `gateway_stop` | capture the served turn — the user's message, its tool calls, and (with `allowConversationAccess`) the answer — and upload on the cadence |
| **`training`**, and only once it can claim | `session_start` · `agent_end` · `session_end` · `gateway_start` · `cron_changed` · `gateway_stop` | claim a rollout, deliver its task, report the result, and run claimed rollouts in their own `percepteye-rollout-*` sessions |

Those six are subscribed only when the driver can actually claim: with no `apiKey` configured, or on a host that exposes no next-turn injection API, it declines and registers none of them. An install with no account is therefore the first row and nothing else.

The DeepSeek Harness entry is unchanged by any of this: it subscribes `tools/result` and nothing more.

## What switches it off, and where it writes

```bash
export PERCEPTEYE_CAPTURE=0   # or …config.capture=false in your OpenClaw config
```

That is the kill switch, and it is the only one: it returns from `register` before a single hook is subscribed, in either mode.

**`PERCEPTEYE_TRAJECTORY_DIR` is not that switch.** It names the directory one *training* rollout records into, and is read **per call, not cached at construction**, so a host that sets it late still gets capture. In production mode it is not consulted at all — a trajectory there belongs to one **turn**, so each call is routed to its own turn directory under `PERCEPTEYE_CAPTURE_DIR` (default `~/.percepteye/captured`), because one shared directory would merge every end user's calls into a single trajectory.

!!! warning "A production install is not inert"
    Production mode captures every served turn to disk **with `PERCEPTEYE_TRAJECTORY_DIR` unset** — the user's message, the tool calls, and the agent's answer. Nothing leaves the machine without both a key and the control plane's per-agent consent, but it is on disk either way. An operator leaving the plugin installed "because it does nothing without the variable" is accumulating their end users' messages. `PERCEPTEYE_CAPTURE=0` is what makes an install record nothing.

!!! note
    `fw.serve()` sets `PERCEPTEYE_TRAJECTORY_DIR` for the child it spawns, already pointed at that rollout's directory — see [CLI harnesses](cli.md). Setting it by hand is for a run you are driving yourself.

## Observe only, structurally

**Both plugins cannot alter, block, delay or replace a tool result.** That is a property of the hooks, not a promise about the code. The rule behind the choice: capturing evidence must never be able to change the behavior being measured.

OpenClaw exposes three places to see a tool result. Only one is both complete and incapable of changing the run:

| surface | can it alter the result? | why not used |
|:--|:--|:--|
| `registerAgentToolResultMiddleware` | **yes** — its return value replaces what the model reads | a recorder must not sit on a mutation seam |
| `registerAgentEventSubscription` | no, but arguments arrive on a separate `phase:"start"` event and exec results are capped before dispatch | correlating two phases needs a pending map that either leaks or drops |
| `api.on("after_tool_call", …)` | no — returns `void`, host runs it best-effort inside its own try/catch | **used**: `params`, `result`, `error`, `durationMs`, `toolCallId` and agent identity arrive in one event |

The safer hook is also the more complete one, which is why the recorder holds no correlation state at all.

dsh's tool pipeline is mostly waterfalls — `tools/execute` wraps dispatch, `tools/post` can accept, replace or **block** a result, `tools/code-dispatch-log` rewrites what gets logged. `tools/result` is declared `@mode emit`, typed to return `undefined`, and the registry **deep-freezes** both the execution and the result before publishing, with an invariant that fails the harness if it ever publishes something unfrozen. There is no return channel and the object is immutable.

??? note "Why two entry modules rather than one"
    Cordis normalizes a plugin with `exports.default ?? exports`. A `default` export in the dsh module would be handed to the harness *instead of* `apply`, and the plugin would fail to load. The OpenClaw entry needs a default; the dsh entry must not have one. Separate modules is the only shape that satisfies both.

## What it writes

One JSON object per executed tool call, appended to `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`. Real output, from the fixtures captured off a live OpenClaw run and a dsh result:

```json
{"name":"exec","arguments":{"command":"exit 3"},"outcome":"failed","status_code":null,"error":"(Command exited with code 3)","error_class":"ToolError","latency_ms":91,"agent_name":"main","tool_call_id":"call_percepteye_1"}
{"name":"mcp__jira__create_issue","arguments":{"project":"ENG"},"outcome":"unknown","status_code":null,"latency_ms":412,"agent_name":"main","tool_call_id":"call_2","entity_ids":{"id":"10001","key":"ENG-7"}}
{"name":"bash","arguments":{"command":"npm run dev &"},"outcome":"unknown","status_code":null,"error_class":"StillRunning","agent_name":"main","tool_call_id":"tc_9"}
```

| field | present | value |
|:--|:--|:--|
| `name` | always | the tool name as the host reports it. A nameless call is dropped, not recorded. |
| `arguments` | always | the arguments the tool **actually ran with** — OpenClaw resolves any `before_tool_call` adjustment before dispatch. `{}` when not an object. |
| `outcome` | always | `ok` · `failed` · `unknown` |
| `status_code` | always | **always `null` from Node.** Neither host's tool stream carries an HTTP status, and inventing one is the failure the tri-state exists to prevent. |
| `error` | when the host stated one | truncated at 1000 characters |
| `error_class` | when known | `ToolError` · `ExitStatus:<n>` · `ExitStatusUnreadable` · `StillRunning` · `Aborted` · `TimedOut` · `Signal:<name>` · dsh's own `info` code |
| `latency_ms` | OpenClaw only | host-measured `durationMs`, so it includes queueing and approval waits. `recordFor()` on the dsh path sets no latency. |
| `agent_name` | when the host names one | `ctx.agentId` (OpenClaw) · `exec.agent.name` then `exec.agent.id` (dsh). Absent means unattributed — never a claim that one agent made everything. |
| `tool_call_id` | when present | `event.toolCallId` · `exec.callId`. The join key back to the completion this call followed. |
| `entity_ids` | when non-empty | which entity a write produced |

**Neither plugin ever writes `output`.** Identity is resolved at capture and the response body is dropped: `id`, `key`, `uuid`, `sid`, `gid` only, shallowest occurrence wins, depth 3 and 2000 nodes at most, a field seen twice at one depth with different values dropped as ambiguous. `name`, `number`, `title`, `label`, `summary`, `description` are never identity — a different entity can share them by coincidence, and matching on one is a false positive dressed as a verified success. `entity_ids` is emitted for **every** outcome including `unknown`, because an unknown call with an identity is exactly the one a later read-back can settle.

Writes are append-only via `O_APPEND` and flushed per call, so concurrent tools interleave whole lines and a killed run keeps every call up to the kill. One JSON object per line, because the Python reader recovers per line.

**Failure never reaches your agent.** An unwritable destination warns once through the host logger and then stays quiet; a malformed event is swallowed by the handler's own try/catch. One lost record, never an interrupted run, never a log full of our name.

## How the outcome is decided

`unknown` means the call executed and the host did not say whether it worked. **It is never rounded up to `ok`** — a call scored as success when nobody said so is training data that teaches the model a failure was fine. See [outcomes](outcomes.md) for what the tri-state buys downstream.

The distinction is not academic. In a real run OpenClaw reports `details.status: "completed"` for a command that exited **3**; `completed` means the process finished, not that it worked. A classifier keying on that field grades every failed command a success. So the plugin reads the exit code.

### OpenClaw

| what we see | outcome |
|:--|:--|
| host sets a non-empty `error` | `failed`, `error_class: "ToolError"` |
| `exec` / `bash` with exit 0 | `ok` |
| `exec` / `bash` with non-zero exit | `failed`, `error_class: "ExitStatus:<n>"` |
| `exec` / `bash`, exit code unreadable | `unknown`, `error_class: "ExitStatusUnreadable"` |
| `read`, `write`, `edit`, `grep`, `ls`, `find` with no error | `ok` |
| anything else — MCP tools, third-party plugin tools | `unknown` |

`apply_patch` is deliberately **not** in the silent-success set: a patch can apply partially, and until a fixture shows how that is reported, grading it `ok` on silence is a guess in the dangerous direction. Tools move out of the last row when a fixture proves how they report failure. Promotion is earned, never assumed.

### DeepSeek Harness

dsh's exec tools declare a real output schema — `kind`, `exitCode`, `signal`, `timedOut`, `aborted` — which lets three cases be told apart that a cruder reading merges.

| what dsh reports | outcome |
|:--|:--|
| `isError: true` | `failed`, carrying dsh's own error class off `error.info` |
| `isError` absent entirely | `unknown` — absent is not `false`; the harness did not say |
| `bash` / `pwsh`, foreground, exit 0 | `ok` |
| `bash` / `pwsh`, foreground, non-zero exit | `failed`, `error_class: "ExitStatus:<n>"` |
| `timedOut: true` | `failed`, `error_class: "TimedOut"` |
| killed by a signal | `failed`, `error_class: "Signal:<name>"` |
| **`kind: "background"`** | **`unknown`**, `error_class: "StillRunning"` — launched, not finished |
| **`aborted: true`** | **`unknown`**, `error_class: "Aborted"` — cancelled, which is not the agent's mistake |
| exit unreadable | `unknown`, `error_class: "ExitStatusUnreadable"` |
| `read`, `write`, `edit`, `read_image`, `str_replace_editor`, `todo_write` with `isError: false` | `ok` |
| everything else — web, terminal session, subagent, plugin-registered tools | `unknown` |

`isError: false` does not mean success. dsh's bash tool renders a failing exit as text (`[exit code: 3]`) appended to the model-facing content and still settles the call as a non-error: the tool call worked, the command did not.

## Describing the agent (OpenClaw only)

The OpenClaw entry also registers `llm_input` and writes `discovered_agent.json` into `PERCEPTEYE_TRAJECTORY_DIR` — the rollout's directory, in either mode, and nothing is written when the variable is unset. This is the only way a CLI agent reaches `introspection_state = described` — Python reflection cannot cross a process *and* a language boundary, so a command entrypoint would otherwise register as `unreadable`, and an agent with no tool catalogue gets no generated workflows.

`llm_input` is typed `(event, ctx) => Promise<void> | void`, the same structurally non-mutating class as `after_tool_call`. The mutation seams OpenClaw does expose — `before_prompt_build`'s `{systemPrompt}`, `before_tool_call`'s `{block: true}` — stay unused. Introspection is a read.

| behavior | why |
|:--|:--|
| `framework: "openclaw"`, `source_type: "openclaw_llm_input"` | lookup keys, not labels; any other spelling degrades to an unknown framework |
| richest catalogue wins, not newest | a sub-agent or post-compaction turn can carry a subset; last-write-wins would overwrite the real catalogue with a fraction of it |
| write-then-`rename` | the Python SDK polls for this file, so a reader sees no file or a complete one, never half a JSON object |
| `tool_definitions: null` vs `[]` | `null` means we could not look; `[]` means the agent has no tools. Collapsing them turns "not observed" into a confident claim. |
| NUL bytes and lone surrogates stripped | Postgres JSONB refuses them, and the payload is customer-authored prompt text and provider-shaped tool JSON |

!!! warning
    `llm_input` is a **conversation hook**. OpenClaw refuses to register it for a non-bundled plugin unless the operator opts in with `plugins.entries.agent-flywheel.hooks.allowConversationAccess=true`. The refusal is a warn-level diagnostic on the host's side, not a throw — so without the opt-in the plugin loads, the recorder works, and the agent registers as `unreadable` forever. The plugin says this out loud at startup, once, and only when a trajectory directory makes it relevant. If `discovered_agent.json` does not appear after the first model call, this is why.

The DeepSeek Harness entry registers no describer. It records tool outcomes and nothing else.

### What it cannot see: `describe`

OpenClaw's `llm_input` carries no tool list, so on its own the plugin registers an agent whose tools are unknown, and no task can be generated for it. The operator knows what the agent sends its model, and declares it in the plugin's config. It is the same declaration as the Python SDK's [`describe=`](how-it-works.md#an-agent-nothing-can-read-describe):

```json
"plugins": { "entries": { "agent-flywheel": { "config": { "describe": {
  "system_prompt": "You triage support tickets.",
  "tools": [{ "name": "lookup_ticket", "input_schema": { "type": "object" }, "read_only": true }]
} } } } }
```

- **Same rules, same wire record, same refusals** — for anything JSON carries, except an integer beyond 2⁵³ inside a schema, which JavaScript holds rounded. `src/declaration.js` is a rule-for-rule port of `declaration.py`. Where the two languages part by default (whitespace, coercion, number types, sort order), the rule is Python's or one both keep: a tool's name and description must be strings, `max_tokens` a whole number from 1 to 2⁵³ − 1, `temperature` finite, and refusals name JSON types. `tests/test_describe_parity.py`, in this repository, runs both over one table: every accepted declaration must produce the identical record, alone and merged over the plugin's observation, and every refused one must refuse with the same sentence.
- **What you declare wins** over the observation where given. The training registration merges it over the startup observation; the production registration sends it instead of `null`.
- **It anchors no execution identity**, on the registration, a rollout report or a production turn. With an `executionSnapshot` configured, a startup warning says the identity is withheld.
- **Three differences, each forced by the host:**
    - A bad declaration: the Python SDK refuses to start, but a throw from an OpenClaw plugin's `register()` fails the plugin load and takes local capture down with it. So the plugin drops it with a warning naming the same cause, and registers only what it observed. For the same reason the config schema accepts any `describe` value, so OpenClaw never refuses the plugin over one.
    - `introspect: false` together with `describe`: Python's `serve(introspect=False, describe=...)` refuses the contradiction; the plugin warns and sends nothing.
    - With nothing observed, the record names the host, `framework: "openclaw"`. Python, which cannot know a custom agent's framework, says `custom`.

**Consent is asked the same way too.** `introspect: false` in the plugin config, or `PERCEPTEYE_INTROSPECT=0`, registers the agent with no description at all, which the control plane records as the opt-out. Either side's refusal stands, and no execution identity is authored.

## Package

| | |
|:--|:--|
| dependencies · peerDependencies · optionalDependencies | **none** — `node:fs` and `node:path` only |
| `engines.node` | `>=22.22.3` |
| module type | ESM (`"type": "module"`) |
| exports | `.` · `./classify` · `./trajectory` · `./dsh` |

Zero dependencies is a hard constraint, not a preference: this runs inside someone else's agent process, and every dependency added here is a version conflict imposed on a customer who did not ask for one. The Node repo's own CI asserts all three dependency keys stay empty.

## The two halves cannot drift

The plugins and the Python SDK are two ends of one ABI — whatever a plugin appends, `agent_flywheel.outcomes.Trajectory` has to read back with the same meaning. **A JS suite that only checked itself would let a plugin stay green while writing a line the reader counts as malformed**, surfacing later as a silently short tool-call list.

**The check lives in the Python repository, because the reader does.** The plugins' own suite moved to `percepteye-ai/agent-flywheel-node` with them and runs there; what can only be checked here is that what they write is what `Trajectory` reads. So the two commands run from two different repositories — clone the Node SDK **beside** the Python one:

```bash
# the plugins' own suite, in their repo
cd /path/to/agent-flywheel/..
git clone https://github.com/percepteye-ai/agent-flywheel-node
(cd agent-flywheel-node && npm test)                # zero skips

# the JS writers against the Python reader, in THIS repo
cd agent-flywheel && pip install -e .
./scripts/cross_language_abi_check.sh
```

The script resolves the Node SDK from `PERCEPTEYE_NODE_SDK`, and otherwise from a sibling checkout named `percepteye-agent-flywheel-node` or `agent-flywheel-node` — which is why the clone goes beside this repo rather than inside it. With none of those present it **skips rather than fails**: a developer who checked out only the Python repo has nothing to check against, and a red suite for that would teach everyone to ignore the failure.

The check drives both plugins' writers into a temp directory, reads the file with `Trajectory`, and asserts `malformed == 0` plus the exact outcome sequence — including that `exec` with `status: "completed"` and exit 3 comes back `failed`, that a backgrounded dsh command comes back `unknown` / `StillRunning`, and that `entity_ids` survives the boundary on an `unknown` call.

The `node-plugins` CI job runs that one step on Node 22 and 24, the ends of OpenClaw's supported engine range. It checks the Node SDK out into the workspace and **names** it to the script through `PERCEPTEYE_NODE_SDK`, so a path that is not there is an error rather than a skip. `npm test` and the zero-dependency assertion run in the Node repo's own workflow, and the Python suite is separate jobs again. `test/host-fixtures.test.js` — in the Node repo — holds payloads dumped verbatim from OpenClaw 2026.7.1-2 executing a shell command end to end: if a future OpenClaw changes the shape, those fail and someone re-reads the payload instead of discovering it months later from bad training data.
