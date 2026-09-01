# Node harnesses

**One npm package, two harnesses, zero dependencies.** `percepteye-agent-flywheel` is a plugin the harness already loads. It records what your agent's tools actually did — `ok`, `failed` or **`unknown`** — into the same trajectory file the Python SDK writes and reads.

```bash
openclaw plugins install percepteye-agent-flywheel            # OpenClaw
dsh plugin --profile <name> add percepteye-agent-flywheel     # DeepSeek Harness
```

| harness | entry module | hook | can it change the run? |
|:--|:--|:--|:--|
| **OpenClaw** | `src/index.js` (default export + `register`) | `api.on("after_tool_call", …)` | no — the hook returns `void` |
| **DeepSeek Harness** | `src/dsh.js` (`apply`, no default) | `ctx.on("tools/result", …)` | no — `@mode emit`, over a deep-frozen object |

No mutation seam is registered. The OpenClaw entry adds exactly one more hook, `llm_input`, to describe the agent — the same void-returning class as `after_tool_call` — and nothing beyond those two. The suite asserts it: no `registerAgentToolResultMiddleware`, and the OpenClaw manifest declares no `contracts`, because an observer claims no capability seam.

## The switch is one environment variable

```bash
export PERCEPTEYE_TRAJECTORY_DIR=/path/to/this/rollout
```

**Unset, the plugin does nothing at all** — no file, no write, no directory created. `createWriter()` reports `active: false` and `write()` returns `false` before touching the filesystem. That is what makes the same install a no-op in production and a recorder during a rollout, so you leave it installed permanently.

The variable is read **per call, not cached at construction**. A host that sets it late still gets capture; a host that never sets it never gets a stray file.

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

The OpenClaw entry also registers `llm_input` and writes `discovered_agent.json` into the same directory. This is the only way a CLI agent reaches `introspection_state = described` — Python reflection cannot cross a process *and* a language boundary, so a command entrypoint would otherwise register as `unreadable`, and an agent with no tool catalogue gets no generated workflows.

`llm_input` is typed `(event, ctx) => Promise<void> | void`, the same structurally non-mutating class as `after_tool_call`. The mutation seams OpenClaw does expose — `before_prompt_build`'s `{systemPrompt}`, `before_tool_call`'s `{block: true}` — stay unused. Introspection is a read.

| behavior | why |
|:--|:--|
| `framework: "openclaw"`, `source_type: "openclaw_llm_input"` | lookup keys, not labels; any other spelling degrades to an unknown framework |
| richest catalogue wins, not newest | a sub-agent or post-compaction turn can carry a subset; last-write-wins would overwrite the real catalogue with a fraction of it |
| write-then-`rename` | the Python SDK polls for this file, so a reader sees no file or a complete one, never half a JSON object |
| `tool_definitions: null` vs `[]` | `null` means we could not look; `[]` means the agent has no tools. Collapsing them turns "not observed" into a confident claim. |
| NUL bytes and lone surrogates stripped | Postgres JSONB refuses them, and the payload is customer-authored prompt text and provider-shaped tool JSON |

!!! warning
    `llm_input` is a **conversation hook**. OpenClaw refuses to register it for a non-bundled plugin unless the operator opts in with `plugins.entries.percepteye-agent-flywheel.hooks.allowConversationAccess=true`. The refusal is a warn-level diagnostic on the host's side, not a throw — so without the opt-in the plugin loads, the recorder works, and the agent registers as `unreadable` forever. The plugin says this out loud at startup, once, and only when a trajectory directory makes it relevant. If `discovered_agent.json` does not appear after the first model call, this is why.

The DeepSeek Harness entry registers no describer. It records tool outcomes and nothing else.

## Package

| | |
|:--|:--|
| dependencies · peerDependencies · optionalDependencies | **none** — `node:fs` and `node:path` only |
| `engines.node` | `>=22.22.3` |
| module type | ESM (`"type": "module"`) |
| exports | `.` · `./classify` · `./trajectory` · `./dsh` |

Zero dependencies is a hard constraint, not a preference: this runs inside someone else's agent process, and every dependency added here is a version conflict imposed on a customer who did not ask for one. CI asserts all three dependency keys stay empty.

## The two halves cannot drift

The plugins and the Python SDK are two ends of one ABI — whatever a plugin appends, `percepteye_agent_flywheel.outcomes.Trajectory` has to read back with the same meaning. **A JS suite that only checked itself would let a plugin stay green while writing a line the reader counts as malformed**, surfacing later as a silently short tool-call list.

```bash
cd packages/percepteye-agent-flywheel && npm test   # 89 tests, zero skips
./scripts/cross_language_abi_check.sh               # the JS writers, the Python reader
```

The check drives both plugins' writers into a temp directory, reads the file with `Trajectory`, and asserts `malformed == 0` plus the exact outcome sequence — including that `exec` with `status: "completed"` and exit 3 comes back `failed`, that a backgrounded dsh command comes back `unknown` / `StillRunning`, and that `entity_ids` survives the boundary on an `unknown` call.

The `node-plugins` CI job runs all three steps on Node 22 and 24, the ends of OpenClaw's supported engine range. `test/host-fixtures.test.js` holds payloads dumped verbatim from OpenClaw 2026.7.1-2 executing a shell command end to end: if a future OpenClaw changes the shape, those fail and someone re-reads the payload instead of discovering it months later from bad training data. The Python suite is 509 tests and is a separate job.
