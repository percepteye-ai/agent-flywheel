# CLI and other languages

Not Python? Pass an **argv list** instead of an import path. A compiled binary, a Node CLI, anything with a command line. There is no protocol to implement, so an existing CLI needs **no changes**.

```python
import percepteye_agent_flywheel as fw

fw.serve(["opencode", "run"], agent_id="coding-agent", input_shape="text")
fw.serve(["dsh", "--profile", "headless", fw.TASK], agent_id="dsh-agent", input_shape="text")
```

A `list` or `tuple` first argument means the agent is a **process**, not an object. Nothing is imported and nothing is resolved — isolation is inherent, because the agent is already its own process. `method=` and `adapters=` have no object to apply to.

## The task, three channels

| channel | when | notes |
|:--|:--|:--|
| **stdin** | always, unless `fw.TASK` is in the argv | written, then closed |
| **`PERCEPTEYE_TURN_INPUT`** | always | inert unless the CLI reads it |
| **an argv position** | only where `fw.TASK` appears | every occurrence is substituted |

The first two are sent together because CLIs differ and guessing wrongly is silent. `fw.TASK` makes the choice a declaration: with the sentinel in the argv, **stdin gets the empty string** so an agent that reads both does not act on the same instruction twice. `PERCEPTEYE_TURN_INPUT` stays set either way.

`fw.TASK` is a **sentinel object, not the string `"{task}"`**. A literal argument that happens to contain the placeholder text survives verbatim, and an unsubstituted sentinel surfaces as a type error rather than as an agent receiving four characters and doing something confident with them.

```python
fw.serve(["agent", "--tmpl", "{task}", fw.TASK], ...)
# argv:  agent --tmpl {task} count to three
```

argv is handed to `Popen` as a list, never through a shell, so a task containing `oops"; rm -rf / #` is one argument.

!!! warning "argv is visible; stdin is not"
    Anything in an argv is readable by other users on the box through `ps`. The task text is the agent's prompt, so `fw.TASK` publishes it to the process table. Use it when the CLI genuinely cannot read stdin or an environment variable — not by default.

## What comes back

**stdout, trimmed, is the final text.** That is the whole return contract.

| what happened | `exclusion_reason` | reported? |
|:--|:--|:--|
| exit 0 | none | yes |
| exit non-zero | `agent_exit_nonzero` — detail is `exit <n>: ` plus the last 1500 chars of stderr | yes, carrying the exclusion |
| killed on the deadline | `deadline_exceeded` | yes, carrying the exclusion |
| binary could not be started | — | no: the spawn raises, and the rollout is abandoned as `flywheel_crash` |

**Excluded is not discarded.** A rollout that wrote a correct deliverable and then exited 1 because it ran out of turns is still reported — the exclusion block travels with it, so the control plane records the work product and still knows not to train on it. The agent's inability to declare itself finished must not decide the score.

`agent_exit_nonzero` is deliberately separate from `entrypoint_error`. `Popen` raises when a binary cannot be started at all, so reaching a non-zero exit means the agent **ran** and chose it: a crash after doing work, or a completion gate it could not satisfy. One label for both sent an afternoon's triage at the process launcher while the detail string read "quality gate still failing".

!!! note "stderr is kept only on the non-zero path"
    On a clean exit and on a timeout kill, stderr is discarded. Anything you want preserved goes on stdout or into a file in the working directory.

## The working directory is the rollout

Each rollout's child starts with its cwd set to that rollout's own directory — the same path it is handed as `PERCEPTEYE_TRAJECTORY_DIR`, namely `<artifacts_dir>/_percepteye/<rollout_id>`. For a coding or REPL agent the files it leaves there **are** the deliverable, so every one of them is collected as an artifact. Two exceptions: `tool_calls.jsonl` is the trajectory, not an artifact; and a symlink is recorded as `skipped:symlink` rather than followed, because the agent authors that directory and `ln -s ~/.aws/credentials creds.txt` would otherwise be read at its target. The directory is removed once the report is out.

`artifacts_dir=` defaults to `$PWD/.percepteye-artifacts`. A CLI agent has a real shell and this SDK's isolation is a process boundary and an environment scrub — it says nothing about the filesystem. **Point `artifacts_dir=` outside any checkout.**

## The child's environment

| variable | value |
|:--|:--|
| `PERCEPTEYE_ROLLOUT_ID` | this rollout's id |
| `PERCEPTEYE_TRAJECTORY_DIR` | this rollout's directory, which is also the cwd |
| `PERCEPTEYE_TURN_INPUT` | the task |
| `OPENAI_BASE_URL` · `OPENAI_API_BASE` · `ANTHROPIC_BASE_URL` | the per-rollout gateway |
| `OPENAI_API_KEY` · `ANTHROPIC_API_KEY` | the rollout's single-use key |

Your environment otherwise passes through — your agent needs your credentials. Two things never do:

- **The `PERCEPTEYE_` namespace is never inherited.** The long-lived control-plane key lives there and stays in the parent, because a rollout child runs customer agent code with a shell.
- **Provider variables the gateway cannot serve are removed** (`UNROUTABLE_PROVIDER_VARS` in [`isolation.py`](https://github.com/percepteye-ai/agent-flywheel/blob/main/src/percepteye_agent_flywheel/isolation.py)) — Azure, Gemini/Vertex, Bedrock's bearer token, and the OpenAI-compatible resellers. `serve()` warns at startup if it finds any set. An agent that finds no provider raises, which is the outcome we want; silent off-gateway sampling is the one we do not.

### `endpoint_env` — when your agent reads other names

```python
fw.serve(["dsh", "--profile", "headless", fw.TASK],
         agent_id="dsh-agent", input_shape="text",
         endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"))
```

The named pair receives the same per-rollout URL and key. **Additive** — the OpenAI and Anthropic names stay set, because an agent may read a mixture. **Declared, never sniffed:** DeepSeek Harness reads `DEEPSEEK_BASE_URL` and grep finds zero hits for `OPENAI_BASE_URL` in its source, so an unmodified integration looks correct from here right up until every completion arrives at the live provider unobserved.

### `gate_flag` — how your binary spells its completion gate

```python
fw.serve(ARGV, ..., gate_flag="--autonomous-gate")
```

Two independent facts, kept apart on purpose. `gate_flag` is a property of **the binary** and belongs to the serve configuration — it cannot be guessed, and handing an unknown option to an arbitrary agent would break it outright. The gate **command** is a property of **the task** and arrives per rollout in `metadata["completion_gate"]`. With both present, the two are appended to the end of the argv:

```
prime-agent … --autonomous-gate 'test -s top_ips.sh'
```

Either side missing leaves the argv untouched, which is exactly the pre-gate behavior.

## Timeouts kill the process group

`rollout_timeout_s` (default `900.0`) is capped by the rollout's own remaining budget less a 10s margin, held back so artifacts and the report can still be delivered against a live lease. On expiry the whole **process group** is killed with `SIGKILL`.

The group, not the process, and that is load-bearing. A real CLI agent spawns freely — a shell, a compiler, a test runner — and killing only the process we launched leaves those children alive holding the stdout pipe open, so `communicate()` blocks and the timeout never returns. Measured: a 1.5s budget against `sh -c 'sleep 120'` took the full 120s. The child is spawned with `start_new_session=True` on POSIX, so a spawning agent leaves nothing behind. Windows has no process groups; only the launched process is killed.

## Tool outcomes with no instrumentation

Two routes in. Neither requires a line of code inside the agent.

### 1. The event stream your CLI already prints

A modern CLI agent emits structured events because it needs them for its own UI. **stdout is parsed for them after every run.** A stream is identified by the event types it actually carries, never by an agent name — a name is a claim, an event type is evidence.

| vocabulary | start / end | id · name · args · result · error |
|:--|:--|:--|
| `prime-agent` | `tool_execution_start` / `tool_execution_end` | `toolCallId` · `toolName` · `args` · `result` · `isError` |
| `otel-genai` | `gen_ai.tool.start` / `gen_ai.tool.end` | `gen_ai.tool.call.id` · `gen_ai.tool.name` · `gen_ai.tool.input` · `gen_ai.tool.output` · `error.type` |

The second row is the one that matters: it belongs to no vendor, so an agent already logging OpenTelemetry GenAI spans as JSON lines gets outcome capture with **no work at all**. Field names are candidate tuples tried in order, because emitters disagree about camelCase versus snake_case far more often than they disagree about meaning. Adding the next CLI is a `StreamFormat` entry — a vocabulary, not a parser.

**The verdict is tri-state by omission.**

| the end event carries | outcome |
|:--|:--|
| the error flag as `true` | `failed` |
| the error flag as a non-empty string (`"error.type": "TimeoutError"`) | `failed` |
| the error flag as `false` | `ok` |
| an empty or whitespace string | `unknown` |
| no error flag, on the event or on its result | `unknown` |

A call that ended without saying whether it worked has not asserted success, and we do not assert it on the agent's behalf. **`unknown` is never rounded up to `ok`** — see [outcomes](outcomes.md). `status_code` is always `null` here: a local tool has no HTTP status, and the contract discards status entirely when the outcome is unknown.

The rest of the extraction rules:

- the pair is joined on the call id; arguments fall back to the start event, because retry detection keys on name + arguments
- a call with no resolvable name is **dropped, not invented** — a fabricated name corrupts every join downstream
- unparseable lines are skipped, because a CLI interleaves its own logging with its event stream
- a stream carrying no tool events leaves `tool_calls` as `null`, never `[]`
- a readable body is preferred (`content[].text`, then `details.stdout`), truncated at 4000 characters; an error body at 1000
- `latency_ms` comes from `result.details.durationMs`; `entity_ids` are lifted from the **structured** result, never from rendered prose
- **only stdout is read.** A CLI that logs its events to stderr reports nothing.

**Capture survives every exit path.** Non-zero exit and timeout kill are finalised exactly like a clean exit — a rollout that died is precisely the one you need the tool calls from. This was learned the hard way: on one run every rollout exited non-zero, so `tool_calls` came back `null` even though the parser worked.

### 2. Append to the trajectory from a shell

A non-Python agent reports outcomes by appending JSONL to a file whose path it is given. No SDK, no import.

```bash
printf '{"name":"deploy","arguments":{"env":"prod"},"outcome":"ok","status_code":201}\n' \
  >> "$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl"
```

One JSON object per line. `name` and `outcome` are required; `outcome` is one of `ok`, `failed`, `unknown`. Everything else is optional: `arguments`, `status_code`, `output`, `error`, `error_class`, `latency_ms`, `agent_name`, `tool_call_id`, `entity_ids`.

**One unreadable line costs that line and nothing else.** A truncated final write is the expected case, not an exotic one — the deadline kill lands mid-write. The lost records are counted and travel as `tool_calls_omitted_count`, because a call we know executed and cannot read is not a call that did not happen.

**The trajectory wins.** If anything was recorded there — including `[]`, which is a positive claim that zero calls were made — the stream parse is not consulted. An agent that wrote to the trajectory made a deliberate assertion about its own behavior, and a parse must never overwrite it.

## Registration: describing a CLI agent

There is no object to reflect on, and reflection cannot cross a process and a language boundary. So the agent is **run once at `serve()` time** in `<artifacts_dir>/_describe`, fed the warm-up prompt `Reply with the single word: ready`, and killed once it has described itself. Not the moment it writes `discovered_agent.json`: the first write is a **lower bound** on the catalogue — an integration that enriches the description after the agent exits once turned 38 tools into 77 — so the child is left to finish, bounded by a 60s settle window once the file appears and a 90s warm-up timeout overall. `fw.TASK` is filled with the warm-up prompt so no sentinel reaches the binary.

The warm-up child gets a deliberately unroutable endpoint (`http://127.0.0.1:1/v1`) and a dummy key. It is not a rollout, nothing captures it, and a real credential would let it sample a live provider unobserved.

Only a host running the companion Node plugin writes that file — the hook fires *before* the model request is dispatched, which is what makes a warm-up with no working completion work at all. **A plain CLI with no plugin writes nothing**, registers `discovered_agent: null`, and `serve()` warns saying exactly that: no tool catalogue means no generated workflows, and every phase downstream is downstream of it. Set `introspect=False` or `PERCEPTEYE_INTROSPECT=0` to skip the warm-up entirely — see [configuration](configuration.md).

## Worked example: prime-agent

[`examples/prime_agent/`](https://github.com/percepteye-ai/agent-flywheel/tree/main/examples/prime_agent) is the end-to-end one — a real TypeScript coding agent, served with no Percepteye code inside it.

| piece | why |
|:--|:--|
| `--mode json` | emits `tool_execution_*`, so outcomes come off stdout |
| `-e route.ts` | prime-agent has no env var for a base URL; an extension reads `OPENAI_BASE_URL` and registers a provider |
| `--provider percepteye --model …` | registering a provider does not select it; without both flags every completion goes somewhere we never see |
| `gate_flag="--autonomous-gate"` | with no gate registered, a *limit* is the only exit that exists, and every rollout ends excluded |
| `artifacts_dir=` outside any checkout | the agent has a shell; a sweep launched from a checkout removed tracked files from it |

??? note "prime-agent's `isError` is not the whole truth"
    In `--mode json`, `tool_execution_end.isError` is hardcoded `false` whenever the tool call returns without throwing — a raising IPython cell still reports `false`, with the real verdict in `result.details.status == "error"`. **The parser reads the error flag and does not consult `details.status`**, so such a call is recorded `ok`. This is a known gap, not a feature: `details.status` is a prime-agent-specific field and the extraction is deliberately vocabulary-driven rather than per-agent. If you need those failures graded, have the agent append them to `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl` — the trajectory wins over the stream.

## What argv does not give you

Stated plainly, because guessing here is expensive:

- **No adapters.** `adapters=` binds to a Python object. A process has none; the event stream and the trajectory file are the two routes.
- **No `method=`, no `input_shape` shaping.** Neither is even validated on this path — there is no object to resolve them against — and a command always receives the raw task text on stdin, in the environment, and at `fw.TASK`.
- **No `record_tool_call()`.** That is an in-process call. Append JSONL instead.
- **No filesystem sandbox.** `isolation` is a process boundary and an environment scrub. Run untrusted agents accordingly.
