# Python agents

The Python SDK has two calls, for two different processes:

- **`attach()`** goes in the process that serves your users. It records the conversations your agent is already having, and is how we learn from them. The [Quickstart](quickstart.md) and [Production](production.md) are the guide; this page is the reference.
- **`serve()`** goes in a process you lend to practice tasks. It claims tasks we write for your agent, runs each one in a fresh child process, and reports what happened.

Both bind the same adapters, so a LangChain, LangGraph or Google ADK agent is recorded with **no Agent Flywheel code inside it**. Not Python? See [CLI and other languages](cli.md).

## Production: `attach()`

`attach()` records what your agent already did, from turns you are already serving: same [adapters](#adapters), same tri-state [outcomes](outcomes.md), same trajectory format as a practice task, scoped around a live user turn.

```python
px = fw.attach(agent_id="support-agent", trajectory_root="/var/lib/percepteye")

with px.turn(task_id=case_id, conversation_id=conv_id,
             input_text=user_text) as turn_id:
    reply = agent.invoke(user_text)       # your agent, unmodified

px.record_answer(turn_id, reply)          # after the scope closes: the answer exists now
```

**It captures. It never reroutes.** That is a property of how the adapters are bound, not a promise about the code: `attach()` binds them with `llm_url=""` and `api_key=""`, and both built-ins treat "no endpoint" as "do not touch routing". Bound by name, `LangChainAdapter` holds no chat model and its `_route()` returns at once; the ADK adapter never reads the URL at all. Attaching cannot re-point a single byte of production traffic.

| parameter | default | notes |
|:--|:--|:--|
| `agent_id` | required | exact 1–255-character production identifier; because it is a URL segment, path separators and `.` / `..` are forbidden; invalid values raise `ConfigurationError` before capture starts |
| `trajectory_root` | **required** | where every turn is written; see [Your copy on disk](#your-copy-on-disk) |
| `adapters` | `"auto"` | binds every built-in whose framework is installed |
| `register` | `True` | `False` skips the control-plane call entirely |
| `retention_turns` | `None` | keep only the most recent N turns on disk; a whole number, 0 or more. The Node plugin's `retentionTurns` |
| `control_plane_url` · `api_key` | env | `PERCEPTEYE_CONTROL_PLANE_URL` (default: the production control plane) / `PERCEPTEYE_API_KEY` |
| `execution_entrypoint` | `None` | the live Python agent/import spec to describe for exact execution identity |
| `execution_input_shape` · `execution_method` | `"text"` · `None` | the declared call surface of `execution_entrypoint` |
| `attested_execution_components` | `None` | opaque exact adapter evidence for implementation/state discovery cannot observe |
| `describe` | `None` | the agent's own `system_prompt` / `tools` / `model` — without `execution_entrypoint`, the whole description this registration sends ([`describe=`](how-it-works.md#an-agent-nothing-can-read-describe)) |
| `rules` | `None` | the Product's workflow rules, declared once after the registration, exactly as for [`serve()`](configuration.md#a-products-workflow-rules-rules); `None` sends nothing, `[]` retracts |

With `register=True` the call makes exactly two outbound requests, once, at startup: `POST /agents` then `GET /policy/current` — and a third, `POST /agents/{agent_id}/rulebook`, only when `rules=` is given, sent in the background after the policy read: `attach()` returns without waiting for its answer. It registers as `mode: "production"` — `serve()` sends `"rollout"`, and a training donor versus a production reader is the one distinction a `mode` on a registration exists to draw. `register=False` skips every one of them, for a process that should make no outbound connection at all.

**Without `execution_entrypoint` there is no agent object to describe, and the registration says so:** `discovered_agent: null` — unless you declared one with `describe=`. A declaration that leaves `tools` out does not erase the tools your `serve()` registration stored: the control plane keeps a catalogue that a later description did not observe. Either way the registration does not omit the key — the control plane reads an absent key as you opting out of introspection, records the agent as `disabled`, and a description your `serve()` registration stored stops driving task generation. Your real opt-out, `PERCEPTEYE_INTROSPECT=0`, still omits it.

**Everything fails open.** An unreachable control plane yields an attachment with `policy == {}` and costs you nothing else; a framework that cannot be bound yields `adapters_bound == ()` rather than a process that will not start. (`serve()` raises there instead, and correctly: a practice task with no capture is wasted, while a production turn with no capture is still a served user.) A registration the control plane *refuses* while the policy read succeeds emits a `RuntimeWarning` and nothing more. Rules it refuses, cannot hear, or answers slowly cost the rules and nothing else: `px.rulebook` stays `None` and `px.rulebook_refusal` says why, in the control plane's own words. It says "not registered" only when that is known: a failure after the rules may have landed (a read timeout, a 502 from a proxy) is named as unknown, and declaring the same rules again is safe either way. Until the control plane answers, the sentence says so, and `px.wait_for_rulebook(timeout)` waits for the answer; `close()` gives an answer still on its way up to five seconds.

| on the `Attachment` | |
|:--|:--|
| `.turn(turn_id=None, conversation_id=None, task_id=None, input_text=None)` | context manager; yields the turn id |
| `.record_answer(turn_id, text)` | what the agent finally answered; callable **after** the scope closes |
| `.read(turn_id)` | that turn's `list[ToolCallOutcome]`, or `None` if nothing was recorded |
| `.read_turn(turn_id)` | the `task` / `answer` text recorded for that turn, or `None` |
| `.flush(include_unanswered=False, batch_size=64)` | uploads now, asking for the permissions again first; larger requested batches are split at the contract's 64-turn limit |
| `.prune(keep_last=None, older_than_s=None)` | deletes turn directories on your disk; with no argument it deletes nothing |
| `.may_upload` · `.upload_refusal` | whether turns may be uploaded, and the sentence explaining a `False` |
| `.may_capture_model_content` · `.model_content_refusal` | whether model calls are recorded and uploaded, and why not |
| `.policy` | what the control plane said to run — **informational here**; `attach()` does not act on it |
| `.agent_fingerprint` | `{"execution_sha256": ...}` only when SDK reconciliation was exact; otherwise `None` |
| `.adapters_bound` | e.g. `("adk", "langchain")`; `()` is a real answer, not a failure |
| `.rulebook` | the control plane's receipt for `rules=` — `status` (`registered` · `retracted` · `unchanged`), `version`, `content_sha256`, `rule_count`, `product_id` — or `None` |
| `.rulebook_refusal` | why the rules passed as `rules=` are not confirmed in force — including "not answered yet" — or `None` when they registered or none were passed |
| `.wait_for_rulebook(timeout=None)` | waits for the control plane's answer to `rules=`; `True` once `.rulebook` / `.rulebook_refusal` say what it did |
| `.close()` | unbinds the adapters and sends what is waiting; idempotent, never raises. Also a context manager |

### Turns

**A turn boundary is declared, not inferred.** Guessing where one request ends and the next begins is the same class of guess this SDK refuses everywhere else, and getting it wrong silently merges two users' tool calls into one record. Keep the yielded id: it is the join key for any outcome that arrives later than the turn does. Production identifiers are exact 1–255-character strings with no leading/trailing whitespace or C0/C1 control characters. Because `turn_id` is also the on-disk directory name, `/`, `\`, `.` and `..` are additionally forbidden; the same path constraints apply to `agent_id` because the control plane addresses it as a URL segment. Absent or invalid `turn_id=` gets a fresh safe `t_` plus 16 hex characters, returned by the context manager; an invalid optional `conversation_id=` stays absent. A valid conversation id is written to a `conversation_id` file beside `tool_calls.jsonl` in the turn's directory.

**Give every turn of one conversation the same `conversation_id`.** Turns are joined into conversations by it, and how a conversation went on is part of what is learned from.

**The turn's text is yours to pass.** `input_text=` records what the user asked and `record_answer()` records what the agent finally said. They are your end user's message and the answer your code returned, facts only your own code holds, which no adapter can tell apart from the rest of a model call. They are separate calls because the answer does not exist yet when the turn opens. A turn with no recorded answer is held back by the 32-turn upload and leaves only at `close()` or exit, and the control plane keeps it but does not learn from it.

**`task_id=` is correlation evidence, not authority.** Pass the opaque id of your case, job, ticket, or work item when a later process needs to join this turn to that exact task. The SDK writes it beside the trajectory, adds it to the optional `task_id` field in `POST /turns`, and emits `percepteye.task.id` on the turn's OTel spans. It never parses the id, creates hints, judges the trace, or treats the turn as trainable. Empty, non-string, C0/C1-control-bearing, whitespace-padded, over-255-character, or conflicting values stay absent.

**Nothing in `turn()` can fail the turn.** If the directory cannot be created, capture is skipped and the body still runs.

`record_tool_call()` works inside a turn exactly as it does inside a practice task. With the `otel` extra installed, a turn's span carries `percepteye.turn.id` rather than `percepteye.rollout.id`, so your own telemetry can tell the two apart.

### What the model saw

With the message-text [permission](dashboard.md#allow-us-to-learn-from-its-conversations) on, the LangChain and Google ADK adapters also observe every model call inside a turn — its system instructions, its whole input, the tool definitions it was offered, what it answered — and every tool's result, and the turn uploads that as one OpenTelemetry GenAI trace in `genai_trace`: an `invoke_agent` span with a `chat` span per model call and an `execute_tool` span per tool call, content per the GenAI semantic conventions. The adapters are observers only — every callback returns nothing, so no call, response or result can be changed — and without the permission nothing of it is recorded, not even to disk. A grant made later is noticed on the upload cadence, and when the control plane refuses a trace by name the SDK stops recording at once. The control plane keeps a trace only for an agent that exactly one agent or product in your dashboard declares. Prime Agent runs as a `serve()` command and its JSON stream carries neither its system prompt nor its tool definitions, so no model content is captured for it.

### Execution identity

**Optional and fail-closed.** `attach()` cannot infer the agent whose call your surrounding application makes. Pass that same live Python agent as `execution_entrypoint=`, plus adapter-authored SHA-256 components that attest its otherwise-unobserved implementation and state, to enable canonical reconciliation. The SDK reuses the practice-task identity algorithm: prompt, tools, model, sub-agents, entrypoint, binding configuration, SDK/contract, and sorted opaque components all enter one domain-separated digest. Missing components, lossy discovery, a long truncated prompt without full-observation proof, or a conflicting turn id leaves `agent_fingerprint` absent. A persisted per-turn sidecar prevents a pending turn from an older process being relabelled with the new process's identity. The same exact digest appears on turn/tool OTel spans as `percepteye.agent.execution_sha256`; no policy identity is emitted because `attach().policy` is informational and does not prove what the application ran.

An argv command is deliberately refused for `execution_entrypoint`: attachment does not launch or enclose that process, so claiming a prepared child was the live agent would be false. Use `serve(execution_snapshot=...)` when the SDK owns the isolated command lifecycle.

### Uploading

Uploading is automatic once the agent has [permission](dashboard.md#allow-us-to-learn-from-its-conversations). Until then turns stay on your disk, and while the answer is no the SDK asks again (`GET /agents/{agent_id}/capture`, a plain read) before each batch it would send, so no restart is needed. Every 32 turns a short-lived daemon thread sends the answered ones; an `atexit` hook sends the remainder, including turns that never recorded an answer. `flush()` and `close()` are serialized against the automatic send, so a turn cannot be sent twice, and the wire admits at most 64 turns per request. Retries, rate limits and personal data are in [Production](production.md#retries-and-rate-limits).

**`PERCEPTEYE_AGENT_MODE` does not gate `attach()`.** That switch gates `serve()`, the call that does training work. `attach()` *is* the production surface, and gating it on the same variable would mean the correct setting for a production agent was `training`. To stop capture, don't call `attach()`, or pass `register=False` to keep it entirely local.

### Which capture am I in?

```python
match fw.current_capture():
    case ("rollout", rid): ...      # inside a serve() practice task, either isolation
    case ("turn", tid):    ...      # inside attach().turn(), in production
    case None:             ...      # neither
```

Three states, three answers, none collapsed into another.

!!! warning "`current_rollout_id()` is not a practice-versus-production test"
    Despite the name, inside `attach().turn()` it returns that **turn's** id, because a production turn records through the same path. So `if fw.current_rollout_id():` is **true in production**. Branch on `current_capture()` instead. The name and the behavior are kept because `record_tool_call()` reads this function to decide where to append.

## The approved model: `apply_current_policy()` and `policy_source()`

Both are for the agent serving your users, never for `serve()`: a practice task samples through the per-rollout gateway, which is what records its model calls, and pointing it at the policy endpoint would bypass that.

### One shot, at startup

```python
from openai import OpenAI

kw = fw.apply_current_policy(agent_id="support-agent")
if kw:                                    # {} means keep your current config
    client = OpenAI(base_url=kw["base_url"], api_key=kw["api_key"])
    model = kw["model"]
```

One `GET /policy/current`. It contributes nothing (no registration, no practice tasks, no capture), so it is the whole integration for a team that wants the approved model and nothing else.

| key | value |
|:--|:--|
| `base_url` | OpenAI-compatible base URL |
| `api_key` | resolved credential (see below) |
| `model` | the model name to send; `""` when the control plane did not name one |

**`{}` means keep your current configuration.** It is the answer for every reason you cannot act: nothing deployed, no `endpoint` in the body, no `base_url`, no credential to authenticate with, an address that is a stored file (`gs://`, `s3://`, `file://`) rather than a running endpoint, an unreachable control plane, or no key configured at all. The reason is logged on the `agent_flywheel` logger. **It never raises into your startup.**

The credential is resolved in this order, and never invented: the `api_key=` argument; the environment variable named by the response's `endpoint.api_key_env`; `PERCEPTEYE_API_KEY`. `set_env=True` additionally exports `OPENAI_BASE_URL` and `OPENAI_API_KEY` for libraries that read the environment; it does **not** export the model.

Underneath sit `current_policy(agent_id=...)`, the raw `/policy/current` body, and `endpoint_kwargs(body, api_key=None)`, the one place a body becomes client arguments; both `apply_current_policy()` and `policy_source()` go through it.

### A handle that keeps up

`apply_current_policy()` is one-shot, so a process runs the policy that was current when it started. The larger cost of that is rollback: a version withdrawn in the dashboard would keep serving until you redeploy. `policy_source()` is the transport for rollback first and promotion second.

```python
policy = fw.policy_source(agent_id="support-agent", ttl_s=300)

snap = policy.current()           # per turn, or per request
if snap.changed:                  # rebuild ONLY on an actual change
    client = OpenAI(base_url=snap.base_url, api_key=snap.api_key)
    model = snap.model
```

**Call `current()` per turn.** A caller who calls it once has re-created the problem the handle exists to solve.

| field | type | meaning |
|:--|:--|:--|
| `base_url` · `api_key` · `model` | `str` | what to run; empty when nothing is deployed or the control plane has never answered |
| `changed` | `bool` | `True` on the **one** snapshot that first reports a change; consumed by whoever observes it, so build one client per handle |
| `stale_for_s` | `float` | `0.0` while refreshes succeed; seconds **past the TTL** while they fail. The number to alert on |
| `last_refresh_ok` | `bool` | `False` after a failed attempt — the snapshot still stands |
| `serving_generation` | `str` | moves when *anything* about what to run moves, the prompt included; a fleet cycles a replica when it differs |

`bool(snap)` is truthy only when there is something to act on, and `snap.as_kwargs()` returns the same shape as `apply_current_policy()`. `on_change=` fires on promotion and rollback alike, never on a failed refresh. `PolicySource` is a context manager, and `close()` is idempotent.

| rule | value | why |
|:--|:--|:--|
| default TTL | `ttl_s=300.0` | |
| server hint | the body's `ttl_s` beats your `ttl_s=` | the control plane knows its own promotion cadence |
| floor | `MIN_TTL_S = 30.0` s, beats both | a misconfigured `ttl_s=1` across a fleet would flood the control plane |
| jitter | ±10%, drawn once per handle | replicas that boot together would otherwise refresh in lockstep |
| refresh budget | `REFRESH_TIMEOUT_S = 1.0` s, one attempt, no retries | it is spending a user's latency |

**Nothing polls in the background.** The refresh runs inside your `current()` call, on the request that observed staleness. Exactly one caller refreshes; everyone else gets the cached snapshot at once. **A failed refresh is a non-event:** the snapshot stands, so a control-plane blip can never move production back to the model you ran before. It never chooses an arm, never damps a flapping control plane, and has no `set_env`.

## The approved prompt: `current_prompt()` and `prompt_decision()`

```python
served = fw.current_prompt(agent_id="support-agent")
decision = fw.prompt_decision(served)     # may it be applied, and if not why
if decision.apply:
    system_prompt = decision.text         # never None here
else:
    log.info("keeping the current prompt: %s", decision.reason)
```

A separate `GET /prompt/current`, not folded into `/policy/current`: that one is polled, and a system prompt is unbounded text.

| field | meaning |
|:--|:--|
| `name` · `version` · `text` | the deliverable |
| `sha256` | prompt identity |
| `candidate_kind` · `status` | e.g. `"replacement"`, `"approved"` |
| `model_prompt_checksum` | the prompt identity stamped on the **model** when it was approved |
| `stale_baseline` | `true` when the prompt you supplied as the baseline has changed since |

The prompt and the model are approved as a **pair**. Do not re-derive the rule from those fields: `prompt_decision()` refuses for four reasons, each naming its own cause.

| it refuses when | because |
|:--|:--|
| there is no text to apply (`{}`, missing, or blank) | nothing has been approved, or the control plane could not answer; the prompt you have stands |
| `stale_baseline` is `true` | your own baseline prompt changed after the pair was approved |
| `status` is not `"approved"` or `"delivered"` | only a version a person approved is applied |
| `sha256` and `model_prompt_checksum` are both present and differ | the pair disagrees with itself |

`text` is `None` on every refusal rather than `""`, so assigning it can never blank the prompt your agent is running on. The Node SDK's `promptDecision` answers the same four questions with the same sentences, and [`tests/test_prompt_decision_parity.py`](https://github.com/percepteye-ai/agent-flywheel/blob/main/tests/test_prompt_decision_parity.py) drives one table of cases through both.

**The text is not applied for you.** Swapping a model endpoint is invisible to an agent's behavior; swapping its system prompt changes what it *does*. That is a decision for whoever owns the agent.

## Your copy on disk

Every captured turn is a directory under `trajectory_root`, and it is your copy: uploading does not delete it, and nothing else does unless you ask. Left alone it grows without bound — roughly **2.4 GB and a million inodes per million recording turns** at typical record sizes, and more if your tool outputs are large. A turn that records nothing costs nothing.

```python
px = fw.attach(agent_id="a", trajectory_root="/var/lib/percepteye",
               retention_turns=50_000)      # keep the most recent 50k turns

px.prune(keep_last=10_000)                  # or do it yourself, when you choose
px.prune(older_than_s=7 * 24 * 3600)
```

`retention_turns` prunes on an interval, one directory scan per 256 turns, so the real ceiling is `retention_turns + 256`. `prune()` with no argument deletes nothing.

Records are **flushed, never fsynced**. A flush puts the record in the OS page cache, which survives this process dying — a crashed agent, a `SIGKILL` on a blown deadline. `fsync` would also survive the machine losing power, at roughly 1–10 ms per record against the ~13 µs the whole write costs, so **a machine crash loses recent outcomes.**

## Practice tasks: `serve()`

```python
import agent_flywheel as fw

fw.serve("my_package.agent:build", agent_id="support-agent", input_shape="text")
```

`serve()` claims practice tasks, runs each one in a fresh child process, reports the result, and blocks until you stop it. It returns an `int`: the number of tasks that completed. Tasks arrive while a run that uses them is going, [started from your agent's page](dashboard.md#run-practice-tasks). It needs `PERCEPTEYE_API_KEY` (or `api_key=`), and refuses to start without it; the control-plane URL defaults to the production one, and `PERCEPTEYE_CONTROL_PLANE_URL` or `control_plane_url=` overrides it.

**Your agent must sample through an OpenAI-compatible client.** During a practice task its model calls go through a per-task gateway, which is what records them; the child's environment routes the client there. See [Configuration](configuration.md#isolation).

### Bound the first run

A practice task is a real run: it executes your agent, in your process tree, on your credentials, against whatever your tools talk to.

```python
fw.serve(
    "my_package.agent:build",
    agent_id="support-agent",
    input_shape="text",
    max_rollouts=8,
    artifacts_dir="/tmp/percepteye-first-run",
    on_event=lambda kind, ev: print(kind, ev),
)
```

| argument | default | what it does |
|:--|:--|:--|
| `max_rollouts` | `None` | stop after this many tasks **complete**, then return. A crash and a graded pass both count |
| `artifacts_dir` | `./.percepteye-artifacts` | created if absent; each task gets `_percepteye/<rollout_id>/` beneath it, deleted once the result is reported |
| `concurrency` | `1` | tasks in flight, under the default subprocess isolation |
| `rollout_timeout_s` | `900.0` | per-task wall clock; subprocess isolation is the only mode with a real timeout |

**`max_rollouts` is the only clean exit `serve()` has.** Nothing in this package installs a SIGINT or SIGTERM handler, so without it the loop runs until the process is killed. See [Stopping a running flywheel](configuration.md#stopping-a-running-flywheel).

### Misconfiguration raises before the first task

| step | raises when |
|:--|:--|
| resolve the mode | `PERCEPTEYE_AGENT_MODE` or `mode=` is a value it cannot parse |
| read the API key | neither `api_key=` nor `PERCEPTEYE_API_KEY` is set |
| create `artifacts_dir` | the path cannot be created |
| `GET /health` | the control plane does not answer, or does not speak `flywheel/1` |
| build the runner | the entrypoint does not import, does not resolve to a callable, or is incompatible with the isolation mode |
| `POST /agents` | registration fails |

One check warns rather than refuses: provider variables the gateway cannot serve (`AZURE_OPENAI_ENDPOINT`, `GOOGLE_API_KEY`, `GROQ_API_KEY` and the rest of `UNROUTABLE_PROVIDER_VARS`) are named at startup and removed from each task's child.

With `PERCEPTEYE_AGENT_MODE=production`, `serve()` returns `0` **before opening any connection** and prints one line saying so, so the same image can run in production without contributing. See [the mode switch](configuration.md#the-mode-switch).

### What you see

By default, nothing: wire `on_event`, because an empty queue and a broken claim route look identical from the outside.

```text
rollout_start {'rollout_id': 'r-1'}
rollout_reported {'rollout_id': 'r-1', 'grade': None, 'duplicate': False, 'exclusion': None}
claim_empty {'poll_after_ms': 500}
```

Every event is in [Configuration](configuration.md#watching-a-flywheel-on_event).

### A result is never silently lost

- **Spooled before the first send.** The result is written to a durable outbox under your `artifacts_dir` *before* it is POSTed, so a process killed mid-request still delivers. The next process over the same directory drains it, and **that drain runs before the next claim**, so a held result is never starved by new work.
- **A permanent refusal is retired, never deleted.** If the control plane refuses in a way redelivery cannot fix, the reason is written into the record and the file is renamed out of the retry set, with the task's trajectory kept beside it. Retired names are uniquified, so a second refusal cannot overwrite the first. After three such refusals in a row, `serve()` stops claiming. A result still undeliverable after seven days is kept and no longer retried.
- **Your agent's turn is bounded well inside the task.** `serve()` derives a per-call ceiling from the task budget and leaves a delivery reserve, so a provider that accepts a connection and never answers costs one timeout instead of the whole task.
- **No bearer credential is written to disk.** The lease token is held only in memory.

`outcome_spooled` and `outcome_undeliverable` name a result that is being held or has been retired.

### The entrypoint is an import path, not an object

The default isolation spawns one child process per task, and a child cannot receive a live Python object. So the entrypoint is a string the child can import. The parent resolves it at startup and the child re-resolves the same string per task, through one shared function.

| `"module:attr"` resolves to | what happens |
|:--|:--|
| a **class** | instantiated with no arguments. One that needs arguments raises `ConfigurationError` naming the missing one |
| a **zero-argument factory** | called; its return value is the agent |
| an **already-built object** with `invoke` / `run` / `execute` / `chat` | used as-is, never called |
| a **function taking the task** | called as a factory, so it raises `TypeError` — wrap it in a class or a factory |

That last row is the sharp edge. `"my_package.agent:answer"`, where `answer(task)` is the agent itself, is indistinguishable from a factory at import time and is treated as one.

A constructed object is accepted directly, but only under `isolation="inprocess"`, which is capped at `concurrency=1` and has no real timeout. Both constraints are checked at startup.

### Which method gets called

Probed in order, first match wins: `invoke`, `run`, `execute`, `chat`, `__call__`. Pass `method="handle"` to name one instead. `async def` is detected and awaited.

**Resolution happens once, at startup, and raises there.** A task that never emits because a method name was wrong biases the score of every sibling it is compared against, so a typo must stop the process in front of the person who made it. The error names what was available:

```text
no entrypoint found on Nothing. Tried, in order: invoke, run, execute, chat,
__call__. Available callables: helper.
Pass method='...' to name one explicitly, or pass a plain callable.
```

### `input_shape`

Declared, never inferred.

| `input_shape` | your entrypoint receives |
|:--|:--|
| `"text"` | `"the task description"` |
| `"messages"` | `[{"role": "user", "content": "the task description"}]` |
| `"dict"` | `{"input": "the task description"}` |

Any other value raises `EntrypointError` at startup.

## Adapters

`adapters="auto"` is the default for both `attach()` and `serve()`. It walks the built-in registry, probes each one, and binds those whose framework is actually installed. An absent framework is skipped and recorded, never an error.

| adapter | distribution | versions | covers |
|:--|:--|:--|:--|
| `langchain` | `langchain-core` | `>=0.3,<2` | LangChain **and** LangGraph |
| `adk` | `google-adk` | `>=1,<3` | Google ADK |

There is no separate LangGraph adapter. LangGraph tool calls run through langchain-core's own callback system, so binding `langchain` captures both; what LangGraph adds is the node name.

### What each one can honestly report

| framework | `ok` | `failed` | `unknown` | call attributed to |
|:--|:--|:--|:--|:--|
| **LangChain** | a `ToolMessage` with `status="success"` | the tool raised, or a `ToolMessage` with `status="error"` | every other return value | an `agent:`-prefixed run tag |
| **LangGraph** | same | same | same | the executing node, from `langgraph_node` run metadata |
| **Google ADK** | **never reported** | the tool raised, or it returned a dict with a truthy `"error"` key | every other return value | `ToolContext.agent_name`, natively |

`status_code` is always `null` from an adapter. Neither framework surfaces an HTTP status at the tool boundary, and recovering one from the payload is exactly how an unobserved call becomes a landed one.

**ADK has no `ok` path, and that is the design.** A returned value proves the tool did not raise; it does not prove the tool worked. ADK's compensation is the better half of the trade: `on_tool_error_callback` separates *raised* from *returned*, so a genuine `failed` is available where LangChain's `on_tool_end` fires for success and swallowed failure alike.

Adapters are best-effort acceleration over [`record_tool_call`](outcomes.md), never a substitute for it. Where you know the answer, say it.

### Adapters add no dependencies

No adapter imports its framework at module scope. Importing the SDK, and even constructing both built-in adapters, pulls in neither `langchain_core` nor `google.adk` — the framework import happens inside `bind()`, and the probe only reaches an import after package metadata has already confirmed the distribution is installed. The runtime dependency list stays at one (`httpx`), asserted by a clean-install CI job.

### `[]`, `"auto"`, and a name are three different instructions

| `adapters=` | behavior |
|:--|:--|
| `"auto"` (default) | bind every built-in whose framework is installed; skip and record the rest |
| `[]` | bind nothing. A real choice, and it must be silent rather than half-instrumented |
| `["langchain"]` | **require** it. A named adapter that cannot bind raises `RoutingNotInstallable` |
| `["nosuchname"]` | `KeyError`, listing the built-ins that do exist |

Naming an adapter is an assertion that it is how your calls get captured, so it fails closed. `"auto"` asserts nothing, so it degrades quietly:

```text
[AdapterFrameworkAbsent] adk: distribution 'google-adk' is not installed

  remedy: install google-adk, or remove the adapter from serve(adapters=[...])
```

`RoutingNotInstallable` carries a machine-readable `reason` — `AdapterFrameworkAbsent`, `VersionOutOfRange` or `SymbolMissing` — plus the evidence and a printed remedy. The version gate is soft: `packaging` is not a runtime dependency, so when it is unavailable the range is skipped and the symbol probe decides. That probe resolves exactly the tuple of `module:attr` paths the adapter will later touch, which is what makes detection and binding inseparable — detection cannot pass while binding is broken.

!!! warning "Under `serve()`, adapters bind in the child, not at startup"

    `serve()` validates the entrypoint, credential and contract version before the first task, but it does **not** probe adapters there. Binding happens inside the task's child, because a callback registered in the parent would never see a child's tool calls. So a named adapter that cannot bind surfaces per task: the child exits non-zero and the task is reported as the typed exclusion `flywheel_crash`, carrying the `RoutingNotInstallable` message in its detail. Under `attach()` the opposite rule applies — a binding failure is swallowed, because a production turn with no capture is still a served user.

## Zero instrumentation, worked end to end

This agent contains no reference to Agent Flywheel of any kind:

```python
# lc_agent.py — your existing agent, untouched
from langchain_core.tools import tool


@tool
def lookup(q: str) -> str:
    """Look something up."""
    return f"found: {q}"


class Agent:
    tools = [lookup]

    def invoke(self, task: str) -> str:
        return str(lookup.invoke({"q": task}))


def build():
    return Agent()
```

Served with:

```python
import agent_flywheel as fw

fw.serve("lc_agent:build", agent_id="research-agent", input_shape="text")
```

On a practice task whose text is `"count to three"`, the reported trajectory is:

```json
"tool_calls": [
  {
    "name": "lookup",
    "arguments": {"q": "count to three"},
    "outcome": "unknown",
    "status_code": null,
    "output": "found: count to three",
    "latency_ms": 0.226
  }
]
```

**`unknown`, not `ok`.** The tool returned a string and nothing observed whether the call succeeded. `Tool.invoke({"q": ...})` with a plain dict has no tool-call id to attach, so LangChain returns raw content rather than a `ToolMessage` — and raw content from a tool that worked is byte-identical to raw content from a tool that caught its own exception and returned prose. Calling that `ok` is the single most consequential lie a capture can tell, so the adapter does not.

That is the honest floor zero instrumentation buys you, and it is enforced by [`tests/test_zero_instrumentation.py`](https://github.com/percepteye-ai/agent-flywheel/blob/main/tests/test_zero_instrumentation.py) inside the 1347-test suite. Drive the same tool through a model-issued tool call and the same adapter reports `"outcome": "ok"` with the model's `tool_call_id` attached, because then there is a `ToolMessage` with a real `status`.

One call where you *do* know the answer gets you the rest:

```python
from agent_flywheel import record_tool_call

record_tool_call("charge_card", {"amount": 4200}, outcome="ok", status_code=201)
```

`outcome` and `status_code` are keyword-only and required, so half an outcome is unconstructible. Outside a practice task or a turn the call is a no-op, which is why the line can stay in a shared code path. See [Outcomes](outcomes.md).

??? note "Routing a chat model you constructed yourself"

    `LangChainAdapter(llm=...)` re-points a chat model at the per-task gateway and reads the value back to check it took. It exists because `ChatOpenAI(base_url=...)` in your own code beats every environment variable: the task completes, the completions go to a live third party, and nothing distinguishes it from a clean run. A client whose endpoint is fixed at construction raises `RoutingNotInstallable(reason="HardcodedBaseUrl")`.

    This is reachable only by passing an adapter **instance**, which means `isolation="inprocess"` — the subprocess payload is JSON, and an adapter object cannot cross it. `adapters=["langchain"]` constructs the adapter with `llm=None`, which touches routing not at all. The ordinary arrangement is the default one: the child's environment routes, because the agent is constructed inside a child that already has those variables set. Build your model inside the factory and it is handled for you.

## Shipping your own adapter

The bar is the one the built-ins meet: no framework import at module scope, report `unknown` rather than guessing, and fail loudly at bind time rather than degrade silently. An adapter is a small object — `name`, `distribution`, `supported`, `required_symbols`, `capabilities()`, and a `bind()` that returns an unbind callable.

Be aware of where the wiring currently stops. `serve()` and `attach()` resolve adapter **names** against the built-in registry only; the `agent_flywheel.adapters` entry-point group is read by `binding.discover_adapters()`, and no public entry point calls it. So today a third-party adapter distribution is not picked up by `adapters="auto"`, and its name is not accepted in `adapters=[...]`. Reach it by passing the instance under `isolation="inprocess"`, or open an issue — a framework worth an adapter is worth wiring properly.

## Related

| page | what it covers |
|:--|:--|
| [Production](production.md) | what is sent and when, retries, rate limits, personal data |
| [Outcomes](outcomes.md) | `ok` · `failed` · `unknown`, and why `unknown` is never rounded up |
| [Configuration](configuration.md) | `PERCEPTEYE_AGENT_MODE`, isolation, concurrency, `on_event`, every environment variable |
| [The contract](contract.md) | the wire format the adapters write into |
