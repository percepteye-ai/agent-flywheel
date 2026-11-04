# Production

Your production agent records the conversations it serves, sends them to us once you allow it, and runs the version you approved. This page covers what is sent and when, what happens when a request fails, and what we do with personal data. Setting it up is the [Quickstart](quickstart.md).

## Recording conversations

<!-- lang -->
`attach()` records each turn your agent serves: what the user asked (`input_text=`), every tool call and its outcome, and the answer (`record_answer()`). For a LangChain, LangGraph or Google ADK agent it also records each model call inside the turn: the system prompt, the whole conversation so far, the tool definitions and every tool result. That is what we learn from, and the turn sends it as one [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) trace. An agent on another framework records tool outcomes and the turn's text, but no model calls; to learn from it, [push its OpenTelemetry traces](otlp.md).

The adapters only observe. No model call, response or tool result is changed, and nothing in `turn()` can fail the turn: if capture breaks, the turn still runs.
<!-- /lang -->

Two [permissions](dashboard.md#allow-us-to-learn-from-its-conversations) decide what leaves your machine: the conversations, and their message text. Without the second, what the model saw is not recorded at all, not even on your disk. Learning needs both.

## When turns are sent

<!-- lang -->
Every 32 turns, a short-lived background thread sends the turns that have an answer. When the process exits, the rest go too, including turns that never recorded one. `px.flush()` sends now. A request carries at most 64 turns, and the network is never on your request path.
<!-- /lang -->

A turn that could not be sent stays on disk and goes with a later batch. Each batch carries an idempotency key, so a batch that arrives twice is stored once.

**`{}` means keep your current configuration.** It is the answer for every reason you cannot act, and they deliberately read the same: nothing deployed, a model the control plane has withdrawn (`withdrawn: true`), no `endpoint` in the body, no `base_url` or `model`, no credential to authenticate with, an unreachable control plane, or no key configured at all. An endpoint you cannot authenticate to is not usable, so it is returned as "nothing to apply" here rather than as 401s at your first request.

Every request to the control plane gets up to four attempts, with a 30-second timeout each.

- **`429`** is always retried, after the server's `Retry-After` (at most 8 seconds), or after a short random backoff when there is none.
- **`408`, `500`, `502`, `503` and `504`, and a connection that drops mid-request,** are retried, except for a claim for practice tasks: that is resent only when it never reached us, because a claim that did may already have handed work out.
- **Any other `4xx`** is final; sending it again would get the same answer. A `403` on upload is a refusal, such as a permission switched off or a key that may not carry this agent's conversations: the SDK stops uploading, says why, keeps the turns on disk, and starts again once the permission is back.

| order | source |
|:--|:--|
| 1 | `endpoint_kwargs(body, api_key=...)`: your own credential for the endpoint |
| 2 | the response's own `endpoint.api_key`, when the control plane sends one |
| 3 | the env var named by the response's `endpoint.api_key_env`; `PERCEPTEYE_API_KEY` there is sent only to an endpoint at the control plane's own address, never as a fallback |

## Personal data

The SDK sends what your agent saw and said as it was; it masks nothing on your side. The control plane scans everything it receives for email addresses, phone numbers, card numbers, US social security numbers and IP addresses before it stores it:

- **In the tool-call records** sent beside each turn, a match is masked, as `[redacted:email]` for example.
- **In the user's message, your agent's answer, and everything the model saw**, a match is kept as sent, because masking would change the conversation. The turn is marked as holding personal data and is never learned from. [Your agent's page](dashboard.md#see-what-arrived) counts these turns.

[Pushed traces](otlp.md) follow the same rule. How long we keep it all is in the [FAQ](faq.md#what-data-do-you-keep-and-for-how-long).

## What stays on your machine

<!-- lang -->
Every turn is a directory under the `trajectory_root` you named, and uploading does not delete it. Nothing does unless you ask: `attach(retention_turns=50_000)` keeps the most recent 50,000 turns, and `px.prune(keep_last=...)` or `px.prune(older_than_s=...)` prunes when you choose. Left alone, it grows by roughly 2.4 GB per million turns, more if your tool outputs are large.

Records are flushed, not fsynced: they survive your process dying, and a machine that loses power can lose the most recent ones.
<!-- /lang -->

## Run the approved version

<!-- lang -->
The SDK hands you the approved model and prompt; applying them is your code's decision.

| call | when | what it returns |
|:--|:--|:--|
| `fw.apply_current_policy(agent_id=...)` | once, at startup | `{"base_url", "api_key", "model"}` for your client, or `{}` |
| `fw.policy_source(agent_id=...)` | `.current()` on each turn | a snapshot; `.changed` is true when you must rebuild your client |
| `fw.prompt_decision(fw.current_prompt(agent_id=...))` | when you load your prompt | `.apply`, `.text`, and `.reason` when it may not be applied |

`policy_source()` refreshes once its answer is about five minutes old, or sooner when the control plane asks, but never more often than every 30 seconds. A refresh is one attempt with a one-second budget, made inside your `current()` call; no thread polls in the background. When a refresh fails, it keeps serving the last answer, so a network blip never moves you back to your old model. Watch `snap.stale_for_s`: it counts the seconds past the refresh interval while refreshes fail.

`prompt_decision()` refuses when there is no approved text, when your own baseline prompt changed after the pair was approved, when the status is not one a person approved, or when the prompt was approved against a different checkpoint. `.text` is `None` on a refusal, never `""`, so applying it cannot blank your prompt.
<!-- /lang -->

Until you approve a new version in [Decisions](dashboard.md#approve-the-winner-in-decisions), your agent keeps the one it has, and you can [try a new version on one agent](dashboard.md#try-a-new-version-on-one-agent) first. An empty answer always means "keep what you run", never "run with nothing".

## When the control plane is unreachable

Your agent keeps serving. Recording carries on to disk, turns wait for a later batch, and your agent keeps the model and prompt it has. Nothing on this page raises into your process because the control plane did not answer.

## Reference

It fires on **promotion and rollback alike** — any move in the `(base_url, api_key, model)` triple, including the first actionable answer, since without that the documented `if snap.changed:` usage would never build a client at all. It does **not** fire when a refresh fails, and it does not fire when the control plane answers "nothing deployed" without `withdrawn: true`. A withdrawal (`withdrawn: true`, which `is_withdrawn(body)` tests) after a served policy fires it with an empty snapshot and `snap.withdrawn` set: go back to your own configuration. Exceptions it raises are swallowed.

### Nothing polls in the background

There is no thread, timer or task behind the POLICY read. The refresh runs **inside your `current()` call**, on the request that observed staleness — which is why its budget is one attempt and one second. A retry ladder would spend several users' latency, and the next stale observation retries naturally.

The one thread this package does start is `attach()`'s turn upload, and it is not a poller either: when capture is enabled it runs a short-lived daemon thread every `_FLUSH_EVERY` turns, triggered by your own traffic, so the network is never on the request path. It is joined by nothing and holds your interpreter open at exit for nothing — the final send is an `atexit` hook.

| rule | value | why |
|:--|:--|:--|
| default TTL | `ttl_s=300.0` | |
| server hint | the body's `ttl_s` beats your `ttl_s=` | the control plane knows its own promotion cadence and you do not |
| floor | `MIN_TTL_S = 30.0` s, beats both | a misconfigured `ttl_s=1` across a fleet is a denial of service aimed at our own control plane |
| jitter | ±10%, drawn **once** at construction | a fleet of replicas that boots together shares a TTL and would re-poll in lockstep, delivering a synchronized spike every TTL |
| refresh budget | `REFRESH_TIMEOUT_S = 1.0` s, one attempt, no retries | it is spending a user's latency |

Jitter is decided once per handle rather than per refresh, because the problem is separating handles from each other, not smearing one handle across the window. A server hint of `5` still polls at 30 s; a hint of `60` against `ttl_s=300` polls at roughly 60 s.

Compare the one-shot functions, which build their client with the transport defaults — 30 s and up to four attempts. They run at startup and can afford it; the handle runs on a user's request path and cannot.

### Serve-stale, always

After the first call, `current()` returns the cached snapshot. **A failed refresh is a non-event**: the snapshot stands, `changed` stays `False`, `last_refresh_ok` goes `False`, and `stale_for_s` accumulates visibly.

The naive reading of "fail open" would be to return the empty answer meaning "keep your current configuration" — but a handle that returns empty after having *had* a policy makes `changed` fire and the caller rebuild against the base model. **A control-plane blip would silently demote production from the trained model back to the untrained one**, which is strictly worse than serving a slightly stale policy.

**No stampede.** Exactly one caller refreshes; everyone else is handed the cached snapshot immediately rather than queueing behind the lock. Fifty concurrent turns cost one request and forty-nine cache reads. Only the very first call can block, because there is nothing cached to serve instead — and even it fails open.

`PolicySource` is a context manager, and `close()` releases the pooled connection and is idempotent.

??? note "What `policy_source()` will not do"
    | | |
    |:--|:--|
    | **It never chooses an arm.** | The agent is *told* its arm, which is what keeps results attributable. No client-side splitting, no fallback picking, no dice. |
    | **It never damps a flapping control plane.** | If the assignment flips, the handle flips. Second-guessing an assignment is how a client stops being a faithful reader of one, and damping belongs on the side that knows why the assignment moved. |
    | **It has no `set_env`.** | `apply_current_policy(set_env=True)` mutates the process environment once, at a moment you chose. Re-exporting `OPENAI_BASE_URL` on a timer is that same global mutation on a schedule, landing under whatever else the process happens to be doing. Read `snap.base_url`. |
    | **It is never for `serve()`.** | See the warning at the top of this page. |

## The prompt

Training produces a better model *and* a better system prompt, co-optimized and gated by the same human approval.

```python
served = fw.current_prompt(agent_id="my-coding-agent")
decision = fw.prompt_decision(served)     # may it be applied, and if not why
if decision.apply:
    system_prompt = decision.text         # never None here
else:
    log.info("keeping the current prompt: %s", decision.reason)
```

A separate `GET /prompt/current`, deliberately not folded into `/policy/current`: that one is polled and identity-sized, and a system prompt is unbounded text, so putting it in the poll would make every tick carry a payload the caller already has.

| field | meaning |
|:--|:--|
| `name` · `version` · `text` | the deliverable |
| `sha256` | prompt identity — read this to tell whether it differs from what you run |
| `candidate_kind` · `status` | e.g. `"replacement"`, `"approved"` |
| `model_prompt_checksum` | the prompt identity stamped on the **model** at certification |
| `stale_baseline` | `true` when the prompt you supplied as the baseline has changed since this was certified |

The contract is a **pair**: this prompt was approved against that checkpoint, and applying it over a different one is not the thing that was approved. `stale_baseline` does not invalidate the deliverable — it was just shaped against a version of your prompt you no longer ship.

**Do not re-derive the rule from those fields — ask `prompt_decision()`.** It is the whole predicate in one call, and it refuses for four distinct reasons, each naming its own cause:

| it refuses when | because |
|:--|:--|
| there is no text to apply (`{}`, missing, or blank) | nothing has passed the approval gate, or the control plane could not answer — the prompt you already have stands |
| `stale_baseline` is `true` | your own baseline prompt changed after the pair was certified, so the approved text was shaped against a prompt you no longer ship. Re-certification is manual |
| `status` is not `"approved"` or `"delivered"` | only a bundle a person approved is applied |
| `sha256` and `model_prompt_checksum` are both present and differ | the bundle disagrees with itself, so it is not the pair that was validated |

`apply` is false unless all four hold; `reason` is the sentence explaining which one failed; `text` is `None` on every refusal rather than `""`, so assigning it can never blank the prompt your agent is running on. A body that is missing, null or not a mapping refuses rather than raising — nothing here throws into your process. The Node SDK's `promptDecision` answers the same four questions with the same sentences, and one parity test drives a table of cases through both and asserts every field agrees.

**The text is still not applied for you, and that is the difference from `apply_current_policy()`.** Swapping a model endpoint is invisible to an agent's behavior; swapping its system prompt changes what it *does*. That is a decision for whoever owns the agent, so this hands back the text and the verdict on it and stops.

`{}` means **keep the prompt you have** — nothing has passed the approval gate, or the control plane could not answer. It never means "use no prompt".

## `attach()` — evidence from live turns

`serve()` does training work. `attach()` records what your agent already did, from turns you are already serving — same [adapters](python.md), same tri-state [outcomes](outcomes.md), same trajectory format, same absence contract, scoped around a live user turn instead of a claimed rollout.

```python
px = fw.attach(agent_id="support-agent", trajectory_root="/var/lib/percepteye")

with px.turn(task_id=case_id, conversation_id=conv_id,
             input_text=user_text) as turn_id:
    reply = agent.invoke(user_text)       # your agent, unmodified

px.record_answer(turn_id, reply)          # after the scope closes — the answer exists now
```

**It captures. It never reroutes.** That is a property of how the adapters are bound, not a promise about the code: `attach()` binds them with `llm_url=""` and `api_key=""`, and both built-ins treat "no endpoint" as "do not touch routing". Bound by name, `LangChainAdapter` holds no chat model and its `_route()` returns at once; the ADK adapter never reads the URL at all. Attaching cannot re-point a single byte of production traffic.

| parameter | default | notes |
|:--|:--|:--|
| `agent_id` | required | exact 1–255-character production identifier; because it is a URL segment, path separators and `.` / `..` are forbidden; invalid values raise `ConfigurationError` before capture starts |
| `trajectory_root` | **required** | see below |
| `adapters` | `"auto"` | binds every built-in whose framework is installed |
| `register` | `True` | `False` skips the control-plane call entirely |
| `control_plane_url` · `api_key` | env | `PERCEPTEYE_CONTROL_PLANE_URL` / `PERCEPTEYE_API_KEY` |
| `execution_entrypoint` | `None` | the live Python agent/import spec to describe for exact execution identity |
| `execution_input_shape` · `execution_method` | `"text"` · `None` | the declared call surface of `execution_entrypoint` |
| `attested_execution_components` | `None` | opaque exact adapter evidence for implementation/state discovery cannot observe |
| `describe` | `None` | the agent's own `system_prompt` / `tools` / `model` — without `execution_entrypoint`, the whole description this registration sends ([`describe=`](how-it-works.md#an-agent-nothing-can-read-describe)) |
| `rules` | `None` | the Product's workflow rules, declared once after the registration, exactly as for [`serve()`](configuration.md#a-products-workflow-rules-rules); `None` sends nothing, `[]` retracts |

**`trajectory_root` is required and that is deliberate.** Every captured turn lands there as a directory, and it is your copy: uploading does not delete it, and without `retention_turns` nothing does. Naming the directory is how **retention becomes a decision somebody made** rather than a surprise. It is yours.

With `register=True` the call makes exactly two outbound requests, once, at startup: `POST /agents` then `GET /policy/current` — and a third, `POST /agents/{agent_id}/rulebook`, only when `rules=` is given, sent in the background after the policy read: `attach()` returns without waiting for its answer. It registers as `mode: "production"` — `serve()` sends `"rollout"`, and a training donor versus a production reader is the one distinction a `mode` on a registration exists to draw. `register=False` skips every one of them, for a process that should make no outbound connection at all.

**Without `execution_entrypoint` there is no agent object to describe, and the registration says so:** `discovered_agent: null` — unless you declared one with `describe=`. A declaration that leaves `tools` out does not erase the tools your `serve()` registration stored: the control plane keeps a catalogue that a later description did not observe. Either way the registration does not omit the key — the control plane reads an absent key as you opting out of introspection, records the agent as `disabled`, and a description your `serve()` registration stored stops driving workflow generation. Your real opt-out, `PERCEPTEYE_INTROSPECT=0`, still omits it.

**Everything fails open.** An unreachable control plane yields an attachment with `policy == {}` and costs you nothing else; a framework that cannot be bound yields `adapters_bound == ()` rather than a process that will not start. (`serve()` raises there instead, and correctly: a rollout with no capture is a wasted rollout, while a production turn with no capture is still a served user.) A registration the control plane *refuses* while the policy read succeeds emits a `RuntimeWarning` and nothing more — the two calls no longer share one `except`, so a payload intake dislikes cannot also demote you to the untrained policy. Rules it refuses, cannot hear, or answers slowly cost the rules and nothing else — not the policy read, not the capture verdict, not the time `attach()` takes: `px.rulebook` stays `None` and `px.rulebook_refusal` says why, in the control plane's own words — a rule it will not accept (named by cause and index), an agent no single Product declares, or no answer at all. It says "not registered" only when that is known: a failure after the rules may have landed (a read timeout, a 502 from a proxy) is named as unknown, and declaring the same rules again is safe either way. Until the control plane answers, the sentence says so, and `px.wait_for_rulebook(timeout)` waits for the answer; `close()` gives an answer still on its way up to five seconds. With `register=False`, or no URL or key, the sentence names that instead.

| on the `Attachment` | |
|:--|:--|
| `.turn(turn_id=None, conversation_id=None, task_id=None, input_text=None)` | context manager; yields the turn id |
| `.record_answer(turn_id, text)` | what the agent finally answered; callable **after** the scope closes |
| `.read(turn_id)` | that turn's `list[ToolCallOutcome]`, or `None` if nothing was recorded |
| `.read_turn(turn_id)` | the `task` / `answer` text recorded for that turn, or `None` |
| `.flush(include_unanswered=False, batch_size=64)` | uploads now; larger requested batches are split at the contract's 64-turn limit |
| `.policy` | what the control plane said to run — **informational here**; `attach()` does not act on it |
| `.agent_fingerprint` | `{"execution_sha256": ...}` only when SDK reconciliation was exact; otherwise `None` |
| `.adapters_bound` | e.g. `("adk", "langchain")`; `()` is a real answer, not a failure |
| `.rulebook` | the control plane's receipt for `rules=` — `status` (`registered` · `retracted` · `unchanged`), `version`, `content_sha256`, `rule_count`, `product_id` — or `None` |
| `.rulebook_refusal` | why the rules passed as `rules=` are not confirmed in force — including "not answered yet" — or `None` when they registered or none were passed |
| `.wait_for_rulebook(timeout=None)` | waits for the control plane's answer to `rules=`; `True` once `.rulebook` / `.rulebook_refusal` say what it did |
| `.close()` | unbinds the adapters; idempotent, never raises. Also a context manager |

**A turn boundary is declared, not inferred.** Guessing where one request ends and the next begins is the same class of guess this SDK refuses everywhere else, and getting it wrong silently merges two users' tool calls into one record. Keep the yielded id: it is the join key for any outcome that arrives later than the turn does — the draft kept an hour afterwards, the pull request merged the next morning. Production identifiers are exact 1–255-character strings with no leading/trailing whitespace or C0/C1 control characters. Because `turn_id` is also the on-disk directory name, `/`, `\`, `.` and `..` are additionally forbidden; the same path constraints apply to `agent_id` because the control plane addresses it as a URL segment. Absent or invalid `turn_id=` gets a fresh safe `t_` plus 16 hex characters, returned by the context manager; an invalid optional `conversation_id=` stays absent. A valid conversation id is written to a `conversation_id` file beside `tool_calls.jsonl` in the turn's directory. Remaining agent-id characters are percent-encoded at URL and idempotency-header seams instead of imposing any one agent product's slug grammar.

**`task_id=` is correlation evidence, not authority.** Pass the opaque id of
your case, job, ticket, or work item when a later control-plane process needs
to join this turn to that exact task. The SDK writes it beside the trajectory,
adds it to the optional `task_id` field in `POST /turns`, and emits
`percepteye.task.id` on the turn's OTel spans. It never parses the id, creates
hints, judges the trace, or treats the turn as trainable. Empty, non-string,
C0/C1-control-bearing, whitespace-padded, over-255-character, or conflicting values
stay absent; control-plane clustering and dataset services validate the
binding before any downstream use.

**Nothing in `turn()` can fail the turn.** If the directory cannot be created, capture is skipped and the body still runs.

**The turn's text is opt-in, and both halves are yours to pass.** `input_text=` records what the user asked and `record_answer()` records what the agent finally said. They are your end user's message and the answer your code returned — facts only your own code holds, which no adapter can tell apart from the rest of a model call. That is what keeps tool outcomes zero-instrumentation while turn text stays a decision you make.

**What the model saw is a second consent tier, off by default.** When an operator grants the agent model-content capture, the LangChain and Google ADK adapters also observe every model call inside a turn — its system instructions, its whole input, the tool definitions it was offered, what it answered — and every tool's result, and the turn uploads that as one OpenTelemetry GenAI trace in `genai_trace`: an `invoke_agent` span with a `chat` span per model call and an `execute_tool` span per tool call, content per the GenAI semantic conventions. The adapters are observers only — every callback returns nothing, so no call, response or result can be changed — and without the grant nothing of it is recorded, not even to disk. `px.may_capture_model_content` is the live answer and `px.model_content_refusal` the sentence explaining a `False`; a grant made later is noticed on the flush cadence, and when the control plane refuses a trace by name the SDK stops recording at once. Prime Agent runs as a `serve()` command and its JSON stream carries neither its system prompt nor its tool definitions, so no model content is captured for it.

They are separate calls because the answer does not exist yet when the turn opens — `record_answer()` is callable after the scope has closed, for the same reason `turn()` yields its id. **Omit them and the periodic flush sends nothing**: a turn with no recorded answer is held back on every 32-turn flush and leaves only at `close()` or the `atexit` hook, carrying no text at all.

`record_tool_call()` works inside a turn exactly as it does inside a rollout. With the `otel` extra installed, a turn's span carries `percepteye.turn.id` rather than `percepteye.rollout.id`, so your own telemetry can tell the two apart too. When supplied, `percepteye.task.id` is also carried on both chat and tool-call spans, so either OTLP ingestion or the turn upload has the same neutral correlation evidence.

**Execution identity is optional and fail-closed.** `attach()` cannot infer the
agent whose call your surrounding application makes. Pass that same live Python
agent as `execution_entrypoint=`, plus adapter-authored SHA-256 components that
attest its otherwise-unobserved implementation and state, to enable canonical
reconciliation. The SDK reuses the rollout identity algorithm: prompt, tools,
model, sub-agents, entrypoint, binding configuration, SDK/contract, and sorted
opaque components all enter one domain-separated digest. Missing components,
lossy discovery, a long truncated prompt without full-observation proof, or a
conflicting turn id leaves `agent_fingerprint` absent. A persisted per-turn
sidecar prevents a pending turn from an older process being relabelled with the
new process's identity. The same exact digest appears on turn/tool OTel spans as
`percepteye.agent.execution_sha256`; no policy identity is emitted because
`attach().policy` is informational and does not prove what the application ran.

An argv command is deliberately refused for `execution_entrypoint`: attachment
does not launch or enclose that process, so claiming a prepared child was the
live agent would be false. Use `serve(execution_snapshot=...)` when the SDK owns
the isolated command lifecycle.

**`PERCEPTEYE_AGENT_MODE` does not gate `attach()`.** That switch gates `serve()`, the call that does training work. `attach()` *is* the production surface, and gating it on the same variable would mean the correct setting for a production agent was `PERCEPTEYE_AGENT_MODE=training`. To stop capture, don't call `attach()`, or pass `register=False` to keep it entirely local. See [Configuration](configuration.md).

**What it ships, and when.** `flywheel/1` gained `POST /turns`, so captured turns
are uploaded — subject to one gate and one cadence.

The gate is **per-agent capture consent, and it is off by default**: a production
turn carries your end users' own messages, so nothing is sent until an operator
enables capture for that agent. The registration response tells the SDK whether
it may, which is why it never finds out by uploading. `px.may_upload` is that
answer and `px.upload_refusal` is the sentence explaining a `False`.

**Enabling it does not require restarting your agent.** While the answer is no,
the SDK re-reads the verdict from `GET /agents/{agent_id}/capture` on its flush
cadence and begins uploading as soon as an operator says yes. That read is a
plain `GET`: it changes nothing, which is precisely why it can be repeated —
the verdict used to be reachable only through a registration, and registering
to ask a permission question had side effects of its own.

The cadence is **every 32 turns, on a short-lived daemon thread**, plus an
`atexit` hook that sends the remainder — including turns that never recorded an
answer, since at shutdown none is coming. Uploading is therefore automatic; you
do not have to call anything. `flush()` and `close()` remain available and are
serialized against the automatic flush, so a turn cannot be sent twice. The
published wire admits at most 64 turns per request; `flush(batch_size=...)`
therefore caps larger requested chunks at 64 rather than emitting an invalid
batch.

Turns that have not been uploaded — no consent, an unreachable control plane, a
low-traffic agent that has not reached the cadence — stay as trajectory
directories under the root you name, and you own them.

## Which capture am I in?

```python
match fw.current_capture():
    case ("rollout", rid): ...      # inside a serve() rollout, either isolation
    case ("turn", tid):    ...      # inside attach().turn(), in production
    case None:             ...      # neither
```

Three states, three answers, none collapsed into another. A `bool` would have been the smaller API and the wrong one: it answers "rollout?" with `False` for both production and no-capture-at-all — the same absence-as-zero collapse this package refuses for `outcome="unknown"` and for `tool_calls=None` versus `[]`.

!!! warning "`current_rollout_id()` is not a rollout-versus-production test"
    Despite the name, inside `attach().turn()` it returns that **turn's** id, because a production turn records through the same path. So `if fw.current_rollout_id():` is **true in production** and takes the rollout branch. Branch on `current_capture()` instead.

    The name and the behavior are kept because `record_tool_call()` reads this function to decide where to append — narrowing it would silently disable production capture, trading a naming bug for a data-loss one.

There is no prefix convention to lean on instead: a rollout id is opaque by contract, and you may pass your own `turn_id` to `Attachment.turn()`.

## Retention

The trajectory is append-only and nothing else removes it. Left alone it grows
without bound — roughly **2.4 GB and a million inodes per million recording
turns** at typical record sizes, and considerably more if your tool outputs are
large, because `output` is stored verbatim.

A turn that records nothing costs nothing: the directory is created by the first
write, not by entering the turn.

```python
px = fw.attach(agent_id="a", trajectory_root="/var/lib/percepteye",
               retention_turns=50_000)      # keep the most recent 50k turns

px.prune(keep_last=10_000)                  # or do it yourself, when you choose
px.prune(older_than_s=7 * 24 * 3600)
```

`retention_turns` prunes on an interval — one directory scan per 256 turns
rather than one per turn — so the real ceiling is `retention_turns + 256`.

!!! note "Nothing is deleted unless you ask"

    Unset, `prune()` with no argument deletes nothing and returns 0. The
    trajectory is your data in a directory you named, and a library that
    silently deleted it would be worse than one that grows. But it *is* a
    decision, so make it.

## Durability

Records are **flushed, never fsynced**. A flush puts the record in the OS page
cache, which survives this process dying — a crashed agent, a `SIGKILL` on a
blown deadline — and that is the failure capture actually has to survive.

`fsync` would additionally survive the machine losing power, at roughly 1–10 ms
per record against the ~13 µs the whole write costs. That would make recording
slower than many of the tool calls being recorded. The trade is deliberate: **a
machine crash loses recent outcomes.**
