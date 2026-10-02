# Configuration

Every knob is a keyword argument on `serve()` or one of seven environment variables. **Two change what the SDK does** — `PERCEPTEYE_AGENT_MODE` and `PERCEPTEYE_INTROSPECT`. The rest are addresses, credentials and locations.

## The mode switch

`PERCEPTEYE_AGENT_MODE` decides whether this process does training work. It is the one knob that turns the flywheel off.

```bash
PERCEPTEYE_AGENT_MODE=training      # claim rollouts and run them (the default when unset)
PERCEPTEYE_AGENT_MODE=production    # serve() returns 0 immediately. No claim, no connection.
```

**The same deployed image does either without a code edit.** "Stop contributing capacity" is an operational decision, and it should not require editing source to express it.

### What `production` actually does

| behavior | why it is that way |
|:--|:--|
| **Returns `0` before opening any connection** | a production deployment needs no control-plane URL and no key. Turning training off must never require being configured for it. |
| **Returns rather than raises** | a switch that crashes the service it was meant to quiet is not a switch. Zero rollouts is the literal truth, so `0` is the literal return. |
| **Prints exactly one line** | a silent no-op is the failure mode this package refuses everywhere else. The operator would otherwise believe capacity was being contributed with nothing saying otherwise. |
| **Resolved before the entrypoint is imported** | `serve("does_not_exist:build", …)` under `production` still returns `0`. Nothing is loaded, nothing is validated, nothing is dialed. |

The line, verbatim:

```text
[percepteye] mode=production — not claiming rollouts. (set PERCEPTEYE_AGENT_MODE=training to contribute capacity)
```

### Accepted spellings

Case-insensitive, surrounding whitespace stripped.

| value | resolves to |
|:--|:--|
| `training` · `train` · `rollout` · `rollouts` · `fine-tuning` · `finetuning` · `improving` | **contribute capacity** |
| `production` · `prod` · `serving` | **do not** |

`rollout` is the legacy spelling of `training` and is unchanged in meaning — on the wire, a donor still registers `mode: "rollout"`, which names the protocol the process speaks rather than the deployment mode.

**An unrecognised value raises `ConfigurationError` rather than guessing.** The set is deliberately small and closed, because both ways of guessing wrong are invisible from the outside: silently contributing capacity the operator switched off, or silently contributing nothing while they believe training is running. Neither is recoverable, because both look exactly like success.

```text
ConfigurationError: PERCEPTEYE_AGENT_MODE='staging' is not a mode. Use 'training'
(this process does training work: it claims rollouts and runs them) or 'production'
(it does not). Recognised spellings: fine-tuning, finetuning, improving, prod,
production, rollout, rollouts, serving, train, training.
```

### Precedence

| rank | source | note |
|:--|:--|:--|
| 1 | `mode=` argument to `serve()` | an empty or whitespace-only string counts as **not passed** and falls through to rank 2 |
| 2 | `PERCEPTEYE_AGENT_MODE` | what lets an operator flip a deployment without editing code |
| 3 | `"training"` | the default when neither is set |

The empty-string rule is load-bearing. `mode=cfg.get("percepteye_mode", "")` is the ordinary way a config layer says "not set"; honoring it as a value pinned the process to `training` without ever reading the variable, so an operator who had flipped the switch kept contributing, on their own credentials, with nothing printed.

**An override is announced in both directions.** When `mode=` in code disagrees with the environment, the line says which won:

```text
[percepteye] mode=training — set by mode= in code, overriding PERCEPTEYE_AGENT_MODE=production.
Remove mode= from the call to let this deployment decide.
```

An unparseable environment value alongside an explicit `mode=` is a `RuntimeWarning`, not fatal — the argument is authoritative and the process has been told what to do, but an unreadable switch must not pass unremarked.

Read the switch yourself with `fw.current_mode()` (environment only) or `fw.resolve_mode(explicit)`. Both raise on an unrecognised value, exactly as `serve()` does, so a typo surfaces wherever it is first read.

!!! note "This switch gates `serve()` and nothing else"
    It does not gate [`attach()`](production.md), which *is* the production surface and captures evidence from turns you already serve. Gating it on the same variable would mean the correct setting for a production agent was `PERCEPTEYE_AGENT_MODE=training`. To stop capture, don't call `attach()`, or pass `register=False` to keep it entirely local. `PERCEPTEYE_AGENT_MODE` is read in exactly one place in the package, and that place is `serve()`.

## Stopping a running flywheel

The switch is read at startup, so flipping it does not stop a loop that is mid-flight.

**`max_rollouts=N` is the only clean exit.** It is a completion counter checked at the top of the loop, and the loop never claims more than it still intends to run. There is no env equivalent and no way to change it on a running process. The count is **completions, not successes** — a crash and a graded pass both increment it.

| you try | what happens |
|:--|:--|
| **`SIGTERM`** | the process ends. **Nothing in this package catches SIGINT or SIGTERM.** In-flight rollouts are orphaned rather than abandoned, and the control plane reclaims them by lease expiry — up to 30 minutes — rather than by an `abandon` call. |
| **Ctrl-C** | not instant. The worker pool is exited with `wait=True`, so in-flight rollouts drain first, for up to `rollout_timeout_s` (default 900s). |
| **Revoking the API key** | does not stop it. A 401 is terminal per request, but the loop catches it, emits `claim_failed`, sleeps 5s and continues — indefinitely. |
| **The control plane** | cannot stop it either. The claim response carries only `rollouts` and `poll_after_ms`; there is no pause or drain field. The strongest server-side levers are an empty queue and a longer backoff, and both leave you polling. |

!!! warning "Ctrl-C is recorded as the SDK crashing"
    A subprocess-isolation child is in the terminal's process group and takes the same SIGINT. It exits non-zero, and a non-zero child is mapped to the typed exclusion `flywheel_crash` — so an operator's Ctrl-C lands in the record as a substrate fault. A **command** entrypoint child is started in a new session, does not receive the terminal's SIGINT, and outlives the parent.

Your production agent is unaffected by any of this. It is a different process calling different functions.

## Watching a flywheel: `on_event`

`serve(on_event=...)` takes a `Callable[[str, dict], None]`. There is no logger in this package; this is the only observability surface `serve()` has.

```python
fw.serve(..., on_event=lambda kind, ev: print(kind, ev))
```

| event | when | payload keys |
|:--|:--|:--|
| `claim_empty` | a successful claim returned no work | `poll_after_ms` |
| `claim_failed` | the claim itself failed | `error` |
| `rollout_start` | a claimed rollout began | `rollout_id` |
| `rollout_reported` | the result was accepted | `rollout_id`, `grade`, `duplicate`, `exclusion` |
| `rollout_crash` | the rollout raised; it was abandoned | `rollout_id`, `error` |
| `report_failed` | the work happened, delivering it did not | `rollout_id`, `error` |
| `artifact_failed` | collecting the rollout's artifacts failed; the rollout is still reported, excluded from training | `rollout_id`, `name` (always `"*"`), `error` |
| `lease_lost` | the lease lapsed or was taken back | `rollout_id` |
| `heartbeat_failed` | a renewal could not be delivered; two more beats remain | `rollout_id`, `error` |

**`claim_empty` is the one worth wiring first.** Without a callback a flywheel is silent, and an empty queue and a broken claim route look identical — `claim_empty` versus `claim_failed` is what tells them apart. This is the answer to "how do I know it's working."

**It runs on three different threads, synchronously.** The serving thread emits `claim_*`; a pool worker emits the per-rollout kinds; the heartbeat thread emits `lease_lost` and `heartbeat_failed`. A slow callback slows that thread, and the heartbeat thread is renewing every in-flight lease. `lease_lost` can arrive from either the heartbeat thread or a pool worker, depending on which noticed first.

**Exceptions it raises are swallowed** and not re-emitted. An observability hook must not be able to change what it observes: a callback that raised inside the heartbeat thread's own handler used to exit the loop and the thread, after which no in-flight lease was renewed again and every subsequent rollout collected a 410 at report time. Keep it cheap and non-blocking.

## Every `serve()` argument

| argument | default | what it does |
|:--|:--|:--|
| `entrypoint` (positional) | — | `"module:attr"`, an argv `list`, or a constructed object |
| `agent_id` | required | the agent this process serves for |
| `input_shape` | required | `"text"` · `"messages"` · `"dict"`. Declared, never inferred |
| `mode` | `None` | see [the mode switch](#the-mode-switch) |
| `transport` | `"attach"` | the only value in contract `flywheel/1`; anything else raises `ModeNotAvailable` |
| `control_plane_url` | `None` | falls back to `PERCEPTEYE_CONTROL_PLANE_URL` |
| `api_key` | `None` | falls back to `PERCEPTEYE_API_KEY` |
| `artifacts_dir` | `None` | falls back to `./.percepteye-artifacts`, created if absent |
| `isolation` | `"subprocess"` | or `"inprocess"`; any other value raises |
| `concurrency` | `1` | rollouts in flight. A single claim asks for at most 16 whatever you set |
| `rollout_timeout_s` | `900.0` | the local ceiling on one rollout |
| `gate_flag` | `None` | how *this* binary spells its completion-gate option, e.g. `"--autonomous-gate"` |
| `endpoint_env` | `None` | `(url_var, key_var)` this agent reads its endpoint from, when they are not the OpenAI/Anthropic names |
| `execution_snapshot` | `None` | generic adapter lifecycle: opaque exact components plus prepare/attest/release for each isolated child |
| `harness_snapshot` | `None` | deprecated compatibility alias for `execution_snapshot`; no harness behavior exists in core |
| `command_output_adapter` | `None` | explicit command stdout projection; without one, stdout is final text and no structured evidence is inferred |
| `attested_execution_components` | `None` | opaque `{stable_name: lowercase_sha256}` evidence from an execution adapter; folded into the generic OPSD execution identity |
| `volatile_command_values` | `None` | command options whose VALUE is transport, not identity, e.g. `("--daemon-socket",)`; the option token stays in `execution_sha256` and only its value is dropped, so a restart that moves a socket path keeps the same digest. Command entrypoints only; a name absent from the command raises |
| `reports_tool_calls` | `False` | assert that this flywheel reports a per-call outcome for every call it makes |
| `method` | `None` | name the entrypoint method instead of probing |
| `adapters` | `"auto"` | `"auto"`, `[]`, or names: `"langchain"`, `"adk"` |
| `introspect` | `True` | [agent discovery](how-it-works.md); also gated by `PERCEPTEYE_INTROSPECT` |
| `describe` | `None` | your agent's own `system_prompt` / `tools` / `model`, for an agent [nothing can read](how-it-works.md#an-agent-nothing-can-read-describe); what you declare wins over reflection |
| `rules` | `None` | [your Product's workflow rules](#a-products-workflow-rules-rules), one prose rule per string, declared once after registration; `None` sends nothing, `[]` retracts |
| `max_rollouts` | `None` | stop after this many rollouts complete |
| `on_event` | `None` | see above |

**Everything that can be misconfigured is checked before the first rollout**, in this order: the mode resolves, the transport is one this contract speaks, a URL and a key exist, `/health` answers and its `contract_versions_supported` includes `flywheel/1`, unroutable provider variables are reported, the entrypoint resolves, and the isolation mode is compatible with the entrypoint you passed. Each raises there, never mid-run.

`adapters="auto"` probes without committing: a framework that is absent is simply not bound. **A named adapter that cannot bind raises**, because naming it asserts that it is how your calls are captured. LangGraph is covered by the `langchain` adapter, which reads `langgraph_node` off call metadata to attribute each call to its node.

When discovery exactly observes the behavior-bearing prompt, tools, model config,
and recursive sub-agent tree **and** an execution adapter supplies at least one
attested component for otherwise-unobserved implementation/state, `serve()` stamps each rollout with
`agent_fingerprint.execution_sha256`. Its canonical descriptor includes those
facts, the entrypoint and serve configuration, the SDK/contract version, and
any `attested_execution_components`. Every argv token of a command entrypoint
is one of those entrypoint facts, except the value of an option named in
`volatile_command_values`, which is excluded — the option token itself is kept,
so a restart that moves a per-process socket path keeps the digest while
running with no socket at all is still a different execution.
Discovery disabled, unreadable, lossy, or
unbacked by attested implementation evidence means the key is absent, so OPSD refuses instead of training on a universal
"unknown agent" identity. Adapter evidence must hash mutable state discovery
cannot see; it is a digest map, never a place for raw configuration or secrets.

The claim long-poll (`wait_ms`, 25s), the requested lease (`lease_ms`, 30 min) and the HTTP timeout (30s) live on the transport and are **not exposed through `serve()`**.

## A Product's workflow rules: `rules=`

```python
fw.serve("my_package.agent:build", agent_id="widget-helper", input_shape="text",
         rules=[
             "When the caller has not said which widget colour they want, ask "
             "for the colour before placing any order.",
         ])
```

A rule is one prose sentence about how your agent should work — the conditional kind a reviewer would write down. **It is training guidance**: when a rule applies to a decision in your agent's traces, training sees it beside that decision. Nothing your agent is served changes — not its prompt, not what its model is sent.

`serve()` and [`attach()`](production.md#attach-evidence-from-live-turns) take the same `rules=`, and send it once, after the registration, to `POST /agents/{agent_id}/rulebook`:

- **The rules are the Product's, not the agent's.** The wire carries no Product: the control plane files them under the one active Product that declares this agent, as that Product's rulebook in force. An agent no Product declares, or more than one does, is refused (`409 rulebook_unattributed`, naming which) and nothing is registered.
- **`None` sends nothing; `[]` retracts.** Leaving `rules=` out leaves the rulebook in force alone, whoever registered it; an empty list takes it away.
- **The last writer wins.** Code that declares rules re-declares them on every start: the same rules are answered `unchanged` and register nothing, different ones are a new version. The Product's own rulebook API (`PUT /api/v2/products/{product_id}/rulebook`) writes the same versions, so whichever of the two wrote last is in force, and every write is audited.
- **The control plane decides what a rule may be** — its length, a duplicate, JSON text, a control character, a credential value, how many there are — and a refusal (`422 rulebook_invalid`) names the cause and the rule's index. The SDK restates none of it: a second copy of those bounds would only drift.
- **Nothing it answers can stop your agent.** The rules are declared in the background, so a refused, undeliverable or slow upload costs the rules and nothing else — never the registration, the policy read, a rollout or your startup. `serve()` prints one line either way (`[percepteye] rulebook: registered as version 3 — 5 rule(s) for Product …`, or `rulebook NOT confirmed — …` in the control plane's own words); `attach()` keeps the receipt on `px.rulebook` and the sentence on `px.rulebook_refusal`, and `px.wait_for_rulebook()` waits for the answer. "Not registered" is said only when it is known: a failure after the rules may have landed is named as unknown, and declaring the same rules again is safe. Only a value that is not a list of strings raises, `ConfigurationError`, at startup — a `set` included, because its order is not stable across processes.
- **Not in `describe=`.** A declaration withholds the agent's execution identity, and these are the Product's rules, not a description of the agent — so they ride their own route. The [Node plugins](node.md#a-products-workflow-rules-rules) declare the same rules with the same request.

## Isolation

`isolation="subprocess"` is the default and **the only mode with a real timeout** — Python cannot interrupt a synchronous call in another thread, so in-process a hung agent means we stop *waiting* while it keeps running. A child process can actually be killed. It is also the only mode where two concurrent rollouts cannot see each other's endpoint or credential.

| mode | entrypoint it accepts | concurrency | timeout |
|:--|:--|:--|:--|
| `"subprocess"` | `"module:attr"` or an argv list | any | real — the child is killed |
| `"inprocess"` | an already-constructed object | **1 only** | best-effort; the wedged thread keeps running |

`isolation="inprocess"` with `concurrency > 1` raises: concurrent in-process rollouts share one set of environment variables, so they would race for the per-rollout endpoint and credential and silently cross-attribute each other's completions.

### What the child gets

**The child's environment is scrubbed, never copied wholesale.** The parent holds the long-lived control-plane key; splatting the parent environment into a child running customer agent code hands the strongest credential in the system to the least trusted process in it.

| rule | effect |
|:--|:--|
| Your environment passes through | your agent needs your credentials. That is the premise of running it in your process. |
| The inherited `PERCEPTEYE_` namespace is scrubbed completely | no customer- or harness-specific setting crosses implicitly; an execution adapter may supply its own non-PerceptEye variables through its child environment, or remove an inherited one (the Prime adapter removes `PYTHONPATH`, `PYTHONHOME`, `PYTHONSTARTUP`, `IPYTHONDIR`, `PYTHONPYCACHEPREFIX`, which points the interpreter's bytecode cache outside the venv, and macOS's `PYTHONEXECUTABLE` and `__PYVENV_LAUNCHER__`, which pick the venv the interpreter starts in, so nothing from outside its sealed kernel runs in it) |
| Two are then set | `PERCEPTEYE_ROLLOUT_ID` and `PERCEPTEYE_TRAJECTORY_DIR`, scoped to this rollout |
| Routing is set last | `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, so they beat anything inherited |
| Unroutable provider config is removed | see below |

**Your agent must sample through an OpenAI-compatible client**, because the per-rollout gateway speaks that API. Routing only wins over the *environment*: a client constructed with an explicit `base_url` in your own code beats all of it, silently, and lands at a live third party.

??? note "The provider variables removed from every child"
    Setting `OPENAI_BASE_URL` does nothing for an agent built on Gemini, Azure OpenAI or Bedrock — those clients read their own names, sail past the gateway, and produce a trajectory that cannot back an on-policy objective while looking exactly like one that can. `serve()` warns at startup when it finds any of these set; an agent that finds no provider configured raises at construction, in front of the person who configured it, which is the outcome we want.

    `AZURE_OPENAI_ENDPOINT` · `AZURE_OPENAI_API_KEY` · `AZURE_OPENAI_AD_TOKEN` · `OPENAI_API_TYPE` · `OPENAI_API_VERSION` · `GOOGLE_API_KEY` · `GEMINI_API_KEY` · `GOOGLE_GENAI_USE_VERTEXAI` · `NVIDIA_API_KEY` · `NVIDIA_NIM_API_KEY` · `AWS_BEARER_TOKEN_BEDROCK` · `GROQ_API_KEY` · `MISTRAL_API_KEY` · `TOGETHER_API_KEY` · `OPENROUTER_API_KEY` · `FIREWORKS_API_KEY` · `DEEPSEEK_API_KEY` · `PERPLEXITY_API_KEY` · `COHERE_API_KEY` · `XAI_API_KEY`

    Deliberately **not** `AWS_ACCESS_KEY_ID` — that is the same credential a customer's agent uses for S3, and taking it would break agents that never touch a model through it. The list names what we know; it is not a proof of completeness. The control plane's reported-vs-served call count is the check that actually decides.

An agent that reads a *different* pair of names declares them rather than being sniffed:

```python
fw.serve(["dsh", "--profile", "headless", fw.TASK], agent_id="dsh-agent",
         input_shape="text", endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"))
```

This is additive — the defaults stay set, because an agent may read a mixture.

### Timeouts

| bound | value |
|:--|:--|
| `rollout_timeout_s` | `900.0` by default |
| the agent's actual budget | `min(rollout_timeout_s, server deadline − 10s)` |
| subprocess hard kill | `budget + 15s` on the child, then `SIGKILL` |
| heartbeat interval | a third of the **granted** lease, floor 1s |
| lease reclaimed by the server | up to 30 minutes after the last beat |

The agent is stopped 10s short of the server deadline so there is time to collect artifacts and land the report while the lease is still good. A rollout whose deadline has already passed when claimed is handed straight back as `deadline_exceeded` rather than run.

A rollout that arrives past its deadline, a lost lease, a crash and a blown budget all come back as **typed exclusions**, so a substrate fault is never scored as a bad decision by your agent: `flywheel_crash` · `substrate_unavailable` · `credential_expired` · `deadline_exceeded` · `entrypoint_error` · `agent_exit_nonzero` · `cancelled` · `capacity`.

## Entrypoints

Probed in order — `invoke`, `run`, `execute`, `chat`, `__call__` — first match wins. Pass `method="..."` to name one. **Resolution happens once, at startup**, and raises there: a rollout that fails to emit because a method name was wrong does not merely lose itself, it biases the score of every sibling it was supposed to be compared against.

| you pass | you get |
|:--|:--|
| `"my_package.agent:build"` | imported per rollout in a fresh child. A class is instantiated; a factory is called; an already-agent-shaped object is used as-is |
| `["opencode", "run"]` | a process. Task on **stdin** and in `PERCEPTEYE_TURN_INPUT`; ordinary stdout is final text, while recognised JSONL yields its terminal assistant text |
| `["dsh", "--profile", "headless", fw.TASK]` | `fw.TASK` marks the argv position for the task — and stdin is then left **empty**, so the task is not delivered twice |
| a constructed object | `isolation="inprocess"`, `concurrency=1` |

A command entrypoint runs in **its own process group**, in a new session, with the per-rollout trajectory directory as its working directory. The whole group is what gets killed on timeout, so a spawning agent leaves nothing behind. See [CLI and other languages](cli.md).

`gate_flag` and a task's `completion_gate` are two independent facts and are kept apart: the flag is how *this binary* spells the option and cannot be guessed, the gate command is a property of the task and arrives per rollout in `metadata`. Either side missing leaves the argv untouched.

## Input shapes

Declared by you, never inferred. An earlier SDK guessed the call convention from whichever method name it happened to find, so `run(str)` and `run(dict)` were indistinguishable until something failed at depth.

| `input_shape` | your entrypoint receives |
|:--|:--|
| `"text"` | `"the task description"` |
| `"messages"` | `[{"role": "user", "content": "..."}]` |
| `"dict"` | `{"input": "..."}` |

## Reporting

Return a string, or a `RolloutOutput` when you have more to say.

```python
from agent_flywheel import RolloutOutput

def invoke(task: str) -> RolloutOutput:
    return RolloutOutput(
        final_text="done",
        llm_call_count=3,       # total model calls this harness observed
        artifacts={},          # usually left empty — files are collected for you
        reward=1.0,
        success=True,
    )
```

| field | default | meaning |
|:--|:--|:--|
| `final_text` | required | what the agent said |
| `tool_calls` | `None` | **`None` is "not reported"; `[]` is "I observed zero"**, which is a positive claim. Merged with what the trajectory recorded |
| `tool_calls_omitted_count` | `0` | calls you saw but did not report |
| `llm_call_count` | `None` | independently observed total model calls. A pinned/on-policy rollout is ineligible unless this is a nonnegative integer equal to the gateway count. Prime JSON mode is counted by its built-in adapter; other harnesses set it explicitly |
| `artifacts` | `{}` | name → digest string, or an inline `{"sha256", "size_bytes", "content"}` body |
| `reward` | `None` | |
| `success` | `None` | |
| `exclusion_reason` / `exclusion_detail` | `None` | a typed exclusion; see the list above |
| `events` | `[]` | |
| `timing` | `{}` | `elapsed_s` is filled in for you |
| `agent_fingerprint` | `None` | |

A dict is accepted and read shallowly — the first string under `final_text`, `output`, `result`, `content`, `text` or `answer`, else the JSON. Anything richer belongs in an explicit `RolloutOutput` or in `record_tool_call`. `to_wire()` raises `ContractViolation` on a key the schema does not declare, so a value the code can emit and the contract does not fails the build rather than a customer's report.

**Artifacts are collected on every path**, including a timeout kill and a non-zero exit — a rollout that died is exactly the one you need the files from. The caps are module constants, not knobs:

| cap | value | over it |
|:--|:--|:--|
| per file, registered | 256 MB | `skipped:too_large:<size>` |
| per file, inlined in the report | 64 KB, UTF-8 only | keeps its digest string |
| total inlined per report | 256 KB | keeps its digest string |

Symlinks are **never** followed — one dropped in the agent's own directory would otherwise be read at its target and inlined into the report. They are recorded as `skipped:symlink` rather than silently dropped, because a link is itself a fact about the rollout.

## Environment variables

| Variable | Meaning |
|:--|:--|
| `PERCEPTEYE_AGENT_MODE` | `training` (default) or `production`. In production `serve()` returns 0 without connecting. Gates capacity contribution only — not `attach()`. A value it cannot parse raises rather than being guessed. |
| `PERCEPTEYE_CONTROL_PLANE_URL` | Control-plane base URL. Overridable with `control_plane_url=`. |
| `PERCEPTEYE_API_KEY` | Your flywheel key. Overridable with `api_key=`. |
| `PERCEPTEYE_INTROSPECT` | Set to `0`, `false`, `no` or `off` to disable agent discovery. |
| `PERCEPTEYE_ROLLOUT_ID` | *Set by us* for the child. Read it to tag your own logs. |
| `PERCEPTEYE_TRAJECTORY_DIR` | *Set by us* for the child — already this rollout's directory. Append to `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`; any other file you leave there is collected as an artifact. Also the variable the Node plugins read. |
| `PERCEPTEYE_TURN_INPUT` | *Set by us* for a command entrypoint. The task, also on stdin. |

Only `PERCEPTEYE_AGENT_MODE` and `PERCEPTEYE_INTROSPECT` change behavior. The optional Prime example also reads its own documented route settings in `examples/prime_agent`; those are example inputs, not SDK child-environment policy. `PERCEPTEYE_INTROSPECT` and `introspect=` are ANDed — either one off disables discovery — and **only a deliberate value disables it**: a typo leaves it on rather than silently switching capture off while you believe it is running.

The one variable this SDK will *write* is under your control: `apply_current_policy(set_env=True)` exports `OPENAI_BASE_URL` and `OPENAI_API_KEY`. It is off by default, because mutating the process environment is a side effect a caller should ask for.

## OpenTelemetry

```bash
pip install "agent-flywheel[otel]"
```

The extra is `opentelemetry-api>=1.20` — **the API only, never the SDK.** This package emits into whatever `TracerProvider` you already registered and **never installs one**. `set_tracer_provider` is once-only, so calling it would either lose all our spans (if yours was configured first) or silently kill your entire GenAI telemetry (if we won). With no provider configured, the API's no-op tracer swallows everything, which is the correct outcome for a library.

Every recorded outcome is additionally emitted as a span named `execute_tool <tool name>`.

| attribute | value |
|:--|:--|
| `gen_ai.operation.name` | `"execute_tool"` |
| `gen_ai.tool.name` | the tool name |
| `gen_ai.tool.call.id` | the model-assigned call id, when there is one |
| `gen_ai.tool.call.arguments` | JSON, truncated at 2000 chars |
| `gen_ai.agent.name` | the sub-agent that made the call, when known |
| `percepteye.tool.outcome` | `ok` · `failed` · `unknown` |
| `percepteye.tool.status_code` | when there is one |
| `percepteye.tool.latency_ms` | when measured |
| `percepteye.rollout.id` | set inside a `serve()` rollout |
| `percepteye.turn.id` | set inside `attach().turn()` — never stamped as a rollout id, or your own telemetry could not answer "how much of my traffic was training" |
| `percepteye.task.id` | optional caller-supplied task/case/work-item correlation from `attach().turn(task_id=...)`; evidence only, never training authority |
| `percepteye.agent.execution_sha256` | exact SDK-reconciled execution identity, only when discovery and adapter evidence were complete |

**Only a reported failure sets the span status to error. An `unknown` leaves the span status unset** — marking it OK would assert a success nobody observed, and marking it ERROR would assert a failure nobody observed. The GenAI conventions have no attribute for "the result of this call was never observed", which is why the tri-state lives under `percepteye.*` rather than being bent into a standard attribute that means something else to every other consumer.

**Nothing is ever read back.** The conventions define no attribute for per-token logprobs or token ids, so spans cannot carry what training needs; [outcomes](outcomes.md) reach the control plane through the trajectory regardless. Your sampler, queue overflow and exporter failures therefore cost us nothing, and span emission never raises — a telemetry failure costs you the span and nothing else.
