# How the loop closes

`fw.serve()` does two things, in that order: it **describes your agent once**, then it **claims work forever**. Everything in between — isolation, routing, capture, reporting — repeats identically on every lap.

**Every connection is opened from your side.** The SDK dials out; nothing dials in. That is not a deployment convenience, it is what makes the package installable inside a corporate network without a change request.

![The flywheel as a closed ring](assets/flywheel-architecture.svg)

| beat | what happens | who does the work |
|:--|:--|:--|
| **⓪ Onboard** | the SDK describes your agent from inside its own process and registers it — once | automatic; `introspect=False` opts out |
| **① Claim** | your process long-polls for a rollout and is granted a lease | you dial out |
| **② Run** | fresh child process, own environment, single-use credential | your agent, unchanged |
| **③ Record** | tool calls land as `ok` / `failed` / `unknown`; artifacts collected on every path | adapters, from the outside |
| **④ Score** | tasks generated from your real tool surface, graded against compiled checks | Percepteye |
| **⑤ Improve** | your production agent picks up the promoted policy | one call at startup |

---

## Once, at startup

`serve()` checks everything that can be misconfigured **before the first rollout**, and raises there. The order is load-bearing.

| # | step | fails with |
|:--|:--|:--|
| 1 | resolve the mode — `mode=` argument, else `PERCEPTEYE_AGENT_MODE`, else `training` | `ConfigurationError` on a value it cannot parse |
| 2 | `transport` must be `"attach"` | `ModeNotAvailable` |
| 3 | `control_plane_url` or `PERCEPTEYE_CONTROL_PLANE_URL` | `ConfigurationError` |
| 4 | `api_key` or `PERCEPTEYE_API_KEY` | `ConfigurationError` |
| 5 | create `artifacts_dir` (default `./.percepteye-artifacts`) | `OSError` |
| 6 | `GET /health`, and assert it serves `flywheel/1` | `ConfigurationError` naming both versions |
| 7 | scan for provider variables the gateway cannot serve | `RuntimeWarning`, never a refusal |
| 8 | resolve the entrypoint and check isolation is compatible with it | `EntrypointError` on an object with no callable to drive; `ConfigurationError` on a bad spec or an isolation mismatch |
| 9 | describe the agent (beat ⓪) | never — discovery cannot fail a startup |
| 10 | `POST /agents` | `TransportError` |
| 11 | `GET /policy/current`, printed and **not applied** | never — a failed read is `{}` |

**Mode is resolved first, before any credential check and before any socket.** A production deployment therefore needs no control-plane URL and no key: turning training off must not require being configured for it. In `production`, `serve()` prints one line and returns `0`.

```bash
PERCEPTEYE_AGENT_MODE=production   # serve() returns 0 immediately. No claim, no connection.
PERCEPTEYE_AGENT_MODE=training     # contribute capacity (the default when unset)
```

Step 8 is why a bad entrypoint stops startup rather than surfacing an hour later as a missing member of a comparison group. Resolution is probed once, in order: `invoke`, `run`, `execute`, `chat`, `__call__`.

---

## ⓪ Onboard — the agent describes itself

An external connector inspects an artifact: source, a manifest, an advertised schema. This one runs **inside the process holding the constructed agent**, so it reads the prompt after its template rendered, the tools after conditional registration, and the sub-agents after the graph was assembled.

That vantage point is the whole point of beat ⓪. **You never write a task list** — the control plane generates tasks from the tool surface you actually have. An agent that reports nothing gets generic tasks.

Three probes, tried in order, first match wins. None imports its framework at module scope, so an absent framework simply does not match.

| probe | matches | reads |
|:--|:--|:--|
| `google_adk` | `google.adk.agents.BaseAgent` | `instruction`, `tools` (via ADK's own `FunctionTool` declaration), `model`, `sub_agents` |
| `langgraph` | a compiled `Pregel` graph | each user node as a sub-agent, with its own prompt, tools and model |
| `langchain` | any object exposing `tools` / `get_tools()` holding a `BaseTool` | tool name, description, JSON Schema, prompt, model |

Sub-agents are walked to depth **4** — deep enough for the graphs people build, shallow enough that a cycle cannot hang startup. The system prompt is truncated at **20 000 characters**, and a callable prompt (ADK resolves `instruction` per turn from state) is **reported as not-observed rather than invoked**.

**Absence is never collapsed into zero.** `tool_definitions: null` means "we could not look"; `[]` means "we looked, and this agent has none". `incomplete` names every part that could not be read. A framework the SDK does not recognize registers `discovered_agent: null` — never an empty agent.

??? note "What `POST /agents` actually carries"

    ```json
    {
      "agent_id": "research-agent",
      "mode": "rollout",
      "sdk": "percepteye-agent-flywheel-python/0.1.0",
      "contract_version": "flywheel/1",
      "reports_tool_calls": false,
      "isolation": "subprocess",
      "concurrency": 1,
      "input_shape": "text",
      "entrypoint": "invoke",
      "discovered_agent": {
        "agent_name": "Agent",
        "framework": "langchain",
        "source_type": "sdk_runtime",
        "role": "standalone",
        "tool_definitions": [
          {"name": "lookup", "description": "Look something up.",
           "input_schema": {"type": "object",
                            "properties": {"q": {"title": "Q", "type": "string"}}},
           "original_format": "langchain"}
        ],
        "system_prompt": "You are a research agent."
      }
    }
    ```

    **Never sent:** your tools' source or return values, your credentials, your data, or anything observed during a rollout. `source_type` is `sdk_runtime` for everything reflection describes — the one value that means "the agent described itself from inside its own process". A host-written description keeps the host's own provenance instead (`openclaw_llm_input`), because the field is a claim about where the description came from.

### A CLI agent has no object to reflect on

`fw.serve(["opencode", "run"], ...)` spawns an OS subprocess. There is nothing to reflect on, so reflection returns `None` and the agent registers as unreadable — which blocks workflow generation, and with it every phase downstream.

Reflecting harder does not help. The **host** is the only party that can describe a CLI agent, because the host assembled the prompt and built the tool list it sent to the model. The [Node plugin](node.md) observes one `llm_input` event and writes `discovered_agent.json`; `serve()` reads it.

The bootstrap is circular, and that is why the warm-up exists:

```text
no launch → no rollout → no CLI spawn → no llm_input → never described → no launch
```

So the agent is run **once at `serve()` time, before registering**, on a trivial prompt (`"Reply with the single word: ready"`). This works because `llm_input` fires *before* the model request is dispatched — the warm-up needs no working completion, no gateway and no API key.

| bound | value |
|:--|--:|
| wait for the first description | 90 s |
| grace period after one appears, so an enriching integration can add to it | 60 s |
| poll interval | 0.1 s |

A host that exits on its own is read immediately; a fast CLI finishes in well under a second. The child is spawned in its own process group and killed with it.

The warm-up child is **not a rollout** and is never captured, graded, or trained on. It gets the same environment scrub a rollout child gets — the whole `PERCEPTEYE_` namespace is stripped so the long-lived key stays in the parent, and `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` are forced to `http://127.0.0.1:1/v1` with a dummy key, so a dispatch that outruns the kill fails closed instead of sampling a live provider unobserved.

### Opting out

```python
fw.serve(..., introspect=False)      # or PERCEPTEYE_INTROSPECT=0
```

Only a deliberate value disables it — `0`, `false`, `no`, `off`. A typo leaves it **on**, rather than silently switching discovery off while you believe it is running.

---

## ① Claim — a long poll you open

`serve()` is **pull-only**. A push transport is deliberately absent rather than half-present: `FlywheelTransport` is a Protocol so one could be added without the runner learning which is in use, and `transport="attach"` is the only value `serve()` accepts — anything else raises `ModeNotAvailable` naming the contract version.

```json
POST /rollouts/claim
{"agent_id": "my-coding-agent", "max": 1, "wait_ms": 25000, "lease_ms": 1800000}
```

| field | clamped to | note |
|:--|:--|:--|
| `max` | `1 … 16` | never more than the slots you still intend to run |
| `wait_ms` | `0 … 25 000` | the long-poll window; HTTP timeout is `wait_ms/1000 + 10 s` |
| `lease_ms` | `1 000 … 1 800 000` | 30 minutes by default |

An empty queue answers **200 with an empty list and a server-chosen `poll_after_ms`** — never a bare 204, which carries no backoff information and leaves the client to invent a fixed sleep.

**A claim is not replayable.** The control plane does not honor `Idempotency-Key` on this route, so resending one that may already have been processed would lease a *second* batch while the first is held by nobody. Only a failure that provably never reached the server (`ConnectError`, `ConnectTimeout`) is retried.

Two events separate the two ways a claim can return nothing. Without an `on_event` callback they are indistinguishable, and that is the answer to "how do I know it's working":

| event | means | then |
|:--|:--|:--|
| `claim_empty` | the claim succeeded, the queue is empty | sleep `poll_after_ms` |
| `claim_failed` | the claim itself failed | sleep 5 s and continue — **indefinitely**, including on a 401 |

### Leases, heartbeats, and redelivery

A claim grants a lease. One daemon thread renews every in-flight lease, ticking each second and beating each rollout at **one third of the lease the server actually granted** (`lease_expires_ts`), so two beats may be lost before anything lapses. The interval is computed once, at claim, against the granted expiry — pacing it against the shrinking remainder turns nine heartbeats into roughly six hundred.

The work budget and the lease are **separate numbers for separate jobs**: `deadline_ts` bounds the work, the lease is liveness. A heartbeat extends the lease by up to 10 minutes; it never moves the deadline.

A `410` from any control-plane call raises `LeaseLost`. The heartbeat thread then marks the rollout, emits `lease_lost`, and **kills the child** — under an earlier design the child kept working for the rest of its timeout against a credential the server had already revoked.

**A crashed worker needs no cleanup call.** Nothing in this package handles `SIGINT` or `SIGTERM`, so a killed process orphans its in-flight rollouts rather than abandoning them; the control plane reclaims them by lease expiry — up to 30 minutes — and hands them to the next claimer. `max_rollouts=N` is the only clean exit, checked at the top of the loop.

---

## ② Run — isolate, then route

| isolation | timeout | concurrency | entrypoint |
|:--|:--|:--|:--|
| `subprocess` *(default)* | real — a child can be killed | any | must be importable (`"module:attr"`) |
| `inprocess` | advisory — Python cannot kill a thread | `1` only | accepts a constructed object |

The constraint is checked at construction, not discovered at rollout time. Under subprocess isolation the parent spawns `python -m percepteye_agent_flywheel.child`, hands it the job as JSON on stdin, and reads the result from the last line of stdout. A [command entrypoint](cli.md) is already its own process: it runs in a new session, in the rollout's own directory, with the task on stdin *and* in `PERCEPTEYE_TURN_INPUT`.

**Configuration crosses into a fresh child at spawn, never into a running parent.** `RolloutRequest` has no `environment` field, and it is a contract violation to send one — an earlier SDK wrote server-supplied values into the host process's `os.environ`, which was both a global mutation in a live application and useless, because the agent's client had already read its base URL when it was constructed.

Three rules govern the child's environment, and none is configurable:

1. **No `PERCEPTEYE_` variable is inherited.** The long-lived control-plane key and every customer- or harness-specific setting stay in the parent. An explicit execution adapter can add its own non-Percepteye child variables; core does not recognize any customer's route vocabulary.
2. **Two rollout-scoped values are set:** `PERCEPTEYE_ROLLOUT_ID` and `PERCEPTEYE_TRAJECTORY_DIR`.
3. **The per-rollout endpoint and its single-use key are set last**, so they beat anything inherited: `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`.

Everything else of yours passes through — your agent needs your credentials; that is the premise of running it in your process — **except** provider configuration the gateway cannot serve. `AZURE_OPENAI_*`, `GOOGLE_API_KEY`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`, `AWS_BEARER_TOKEN_BEDROCK`, `GROQ_API_KEY` and the rest of [`UNROUTABLE_PROVIDER_VARS`](https://github.com/percepteye-ai/agent-flywheel/blob/main/src/percepteye_agent_flywheel/isolation.py) are removed, and `serve()` warns at startup if it finds any. An agent that finds no provider configured raises in front of the person who configured it. **Silent off-gateway sampling is the outcome that must not happen**: a trajectory from an unobserved completion cannot back an on-policy objective while looking exactly like one that can.

Two limits, stated plainly:

- An agent reading a **different** variable name is invisible from here. Declare it: `endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY")`. Sniffing would be a guess, and a wrong guess is silent.
- A client constructed with an explicit `base_url=` **in your own code** beats every environment variable. The control plane's reported-versus-served call count is the check that actually decides; the variable list is not a proof of completeness.

---

## ③ Record, then report

Adapters bind **inside whichever process runs the agent**, because a callback registered in the parent would never see a child's tool calls. `adapters="auto"` binds every built-in whose framework is installed; a *named* adapter that cannot bind raises, because naming it asserts that it is how your calls are captured.

Anything that can append a JSON line can record. `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl` is the record the control plane grades; any other file left in that directory is collected as an artifact. A truncated final line is counted as malformed, not raised — the deadline kill lands mid-write, and a call we know executed and cannot read is not a call that did not happen.

For a CLI agent, tool calls are also recovered from its own [event stream](cli.md) — but **only when the trajectory recorded nothing**. A `record_tool_call` is a deliberate assertion about the agent's own behavior, and a stream parse must never overwrite one.

Reporting happens in a fixed order, and every step of it is deliberate.

**Artifacts first, whatever the outcome.** For a coding or REPL agent the files *are* the deliverable, and collecting only on the clean-exit path meant a rollout could write a correct solution and have it discarded because the process did not terminate tidily. Registration carries a digest and a size; small UTF-8 files travel *with* the report, because a grader that can verify a digest but never read a file proves integrity while grading nothing.

| cap | value | over it |
|:--|--:|:--|
| any single artifact | 256 MB | recorded as `skipped:too_large:<size>` |
| inlined per file | 64 KB | registered and named, not inlined |
| inlined per report | 256 KB total | registered and named, not inlined |

Symlinks are **never** collected — `is_file()` and `read_text()` both follow them, so a link the agent drops in its own directory would be read at its target and inlined into the report. They are recorded as `skipped:symlink` rather than dropped, because a link is itself a fact about the rollout.

**Then the substrate check.** A rollout whose every observed tool call failed is not a low-scoring attempt; it is an attempt that never got to happen, and reporting it as a legitimate zero teaches the policy from an environment fault. The rule is strict on purpose:

| tool calls | result |
|:--|:--|
| all `failed`, at least one | `substrate_unavailable` |
| any `unknown` | left alone — `unknown` is not failure |
| any `ok` | left alone — partial failure is ordinary agent behavior |
| `[]` or `null` | left alone — nothing was learned about the substrate |

It never overwrites an exclusion already set; an earlier, more specific reason knows more.

**Then the report — for every rollout, whatever happened to it.** A rollout that went badly is reported *with* its typed exclusion, so the server records the work product and still knows not to train on it. Abandoning would suppress the record entirely, and an operator asking "what did my agent do" would get silence for exactly the rollouts that went worst. Whether a typed reason means retry, exclude or train is the control plane's call, and it has the reason to make it with.

| exclusion | means |
|:--|:--|
| `entrypoint_error` | the Python entrypoint **raised**. Nothing usable ran; retry elsewhere |
| `agent_exit_nonzero` | the agent **ran** and exited non-zero. Its own failure, and evidence |
| `deadline_exceeded` | the budget was gone before or during the run |
| `credential_expired` | the lease lapsed mid-flight and the child was killed |
| `substrate_unavailable` | every observed tool call failed |
| `flywheel_crash` | the SDK's own failure — a non-zero child, an unparseable result |

`flywheel/1` also declares `cancelled` and `capacity`. **Nothing in this package emits them.**

The agent is stopped **10 seconds short of the deadline**, so there is time left to collect artifacts and land the report while the lease is still good. Running to the deadline exactly meant the kill and the expiry landed together, and the work was thrown away at the last step.

---

## ④ Score — the part you do not build

Grading happens on the control plane, against tasks generated from the tool surface beat ⓪ reported. The SDK's job ends at faithful observation; `POST /rollouts/{id}/result` returns the acknowledgment, and `rollout_reported` carries its `grade` and `duplicate` flag straight to your `on_event` callback.

**The control plane is the part you don't build — not a part you can't see.** Sign in and the tasks it generated, how each rollout scored, and what got promoted are all there to inspect.

---

## ⑤ Improve — the better agent runs the next lap

`serve()` reads `GET /policy/current` at startup and **prints it without applying it**. That is a correctness boundary, not a style choice: a donor rollout samples through the **per-rollout gateway**, which is what attributes and captures the completion, and the gateway already rewrites the model field to the policy under test. Pointing a donor at the policy endpoint would bypass capture and produce a run with nothing to train on.

Your *production* agent is where the policy is applied, in a different process calling different functions.

```python
kw = fw.apply_current_policy(agent_id="my-coding-agent")
if kw:                                    # {} means keep your current config
    client = OpenAI(base_url=kw["base_url"], api_key=kw["api_key"])
    model = kw["model"]
```

Training produces a better model **and** a better system prompt, gated by the same approval. They are separate reads on purpose: the policy is polled and identity-sized, a system prompt is unbounded text, and putting it in the poll would make every tick carry a payload the caller already has.

```python
src = fw.policy_source(agent_id="my-coding-agent", on_change=reconfigure)
snap = src.current()                      # per turn; between TTLs it is a cache read
prompt = fw.current_prompt(agent_id="my-coding-agent")
```

`policy_source()` exists for **rollback first, promotion second**. A delayed promotion is an inconvenience; a regressed candidate that can only be withdrawn by redeploying the customer's production agent is an incident, and it is an incident whose remedy sits in someone else's deploy pipeline. Nothing polls in the background — the refresh runs inside your call, no timer, no poller — and after its first success the handle keeps serving the last good policy, so a network blip can never silently demote production back to the untrained model.

This is the beat that makes it a flywheel rather than a pipeline: **the better agent generates the better evidence**, and the ring closes.

---

## The five lanes

Lane ⓪ opens once, and lane ④ once from `attach()`. The rest repeat, and **every one of them is a request your process makes**.

| lane | routes | opened |
|:--|:--|:--|
| **⓪ register** | `GET /health` · `POST /agents` | once, at startup |
| **① claim** | `POST /rollouts/claim` | in a loop, long-polled |
| **② return the rollout** | `POST /rollouts/{id}/heartbeat` · `/artifacts` · `/result` · `/abandon` | per rollout |
| **③ policy back** | `GET /policy/current` · `GET /prompt/current` | at startup, or on a TTL |
| **④ live evidence** | `POST /agents` · `GET /policy/current` · capture to disk · `POST /turns` **with consent only** | at startup, then every 32 turns and at exit |

!!! note "There is no listening socket in this package"

    ```bash
    grep -rnE "\.listen\(|HTTPServer|uvicorn|fastapi|flask" src/    # exits 1
    ```

    Zero inbound ports, zero public DNS, zero TLS termination on your side. For a flywheel inside a corporate network that is the difference between "install a package" and "file a change request with the network team."

!!! warning "Lane ④ does not ship yet"

    `attach()` binds the same adapters, writes the same tri-state into the same trajectory format, and scopes it to a live user turn instead of a claimed rollout. `flywheel/1` gained `/turns` for exactly this, so captured turns are uploaded — but only once an operator enables capture for that agent, which is off by default because a production turn carries your end users' own messages. Until then, and whenever the control plane is unreachable, they accumulate as directories under the `trajectory_root` you name and you own them. `trajectory_root` is required and undefaulted for exactly that reason: nothing deletes them unless you set `retention_turns`.

    Writing a client for a route the server does not serve is how an earlier SDK ended up with five modes, four of which were strings nobody read.

Everything fails open. An unreachable control plane costs you the observation and nothing else; `attach()` never raises on a control-plane failure, and `policy_current()` answering `{}` means *keep doing what you were doing*.

---

**Next:** [the tri-state outcome contract](outcomes.md) · [closing the loop in production](production.md) · [every knob and variable](configuration.md)
