<div align="center">

# 🎡 percepteye-agent-flywheel

### **Self-improving agents through continual learning.**

**Your agent keeps running in your process, behind your firewall, on your credentials — unchanged.**<br/>
One import. One call. One runtime dependency.

[![ci](https://img.shields.io/github/actions/workflow/status/percepteye-ai/agent-flywheel/ci.yml?branch=main&label=ci&style=for-the-badge&logo=githubactions&logoColor=white&color=22c55e)](https://github.com/percepteye-ai/agent-flywheel/actions)
[![pypi](https://img.shields.io/pypi/v/percepteye-agent-flywheel?style=for-the-badge&logo=pypi&logoColor=white&label=pypi&color=3b82f6)](https://pypi.org/project/percepteye-agent-flywheel/)
[![python](https://img.shields.io/badge/python-3.10%20→%203.13-3b82f6?style=for-the-badge&logo=python&logoColor=white)](https://pypi.org/project/percepteye-agent-flywheel/)
[![node](https://img.shields.io/badge/node-plugin-8b5cf6?style=for-the-badge&logo=nodedotjs&logoColor=white)](packages/percepteye-agent-flywheel/)
<br/>
[![runtime dependencies](https://img.shields.io/badge/runtime_deps-1-f59e0b?style=for-the-badge)](#what-it-costs-to-integrate)
[![lines changed in your agent](https://img.shields.io/badge/lines_changed_in_your_agent-0-ec4899?style=for-the-badge)](#zero-instrumentation-concretely)
[![tests](https://img.shields.io/badge/tests-509_·_0_skips-22c55e?style=for-the-badge)](#development)
[![license](https://img.shields.io/badge/license-MIT-14b8a6?style=for-the-badge)](LICENSE)

<img src="https://raw.githubusercontent.com/percepteye-ai/agent-flywheel/main/docs/assets/flywheel-overview.gif"
     width="900"
     alt="Your agent — prompt, tools and model — sits inside your infrastructure. fw.serve() wraps it from the outside with zero lines changed inside. Rollouts and observed outcomes go out; a trained policy comes back. The model row flips from your-model to a policy trained on this agent's own work, while the prompt and tools rows stay untouched.">

**[Quickstart](#quickstart-three-lines)** · [How the loop closes](#how-the-loop-closes) · [Python · Node · CLI](#runs-whatever-you-already-run) · [Zero instrumentation](#zero-instrumentation-concretely) · [Production](#close-the-loop-in-production) · [Harnesses](#harnesses-already-supported) · [Docs](https://docs.percepteye.ai/flywheel)

</div>

---

**Your agent already does real work. This turns that work into training data, and delivers what it learns back as an endpoint your agent already calls.**

No container to build. No cluster to run. No GPU to provision. No inbound port to open. Your agent does not move.

<br/>

## Quickstart: three lines

```bash
pip install percepteye-agent-flywheel        # one runtime dependency: httpx
```

```python
import percepteye_agent_flywheel as fw

fw.serve("my_package.agent:build", agent_id="my-coding-agent", input_shape="text")
```

**That is the whole integration.** You pass an importable `"module:attr"` factory so each rollout can rebuild your agent in a fresh child process. `serve()` claims rollouts, runs them, reports outcomes, and blocks until you stop it — see [starting and stopping](#configuration) for what "stop it" means.

Everything that can be misconfigured — entrypoint, credential, contract version, isolation mode — is checked **before the first rollout** and raises there, never mid-run.

> [!TIP]
> Rollouts are real runs with your real credentials, so bound the first one with `max_rollouts=8` and point `artifacts_dir=` somewhere you don't mind it writing. To stand the same image up without contributing, set `PERCEPTEYE_AGENT_MODE=production` — [one switch, both ways](#configuration).

<br/>

## CLI agents: three commands

**Your agent is a binary — goose, opencode, a Node CLI — so there is no `import` to add.** There is also nothing to run alongside it: no worker, no sidecar, no wrapper process. You start your agent the way you always did.

```bash
pip install percepteye-agent-flywheel     # 1. the wheel, one dependency
percepteye install-hooks                  # 2. register the plugin with your agent
export PERCEPTEYE_API_KEY=pefw_...        # 3. optional — see below
```

```bash
goose session                             # unchanged. That is the whole point.
```

**Step 3 is optional, and that is the design.** With no key, the plugin records your agent's real tool-failure rate to `~/.percepteye/turns/` and nothing leaves the machine — no account, no network, useful on day one. Add a key and the same install starts claiming rollouts and contributing them. One decision, not two.

### One switch owns the endpoint

Capturing a completion means your agent's traffic has to reach us — and for a binary, the only place that lives is its own config. So `install-hooks` sets it, and **the mode you are in decides what it says**:

> [!NOTE]
> **No key, no routing.** Your traffic is re-pointed only when you are in `training` **and** `PERCEPTEYE_API_KEY` is set. A keyless install records locally and leaves your endpoint exactly where it was — and unsetting the key later hands it back, with no mode flip needed. The mode is the switch; the key is the consent, and routing needs both.

```bash
PERCEPTEYE_AGENT_MODE=training      # our gateway — completions captured (the default)
PERCEPTEYE_AGENT_MODE=production    # your own endpoint, exactly as it was
```

**`production` restores the value you had**, not a default we picked. It is read once, before the first time we ever write to your config, and kept — so upgrading the wheel and re-running `install-hooks` cannot lose it.

**No command is needed to flip.** Set the variable and start your agent; the `SessionStart` hook reconciles the config before the session begins.

> [!IMPORTANT]
> goose resolves its endpoint **once, at process start**. A flip therefore takes effect on the **next** session, not the one already running. The hook re-checks on every session start, so the config is self-correcting rather than something you have to remember to repair.

**Your environment always wins.** goose ranks `OPENAI_HOST` and `OPENAI_BASE_URL` from the environment *above* its config file, so the instant escape hatch needs no command from us at all:

```bash
OPENAI_BASE_URL=https://my-own-proxy/v1 goose session   # we are not in the path
```

`percepteye status` prints where traffic will actually go on the next run — and says so explicitly if one of those variables is shadowing us, which is the failure that otherwise costs an hour.

<br/>

## How the loop closes

**Register once, then loop forever.** Three lanes cross the trust boundary, and **every one of them is opened from your side** — there is no listening socket anywhere in this package.

<p align="center">
  <img src="docs/assets/flywheel-architecture.svg" width="960"
       alt="Onboarding once, then the flywheel as a closed ring, split by a dashed trust boundary. At the top, once at startup: the SDK describes your agent — system prompt, tool schemas, sub-agent tree, read from inside the process — and lane ⓪ registers it on its own, giving the control plane the tasks worth training on, derived from the tool surface you actually have. A CLI agent with no object to read is warmed up once, well under a second, and described from that. Below, every lap: your agent runs the rollout and your production agent applies the policy on the left, inside your infrastructure; the control plane scores observed outcomes and trains the policy on the right. Three numbered lanes cross the boundary — ① it claims a task, ② your agent returns the rollout and its per-call outcomes, ③ the promoted policy comes back as an endpoint. A green arrow up the left side closes the ring: the better agent runs the next lap.">
</p>

| beat | what happens | who does the work |
|:--|:--|:--|
| **⓪ Onboard** | the SDK describes your agent from inside its own process and registers it — once, before the first rollout | automatic; `introspect=False` opts out |
| **① Claim** | your harness long-polls for a rollout; leases heartbeat, crashed workers redeliver | you dial out |
| **② Run** | fresh child process, own environment, single-use credential, own trajectory | your agent, unchanged |
| **③ Record** | every tool call lands as `ok` / `failed` / **`unknown`**; artifacts collected on every path | adapters, from the outside |
| **④ Score** | tasks generated from your real tool surface, graded against compiled checks | **Percepteye — you build none of it** |
| **⑤ Improve** | your production agent picks up the promoted policy; the next lap is fed by the next round of rollouts | one call at startup |

That last beat is what makes it a flywheel rather than a pipeline: **the better agent generates the better evidence**. And beat ⓪ is why you never write a task list: the tasks come from **the tool surface you actually have**, read after templating and conditional registration — the prompt your agent really assembled, the tools it really registered, the sub-agents it really built. A CLI agent with no object to read is warmed up once, well under a second, and described from that.

**And you can watch every lap of it.** The control plane is the part you don't build — not a part you can't see: sign in and the tasks it generated, how each rollout scored, and what got promoted are all there to inspect.

<br/>

## Runs whatever you already run

Three surfaces, one contract. Pick the row that matches your stack — all of them report the same tri-state outcomes into the same trajectory format, checked against the same reader in CI.

### 🐍 Python — an import path

```python
fw.serve("my_package.agent:build", agent_id="research-agent", input_shape="text")
```

Adapters are **on by default** (`adapters="auto"`) and bind every built-in whose framework is installed:

| framework | outcomes captured with **zero code in your agent** |
|:--|:--|
| **LangChain** | `ok` on `ToolMessage(status="success")` · `failed` on raise or `status="error"` · `unknown` otherwise |
| **LangGraph** | same, with each call attributed to the node that made it |
| **Google ADK** | `failed` on raise **and** on `{"error": ...}` · `unknown` otherwise, attributed via `ToolContext.agent_name` |

No adapter imports its framework until you bind it, so they add **zero dependencies**. Shipping your own is one entry-point factory in the `percepteye_agent_flywheel.adapters` group.

### ⬢ Node — a plugin the harness already loads

One npm package, two harnesses, **zero dependencies** (Node builtins only):

```bash
openclaw plugins install percepteye-agent-flywheel            # OpenClaw
dsh plugin --profile <name> add percepteye-agent-flywheel     # DeepSeek Harness
```

Then set `PERCEPTEYE_TRAJECTORY_DIR` on the process running your agent. **Unset, the plugin does nothing at all** — so the same install is a no-op in production and a recorder during a rollout. Leave it installed permanently.

**Both plugins observe and cannot do anything else** — a property of the hooks, not a promise about the code. OpenClaw's `after_tool_call` returns `void`; DeepSeek Harness publishes `tools/result` emit-only over a deep-frozen object. Capturing evidence can never change the behaviour being measured. → [`packages/percepteye-agent-flywheel/`](packages/percepteye-agent-flywheel/)

### ⌘ Any CLI, any language — an argv list

Not Python? A compiled binary, a Node CLI, anything with a command line. There is no protocol to implement, so an existing CLI needs **no changes**:

```python
fw.serve(["opencode", "run"], agent_id="coding-agent", input_shape="text")
fw.serve(["dsh", "--profile", "headless", fw.TASK], agent_id="dsh-agent", input_shape="text")
```

The task arrives on **stdin** and as `PERCEPTEYE_TURN_INPUT`; **stdout** is the final text. Mark an argv position with `fw.TASK` if your CLI takes the task as an argument. The whole **process group** is what gets killed on timeout, so a spawning agent leaves nothing behind.

**And you probably get tool outcomes for free**, read straight off the stream your CLI already prints for its own UI:

| vocabulary | events | emitted by |
|:--|:--|:--|
| `otel-genai` | `gen_ai.tool.start` / `gen_ai.tool.end` | **any agent already logging OpenTelemetry GenAI spans as JSON lines** |
| `prime-agent` | `tool_execution_start` / `tool_execution_end` | prime-agent under `--mode json`, verified against a live run |

The first row is the one that matters: it belongs to no vendor, so an agent instrumented for observability gets outcome capture with **no work at all**. Adding the next CLI is a `StreamFormat` entry — a vocabulary, not a parser. Capture survives clean exits, non-zero exits and timeout kills, because a rollout that died is exactly the one you need the tool calls from.

<br/>

## Zero instrumentation, concretely

This LangChain agent contains no reference to Percepteye of any kind:

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

Served with the three lines above, the rollout reports:

```json
"tool_calls": [{"name": "lookup", "arguments": {"q": "count to three"},
                "outcome": "unknown", "status_code": null,
                "output": "found: count to three", "latency_ms": 0.199}]
```

`unknown`, not `ok` — nothing observed whether the call actually succeeded, so nothing claims it did. **That is the honest floor zero instrumentation buys you**, and it is enforced by [`tests/test_zero_instrumentation.py`](tests/test_zero_instrumentation.py). One `record_tool_call` where you *do* know the answer gets you the rest:

```python
from percepteye_agent_flywheel import record_tool_call

record_tool_call("charge_card", {"amount": 4200}, outcome="ok", status_code=201)
```

<br/>

## `ok` · `failed` · **`unknown`** — the outcome nobody else has

Rewards in agent RL are usually computed from transcripts, and **transcripts lie about tool outcomes**. A LangChain tool with `handle_tool_error=True` swallows its exception and returns a string that reads like a normal result. HTTP 200. The tool failed. Score the status and a policy learns to believe it.

*Fuzzing RLVR Verifiers* ([arXiv:2606.01066](https://arxiv.org/abs/2606.01066)) measured false-positive rates of **0.869 on JSON tool calls** for buggy verifiers, against 0.000 for strict ones.

So this SDK makes the dangerous shape unconstructible:

- `outcome` and `status_code` are **required and non-defaultable** — you cannot record half an outcome by forgetting a keyword.
- `unknown` is legal, supported and expected. Verifiers that need a per-call outcome **abstain** on it. It is **never rounded up to `ok`**.
- A status may confirm or contradict an observed outcome. It may **never manufacture one**.
- A permission denial your API returns as HTTP 200 is `outcome="failed", status_code=200`. Both halves are true, both recorded.

Because the SDK is *in* your process, the trajectory can also carry **which entity a write produced** — a handful of opaque identifiers, never the response body — so a later check can settle whether the change actually stuck:

```json
{"name": "POST /rest/api/2/issue", "outcome": "unknown", "status_code": null,
 "entity_ids": {"id": "10001", "key": "ENG-7"}}
```

That is a different **class** of evidence, not a better rubric — and it is available because we run beside your agent rather than in place of it.

<br/>

## Close the loop in production

Training produces a better model *and* a better system prompt. This is where your agent picks them up.

**One shot, at startup:**

```python
kw = fw.apply_current_policy(agent_id="my-coding-agent")
if kw:                                    # {} means keep your current config
    client = OpenAI(base_url=kw["base_url"], api_key=kw["api_key"])
    model = kw["model"]
```

**A handle that keeps up** — re-polls on a TTL, so a promotion *or a rollback* reaches production with no redeploy:

```python
src = fw.policy_source(agent_id="my-coding-agent", on_change=reconfigure)
snap = src.current()                      # per turn; between TTLs it is a cache read
prompt = fw.current_prompt(agent_id="my-coding-agent")   # the approved prompt, co-optimized
```

**Nothing polls in the background.** No thread in this package — the refresh runs inside your call. After its first success, `policy_source()` keeps serving the last good policy, so a network blip can never silently demote production back to the untrained model.

**And your live traffic can be captured too — locally.** `attach()` records the turns you are already serving, with the same adapters, the same tri-state and the same trajectory format:

```python
px = fw.attach(agent_id="support-agent", trajectory_root="/var/lib/percepteye")

with px.turn(conversation_id=conv_id) as turn_id:
    reply = agent.invoke(user_text)       # your agent, unmodified
```

> [!IMPORTANT]
> **Turns captured by `attach()` are uploaded only if you turn capture on for that agent — and it is off by default.** A production turn is your end users' own message and the answer you gave them, so nothing is sent until an operator enables capture for the agent; the registration response tells the SDK whether it may. Until then turns accumulate as trajectory directories under the `trajectory_root` you name and go nowhere.
>
> When capture is on, `flush()` uploads turns that have an answer and `close()` flushes the rest. Nothing is sent at turn-scope exit — `record_answer()` is callable after the block closes, so an upload there would ship every turn before its answer existed. Uploads never raise into your process and never block a turn on the network.
>
> Training is fed by **rollouts**, not by production traffic: `serve()` (or the CLI hooks) claims a rollout, runs it, and reports the result — that report is the upload. So a continual-learning loop is repeated rounds of rollouts, and `attach()` today is local observability plus the policy read.
>
> Sending production traces to the control plane needs a mode that does not exist yet. When it lands, this section changes; until then, treat `attach()` as local.

**It captures. It never reroutes.** Adapters bind with no endpoint at all, so attaching cannot re-point a single byte of production traffic. Everything fails open: an unreachable control plane costs you the observation and nothing else.

> [!NOTE]
> That is a promise about **`attach()`**, and it holds because your Python process already hands us the turn. A **CLI agent** has no such seam — a binary's completions only become capturable if its endpoint points at us — so `percepteye install-hooks` *does* set one, and [`PERCEPTEYE_AGENT_MODE` owns it in both directions](#cli-agents-three-commands). Different mechanism, same rule: you can always see it, and you can always take it back.

**Telling the two apart in your own code** — `current_capture()` answers with three states and never collapses them:

```python
match fw.current_capture():
    case ("rollout", rid): ...      # inside a serve() rollout
    case ("turn", tid):    ...      # inside attach().turn(), in production
    case None:             ...      # neither
```

Branch on this, not on `current_rollout_id()` — that one answers the narrower "which trajectory am I appending to", and inside `attach().turn()` it returns that **turn's** id, so `if fw.current_rollout_id():` is true in production.

<br/>

## Harnesses already supported

| harness | surface | integration |
|:--|:--|:--|
| **LangChain · LangGraph · Google ADK** | 🐍 Python | adapter auto-binds — [zero code in your agent](#zero-instrumentation-concretely) |
| **OpenClaw** | ⬢ Node | `openclaw plugins install percepteye-agent-flywheel` |
| **DeepSeek Harness** | ⬢ Node | `dsh plugin --profile <name> add percepteye-agent-flywheel` |
| **prime-agent** | ⌘ CLI | argv + `--mode json` event stream — **[full worked example →](examples/prime_agent/)** |
| **goose** | ⌘ CLI | `percepteye install-hooks` — **[three commands, no worker →](#cli-agents-three-commands)** |
| **opencode** | ⌘ CLI | `fw.serve(["opencode", "run"], ...)` — nothing added to the binary |
| **anything else with a CLI** | ⌘ CLI | argv list; outcomes free if it logs [OTel GenAI events](#runs-whatever-you-already-run) |

[`examples/prime_agent/`](examples/prime_agent/) is the end-to-end one: a real TypeScript coding agent, with routing, event-stream parsing and completion gates.

<br/>

## What it costs to integrate

Every row is a structural property of this repository, checkable from it.

| | | how to verify |
|:--|:--|:--|
| Lines changed **inside** your agent | **0** | [`tests/test_zero_instrumentation.py`](tests/test_zero_instrumentation.py) |
| Lines of integration code | **1 import + 1 call** | [Quickstart](#quickstart-three-lines) |
| Runtime dependencies | **1** (`httpx`) | [`pyproject.toml`](pyproject.toml) + a clean-install CI job |
| Node plugin dependencies | **0** | Node builtins only |
| Inbound ports · public DNS · TLS | **0 · 0 · 0** | `grep -rnE "\.listen\(\|HTTPServer\|uvicorn\|fastapi\|flask" src/` exits 1 |
| Containers to build · GPUs to provision | **0 · 0** | your process; the control plane holds the GPUs |
| Environment variables required | **2** — URL and key, both overridable in code | [Configuration](#configuration) |
| Frameworks captured with no instrumentation | **LangChain · LangGraph · Google ADK · OpenClaw · DeepSeek Harness · any OTel-GenAI CLI** | [`adapters/`](src/percepteye_agent_flywheel/adapters/), [`event_stream.py`](src/percepteye_agent_flywheel/event_stream.py) |
| Languages | **Python by import path, anything else by argv** | [above](#runs-whatever-you-already-run) |
| Tests | **509, zero skips permitted in CI** | `pytest` + [`ci.yml`](.github/workflows/ci.yml) |
| License | **MIT** | [LICENSE](LICENSE) |

<details>
<summary><b>One dependency, in context</b> — core runtime deps, measured from the PyPI JSON API</summary>

<br/>

| SDK | core deps | with extras |
|:--|--:|--:|
| **percepteye-agent-flywheel** | **1** | **1** |
| langfuse | 8 | 8 |
| openpipe-art | 8 | 62 |
| braintrust | 10 | 32 |
| langsmith | 14 | 45 |
| weave | 15 | 97 |
| hud | 18 | 38 |
| verifiers | 26 | 36 |

Counts are not a quality judgement — but inside a live production process, an exact pin is not a dependency, it is a veto. This package ships one floor and no pins.
</details>

<br/>

## Configuration

<details>
<summary><b>Starting and stopping a flywheel</b> — there is no off switch, and that is the design</summary>

<br/>

**Starting is a call, not a mode.** Which surface you use *is* the boundary:

| you call | you contribute | outbound |
|:--|:--|:--|
| `fw.serve(...)` | **capacity** — control-plane tasks run on your credentials | `/health`, `POST /agents`, `GET /policy/current`, then claim in a loop |
| `fw.attach(...)` | **evidence** — outcomes from turns you already serve | `POST /agents` + `GET /policy/current`, once |
| `fw.apply_current_policy(...)` | **nothing** | one `GET /policy/current` |

**Contribute nothing, still get the trained model.** No `serve()`, no `attach()`, no registration:

```python
kw = fw.apply_current_policy(agent_id="my-agent")
prompt = fw.current_prompt(agent_id="my-agent")
```

**One switch turns capacity contribution on and off**, so the same deployed image does either without a code edit:

```bash
PERCEPTEYE_AGENT_MODE=production    # serve() returns 0 immediately. No claim, no connection.
PERCEPTEYE_AGENT_MODE=training      # contribute capacity (the default when unset)
```

**The same switch governs a CLI install**, so one variable means one thing everywhere:

| | `training` | `production` |
|:--|:--|:--|
| `fw.serve(...)` | claims and runs rollouts | returns 0 immediately |
| `percepteye install-hooks` | endpoint → our gateway; sessions claim and report | endpoint → **the one you had**; nothing claimed, nothing reported |

An unrecognised value **raises** rather than guessing — a typo that silently picked `training` would keep contributing on your credentials with nothing printed. See [CLI agents](#cli-agents-three-commands) for the one timing caveat.

In `production` the call returns **before opening any connection**, so a production deployment needs no control-plane URL and no key — turning training off never requires being configured for it. It returns rather than raising, because a switch that crashes the service it was meant to quiet is not a switch, and it prints one line saying which mode it took, because a silent no-op is the failure mode this package refuses everywhere else. An unrecognised value raises instead of guessing: both ways of guessing wrong are invisible from the outside.

`training` · `train` · `rollout` · `rollouts` · `fine-tuning` · `finetuning` · `improving` all mean contribute. `production` · `prod` · `serving` all mean don't. Passing `mode=` in code beats the variable, and the printed line tells you when that happened — in both directions.

> [!NOTE]
> This switch gates **`serve()`** — the surface that contributes capacity. It does not gate `attach()`, which *is* the production surface and captures evidence from the turns you already serve. Gating that on the same variable would mean the correct setting for a production agent was `PERCEPTEYE_AGENT_MODE=training`. To stop capture, don't call `attach()`, or pass `register=False` to keep it entirely local.

**Stopping a run that is already going.** The switch is read at startup, so flipping it does not stop a loop that is mid-flight. For that:

- **`max_rollouts=N` is the only clean exit.** It is a completion counter checked at the top of the loop; there is no env equivalent and no way to change it on a running process.
- **There is no signal handler.** Nothing in the package catches SIGINT or SIGTERM. `SIGTERM` ends the process; in-flight rollouts are **orphaned rather than abandoned**, and the control plane reclaims them by lease expiry (up to 30 minutes) rather than by an `abandon` call.
- **Ctrl-C is not instant.** In-flight rollouts drain first, for up to `rollout_timeout_s` (default 900s). A subprocess-isolation child takes the same SIGINT and is reported as `flywheel_crash`, so an operator's Ctrl-C is recorded as the SDK crashing. A command-entrypoint child runs in a new session, does not receive the terminal's SIGINT, and outlives the parent.
- **Revoking the API key does not stop it.** A 401 is terminal per request, but the loop catches it, emits `claim_failed`, sleeps 5s and continues — indefinitely.
- **The control plane cannot stop it either.** The claim response carries only `rollouts` and `poll_after_ms`; there is no pause or drain field. The strongest server-side levers are an empty queue and a longer backoff, and both leave you polling.

Your production agent is unaffected by any of this — it is a different process calling different functions.
</details>

<details>
<summary><b>Watching a flywheel</b> — <code>on_event</code> and the full event vocabulary</summary>

<br/>

`serve(on_event=...)` takes a `Callable[[str, dict], None]`. It is called **synchronously, on whichever thread produced the event** — the serving thread for `claim_*`, a pool worker for the per-rollout kinds, and the heartbeat thread for `lease_lost`/`heartbeat_failed` — so a slow callback slows that thread. Exceptions it raises are swallowed: an observability hook must not be able to change what it observes, and the heartbeat thread is renewing every in-flight lease. Keep it cheap.

```python
fw.serve(..., on_event=lambda kind, ev: print(kind, ev))
```

| event | when |
|:--|:--|
| `claim_empty` | a successful claim returned no work — carries `poll_after_ms` |
| `claim_failed` | the claim itself failed — carries `error` |
| `rollout_start` | a claimed rollout began |
| `rollout_reported` | the result was accepted; carries the rollout's typed exclusion if it had one |
| `rollout_crash` | the rollout raised; it was abandoned |
| `report_failed` · `artifact_failed` | the work happened, delivering it did not |
| `lease_lost` · `heartbeat_failed` | the lease lapsed or could not be renewed |

Without a callback a flywheel is silent, and **an empty queue and a broken claim route look identical** — `claim_empty` versus `claim_failed` is what tells them apart. This is the answer to "how do I know it's working."
</details>

<details>
<summary><b>Isolation, entrypoints, input shapes, environment variables</b></summary>

<br/>

**Isolation.** `isolation="subprocess"` is the default and the only mode with a real timeout — a child process can actually be killed. Your environment passes through (your agent needs your credentials); the `PERCEPTEYE_` namespace never is, so the long-lived key stays in the parent. `isolation="inprocess"` accepts an already-constructed object, capped at `concurrency=1`.

**Entrypoints.** Probed in order: `invoke`, `run`, `execute`, `chat`, `__call__`. Pass `method="..."` to name one. Resolution happens once, at startup.

| `input_shape` | your entrypoint receives |
|:--|:--|
| `"text"` | `"the task description"` |
| `"messages"` | `[{"role": "user", "content": "..."}]` |
| `"dict"` | `{"input": "..."}` |

**Reporting.** Return a string, or a `RolloutOutput(final_text=..., artifacts={...}, reward=...)` when you have more to say. Crashes, expired credentials and blown deadlines come back as **typed exclusions**, so a substrate fault is never scored as a bad decision by your agent.

**Concurrency.** `concurrency=N` under the default subprocess isolation.

| Variable | Meaning |
|:--|:--|
| `PERCEPTEYE_CONTROL_PLANE_URL` | Control-plane base URL. |
| `PERCEPTEYE_API_KEY` | Your flywheel key. |
| `PERCEPTEYE_AGENT_MODE` | `training` (default) or `production`. In production `serve()` returns 0 without connecting. Gates capacity contribution only — not `attach()`. A value it cannot parse raises rather than being guessed. |
| `PERCEPTEYE_INTROSPECT` | Set to `0` to disable [agent discovery](#configuration). |
| `PERCEPTEYE_ROLLOUT_ID` | *Set by us* for the child. Read it to tag your own logs. |
| `PERCEPTEYE_TRAJECTORY_DIR` | *Set by us* for the child — already this rollout's directory. Append to `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`; any other file you leave there is collected as an artifact. |
| `PERCEPTEYE_TURN_INPUT` | *Set by us* for a command entrypoint. The task, also on stdin. |

Two variables change what the SDK does — `PERCEPTEYE_AGENT_MODE` and `PERCEPTEYE_INTROSPECT`. The rest are addresses and credentials.

**Your agent must sample through an OpenAI-compatible client**, because the per-rollout gateway speaks that API. Provider variables we cannot point at it are removed from the child ([`UNROUTABLE_PROVIDER_VARS`](src/percepteye_agent_flywheel/isolation.py)), and `serve()` warns at startup if it finds any — a loud failure beats an agent quietly sampling its live provider unobserved.
</details>

<details>
<summary><b>Agent discovery</b> — on by default, and how to turn it off</summary>

<br/>

Once at startup, the SDK reads the *shape* of the agent you handed `serve()` and sends it up: system prompt (truncated at 20k, callables never invoked), tool names/descriptions/JSON Schemas, provider and model name, sub-agents to depth 4, and the framework matched.

**Never sent:** your tools' source or return values, your credentials, your data, or anything observed during a rollout.

**Why it's on:** it is what lets the platform generate tasks from the workflows your tools actually make possible, instead of asking you to enumerate them. An agent that reports nothing gets generic tasks.

```python
fw.serve(..., introspect=False)      # or PERCEPTEYE_INTROSPECT=0
```

Only a deliberate value disables it (`0`, `false`, `no`, `off`) — a typo leaves it on rather than silently switching capture off while you believe it is running. A framework we cannot read reports `discovered_agent: null`, never an empty agent.
</details>

<details>
<summary><b>OpenTelemetry</b> — optional, egress only</summary>

<br/>

```bash
pip install "percepteye-agent-flywheel[otel]"
```

Every outcome is additionally emitted as a span, so it lands in whatever you already run — Datadog, Phoenix, Grafana — beside the rest of your agent's telemetry. The extra is the **API only, never the SDK**: this package emits into your registered `TracerProvider` and **never installs one**. Spans carry the GenAI semantic conventions plus `percepteye.tool.outcome` for the tri-state, and an `unknown` leaves the span status unset. Nothing is ever read back — outcomes reach the control plane through the trajectory regardless.
</details>

<br/>

## The contract

[`schema/flywheel-1.json`](schema/flywheel-1.json) is the authority, and this repository owns it — server implementations pin against the published schema, never the other way round. The Python types in [`contract.py`](src/percepteye_agent_flywheel/contract.py) are hand-written to match it and the suite asserts the two agree, so a value the code can emit and the schema does not declare fails the build rather than a customer's report.

Absence is never collapsed into zero: `tool_calls=None` means "I did not report tool calls"; `[]` means "I observed zero", which is a positive claim.

<br/>

## Development

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev,adapters-test,otel,otel-test]"
.venv/bin/pytest
./scripts/cross_language_abi_check.sh     # the JS suite, against the same Python reader
```

509 tests at roughly a 1:1 test-to-source line ratio. Five CI jobs, each protecting a specific claim on this page:

| job | what it catches |
|:--|:--|
| `test` | a bare install — nothing but `httpx` — breaking on Ubuntu or macOS, Python 3.10→3.13 |
| `test-full` | **zero skips permitted** — a skipped test and a passing test look identical in a green check |
| `clean-install` | the runtime dependency list growing past `httpx`, asserted from installed metadata |
| `boundary` | an internal system name reaching the published package, including inside an error string |
| `node-plugins` | the OpenClaw and DeepSeek Harness plugins drifting from the Python reader |

The animations are generated, not drawn — [`docs/assets/render.sh`](docs/assets/render.sh) rebuilds them so the checked-in GIFs cannot drift from their source.

## Contributing

Issues and pull requests welcome. For a new framework adapter the bar is the one the built-ins meet: no framework import at module scope, report `unknown` rather than guessing, and fail loudly at bind time rather than degrade silently.

## Security

Please report vulnerabilities privately to **security@percepteye.ai** rather than in a public issue.

## License

[MIT](LICENSE) © Percepteye.
