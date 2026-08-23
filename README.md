# percepteye-agent-flywheel — in-process agent RL, outbound-only

**percepteye-agent-flywheel is a Python SDK that runs your own agent against evaluation and reinforcement-learning rollouts supplied by the Percepteye control plane — without moving your agent.** It keeps running in your process, behind your firewall, on your credentials. One runtime dependency, one import, one call. No inbound port, no container to build, no GPU to provision.

<p align="center">
  <img src="https://raw.githubusercontent.com/percepteye-ai/agent-flywheel/main/docs/assets/flywheel-loop.gif"
       width="900"
       alt="The Percepteye agent flywheel loop. On the left, inside a box labelled YOUR INFRASTRUCTURE, three components: a flywheel harness running fw.serve(), the agent it spawns as an isolated child process with a tool-outcome trajectory recording ok, failed and unknown, and the customer's own production agent calling apply_current_policy(). On the right, across a dashed trust boundary, a single opaque box labelled Percepteye control plane, which generates the tasks, scores every rollout and trains the policy. Four lanes cross the boundary — claim a rollout, completions, result and outcomes, and current policy — and every one of them is opened from the left, with responses returning along the same connection. No arrow originates on the Percepteye side. Caption: no inbound port, no public DNS, no TLS to terminate.">
</p>

[![ci](https://img.shields.io/github/actions/workflow/status/percepteye-ai/agent-flywheel/ci.yml?branch=main&label=ci&style=flat-square)](https://github.com/percepteye-ai/agent-flywheel/actions)
[![pypi](https://img.shields.io/pypi/v/percepteye-agent-flywheel?style=flat-square&label=pypi)](https://pypi.org/project/percepteye-agent-flywheel/)
[![python](https://img.shields.io/pypi/pyversions/percepteye-agent-flywheel?style=flat-square)](https://pypi.org/project/percepteye-agent-flywheel/)
[![runtime dependencies: 1](https://img.shields.io/badge/runtime%20dependencies-1-blue?style=flat-square)](#what-does-it-cost-to-integrate)
[![license: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

[Quickstart](#quickstart-one-import-one-call) · [How it works](#how-the-flywheel-works) · [Zero instrumentation](#zero-instrumentation-langchain-langgraph-google-adk-and-cli-agents) · [Why `unknown` matters](#why-unknown-is-a-first-class-tool-outcome) · [Comparison](#how-is-this-different-from-langsmith-braintrust-hud-art-or-nemo-data-flywheel) · [FAQ](#faq) · [Docs](https://docs.percepteye.com/flywheel)

---

## Install

```bash
pip install percepteye-agent-flywheel
```

Python 3.10, 3.11, 3.12 or 3.13. The install pulls exactly one runtime dependency, `httpx`. That is not a boast — it is a constraint the test suite enforces, because this package runs inside your production process and every dependency it adds is a version conflict it imposes on you.

## Quickstart: one import, one call

Point your credentials at the control plane and serve:

```bash
export PERCEPTEYE_CONTROL_PLANE_URL="https://api.percepteye.com/api/flywheel/v1"
export PERCEPTEYE_API_KEY="pek_..."
```

```python
import percepteye_agent_flywheel as fw

fw.serve("my_package.agent:build", agent_id="my-coding-agent", input_shape="text")
```

That is the whole integration. **Zero lines change inside your agent.** You pass an importable path — `"module:attr"` — so each rollout can be rebuilt in a fresh child process; `input_shape` declares how the task text should reach your entrypoint.

`build` must be a **zero-argument factory or a class**, because the child calls it to construct a fresh agent per rollout. Pointing the path at the agent callable itself would invoke it with no arguments.

`serve()` blocks, claiming rollouts until you stop it, and returns the number completed. It is quiet by design — the one line it prints is the policy your rollouts will be graded against, when the control plane has one deployed:

```
[percepteye] policy: candidate-3 — promoted 2026-08-19 → policy at https://gw.percepteye.com/r/...
```

To watch individual rollouts, pass `on_event`. It fires with `("rollout_start" | "rollout_reported" | "rollout_crash" | "claim_failed" | "lease_lost" | "heartbeat_failed" | "report_failed" | "artifact_failed", payload)`:

```python
fw.serve(
    "my_package.agent:build",
    agent_id="my-coding-agent",
    input_shape="text",
    on_event=lambda kind, ev: print(kind, ev),
)
```

If nothing arrives at all, `serve()` has already told you why. Everything that can be wrong with the configuration — the entrypoint does not resolve, the credential is missing, the control plane speaks a different contract version, the isolation mode is incompatible with the entrypoint you passed — is checked **before the first rollout**, and raises there rather than mid-run.

<details>
<summary><b>Prefer to keep credentials out of the environment?</b></summary>

```python
import percepteye_agent_flywheel as fw

fw.serve(
    "my_package.agent:build",
    agent_id="my-coding-agent",
    input_shape="text",
    control_plane_url="https://api.percepteye.com/api/flywheel/v1",
    api_key="pek_...",
)
```
</details>

## How the flywheel works

The diagram above animates one full lap. In prose, because a picture is not indexable and half of you are reading this through a model:

1. **Claim.** Your harness long-polls the control plane for a rollout. The lease is renewed by heartbeat, and a crashed worker's rollout is redelivered rather than lost. Nothing listens on your side — you opened the connection.
2. **Isolate.** Each rollout gets its own child process, with its own environment, its own single-use credential and its own trajectory directory. A child process can actually be killed, which is why it is the default: Python cannot interrupt a synchronous call in another thread, so in-process a hung agent means you stop *waiting* while it keeps running.
3. **Route.** Your agent's LLM completions go to a per-rollout endpoint. That is what attributes the work, captures the completion wire, and lets the platform score what the model actually sampled rather than what a transcript says it did.
4. **Record.** Every tool call lands in the trajectory as `ok`, `failed` or `unknown`. See [why that third value exists](#why-unknown-is-a-first-class-tool-outcome).
5. **Report.** Final text, artifacts, timings and outcomes go back. Crashes, expired credentials and blown deadlines are reported as **typed exclusions**, so a substrate fault is never scored as a bad decision by your agent.
6. **Score.** Percepteye grades the rollout against its own compiled checks. This is the part you do not have to build.
7. **Improve.** Your *production* agent — a different process from the one donating capacity — calls `apply_current_policy()` and picks up what training produced. That closes the loop.

**Every arrow that crosses the trust boundary is opened from your side.** There is no listening socket anywhere in this package; you can check that claim in about five seconds:

```bash
grep -rnE "socket\.socket|\.listen\(|HTTPServer|socketserver|uvicorn|fastapi|flask|start_server" src/
# exits 1 — no matches. `import socket` appears nowhere in this package.
```

## What does it cost to integrate?

The claims in this table are structural properties of the package, not estimates. Each one is checkable from the repository.

| | percepteye-agent-flywheel | how to verify |
|---|---|---|
| Runtime dependencies | **1** (`httpx`) | [`pyproject.toml`](pyproject.toml); enforced by [`tests/test_oss_boundary.py`](tests/test_oss_boundary.py) and the `clean-install` CI job |
| Lines of integration code | **1 import + 1 call** | [Quickstart](#quickstart-one-import-one-call) |
| Lines changed inside your agent | **0** | [`tests/test_zero_instrumentation.py`](tests/test_zero_instrumentation.py) |
| Inbound ports opened | **0** | the `grep` above |
| Public DNS records required | **0** | — |
| Containers to build | **0** | your agent runs as your process, not our image |
| GPUs you provision | **0** | the control plane holds them |
| Zero-instrumentation capture | **3 Python frameworks** — LangChain, LangGraph, Google ADK — **plus any CLI agent** that already prints a structured event stream | [`adapters/`](src/percepteye_agent_flywheel/adapters/), [`event_stream.py`](src/percepteye_agent_flywheel/event_stream.py) |
| Environment variables required | **2** (URL and key), both overridable in code | [Configuration](#configuration-reference) |
| Tests | **278 across 23 files**, all of them run in CI with zero skips permitted; the bare-install matrix adds 2 operating systems × 4 Python versions | `pytest` and [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| License | MIT | [LICENSE](LICENSE) |

### One dependency, in context

The dependency count is the number that decides whether an SDK can go into a live service at all. Measured directly from the PyPI JSON API on 2026-08-22, latest published version of each package:

| SDK | core runtime deps | total incl. extras |
|---|---|---|
| **percepteye-agent-flywheel** | **1** | **1** |
| [langfuse](https://pypi.org/project/langfuse/) | 8 | 8 |
| [openpipe-art](https://pypi.org/project/openpipe-art/) | 8 | 62 |
| [braintrust](https://pypi.org/project/braintrust/) | 10 | 32 |
| [langsmith](https://pypi.org/project/langsmith/) | 14 | 45 |
| [weave](https://pypi.org/project/weave/) | 15 | 97 |
| [hud](https://pypi.org/project/hud/) | 18 | 38 |
| [verifiers](https://pypi.org/project/verifiers/) | 26 | 36 |
| [arize-phoenix](https://pypi.org/project/arize-phoenix/) | 51 | 87 |

Counts are not a quality judgement — Phoenix is a server and is *supposed* to be heavy. But three of those SDKs ship exact pins (`hud` pins `fastmcp==3.2.0`, `prompt-toolkit==3.0.51` and `questionary==2.1.0`) and several ship model-vendor SDKs of their own. Inside a production process, an exact pin is not a dependency, it is a veto.

## Zero instrumentation: LangChain, LangGraph, Google ADK, and CLI agents

Adapters are **on by default**. `adapters="auto"` binds every built-in whose framework is installed, so a LangChain, LangGraph or Google ADK agent reports per-call outcomes with no Percepteye code in it at all — not one import, not one `record_tool_call`.

| Adapter | Covers | Outcomes it can report | Routes LLM calls |
|---|---|---|---|
| `adk` | Google ADK | `failed` on raise **and** on `{"error": ...}`; `unknown` otherwise | no — the child environment routes |
| `langchain` | LangChain, LangGraph | `failed` on raise or an explicit `ToolMessage` status; `unknown` otherwise | only when you pass `llm=` |

Adapters are selected by **name**, because an adapter cannot cross a process boundary and the default isolation runs your agent in a child. `adapters=[]` binds none. Naming one explicitly is a stronger statement — you are asserting that is how your calls are captured, so an adapter that cannot bind raises `RoutingNotInstallable` at startup rather than degrading silently.

They add **no dependencies**: no adapter imports its framework until you bind it, so an absent framework simply fails that adapter's probe.

Here is what that actually produces. This LangChain agent contains no reference to Percepteye of any kind:

```python
# lc_agent.py — your existing agent, unmodified
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

Served with the three-line script above, the rollout reports:

```json
"tool_calls": [{"name": "lookup", "arguments": {"q": "count to three"},
                "outcome": "unknown", "status_code": null,
                "output": "found: count to three", "latency_ms": 0.199}]
```

**`unknown`, not `ok`** — because LangChain returned raw content with no `ToolMessage` status, so nothing observed whether that call succeeded. That is the honest floor zero instrumentation buys you. One `record_tool_call` where you actually know the answer gets you the rest.

### …and CLI agents, from the stream they already print

An adapter binds to a Python object. A Node CLI or a compiled binary has no object to bind to, so its tool calls used to be unobservable and `reports_tool_calls=False` was the only honest setting.

But a modern CLI agent already emits a structured event stream, because it needs one for its own UI. So [`event_stream.py`](src/percepteye_agent_flywheel/event_stream.py) reads the stream you are already printing. Two vocabularies are recognised today:

| vocabulary | events | emitted by |
|---|---|---|
| `prime-agent` | `tool_execution_start` / `tool_execution_end` | prime-agent under `--mode json`, verified against a live run |
| `otel-genai` | `gen_ai.tool.start` / `gen_ai.tool.end` | any agent that logs OpenTelemetry GenAI spans as JSON lines |

The second matters more than the first. It is the one vocabulary that belongs to no single vendor, so **an agent already instrumented for observability gets outcome capture with no work at all** — which is the point.

Four rules govern it, and they are the same rules as everywhere else in this package:

- **A stream is matched by the events it carries**, never by position and never by an agent name you declared. Telling us what you are does not make it so.
- **Tri-state by omission.** `isError: true` → `failed`, `false` → `ok`, **absent → `unknown`**. An agent that ends a call without saying whether it worked has not asserted success, and we do not assert it on the agent's behalf.
- **The trajectory wins.** A `record_tool_call` is a deliberate assertion about your own behaviour, so a stream parse never overwrites it. The stream is consulted only where the trajectory recorded nothing — and a stream carrying no tool events leaves `tool_calls` as `None`, never `[]`.
- **Capture survives failure.** Outcomes are recovered on a clean exit, on a non-zero exit, and on a timeout kill — because a rollout that died is exactly the one you need the tool calls from.

Supporting the next CLI agent is a `StreamFormat` entry — a vocabulary, not a parser. Nothing about the extraction, the tri-state or the absence rules gets reimplemented per agent.

In a multi-agent graph each call is attributed to the agent that made it, read from the framework's own run metadata — ADK exposes `ToolContext.agent_name` directly, LangGraph stamps the node it is executing. A call nothing names stays unattributed, because filling it in would train a planner on a coder's tool calls.

To ship your own, expose a factory in the `percepteye_agent_flywheel.adapters` entry-point group.

## Your agent doesn't have to be Python

If it isn't — a compiled binary, a Node CLI, anything with a command line — pass an argv list instead of an importable path. There is no protocol to implement, so an existing CLI needs no changes:

```python
fw.serve(["opencode", "run"], agent_id="coding-agent", input_shape="text")
```

The task arrives two ways, because CLIs differ and guessing wrongly is silent: on **stdin**, and as `PERCEPTEYE_TURN_INPUT`. Its **stdout** is the final text. Its exit code is a result — non-zero is excluded from training, because a crashed run says nothing about the policy.

**You probably get tool outcomes for free.** If your CLI already prints a structured event stream — prime-agent's `--mode json`, or anything logging OpenTelemetry GenAI events as JSON lines — they are [read straight off stdout](#zero-instrumentation-langchain-langgraph-google-adk-and-cli-agents) with nothing added to the agent.

If it prints nothing useful, the trajectory is still there and a shell can append to it:

```sh
printf '{"name":"deploy","arguments":{},"outcome":"ok","status_code":200}\n' \
  >> "$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl"
```

The command runs in its own **process group**, and that group is what gets killed on timeout — a real CLI agent spawns freely, and killing only the process we launched leaves its children holding the output pipe open.

`gate_flag=` names how your CLI spells its completion-gate option (prime-agent: `--autonomous-gate`). The flag is a fact about your binary and cannot be guessed; the gate *command* is a property of the task and arrives per rollout. Leave it unset for an agent with no gate concept.

A worked example against a real TypeScript coding agent — routing, event parsing, gates and all — is in [`examples/prime_agent/`](examples/prime_agent/).

## Why `unknown` is a first-class tool outcome

This is the part worth reading carefully, and it is the part nothing else in this category does.

**Rewards in agent RL are usually computed from transcripts. Transcripts lie about tool outcomes.** Consider a tool configured to swallow its own exceptions and return a string that reads like a normal result — in LangChain, that is `handle_tool_error=True`, a single documented keyword. The HTTP status is 200. The tool failed. Score the status and that failed call becomes a successful one in your training data, and a policy trained on it learns to believe it.

The literature calls this a **silent false positive**. *Before the Model Learns the Bug: Fuzzing RLVR Verifiers* ([arXiv:2606.01066](https://arxiv.org/abs/2606.01066)) measured false-positive rates for buggy verifiers of **0.869 on JSON tool calls**, 0.832 on math and 0.557 on code unit tests, against 0.000 for strict reference verifiers.

So this SDK makes the dangerous shape unconstructible:

```python
from percepteye_agent_flywheel import record_tool_call

record_tool_call(
    "charge_card",
    {"amount": 4200},
    outcome="unknown",     # "ok" | "failed" | "unknown" — required, no default
    status_code=None,      # required keyword, may be None
)
```

- `outcome` and `status_code` are both **required and non-defaultable**. You cannot record half an outcome by forgetting a keyword.
- `outcome="unknown"` is legal, supported and expected. It is recorded as unknown, the verifiers that depend on a per-call outcome **abstain** on it, and it is **never rounded up to `"ok"`**.
- When the outcome is `unknown`, any status is discarded — from the field *and* from the result body, because a scorer that recovers a status by parsing the envelope would find it there. **A status may confirm or contradict an observed outcome. It may never manufacture one.**
- The corollary holds too: an application-level permission denial that the API returns as HTTP 200 is `outcome="failed", status_code=200`. Both halves are true and both are recorded.

Unknown outcomes lower the grade — they do not exclude the rollout. So capture with no code gets you an honest floor, and one `record_tool_call` at the point where you actually know the answer gets you the rest.

The same refusal to collapse absence into zero runs through the whole contract: `tool_calls=None` means "I did not report tool calls"; `tool_calls=[]` means "I observed zero", which is a positive claim. The default is `None`. `tool_definitions: null` means "we could not look"; `[]` means "we looked, and there are none".

## The last mile: your production agent picks up what training produced

Training produces a better model. This is where your agent finds out.

```python
import percepteye_agent_flywheel as fw
from openai import OpenAI

kw = fw.apply_current_policy(agent_id="my-coding-agent")
if kw:                                  # {} means nothing is deployed — keep your config
    client = OpenAI(base_url=kw["base_url"], api_key=kw["api_key"])
    model = kw["model"]
```

What comes back is an **endpoint**, because that is the only form an agent can act on — an artifact URI is not callable. It returns `{}` when the control plane has nothing deployed *or* cannot answer, and you should read that as "keep your current configuration". An unreachable control plane must never stop your agent from serving its users.

> [!IMPORTANT]
> `apply_current_policy()` is for your **production** agent — never for `serve()`. A serve rollout samples through the per-rollout gateway, which is what attributes and captures the completion, and the gateway already rewrites the model field to the policy under test. Pointing a donor agent at the policy endpoint directly would bypass capture entirely and produce a run with nothing to train on.

## What results should you expect?

An honest answer, in three parts.

**What this repository can prove today.** Everything in [the integration table above](#what-does-it-cost-to-integrate): one dependency, one call, zero agent changes, zero inbound ports, 278 passing tests (195 of them on a bare install — the adapter and OpenTelemetry suites skip unless you install those extras). Those are properties of the code and you can check every one of them yourself.

**What it cannot prove yet.** Task success rate, reward improvement, latency or cost deltas for *your* agent. This repository ships no benchmark numbers, and we would rather have the empty table below than a number you cannot reproduce. When we publish, we publish in the form the category's credible results take — `before → after, on a named task, with a named model`:

| Agent | Task suite | Model | Before | After | Rollouts | Date |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

**What this class of technique has produced elsewhere.** These are other vendors' published numbers on their own systems, linked so you can read the methodology, and they are not claims about Percepteye:

- OpenAI reinforcement fine-tuning: Ambience Healthcare ICD-10 assignment, o3-mini **0.39 → 0.57** against a 0.45 physician baseline; Harvey legal-citation F1 **0.563 → 0.6765** ([RFT use cases](https://developers.openai.com/api/docs/guides/rft-use-cases)).
- NVIDIA NeMo Data Flywheel with AT&T: "up to 40% improvement in accuracy" and an "84% decrease in call center analytics cost" ([case study](https://www.nvidia.com/en-us/case-studies/att-drives-ai-agents-with-nemo/)).
- And the counterweight, because a section that only cites wins is marketing: TensorZero found RFT **100–700× more expensive than SFT** on equivalent datasets, and that on customer-service tasks it *degraded* performance ([write-up](https://www.tensorzero.com/blog/is-openai-reinforcement-fine-tuning-rft-worth-it/)).

### Measure it yourself

You do not have to take a number on faith when the SDK can produce yours. Serve a bounded batch against the base policy, then against the deployed one, and compare:

```python
import percepteye_agent_flywheel as fw

rows = []
fw.serve(
    "my_package.agent:build",
    agent_id="my-coding-agent",
    input_shape="text",
    max_rollouts=100,                       # bounded — returns when the batch is done
    reports_tool_calls=True,                # assert per-call outcomes, so outcome-dependent checks can grade
    on_event=lambda kind, ev: rows.append((kind, ev)),
)
```

`on_event` gives you every rollout as it completes, including its typed exclusion if it had one. Score the two batches with your own metric, and note the exclusions separately — a rollout excluded as `substrate_unavailable` or `entrypoint_error` is not a loss, it is a run that never happened.

## How is this different from LangSmith, Braintrust, HUD, ART or NeMo Data Flywheel?

The short version: **the observability tools watch your agent but never close the loop to weights; the RL platforms close the loop but make your agent move into their container or their cluster.** This one leaves the agent where it is.

| | agent runs in | supplies the tasks | supplies the scoring | core deps | inbound port |
|---|---|---|---|---|---|
| **percepteye-agent-flywheel** | **your own process** | yes | yes | **1** | **none** |
| HUD | their container image | yes | yes | 18 | none |
| Prime Intellect `verifiers` | wherever, you build the env | community hub | you write the rubric | 26 | none |
| OpenPipe ART / Tinker | wherever, you write the loop | no | no | 8 / hosted | none |
| LangSmith · Braintrust · Langfuse · Phoenix | your own process | no | you write the evals | 14 · 10 · 8 · 51 | none |
| NVIDIA NeMo Data Flywheel | your Kubernetes cluster | no | yes | k8s + 6× H100 to start | n/a |
| AWS Bedrock AgentCore | your AWS account | no | yes | AWS-bound | n/a |

Three honest caveats on that table. **Line count is table stakes** — HUD's decorator is about eight lines and Weave advertises one, so nobody wins on that alone; the differentiator is one call *with one dependency and no container*, which is why the table has a deps column. Read the row, not the cell. **Task supply is not equivalent** — everyone in the "yes" column supplies tasks for environments they built, whereas [agent discovery](#agent-discovery-on-by-default) derives them from your actual tool surface. And **NeMo and AgentCore are not really competitors** so much as the enterprise-platform version of the same idea; if you already run the cluster, they are a reasonable answer.

## Agent discovery, on by default

At startup — once, before any rollout — this SDK inspects the agent object you handed `serve()` and sends a description of it to the control plane. **This is on by default and you should decide whether you want it.** It is also the sharpest edge on the outbound-only claim above, which is why it is documented here and not in a footnote.

**What is sent:**

| | |
|---|---|
| `system_prompt` | Your agent's instruction text, truncated at 20,000 characters. A callable instruction is **not invoked** — it is reported as unread. |
| `tool_definitions` | Each tool's name, description and input JSON Schema — its declaration, not its implementation. |
| `discovered_model_config` | Provider and model name; temperature when readable. |
| `sub_agents` | The same, recursively, for a multi-agent graph, to a depth of 4. |
| `agent_name`, `framework` | What your agent calls itself, and which framework we matched. |

**What is never sent:** your tools' source or return values, your credentials, your data, or anything observed during a rollout. This is a description of the agent's *shape*, taken once, at startup.

**Why it is on.** It is what lets the platform work out which workflows are worth generating for your agent — the universe of tasks its tools actually make possible — instead of asking you to enumerate them. An agent that reports nothing gets generic tasks.

**Turning it off**, either way:

```python
fw.serve("my_package.agent:build", agent_id="a", input_shape="text", introspect=False)
```

```bash
PERCEPTEYE_INTROSPECT=0        # for operators who cannot edit the call site
```

Only a deliberate value disables it (`0`, `false`, `no`, `off`); a typo leaves it on rather than silently switching capture off while you believe it is running.

**If we could not read your agent, we say so rather than guessing.** A framework we do not recognise, or a command entrypoint (a separate process has no object to read), reports `discovered_agent: null` — never an empty agent. Supported today: **Google ADK** and **LangChain / LangGraph**.

## Configuration reference

### Isolation

`isolation="subprocess"` is the default, and it is the only mode with a real timeout — a child process can actually be killed. The trade-off is that a child cannot receive a live Python object, so it needs an entrypoint it can import. Passing an already-constructed object is supported with `isolation="inprocess"`, which is capped at `concurrency=1`, because concurrent in-process rollouts share one set of environment variables and would race for the per-rollout endpoint. Both are checked when `serve()` starts.

**What the child inherits.** Your agent needs your credentials — that is the entire point of running it in your own process — so your environment passes through. Ours does not: no variable in the `PERCEPTEYE_` namespace is inherited, so the long-lived control-plane key stays in the parent and never reaches code the model can influence. Exactly two are set for the child, the rollout id and the trajectory directory, and the per-rollout endpoint and its single-use key are set last so they win over anything inherited.

> [!WARNING]
> `isolation` is a process boundary and an environment scrub. It says nothing about the filesystem. Point `artifacts_dir` at a directory outside any source tree before you run an untrusted agent, and use an OS-level sandbox if you need one.

### Entrypoints

Probed in order: `invoke`, `run`, `execute`, `chat`, `__call__`. Pass `method="..."` to name one. A plain function is its own entrypoint. Resolution happens once, at startup, and raises there — never mid-rollout, because a rollout is one member of a comparison group, and one that fails to emit does not merely lose itself, it distorts the scores of the rollouts it was supposed to be compared against.

| `input_shape` | your entrypoint receives |
|---|---|
| `"text"` | `"the task description"` |
| `"messages"` | `[{"role": "user", "content": "..."}]` |
| `"dict"` | `{"input": "..."}` |

### Reporting results

Return a string and it is used as the final output. Return a `RolloutOutput` when you have more to say:

```python
from percepteye_agent_flywheel import RolloutOutput

return RolloutOutput(
    final_text=answer,
    artifacts={"workspace": "workspace.tar.gz"},
    reward=my_own_score,        # advisory; recorded, not used to train
)
```

If you cannot classify every executed call, report none of them — a partial list is indistinguishable from a short one. Either emit the unclassifiable call with `outcome="unknown"`, or set `tool_calls=None` for the whole rollout.

### Environment variables

There are no feature flags, and no variable selects behaviour. These are addresses and credentials.

| Variable | Meaning |
|---|---|
| `PERCEPTEYE_CONTROL_PLANE_URL` | Control-plane base URL. |
| `PERCEPTEYE_API_KEY` | Your flywheel key. |
| `PERCEPTEYE_INTROSPECT` | Set to `0` to disable [agent discovery](#agent-discovery-on-by-default). |
| `PERCEPTEYE_ROLLOUT_ID` | *Set by us* for the child. Read it to tag your own logs. |
| `PERCEPTEYE_TRAJECTORY_DIR` | *Set by us* for the child, and it already **is** this rollout's directory — nothing to join, nothing to create. Append tool calls to `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`; any other file you leave there is collected as an artifact. |
| `PERCEPTEYE_TURN_INPUT` | *Set by us* for a command entrypoint. The task, also on stdin. |

### OpenTelemetry (optional)

```bash
pip install "percepteye-agent-flywheel[otel]"
```

Every outcome you record is additionally emitted as an OpenTelemetry span, so it appears in whatever you already run — Datadog, Phoenix, Grafana — beside the rest of your agent's telemetry. The extra is the **API only, never the SDK**: this package emits into whatever `TracerProvider` you have already registered and **never installs one**, because `set_tracer_provider` is once-only and a library that called it would silently replace your entire GenAI telemetry.

Spans carry the GenAI semantic conventions where they name the thing, plus `percepteye.tool.outcome` for the tri-state. That needs its own namespace, and that is the honest reason it has one: the GenAI conventions have no way to express "the result of this call was never observed". For the same reason an `unknown` outcome leaves the **span status unset** — marking it OK asserts a success nobody observed, and marking it ERROR asserts a failure nobody observed.

**This is egress, not a transport.** Nothing is ever read back, so your sampler, your queue overflow and your exporter being down cost you nothing here; outcomes reach the control plane through the trajectory regardless.

## FAQ

**Does my agent leave my infrastructure?**
No. It runs as a child process of the harness you start, in your own environment, with your own credentials. The only things that cross the boundary are the four lanes in the diagram, and every one of them is a connection your side opened.

**Do I have to open a firewall port?**
No. There is no listening socket in this package — no inbound rule, no public DNS, no TLS to terminate on your side.

**Do I need a GPU?**
No. The control plane holds them. This is the difference between this and running a trainer like ART or `prime-rl` yourself.

**Do I have to rewrite my agent, or port it into a container?**
No. You pass an importable path or an argv list. For LangChain, LangGraph and Google ADK, per-call outcomes are captured with zero code in your agent.

**What if my agent isn't written in Python?**
[Pass an argv list.](#your-agent-doesnt-have-to-be-python) The task arrives on stdin and in `PERCEPTEYE_TURN_INPUT`; stdout is the final text. There is no protocol to implement, so an existing CLI needs no changes.

**Can a CLI or REPL agent report per-tool outcomes, or only Python frameworks?**
It can, with no instrumentation. If your agent prints a structured event stream — prime-agent's `--mode json`, or OpenTelemetry GenAI events as JSON lines — outcomes are parsed off stdout, including when the run exits non-zero or is killed on the deadline. Adding another CLI agent is a `StreamFormat` entry, not a parser. [Details.](#zero-instrumentation-langchain-langgraph-google-adk-and-cli-agents)

**What happens when a rollout crashes, or my credential expires?**
It is still reported — every rollout is, whatever happened to it — carrying a **typed exclusion** so the platform records the work product and still knows not to train on it. The SDK emits six: `flywheel_crash`, `substrate_unavailable`, `credential_expired`, `deadline_exceeded`, `entrypoint_error` and `agent_exit_nonzero`. (`cancelled` and `capacity` are declared vocabulary the control plane uses; this SDK never emits them.) `entrypoint_error` — the agent could not be *started* — and `agent_exit_nonzero` — it ran and exited non-zero — are deliberately distinct, because one is retryable and untrainable and the other is a real attempt whose failure is exactly what training must see.

Artifacts are collected **before** the report on every path, including the failing ones: a rollout that exhausted its budget still wrote files, and for a coding agent those files *are* what the reward reads.

**What is `substrate_unavailable`?**
Every tool call in the rollout failed, which means the environment was broken rather than the policy. It is deliberately strict — one `unknown`, one success, or no tool calls at all, and the rollout is graded normally, because partial failure is ordinary agent behaviour and exactly what you want to train on.

**What does `outcome="unknown"` actually do to my score?**
It lowers the grade rather than excluding the rollout. Verifiers that depend on a per-call outcome abstain on it. It is never rounded up to `ok`. [Why this exists.](#why-unknown-is-a-first-class-tool-outcome)

**Can I run more than one rollout at a time?**
Yes — `concurrency=N` with the default subprocess isolation. In-process isolation is capped at 1, because concurrent rollouts would race for the per-rollout endpoint. If your tasks arrive as a group with a shared deadline, match the concurrency to the group width or the group's own clock is what will kill it.

**Does this send my system prompt anywhere?**
Yes, once at startup, unless you turn it off — and that is [documented in full above](#agent-discovery-on-by-default) rather than buried. `introspect=False` or `PERCEPTEYE_INTROSPECT=0`.

**How is this different from an eval platform like LangSmith or Braintrust?**
They observe your agent and let you write evals; they do not supply tasks or close the loop to weights. [Full comparison.](#how-is-this-different-from-langsmith-braintrust-hud-art-or-nemo-data-flywheel)

**Is it really only one dependency?**
Yes, `httpx`. `tests/test_oss_boundary.py` fails the build if the runtime list grows past one entry, and a separate CI job installs the package into a clean environment and asserts the same thing from `importlib.metadata`.

**What is the licence, and can I use it commercially?**
MIT. Yes.

## The contract

[`schema/flywheel-1.json`](schema/flywheel-1.json) is the authority. This repository owns it; server implementations pin against the published schema, never the other way round. Python types are generated from it, and a TypeScript package will be generated from the same file.

Fields derived from the inference wire — token-level data, sampling parameters, grouping keys, scoring provenance — are absent from the agent contract by construction. An agent cannot know them, so it is never asked to assert them.

## Development

```bash
python -m venv .venv
# [dev] alone runs 195 tests; the adapter and tracing suites need the
# frameworks they bind to, and skip themselves when those are absent.
.venv/bin/pip install -e ".[dev,adapters-test,otel,otel-test]"
.venv/bin/pytest
```

278 tests across 23 files, at a 1:1 test-to-source line ratio (4,819 lines of tests against 4,799 of source). Four CI jobs, and each one is there to protect a specific claim in this README:

| job | what it would catch |
|---|---|
| `test` | a bare install — nothing but `httpx` — breaking on Ubuntu or macOS, on Python 3.10 through 3.13 |
| `test-full` | every suite, with every optional framework present, and **zero skips permitted** — because a skipped test and a passing test look identical in a green check mark |
| `clean-install` | the runtime dependency list growing past `httpx`, asserted from installed metadata rather than from the source |
| `boundary` | an internal system name reaching the published package, including inside an error string, where it would print into your logs and disclose something you cannot act on |

## Contributing

Issues and pull requests welcome. If you are adding a framework adapter, the bar is the one the built-ins meet: it must not import its framework at module scope, it must report `unknown` rather than guessing, and it must fail loudly at bind time rather than degrade silently.

## Security

Please report vulnerabilities privately to security@percepteye.com rather than in a public issue.

## License

[MIT](LICENSE) © Percepteye.
