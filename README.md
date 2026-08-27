<div align="center">

# percepteye-agent-flywheel

**Self-improving agents through continual learning.**<br>
Your agent keeps running in your process, behind your firewall, on your credentials — unchanged.<br>
One import. One call. One runtime dependency.

[![ci](https://img.shields.io/github/actions/workflow/status/percepteye-ai/agent-flywheel/ci.yml?branch=main&label=ci&style=flat-square)](https://github.com/percepteye-ai/agent-flywheel/actions)
[![pypi](https://img.shields.io/pypi/v/percepteye-agent-flywheel?style=flat-square&label=pypi)](https://pypi.org/project/percepteye-agent-flywheel/)
[![python](https://img.shields.io/pypi/pyversions/percepteye-agent-flywheel?style=flat-square)](https://pypi.org/project/percepteye-agent-flywheel/)
[![runtime dependencies: 1](https://img.shields.io/badge/runtime%20dependencies-1-blue?style=flat-square)](#what-does-it-cost-to-integrate)
[![skips permitted in CI: 0](https://img.shields.io/badge/skipped%20tests%20permitted%20in%20CI-0-brightgreen?style=flat-square)](#development)
[![license: MIT](https://img.shields.io/badge/license-MIT-green?style=flat-square)](LICENSE)

<img src="https://raw.githubusercontent.com/percepteye-ai/agent-flywheel/main/docs/assets/flywheel-overview.gif"
     width="900"
     alt="An animation of one customer's agent going through the flywheel, in seven beats. A single card labelled &quot;your agent&quot; sits inside a zone marked YOUR INFRASTRUCTURE, listing three rows: system prompt — yours, unchanged; tools — yours, unchanged; and model — your-model. A rounded bracket then draws itself around the card, labelled fw.serve(&quot;my.agent:build&quot;), with a green badge reading &quot;0 lines changed inside&quot; — the SDK wraps the agent and never enters it. A dashed trust boundary appears and a single opaque box labelled Percepteye control plane fades in on the right. Two channels cross the boundary, both opened from the left: rollouts and observed outcomes going out, and the trained policy coming back, carrying base_url, model and generation. The agent runs real tasks and a row of outcome chips fills in on its own card — ok, ok, unknown, failed, ok — captioned &quot;unknown is never rounded up to ok&quot;. The control plane lights three lines: generates the tasks, scores observed outcomes, trains the policy. Finally a highlight sweeps across the agent card's model row and the value changes from your-model to a Percepteye policy endpoint, while the prompt and tools rows stay untouched. A closing comparison reads: BEFORE, your prompt and a general-purpose model; AFTER, your prompt and a policy trained on this agent's own work. Caption: same agent, same code, better policy.">

**[Quickstart](#quickstart-one-import-one-call)** · [How it works](#how-the-flywheel-works) · [Why `unknown` matters](#why-unknown-is-a-first-class-tool-outcome) · [Zero instrumentation](#zero-instrumentation-langchain-langgraph-google-adk-and-cli-agents) · [Comparison](#how-is-this-different-from-applied-compute-langsmith-braintrust-hud-or-nemo-data-flywheel) · [FAQ](#faq) · [Docs](https://docs.percepteye.com/flywheel)

</div>

---

`percepteye-agent-flywheel` turns the work your agent already does into training data, and delivers what it learns back as an endpoint your agent already calls.

***Observed, not inferred.*** Every tool call is recorded as `ok`, `failed` or **`unknown`**, and `unknown` is never rounded up to `ok`. That is the difference between learning from what your agent *did* and learning from what its transcript *claimed* — and [nothing else in this category can make the distinction](#how-is-this-different-from-applied-compute-langsmith-braintrust-hud-or-nemo-data-flywheel), because a trajectory reconstructed from the model wire cannot see whether a tool actually worked.

Under the hood this is **in-process agentic RL** — reinforcement learning on rollouts your own agent runs, outbound-only, with per-call tri-state outcomes and a policy endpoint the control plane chooses. You never have to think about any of that to use it.

## Install

```bash
pip install percepteye-agent-flywheel
```

Python 3.10, 3.11, 3.12 or 3.13. The install pulls exactly one runtime dependency, `httpx`. That is not a boast — it is a constraint the test suite enforces, because this package runs inside your production process and every dependency it adds is a version conflict it imposes on you.

> [!IMPORTANT]
> **Your agent has to sample through an OpenAI-compatible client.** The per-rollout gateway speaks the OpenAI completions API, so provider configuration we cannot point at it is **removed from each rollout's child process** — twenty variables covering Azure OpenAI, Gemini/Vertex, Bedrock, NVIDIA NIM, Groq, Mistral, Together, OpenRouter, Fireworks, DeepSeek, Perplexity, Cohere and xAI ([`UNROUTABLE_PROVIDER_VARS`](src/percepteye_agent_flywheel/isolation.py)).
>
> An agent built on one of those then raises when it constructs its client, which is the outcome we want: the alternative is that it sails past the gateway, samples its live provider unobserved, and returns a trajectory that cannot back an on-policy objective while looking exactly like one that can. `serve()` warns at startup naming any of them it finds in your environment.

## Quickstart: one import, one call

> [!WARNING]
> `serve()` is not a dry run. Each rollout is a task the **control plane generated**, executed by your agent, in a child process that inherits your environment — so its tool calls are real, and a generated task can create, modify or delete anything your agent's credentials can reach. Bound the first run with `max_rollouts=`, point `artifacts_dir=` outside any source tree, and give it credentials scoped to somewhere you don't mind it writing.

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

`serve()` blocks, claiming rollouts until you stop it, and returns the number completed. It is quiet by design — the one line it prints is the policy your rollouts will be graded against, when the control plane has one deployed (a `RuntimeWarning` about unroutable providers is the only other thing it says):

```
[percepteye] policy: candidate-3 — promoted 2026-08-19 → policy at https://gw.percepteye.com/r/...
```

If nothing arrives at all, `serve()` has already told you why. Everything that can be wrong with the configuration — the entrypoint does not resolve, the credential is missing, the control plane speaks a different contract version, the isolation mode is incompatible with the entrypoint you passed — is checked **before the first rollout**, and raises there rather than mid-run.

<details>
<summary><b>Watch individual rollouts as they complete</b></summary>

Pass `on_event`. It fires with `("rollout_start" | "rollout_reported" | "rollout_crash" | "claim_failed" | "lease_lost" | "heartbeat_failed" | "report_failed" | "artifact_failed", payload)`:

```python
fw.serve(
    "my_package.agent:build",
    agent_id="my-coding-agent",
    input_shape="text",
    on_event=lambda kind, ev: print(kind, ev),
)
```

`rollout_reported` carries the rollout's typed exclusion if it had one, which is what lets you score a batch and count the runs that never really happened separately from the ones that failed.
</details>

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

## At a glance

| | |
|---|---|
| **Lines changed inside your agent** | **0** — adapters bind from the outside |
| **Runtime dependencies** | **1** (`httpx`), enforced by a test and a clean-install CI job |
| **Inbound ports / public DNS / TLS to terminate** | **0 / 0 / 0** — there is no listening socket in this package |
| **Containers to build, GPUs to provision** | **0 / 0** — your agent runs as your process; the control plane holds the GPUs |
| **Frameworks captured with no instrumentation** | LangChain, LangGraph, Google ADK, plus any CLI agent that already prints a structured event stream |
| **Languages** | Python by import path; anything else by argv |
| **Per-tool-call fidelity** | `ok` / `failed` / **`unknown`** — never rounded up |
| **Licence** | MIT |

Every row is checkable from this repository. [How to check each one.](#what-does-it-cost-to-integrate)

## How the flywheel works

The figure at the top sketches one lap. Here it is step by step, because a picture is not indexable and half of you are reading this through a model:

1. **Claim.** Your harness long-polls the control plane for a rollout. The lease is renewed by heartbeat, and a crashed worker's rollout is redelivered rather than lost. Nothing listens on your side — you opened the connection.
2. **Isolate.** Each rollout gets its own child process, with its own environment, its own single-use credential and its own trajectory directory. A child process can actually be killed, which is why it is the default: Python cannot interrupt a synchronous call in another thread, so in-process a hung agent means you stop *waiting* while it keeps running.
3. **Route.** Your agent's LLM completions go to a per-rollout endpoint. That is what attributes the work, captures the completion wire, and lets the platform score what the model actually sampled rather than what a transcript says it did.
4. **Record.** Every tool call lands in the trajectory as `ok`, `failed` or `unknown`. See [why that third value exists](#why-unknown-is-a-first-class-tool-outcome).
5. **Report.** Final text, artifacts, timings and outcomes go back. Crashes, expired credentials and blown deadlines are reported as **typed exclusions**, so a substrate fault is never scored as a bad decision by your agent.
6. **Score.** Percepteye grades the rollout against its own compiled checks. This is the part you do not have to build.
7. **Improve.** Your *production* agent — a different process from the one donating capacity — [picks up what training produced](#the-last-mile-your-production-agent-picks-up-what-training-produced). That closes the loop.

The same lap again, drawn as architecture rather than as a story — each component in the trust zone that actually owns it, and each lane labelled with what crosses it:

<p align="center">
  <img src="https://raw.githubusercontent.com/percepteye-ai/agent-flywheel/main/docs/assets/flywheel-loop.gif"
       width="880"
       alt="The same loop drawn as an architecture diagram, animated in ten beats on a dark canvas. It opens on a single box labelled 'your agent' inside a zone labelled YOUR INFRASTRUCTURE — your process, your credentials, your firewall — with its tool outcomes marked unobserved. A terminal runs pip install percepteye-agent-flywheel, pulling one runtime dependency, httpx. A flywheel harness box appears above the agent, types an import and fw.serve, and spawns the agent as a child process, now badged unchanged, 0 lines. A dashed trust boundary draws itself down the middle and a single opaque box labelled Percepteye control plane fades in on the right. Four lanes cross the boundary and light in turn — claim a rollout, completions through a per-rollout endpoint, result and outcomes, and current policy — and every one is opened from the left, with responses returning dimmer along the same connection. No arrow originates on the Percepteye side. A trajectory rail inside the agent fills with chips reading ok, ok, unknown, failed, ok, captioned 'unknown is never rounded up to ok'. Finally a production agent box appears calling fw.apply_current_policy() and its badge flips from policy v1 to policy v2. Footer: outbound only, no inbound port, no public DNS.">
</p>

**Every arrow that crosses the trust boundary is opened from your side.** There is no listening socket anywhere in this package; you can check that claim in about five seconds:

```bash
grep -rnE "socket\.socket|\.listen\(|HTTPServer|socketserver|uvicorn|fastapi|flask|start_server" src/
# exits 1 — no matches. `import socket` appears nowhere in this package.
```

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

## Zero instrumentation: LangChain, LangGraph, Google ADK, and CLI agents

Adapters are **on by default**. `adapters="auto"` binds every built-in whose framework is installed, so a LangChain, LangGraph or Google ADK agent reports per-call outcomes with no Percepteye code in it at all — not one import, not one `record_tool_call`.

| Adapter | Covers | Outcomes it can report | Routes LLM calls |
|---|---|---|---|
| `adk` | Google ADK | `failed` on raise **and** on `{"error": ...}`; `unknown` otherwise | no — the child environment routes |
| `langchain` | LangChain, LangGraph | `ok` on a `ToolMessage` with `status="success"`; `failed` on raise or `status="error"`; `unknown` for any other result shape | only when you pass `llm=` |

They add **no dependencies**: no adapter imports its framework until you bind it, so an absent framework simply fails that adapter's probe.

Adapters are selected by **name**, because an adapter cannot cross a process boundary and the default isolation runs your agent in a child. Naming one explicitly is a stronger statement than `"auto"` — you are asserting that is how your calls are captured, so an adapter that cannot bind raises `RoutingNotInstallable` rather than degrading silently.

That raise happens where the binding happens: **inside the process your agent runs in, on every rollout** — not in `serve()`'s startup preflight, which never touches adapters. Under the default subprocess isolation the child raises, the parent sees a non-zero exit, and the rollout comes back as a `flywheel_crash` exclusion whose detail is the child's stderr. So a named adapter you have not installed costs you every rollout one at a time rather than refusing to start — pass `on_event` and the first `rollout_reported` payload carries the traceback.

`adapters=[]` and `adapters=None` both bind nothing.

<details>
<summary><b>What zero instrumentation actually produces</b></summary>

This LangChain agent contains no reference to Percepteye of any kind:

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

**`unknown`, not `ok`** — because this tool returned raw content rather than a `ToolMessage` carrying a status, so nothing observed whether the call succeeded. That is the honest floor zero instrumentation buys you. One `record_tool_call` where you actually know the answer gets you the rest.
</details>

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
- **Tri-state by omission.** The verdict is one error flag and nothing else — `isError`, `is_error` or `error`, and under `otel-genai` also `error.type`. `true` → `failed`, `false` → `ok`, **absent → `unknown`**. There is no separate success flag to send, because an agent that ends a call without saying whether it worked has not asserted success, and we do not assert it on the agent's behalf.
- **The trajectory wins.** A `record_tool_call` is a deliberate assertion about your own behaviour, so a stream parse never overwrites it. The stream is consulted only where the trajectory recorded nothing — and a stream carrying no tool events leaves `tool_calls` as `None`, never `[]`.
- **Capture survives failure.** Outcomes are recovered on a clean exit, on a non-zero exit, and on a timeout kill — because a rollout that died is exactly the one you need the tool calls from.

Supporting the next CLI agent is a `StreamFormat` entry — a vocabulary, not a parser. Nothing about the extraction, the tri-state or the absence rules gets reimplemented per agent.

### …and Node harnesses, through a plugin they already load

Some harnesses do not print a tool-outcome stream at all. Their tool results exist only inside the process, on a plugin seam. For those, [`packages/percepteye-agent-flywheel/`](packages/percepteye-agent-flywheel/) ships a plugin — one npm package, one install, currently covering **OpenClaw** and **DeepSeek Harness**:

```bash
openclaw plugins install percepteye-agent-flywheel     # OpenClaw
dsh plugin --profile <name> add percepteye-agent-flywheel   # DeepSeek Harness
```

Then set `PERCEPTEYE_TRAJECTORY_DIR` on the process running your agent. With it unset the plugin does nothing at all, so the same install is a no-op in production and a recorder during a rollout.

**Both plugins observe and cannot do anything else**, and that is a property of the hooks rather than a promise about the code. Each harness offers seams that *replace* a tool result before the model reads it; the plugins take neither. OpenClaw's `after_tool_call` returns `void`, and DeepSeek Harness publishes `tools/result` as an emit-only listener over a **deep-frozen** execution and result — there is no return channel and the object is immutable. Capturing evidence must never be able to change the behaviour being measured.

> [!NOTE]
> Zero dependencies — Node builtins only (`node:fs` and `node:path`). This runs inside your agent's process, and every dependency added there is a version conflict imposed on someone who did not ask for one — the same rule as the Python package's single dependency.

Both write the same file in the same format, and CI runs the JavaScript suite against the *same* Python reader, so the two halves cannot drift apart quietly. That check is a script rather than workflow-only YAML, so you can run exactly what CI runs: [`./scripts/cross_language_abi_check.sh`](scripts/cross_language_abi_check.sh).

In a multi-agent graph each call is attributed to the agent that made it, read from the framework's own run metadata — ADK exposes `ToolContext.agent_name` directly, LangGraph stamps the node it is executing. A call nothing names stays unattributed, because filling it in would train a planner on a coder's tool calls.

To ship your own adapter, expose a factory in the `percepteye_agent_flywheel.adapters` entry-point group.

## Your agent doesn't have to be Python

If it isn't — a compiled binary, a Node CLI, anything with a command line — pass an argv list instead of an importable path. There is no protocol to implement, so an existing CLI needs no changes:

```python
fw.serve(["opencode", "run"], agent_id="coding-agent", input_shape="text")
```

The task arrives two ways, because CLIs differ and guessing wrongly is silent: on **stdin**, and as `PERCEPTEYE_TURN_INPUT`. Its **stdout** is the final text. Its exit code is a result — non-zero is reported as `agent_exit_nonzero` and excluded from training, because a crashed run says nothing about the policy.

> [!NOTE]
> `input_shape` is still required by `serve()`, but it describes how a **Python** entrypoint is called. A command entrypoint receives the task on stdin and in the environment regardless of what you declare.

**If your CLI takes the task as an argument**, mark the position with `fw.TASK`:

```python
fw.serve(["dsh", "--profile", "headless", fw.TASK], agent_id="dsh-agent", input_shape="text")
```

A sentinel rather than a `"{task}"` string, so a literal argument that happens to contain the placeholder text cannot be mistaken for the marker. When it appears, **stdin is left empty** — an agent told the task twice acts on it twice. Leave it out and nothing changes: stdin and the environment variable, exactly as before.

> [!NOTE]
> A task delivered through argv is visible to other users on the machine via `ps`. One delivered on stdin is not. That is a real trade-off and it is yours to make — which is why this is declared rather than inferred from your binary.

**If your agent reads its endpoint from other variable names**, say so:

```python
fw.serve(["dsh", "--profile", "headless", fw.TASK],
         agent_id="dsh-agent", input_shape="text",
         endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"))
```

By default the child gets `OPENAI_BASE_URL`, `OPENAI_API_BASE`, `OPENAI_API_KEY`, `ANTHROPIC_BASE_URL` and `ANTHROPIC_API_KEY`. An agent that reads none of them looks identical from here to one that is correctly routed — right up until its completions arrive at your live provider instead of the gateway, unobserved. `endpoint_env` is additive, so the defaults stay for an agent that reads a mixture.

**You probably get tool outcomes for free.** If your CLI already prints a structured event stream — prime-agent's `--mode json`, or anything logging OpenTelemetry GenAI events as JSON lines — they are [read straight off stdout](#zero-instrumentation-langchain-langgraph-google-adk-and-cli-agents) with nothing added to the agent.

If it prints nothing useful, the trajectory is still there and a shell can append to it:

```sh
printf '{"name":"deploy","arguments":{},"outcome":"ok","status_code":200}\n' \
  >> "$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl"
```

The command runs in its own **process group**, and that group is what gets killed on timeout — a real CLI agent spawns freely, and killing only the process we launched leaves its children holding the output pipe open.

`gate_flag=` names how your CLI spells its completion-gate option (prime-agent: `--autonomous-gate`). The flag is a fact about your binary and cannot be guessed; the gate *command* is a property of the task and arrives per rollout. Leave it unset for an agent with no gate concept.

A worked example against a real TypeScript coding agent — routing, event parsing, gates and all — is in [`examples/prime_agent/`](examples/prime_agent/).

## Did it work, or did it *stick*?

An outcome says whether a call **worked**. It cannot say whether the change **survived**. A create that returns `201` and then silently fails to persist is indistinguishable, from inside your process, from one that landed — and a policy trained on the difference between them learns that the failure was fine.

Nothing observed from inside the agent can close that gap. The only thing that settles it is looking at the resource afterwards, in the system that owns it.

So when a tool call produces something, the trajectory records **which entity it produced** — a handful of opaque identifiers, never the response body:

```json
{"name": "POST /rest/api/2/issue", "outcome": "unknown", "status_code": null,
 "entity_ids": {"id": "10001", "key": "ENG-7"}}
```

That is deliberately the *identity*, not the payload. The body is your production data and does not belong in a trajectory; the identity is what makes the record checkable later. Two properties follow:

- **It is recorded even when the outcome is `unknown`.** An identity claims nothing about whether the call worked — it names an entity so that something else can go and observe one. It is the only evidence that can turn an honest `unknown` into an honest `ok`, so withholding it would starve exactly the records that need it most.
- **Absent means absent.** No identity is an abstention, never a failure. Prose yields nothing rather than a guessed id, and low-entropy fields like `name` are refused outright — a human name that a *different* entity happens to share confirms the wrong resource, which is a false positive wearing a verified badge.

What this SDK ships today is the identity: [`evidence.py`](src/percepteye_agent_flywheel/evidence.py) extracts it and the contract carries it. Nothing in this package reads your systems back, and there is no credential in it that could.

**Confirming those identities against your own system of record is a control-plane feature configured outside this package — and configuring it means giving Percepteye a read credential for the environment your agent operates on.** That is worth stating plainly rather than folding into the outbound-only claim above, because it is the opposite direction: verification traffic arrives at your API and shows up in your audit log. It is additive, it is optional, and **it is off unless you turn it on**. If you never do, nothing changes — the flywheel scores exactly what it scores today.

> [!NOTE]
> Confirmation is partial by nature. Some operations leave a resource you can read back and some do not, which is why coverage is reported per episode instead of implied. A number you can check beats a claim you cannot.

## Capture in the process that serves your users

`serve()` donates capacity. `attach()` donates **evidence**, from the turns your agent is already serving — same adapters, same tri-state, same trajectory file, scoped around a live user turn instead of a claimed rollout:

```python
import percepteye_agent_flywheel as fw

px = fw.attach(agent_id="support-agent", trajectory_root="/var/lib/percepteye")

with px.turn(conversation_id=conv_id) as turn_id:
    reply = agent.invoke(user_text)        # your agent, unmodified
```

**It captures. It never reroutes.** Adapters are bound with no endpoint at all, so nothing your agent sends can be re-pointed by attaching. Changing where production traffic goes is [the last mile](#the-last-mile-your-production-agent-picks-up-what-training-produced)'s job, and it stays your explicit decision.

Everything here fails open. An unreachable control plane yields `policy == {}` — read that as *keep your current configuration* — and a turn whose capture cannot write still runs. A capture bug costs the observation and nothing else. `register=False` skips the control-plane call entirely, for a process that should make no outbound connection at all.

Turn boundaries are **declared, not inferred**: guessing where one request ends and the next begins would silently merge two users' tool calls into one record. The turn id it yields is the join key for an outcome that arrives later than the turn does — the draft kept an hour afterwards, the pull request merged next morning.

> [!NOTE]
> `trajectory_root` is required rather than defaulted, and turns accumulate there. Nothing removes them, because the `flywheel/1` contract has no route for a production turn yet — so you own retention until it does. Naming the directory is how that becomes a decision rather than a surprise.

## The last mile: your production agent picks up what training produced

Training produces a better model. This is where your agent finds out. There are two surfaces, and which one you want depends on whether your process is long-lived.

**One shot, at startup** — `apply_current_policy()` returns keyword arguments you can hand a client:

```python
import percepteye_agent_flywheel as fw
from openai import OpenAI

kw = fw.apply_current_policy(agent_id="my-coding-agent")
if kw:                                  # {} means nothing is deployed — keep your config
    client = OpenAI(base_url=kw["base_url"], api_key=kw["api_key"])
    model = kw["model"]
```

**A handle that keeps up** — `policy_source()` re-polls on a TTL and calls you back when the answer changes. This is the one you want for a service that stays up across a promotion, and it is what makes a **rollback** reach your agent without a redeploy:

```python
src = fw.policy_source(
    agent_id="my-coding-agent",
    on_change=lambda snap: reconfigure(snap),   # fires on promotion AND on rollback
)

snap = src.current()                    # call this per turn — see below
```

**Nothing polls in the background.** There is no thread in this package; the refresh, and the `on_change` callback with it, run inside your call to `current()`. Between TTLs that call is a cache read, so calling it per turn costs nothing — and calling it once at startup rebuilds exactly the problem the handle exists to solve. Your `ttl_s=` is a default the control plane may **replace in either direction**, because it knows its own promotion cadence and you do not; the 30-second floor wins over both, and a little jitter is applied on top.

What comes back either way is an **endpoint**, because that is the only form an agent can act on — an artifact URI is not callable.

The two differ in how they fail, and the difference matters. `apply_current_policy()` returns `{}` when the control plane has nothing deployed *or* cannot answer — read that as "keep your current configuration". `policy_source()` is empty only before its first successful answer; after that it **keeps serving the last good policy**, because a blip that emptied the snapshot would silently demote production back to the untrained model. A failed refresh leaves `snap.last_refresh_ok` false and `snap.stale_for_s` growing, and the endpoint stands. Either way, an unreachable control plane must never stop your agent from serving its users.

> [!IMPORTANT]
> These are for your **production** agent — never for `serve()`. A serve rollout samples through the per-rollout gateway, which is what attributes and captures the completion, and the gateway already rewrites the model field to the policy under test. Pointing a donor agent at the policy endpoint directly would bypass capture entirely and produce a run with nothing to train on.

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
| Tests | **395 across 28 files**, all of them run in CI with zero skips permitted; the bare-install matrix adds 2 operating systems × 4 Python versions | `pytest` and [`.github/workflows/ci.yml`](.github/workflows/ci.yml) |
| License | MIT | [LICENSE](LICENSE) |

<details>
<summary><b>One dependency, in context</b> — measured from the PyPI JSON API</summary>

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
</details>

## What results should you expect?

An honest answer, in three parts.

**What this repository can prove today.** Everything in [the integration table above](#what-does-it-cost-to-integrate): one dependency, one call, zero agent changes, zero inbound ports, and 395 passing tests — 310 of them on a bare install, where the adapter and OpenTelemetry suites skip because their frameworks are absent. Those are properties of the code and you can check every one of them yourself.

**What it cannot prove yet.** Task success rate, reward improvement, latency or cost deltas for *your* agent. This repository ships no benchmark numbers, and we would rather have the empty table below than a number you cannot reproduce. When we publish, we publish in the form the category's credible results take — `before → after, on a named task, with a named model`:

| Agent | Task suite | Model | Before | After | Rollouts | Date |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

**What this class of technique has produced elsewhere.** These are other vendors' published numbers on their own systems, linked so you can read the methodology, and they are not claims about Percepteye:

- OpenAI reinforcement fine-tuning: Ambience Healthcare ICD-10 assignment, o3-mini **0.39 → 0.57** against a 0.45 physician baseline; Harvey legal-citation F1 **0.563 → 0.6765** ([RFT use cases](https://developers.openai.com/api/docs/guides/rft-use-cases)).
- NVIDIA NeMo Data Flywheel with AT&T: "up to 40% improvement in accuracy" and an "84% decrease in call center analytics cost" ([case study](https://www.nvidia.com/en-us/case-studies/att-drives-ai-agents-with-nemo/)).
- And the counterweight, because a section that only cites wins is marketing: TensorZero found RFT **100–700× more expensive than SFT** on equivalent datasets, and that on customer-service tasks it *degraded* performance ([write-up](https://www.tensorzero.com/blog/is-openai-reinforcement-fine-tuning-rft-worth-it/)).

<details>
<summary><b>Measure it yourself</b></summary>

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
</details>

## How is this different from Applied Compute, LangSmith, Braintrust, HUD or NeMo Data Flywheel?

The short version: **the observability tools watch your agent but never close the loop to weights; most RL platforms close the loop but make your agent move into their container or their cluster.** One other company leaves the agent where it is — and the difference from them is what your capture can actually see.

| | agent runs in | supplies the tasks | supplies the scoring | what it sees per tool call | core deps |
|---|---|---|---|---|---|
| **percepteye-agent-flywheel** | **your own process** | yes | yes | **`ok` / `failed` / `unknown`, plus the entity a write produced when one can be identified** | **1** |
| Applied Compute (BYOH) | your own process | yes | yes | the model wire only | — |
| HUD | their container image | yes | yes | their environment grades | 18 |
| Prime Intellect `verifiers` | wherever, you build the env | community hub | you write the rubric | you write it | 26 |
| OpenPipe ART / Tinker | wherever, you write the loop | no | no | you write it | 8 / hosted |
| LangSmith · Braintrust · Langfuse · Phoenix | your own process | no | you write the evals | whatever the framework said | 14 · 10 · 8 · 51 |
| NVIDIA NeMo Data Flywheel | your Kubernetes cluster | no | yes | — | k8s + 6× H100 to start |
| AWS Bedrock AgentCore | your AWS account | no | yes | — | AWS-bound |

**Applied Compute is the closest thing to this, and worth being precise about.** Their BYOH also keeps the harness where it already runs, also connects over an outbound WebSocket with no public endpoint, and is SOC 2 and ISO 42001 certified. If you are choosing between the two, "runs in my own infrastructure" is not the axis — both do that.

The axis is the last column. Their design is explicit that the harness *"does not need to emit any additional observability data"*: traces are reconstructed from the model wire alone. That sees what the model was asked and what it answered. It does not see whether the tool **worked** — so a LangChain tool with `handle_tool_error=True`, or a permission denial your API returns as HTTP 200, is invisible to it. [That distinction is the whole reason `unknown` exists](#why-unknown-is-a-first-class-tool-outcome).

And there is a second axis, one row further out than the table goes. Everyone above scores from **evidence produced by the run** — a transcript, a model wire, or an environment the vendor supplies and you port your task into. Because this SDK is in your process, the record can also carry **which entity a write actually produced**, wherever one can be identified — the only thing that lets a later check settle whether the work stuck. That is not a better rubric; it is a different class of evidence, and it is available to us because we are in your process rather than in place of it. [How that works, and what it costs you](#did-it-work-or-did-it-stick).

Three further caveats. **Line count is table stakes** — HUD's decorator is about eight lines and Weave advertises one, so nobody wins on that alone; the differentiator is one call *with one dependency and no container*, which is why the table has a deps column. **Task supply is not equivalent** — everyone in the "yes" column supplies tasks for environments they built, whereas [agent discovery](#agent-discovery-on-by-default) derives them from your actual tool surface. And **NeMo and AgentCore are not really competitors** so much as the enterprise-platform version of the same idea; if you already run the cluster, they are a reasonable answer.

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

**What the child inherits.** Your agent needs your credentials — that is the entire point of running it in your own process — so your environment passes through, minus two classes.

*Ours*: no variable in the `PERCEPTEYE_` namespace is inherited, so the long-lived control-plane key stays in the parent and never reaches code the model can influence.

*Provider configuration this gateway cannot route*: the twenty names in [`UNROUTABLE_PROVIDER_VARS`](src/percepteye_agent_flywheel/isolation.py) are removed rather than inherited, [for the reason given under Install](#install). That list is what we know of, not a proof of completeness — the control plane also compares the calls an agent reports against the calls the gateway actually served, and a mismatch excludes the rollout however the traffic escaped.

Two variables are then set for the child — the rollout id and the trajectory directory — and a command entrypoint gets a third, `PERCEPTEYE_TURN_INPUT`. The per-rollout endpoint and its single-use key are set last, so they win over anything inherited. Note that this only beats the *environment*: a client your own code constructs with an explicit `base_url` argument beats all of it, silently, and lands at a live third party. Catching that is what binding an adapter is for.

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

There are no feature flags. Exactly one of these changes what the SDK does — `PERCEPTEYE_INTROSPECT`, the kill switch for [agent discovery](#agent-discovery-on-by-default). The rest are addresses, credentials, and values we set for the child.

| Variable | Meaning |
|---|---|
| `PERCEPTEYE_CONTROL_PLANE_URL` | Control-plane base URL. |
| `PERCEPTEYE_API_KEY` | Your flywheel key. |
| `PERCEPTEYE_INTROSPECT` | Set to `0` to disable agent discovery. |
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

<details>
<summary><b>Does my agent leave my infrastructure?</b></summary>

No. It runs as a child process of the harness you start, in your own environment, with your own credentials. The only things that cross the boundary are the four lanes in the animation, and every one of them is a connection your side opened.
</details>

<details>
<summary><b>Do I have to open a firewall port?</b></summary>

No. There is no listening socket in this package — no inbound rule, no public DNS, no TLS to terminate on your side.
</details>

<details>
<summary><b>Do I need a GPU?</b></summary>

No. The control plane holds them. This is the difference between this and running a trainer like ART or `prime-rl` yourself.
</details>

<details>
<summary><b>Do I have to rewrite my agent, or port it into a container?</b></summary>

No. You pass an importable path or an argv list. For LangChain, LangGraph and Google ADK, per-call outcomes are captured with zero code in your agent.
</details>

<details>
<summary><b>What if my agent isn't written in Python?</b></summary>

[Pass an argv list.](#your-agent-doesnt-have-to-be-python) The task arrives on stdin and in `PERCEPTEYE_TURN_INPUT`; stdout is the final text. There is no protocol to implement, so an existing CLI needs no changes.
</details>

<details>
<summary><b>Can a CLI or REPL agent report per-tool outcomes, or only Python frameworks?</b></summary>

It can, with no instrumentation. If your agent prints a structured event stream — prime-agent's `--mode json`, or OpenTelemetry GenAI events as JSON lines — outcomes are parsed off stdout, including when the run exits non-zero or is killed on the deadline. Adding another CLI agent is a `StreamFormat` entry, not a parser. [Details.](#zero-instrumentation-langchain-langgraph-google-adk-and-cli-agents)
</details>

<details>
<summary><b>What happens when a rollout crashes, or my credential expires?</b></summary>

It is still reported — every rollout is, whatever happened to it — carrying a **typed exclusion** so the platform records the work product and still knows not to train on it. The SDK emits six: `flywheel_crash`, `substrate_unavailable`, `credential_expired`, `deadline_exceeded`, `entrypoint_error` and `agent_exit_nonzero`. (`cancelled` and `capacity` are declared vocabulary the control plane uses; this SDK never emits them.) `entrypoint_error` — the agent could not be *started* — and `agent_exit_nonzero` — it ran and exited non-zero — are deliberately distinct, because one is retryable and untrainable and the other is a real attempt whose failure is exactly what training must see.

Artifacts are collected **before** the report on every path, including the failing ones: a rollout that exhausted its budget still wrote files, and for a coding agent those files *are* what the reward reads.
</details>

<details>
<summary><b>What is <code>substrate_unavailable</code>?</b></summary>

Every tool call in the rollout failed, which means the environment was broken rather than the policy. It is deliberately strict — one `unknown`, one success, or no tool calls at all, and the rollout is graded normally, because partial failure is ordinary agent behaviour and exactly what you want to train on.
</details>

<details>
<summary><b>What does <code>outcome="unknown"</code> actually do to my score?</b></summary>

It lowers the grade rather than excluding the rollout. Verifiers that depend on a per-call outcome abstain on it. It is never rounded up to `ok`. [Why this exists.](#why-unknown-is-a-first-class-tool-outcome)
</details>

<details>
<summary><b>Can I run more than one rollout at a time?</b></summary>

Yes — `concurrency=N` with the default subprocess isolation. In-process isolation is capped at 1, because concurrent rollouts would race for the per-rollout endpoint. If your tasks arrive as a group with a shared deadline, match the concurrency to the group width or the group's own clock is what will kill it.
</details>

<details>
<summary><b>Does this send my system prompt anywhere?</b></summary>

Yes, once at startup, unless you turn it off — and that is [documented in full above](#agent-discovery-on-by-default) rather than buried. `introspect=False` or `PERCEPTEYE_INTROSPECT=0`.
</details>

<details>
<summary><b>How is this different from an eval platform like LangSmith or Braintrust?</b></summary>

They observe your agent and let you write evals; they do not supply tasks or close the loop to weights. [Full comparison.](#how-is-this-different-from-applied-compute-langsmith-braintrust-hud-or-nemo-data-flywheel)
</details>

<details>
<summary><b>Is it really only one dependency?</b></summary>

Yes, `httpx`. `tests/test_oss_boundary.py` fails the build if the runtime list grows past one entry, and a separate CI job installs the package into a clean environment and asserts the same thing from `importlib.metadata`.
</details>

<details>
<summary><b>What is the licence, and can I use it commercially?</b></summary>

MIT. Yes.
</details>

## The contract

[`schema/flywheel-1.json`](schema/flywheel-1.json) is the authority. This repository owns it; server implementations pin against the published schema, never the other way round. The Python types in [`contract.py`](src/percepteye_agent_flywheel/contract.py) are **hand-written to match it, not generated from it**, and the test suite asserts the two agree — so a value the code can emit and the schema does not declare fails the build rather than a customer's report.

Fields derived from the inference wire — token-level data, sampling parameters, grouping keys, scoring provenance — are absent from the agent contract by construction. An agent cannot know them, so it is never asked to assert them.

## Development

```bash
python -m venv .venv
# [dev] alone passes 310 tests; the adapter and tracing suites need the
# frameworks they bind to, and skip themselves when those are absent.
.venv/bin/pip install -e ".[dev,adapters-test,otel,otel-test]"
.venv/bin/pytest
```

395 tests across 28 files, at roughly a 1:1 test-to-source line ratio (6,236 lines of tests against 5,812 of source). The Node plugins are checked against the *same* Python reader, and that check is a script rather than workflow-only YAML, so you can run exactly what CI runs:

```bash
./scripts/cross_language_abi_check.sh
```

Five CI jobs, and each one is there to protect a specific claim in this README:

| job | what it would catch |
|---|---|
| `test` | a bare install — nothing but `httpx` — breaking on Ubuntu or macOS, on Python 3.10 through 3.13 |
| `test-full` | every suite, with every optional framework present, and **zero skips permitted** — because a skipped test and a passing test look identical in a green check mark |
| `clean-install` | the runtime dependency list growing past `httpx`, asserted from installed metadata rather than from the source |
| `boundary` | an internal system name reaching the published package, including inside an error string, where it would print into your logs and disclose something you cannot act on |
| `node-plugins` | the OpenClaw and DeepSeek Harness plugins drifting from the Python reader — their JS suite runs against the *same* `Trajectory.read`, so a line one side writes and the other counts as malformed fails here rather than surfacing as a silently short tool-call list |

The hero animation is generated, not drawn. [`docs/assets/render.sh`](docs/assets/render.sh) rebuilds it from [`render_flywheel_loop.py`](docs/assets/render_flywheel_loop.py); run it after touching the renderer so the checked-in GIF cannot drift from its source.

## Contributing

Issues and pull requests welcome. If you are adding a framework adapter, the bar is the one the built-ins meet: it must not import its framework at module scope, it must report `unknown` rather than guessing, and it must fail loudly at bind time rather than degrade silently.

## Security

Please report vulnerabilities privately to security@percepteye.com rather than in a public issue.

## License

[MIT](LICENSE) © Percepteye.
