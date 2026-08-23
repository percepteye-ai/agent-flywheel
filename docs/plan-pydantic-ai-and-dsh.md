# Plan: the capture ABI, the mode question, and four integrations

**Status: parked.** Written 2026-08-22, extended 2026-08-23. Nothing here is
implemented. No code has been written against any of it.

Everything marked **[verified]** was read from primary source — npm tarballs, a
shallow clone of the upstream repo, the GitHub API, or this repository.
Everything marked **[open]** is not yet established and must not be assumed.

> The filename is narrower than the contents. This began as a PydanticAI + DeepSeek
> Harness plan and grew to cover OpenClaw, ACP, and the training/production mode
> question. Rename when convenient.

---

## 0. The one idea this document is organised around

**The trajectory is the ABI.** Every integration, in every language, reduces to a shim
that does two things: append JSON lines to
`$PERCEPTEYE_TRAJECTORY_DIR/$PERCEPTEYE_ROLLOUT_ID/tool_calls.jsonl`, and make the
agent's completions go to an endpoint we supply.

This is not aspirational. It is already true, and **[verified]**:
`adapters/adk.py:132` and `adapters/langchain.py:128` both call
`record_tool_call()`, which appends to that file. The README already documents a
**shell one-liner** as a conforming client. A Cordis plugin and an OpenClaw
plugin would write *the same lines*.

So there is one mechanism with several hosts, not several integrations:

| framework | capture shim | status | routing | status |
|---|---|---|---|---|
| LangChain / LangGraph | in-process adapter | ✅ built | env, or `llm=` | ✅ |
| Google ADK | in-process adapter | ✅ built | env | ✅ |
| PydanticAI | in-process adapter | planned | **[open]** — see §8 | |
| prime-agent | stdout `StreamFormat` | ✅ built | `route.ts` extension | ✅ |
| **DeepSeek Harness** | **Cordis plugin** | planned §6 | `DEEPSEEK_BASE_URL` | ✅ verified §5 |
| **OpenClaw** | **OpenClaw plugin** | planned §7 | **✅ already solved** | see §7 |

Consequences worth stating plainly:

- dsh and OpenClaw are **two instances of one mechanism**, not two projects.
  Build it once for dsh and OpenClaw is incremental.
- The existing Python adapters need no change. They are the Python instance of
  the same contract.
- Promoting the trajectory from an implementation detail to a **documented, versioned
  wire format** is the reusable deliverable. It is already half-specified as
  `ToolCallRecord` in `schema/flywheel-1.json`.

---

## 1. BLOCKER — fix before anything writes the trajectory from outside Python

`Trajectory.read()` is not crash-tolerant. `outcomes.py:195`:

```python
for line in fh:
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)          # <- no guard
```

`_attach_trajectory` calls it unguarded, **unlike** `_attach_stream_tool_calls`,
which is explicitly wrapped with the comment *"never fail a rollout over
telemetry"*. That asymmetry is the bug.

**Failure mode.** A `SIGKILL` on deadline truncates the final line mid-write →
`json.JSONDecodeError` → propagates out of `run_one` → the rollout is reported as
`flywheel_crash`. **Every** tool call is lost, not just the partial one, and a
completed rollout's work is discarded.

Today this is unlikely: `record_tool_call` holds a lock and writes short lines
from the same process. **The moment an external Node process appends to the
trajectory under a kill signal it becomes probable.**

**Fix.** Skip malformed lines, count them, surface the count. Absence must stay
distinguishable from zero: a file that existed but yielded no parseable records
is not the same as no file.

---

## 2. The mode question, and the lesson from the prior SDK

### What is recoverable about the prior SDK

**Nothing, from this repository, by design.** Its codename appears in exactly one
place: the internal-names tuple in `tests/test_oss_boundary.py`, which the
boundary test uses to keep those names *out* of anything this repo publishes.
Naming it here would fail that test — as it did on the first draft of this
document, which is the check working.

**[open]** — if someone can supply, out of band, what its five modes were and
which one was real, this section can be sharpened.

What *is* recorded is the lesson, in the module docstring:

> *"Modes that do not change what the SDK actually does are not offered, because
> an earlier SDK shipped five of them, of which four were byte-identical strings
> in one payload field the server never read."*

### The lesson applied

**Do not add `mode="training"` / `mode="production"`.** It would be the fifth
fake mode. `FlywheelMode` is `Literal["rollout"]` and should stay that way.

Ask instead what genuinely differs between the two roles. Two things do, and
neither is a mode:

1. **Where the task came from** — a user, or the control plane. A per-turn
   **fact**, not a process setting.
2. **Whether the completion went through a capturing endpoint.** A per-turn
   **observation**, which can differ turn to turn inside one process.

Model those two and the mode disappears.

---

## 3. The production design — no donor worker

**Constraint: deploying a standing donor worker is a non-starter in the
enterprise.** The design must put the SDK *inside* the agent, and let the control
plane switch the endpoint to a fine-tuned model.

### The reframe

If the SDK sits in the production agent and completions flow through a
Percepteye-supplied endpoint, **production traffic is the training corpus.** No
second process is required to *capture* anything.

### What already exists

- `apply_current_policy()` — the control plane chooses the endpoint. This is
  exactly the mechanism required, already built, and already documented as
  *"Use this in your PRODUCTION agent, the one serving your users."*
- The adapters — in-process tool capture, zero instrumentation.
- The trajectory, and introspection.

### What is missing

1. Adapters bind only inside `serve()`'s runner. There is no public "bind these
   in my production process" entry point.
2. The contract is rollout-shaped (`RolloutRequest`, `rollout_id`). There is no
   production-turn shape.
3. Policy adoption is startup-only, deliberately — a policy that changes under a
   running agent is the same confound the control plane refuses on its own
   training runs.
4. No way to mark a turn that **bypassed** capture.

### The proposed surface

```python
import percepteye_agent_flywheel as fw

fw.attach(agent_id="my-agent")   # register + resolve policy + bind adapters
```

One call at startup. It should register the agent (with introspection), resolve
the current policy into the endpoint configuration, bind the framework adapters
in-process, and return a handle whose policy can be refreshed explicitly rather
than polled.

### Two hard problems, stated rather than papered over

**(a) The gateway lands in the users' critical path.** Enterprises will resist
that as hard as a donor worker — Percepteye's availability becomes their
availability. The house position already exists and should extend: `current_policy`
fails open because *"an unreachable control plane must never stop a customer's
agent from serving its users."*

But a bypassed turn **must be recorded as bypassed, not absent**. That is the
absence≠zero rule one layer up, and getting it wrong yields a training set that
silently over-represents the turns that happened to work.

**(b) Production traffic is one sample per task.** Group-relative advantage needs
N samples of the *same* task. So "keeps training itself" has a real staging
constraint:

| stage | what improves | what it needs |
|---|---|---|
| 1 | prompts and tool descriptions, from production traces | nothing new — this is AWS AgentCore's model and it is genuinely friction-free |
| 2 | SFT'd weights behind the same endpoint | capture only; the endpoint switch is invisible to the agent |
| 3 | on-policy RL | sampling capacity — **a burst worker run for a training window, not standing infrastructure** |

Stage 3 reframes the donor worker from "deploy this forever" to "run this for an
afternoon", which is a materially different enterprise conversation.

---

## 4. Two general SDK features

Both chosen over vendor-specific hacks.

### 4.1 `fw.TASK` — argv templating

Some CLIs take the task as a **positional argv argument**; the SDK delivers it on
stdin and in `PERCEPTEYE_TURN_INPUT`, neither of which reaches them.

```python
fw.serve(["dsh", "--profile", "percepteye", fw.TASK], agent_id=..., input_shape="text")
```

A module-level sentinel rather than a `"{task}"` string: a sentinel cannot collide
with a literal argument and can be validated at startup — consistent with the
house rule that the dangerous shape should be unconstructible.

Semantics, all checked at startup:

- **No `TASK` in argv → behaviour unchanged.** Fully backwards compatible.
- **`TASK` present → the task is delivered by argv, and stdin gets nothing.**
  Otherwise an agent reading both processes it twice. Declared, not guessed.
- `PERCEPTEYE_TURN_INPUT` stays set either way; it is inert unless read.

Two implementation notes:

- `serve()` builds its registration label with `" ".join(runner._command)`, which
  raises on a non-string sentinel. Same class of bug the code comments record
  shipping once before. Render it as `<task>`.
- **Security:** argv is visible to other users via `ps`; stdin is not. That
  trade-off belongs in the docs beside the feature.

### 4.2 `endpoint_env=` — configurable endpoint variable names

**[verified]** dsh reads **no** `OPENAI_BASE_URL` — grepped across the whole repo.
The SDK sets that name today, so an unmodified integration would look correct
while every completion went to DeepSeek unobserved.

```python
fw.serve(..., endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"))
```

Names declared, never guessed. Useful well beyond dsh.

---

## 5. DeepSeek Harness — routing **[verified] viable**

Sources: npm `@deepseek-ai/dsh@0.1.1-rc.2`, `dsh-llm`, `dsh-llm-pi-ai`,
`dsh-llm-deepseek`, `dsh-headless`, `dsh-base`; repo @ `b150a55`.
`@deepseek-ai/dsh-core` and `dsh-config` **do not exist** (404).

| lever | mechanism |
|---|---|
| Base URL | `DEEPSEEK_BASE_URL` — `llm-deepseek/src/index.ts:185,360` |
| API key | `DEEPSEEK_API_KEY` — `index.ts:79` |
| Model id | **not settable by env** — needs `--patch` or `settings.yaml` |

`DEEPSEEK_BASE_URL` is on dsh's **bootstrap-only** list
(`packages/boot/app-boot/src/index.ts:93-116`): only the inherited process
environment may supply it, and a `.env` naming it is rejected at load. Credential
precedence is **inherited process environment always wins**. That is exactly our
spawn model.

**Wire** (`adapter.ts:607`): `fetch(\`${baseURL}/chat/completions\`)` with
`authorization: Bearer`. Point `DEEPSEEK_BASE_URL` at `https://host/v1`.
`WireRequest` hardcodes `stream: true` and `stream_options: {include_usage: true}`
— **the gateway must serve SSE**.

**Model id** needs `--patch <path>`, repeatable, applied after the profile layer,
supporting `!!js process.env.X`. `--dump-config` verifies the composed tree.
Caveat: an id-targeted patch **replaces the named entry's whole `config`**.

### Ecosystem

**[verified]** `deepseek-ai/deepseek-harness`, **185,522 ★**, MIT, created
2026-08-13, description *"DeepSeek Harness: Everything is a Plugin."*
`dsh-plugin` GitHub topic: **10,762 repos**. `CONTRIBUTING.md:15` and
`README.md:42` both instruct plugin authors to use that topic.

---

## 6. `percepteye-dsh` — the plugin

One npm package, **zero runtime dependencies**, mounting as an ordinary Cordis
plugin.

### 6.1 Subscribe to exactly one event

**[verified]** `packages/core/tools/src/index.ts:135-206`. Six `tools/*` events
exist; five are unusable for observation.

| event | `@mode` | use it? |
|---|---|---|
| `tools/pre-execute` | **waterfall** | **never** — can `deny` / `ask` |
| `tools/execute` | **waterfall** | **never** — around-dispatch wrapper |
| `tools/post-execute` | **waterfall** | **never** — can replace or **block** the result |
| `tools/code-dispatch-log` | **waterfall** | **never** — rewrites the logged copy |
| **`tools/result`** | **emit** | **yes — the only one** |
| `tools/change` | emit | n/a — registry churn |

`tools/result` is documented as *"Observe the frozen, lossless-JSON final
outcome. Listener failures are contained. Scope-filtered dispatch: keyed by
`exec.agent`."* That last clause gives **multi-agent attribution for free**.

**Payload [verified]:** `ToolExecution` carries `callId`, `rootCallId`, `name`,
`arguments`, `agent?`, `parent?`, `token`, `signal`. `ToolExecutionResult` is a
discriminated union on `isError`.

### 6.2 Outcome mapping — the crux

dsh's bash tool documents *"non-zero exits are reported, not errored; only
infrastructure failures surface as `isError`."* **So `isError: false` means "the
dispatch completed", not "the operation succeeded."** A flat mapping scores every
failed `npm test` as a success.

| case | → | why |
|---|---|---|
| `isError: true` | `failed` | `ToolExecutionFailure` never carries a value; the registry normalizes throws into it |
| `isError: true`, cancellation/timeout code | **`unknown`** | cancellation arrives as `isError:true` *even when the body ran and its side effect landed* |
| `isError: false`, `bash` / `run_code` / `terminal_run` | parse exit code; **`unknown`** if unreadable | in-band verdict. An exit code is **not** an HTTP status — leave `status_code` null or `__post_init__` raises |
| `isError: false`, verified tools (`read`, `write`, `edit`, `glob`, `grep`) | `ok` | no in-band status channel exists, so failure has nowhere to go but the failure variant. **This is where the signal comes from** |
| everything else — MCP, third-party, subagents | **`unknown`** | an MCP server returning `{"error": ...}` as content is the swallowed-error case verbatim |

Tools move from `unknown` to `ok` by a fixture plus a contract test, never by
assumption. Config may widen the verified list; it can never override `isError: true`.

### 6.3 Safety rules

- **Inert when `PERCEPTEYE_TRAJECTORY_DIR` / `PERCEPTEYE_ROLLOUT_ID` are absent** —
  register no listener, so the plugin is always safe to leave installed.
- **No tool events → write nothing**, never an empty file.
- **One synchronous `write(2)` on a single `O_APPEND` fd per record**, built fully
  in memory first. That is what stops concurrent tools interleaving.
- **`try/catch` internally anyway** — dsh guarantees containment; we do not rely
  on it, exactly as `adk.py` does not rely on ADK.
- Never write to stdout/stderr — stdout is the agent's `final_text`.
- **Structural mirror** of dsh's payload types instead of `import type`, plus a
  type-level contract test pinning it. Upstream drift becomes one red test.

### 6.4 Distribution

**[verified]** A package is a dsh plugin **iff `package.json` declares
`dsh.bundle.patch`**. No enforced name convention; `dsh-*` is habit and
`dsh-plugin` is a GitHub topic, not a prefix. Dev loop: `--patch` with an absolute
path, no publishing needed. Install: `dsh plugin --profile <name> add <package>`,
a thin pnpm forwarder, so local paths, tarballs and `github:` specs all work.

---

## 7. OpenClaw

**[verified]** `openclaw/openclaw`, **387,190 ★**, MIT © OpenClaw Foundation,
TypeScript, CalVer `2026.7.1-2` with `extended-stable` and `beta` channels — a
**more mature** target than dsh, which is RC-only.

### 7.1 Is it rollout-shaped? Narrowly

Not as a product. `VISION.md` disclaims the framework role; the command surface is
`channels devices message pairing qr matrix cron wiki` — bound to a person's phone
number, not something you sample 200 trajectories from.

But one slice is: **`openclaw agent --local`**, a documented one-shot. And its own
org already runs scored rollouts against it — `openclaw/shellbench` (Python) has
`EXECUTABLE_ADAPTERS = {"openclaw"}` and drives it over the Gateway WebSocket.

**Support "OpenClaw headless task runs", never "OpenClaw".**

### 7.2 Routing — **already solved in this repo**

See commit `684392c` and [`nemoclaw-sandbox.md`](nemoclaw-sandbox.md). OpenClaw
substitutes `${VAR}` into any config string at load time, so:

```json
"baseUrl": "${OPENAI_BASE_URL}",
"apiKey":  "${OPENAI_API_KEY}"
```

written **once** to `~/.openclaw/openclaw.json` resolves per rollout from
`build_child_env`, with no file write and no concurrency problem.

An earlier revision proposed generating a merged config per rollout; commit
`684392c` rejects that because it *"would serialise every rollout through one
shared file."* **Do not revisit it.**

**[open]** Verified against `openclaw@2026.3.11`; latest is `2026.7.1-2`.
Re-confirm the substitution behaviour before relying on it.

### 7.3 Capture — a plugin, same shape as dsh

**Do not add a `StreamFormat` for OpenClaw.** **[verified]** It emits nothing
structured on stdout: with `--json`, `protectJsonStdout` routes logs to stderr and
stdout carries one terminal envelope; without it, logs interleave unparseably.
Per-tool telemetry goes to a **file**. There is no stream to declare, and a
`StreamFormat` entry would match nothing forever.

The native plugin system is the path. **[verified]** Every plugin ships
`openclaw.plugin.json`, which OpenClaw reads *"to validate configuration without
executing plugin code"*; entry points come from `package.json#openclaw`. The
capability registry has 70+ registration methods; the relevant ones are
`registerAgentToolResultMiddleware`, `registerAgentEventSubscription` and
`registerRuntimeLifecycle`. **[open]** — which to use is not yet settled.

### 7.4 Two traps

**A swallowed-error default, in OpenClaw's own emitter [verified]:**

```js
function itemStatus(item) {
  if (status === "failed" || status === "error") return "failed";
  ...
  return "completed";   // ← default branch swallows everything else
}
```

Unrecognised or absent statuses — including `"cancelled"` and `"unknown"`, which
OpenClaw's own `auditNativeToolTerminalStatus` preserves as *distinct* — become
`"completed"`. Feeding that file to `_outcome_of()` would round unknown up to ok.

**`approval-unavailable` is deliberately excluded from the error flag**
(`dist/selection-JInn13lc.js:1938`). A tool that never ran because approval could
not be obtained arrives with no `error` field.

### 7.5 A number that should shape the product, not just this integration

Claw-SWE-Bench (arXiv 2606.12344): OpenClaw scores **19.1% Pass@1 with a minimal
adapter vs 73.4% with a full adapter — same GLM 5.1 backbone.** A 54-point spread
from harness configuration alone.

**Implication beyond OpenClaw:** if harness config drifts across a comparison
group, the gradient is noise rather than signal. Pinning harness config per group
is a general requirement of the grouping design.

---

## 8. Rejected: Agent Client Protocol (ACP) as a sixth mechanism

Investigated because `openclaw/acpx` (3,153 ★) suggested one protocol client could
reach ~40 agents at once. **Rejected.** Recording it so nobody re-runs the
analysis.

ACP is real: created by Zed, now jointly governed by Zed and JetBrains, Apache 2.0,
stable wire protocol version 1, NDJSON over stdio, ~3.9M crate downloads.

**The tool-call status enum genuinely maps cleanly**, which was the surprise:

```
ToolCallStatus (closed oneOf, stable v1): "pending" | "in_progress" | "completed" | "failed"
```

`pending` / `in_progress` / never-updated → `unknown`, correctly, for free. v1 has
no `cancelled`, so a cancelled call sits at `in_progress` and lands on `unknown`.

**Four blockers, in descending severity:**

1. **ACP carries no routing lever at all — disqualifying.** No method, field or
   capability lets a client tell an agent which endpoint to sample from. The only
   model-adjacent surface, `SessionConfigOptionCategory`, is disclaimed in the
   schema: *"for UX purposes… MUST NOT be required for correctness."* So "one
   integration reaches 40 agents" is false. It is *one capture integration reaches
   40 agents, each still needing its own routing work* — and routing is where all
   the risk and cost live.
2. **No machine-readable tool name in stable v1.** `ToolCall` has `title`
   (human prose) and `kind` (a 10-value bucket for *"choosing appropriate icons"*).
   `name` is unstable-only. Our contract requires a non-empty name.
   Worse: OpenClaw's ACP bridge *has* the real name and destroys it —
   `const title = formatToolTitle(name, args)`. Going through ACP is a **lossy
   downgrade of data the agent already had**.
3. **`status` is not required**, defaults to `pending`, absence unrecoverable.
4. Much of the interesting surface (`session/fork`, `providers/*`) is behind the
   unstable schema.

**Note for anyone re-reading the schema:** the appealing `Cancelled` and
`Other(String)` variants are **v2 (`2.0.0-alpha.3`, draft)**, not stable v1. Do
not quote them as available.

---

## 9. PydanticAI — **[open]**

Research did not complete. Nothing below is established.

Planned shape: an adapter mirroring `adk.py` / `langchain.py` — no module-scope
framework import, `bind()` returns an unbind, errors contained,
`RoutingNotInstallable` when named explicitly. Plus `_probe_pydantic_ai` in
`introspection.py`, with the constraint that PydanticAI supports **dynamic system
prompts**, which must be reported as unread rather than invoked.

Three questions to answer before writing code:

1. Which interception point fires on **every** tool call and exposes failure —
   `FunctionToolResultEvent`, a `WrapperToolset`, or instrumentation hooks?
2. What is genuinely **unknowable**? Where do `ModelRetry` and `RetryPromptPart`
   sit, and is there a `handle_tool_error` equivalent that swallows exceptions?
3. **Does PydanticAI honour `OPENAI_BASE_URL` / `OPENAI_API_KEY` from the
   environment?** Decisive: if yes, routing needs *zero* adapter code.

`pydantic-ai` is 2.33.0, requires Python ≥3.10 — compatible.

---

## 9b. Verifying an agent that acts on someone else's API

The prime-agent case is a **coding** agent: the deliverable is a file, the gate is
`test -s top_ips.sh`, and a hidden test grades the artifact. A **tool-calling**
agent — Jira, Confluence, Salesforce — is different in kind: the deliverable is
**state in a third-party system**, and verifying it means calling that system's
API.

Most of the machinery survives unchanged. One thing breaks hard.

### 9b.1 What survives

`gate_flag` + `metadata["completion_gate"]` already accepts an arbitrary shell
command, so a gate becomes a request instead of a file test:

```sh
curl -sf "$JIRA/rest/api/3/issue/$KEY" | jq -e '.fields.status.name == "Done"'
```

Still an existence-and-smoke check. The `--autonomous-gate-timeout-ms` bound
matters more here — the gate is now a network call against a rate-limited system,
re-run on every continuation.

### 9b.2 What breaks: the task cannot be repeated

Sixteen rollouts of *"resolve PROJ-412"* against production Jira produce sixteen
transitions, sixteen notification storms, sixteen SLA clocks. **Trajectory-level
repetition is unavailable on production for any write-acting agent.**

This makes the step-level design (§3) *more* necessary, not less. At step level no
tool is ever executed: sample k completions at a fixed prefix, score which call
the model chose and how it built the arguments, discard k−1. Zero side effects by
construction. For a Jira agent that is most of the signal anyway — the failures
are wrong endpoint, wrong transition id, malformed body, missing required field.

**Two tiers, answering different questions:**

| tier | question | needs | cadence |
|---|---|---|---|
| step, from logged prefixes | right call, valid arguments? | nothing — no API access | daily, on production traffic |
| trajectory, in a disposable instance | right end state? | a resettable target | per training batch |

### 9b.3 τ-bench, and the blind spot that argues for the tri-state

**[verified]** from source (arXiv 2406.12045). τ-bench's reward is **SHA-256
equality of the entire canonicalized database** — not a subset check, not
per-field assertions. `calculate_reward` hashes the DB, replays the ground-truth
actions from a fresh load, hashes again, and compares.

Three consequences read straight off the code:

- **Extra reads are free.** Read tools never mutate `data`.
- **List order is load-bearing.** `to_hashable` preserves list order and writes
  append, so two orderings of the same logical writes hash differently. "Any path
  to the same state passes" holds for the *state*, not unconditionally.
- **A failed write leaves no trace and does not fail the task.** `step()` wraps
  tools in `try/except` returning `f"Error: {e}"`, and tools guard early. The
  agent attempts a forbidden transition, is rejected, the DB is unchanged — and
  the hash still matches.

That last one is the argument. **State-hash verification cannot see
attempted-and-rejected actions; per-call tri-state outcomes can.** It is the same
blindness as `handle_tool_error`, one layer up, sitting in the field's leading
benchmark for exactly this agent shape.

**Adopt `pass^k`** — the probability that *all* k trials succeed, the inverse of
`pass@k`. τ-bench introduced it because customer service needs reliability, not
discoverability, and the measured collapse is the headline: **gpt-4o exceeds 60%
at pass^1 and falls below 25% at pass^8.** For an enterprise tool-calling agent
that is the number that blocks deployment, and it is a better daily claim than
accuracy.

### 9b.4 The environment: per-run namespacing, not vendor sandboxes

**[verified]** Vendor sandboxes are not a per-rollout reset primitive:

| | reset | API? |
|---|---|---|
| Atlassian Cloud Sandbox (Premium/Enterprise only) | *"up to 14 days to copy data"* | **none — UI only** |
| Salesforce Full sandbox | 29-day refresh interval | — |
| ServiceNow PDI | "a few hours", UI-triggered | none |

Atlassian sandboxes also **do not copy** Marketplace apps, webhooks, API tokens or
third-party credentials, and copy automation rules **disabled** — a sim-to-real
gap inside the vendor's own sandbox.

What published benchmarks actually do is **a real instance with per-run
namespacing and programmatic teardown**. Salesforce's own CRMArena-Pro
(arXiv 2505.18878) runs on live org sandboxes seeded with ~54k synthetic records
rather than a mock — a strong signal from a vendor with total control over that
tradeoff.

**For Jira: a free developer site, one project per rollout, deleted by API
afterwards.** Provisioning is manual and once; teardown is seconds and scriptable.

### 9b.5 The credential problem — genuinely open space

**[verified]** No eval or RL platform surveyed — HUD, Veris, LangSmith, Braintrust,
Galileo/Splunk, AgentCore, Salesforce, Microsoft — calls a customer's live
third-party API to verify an agent action. They avoid it four ways: simulate the
system, containerize a replica, *be* the system of record, or grade the trace
instead of the world.

The closest precedent for the shape below is **Braintrust remote evals**: the
customer runs the eval server, Braintrust triggers it and displays results —
*"Your evaluation code runs on your machine or server… Only results are sent to
Braintrust."* Their own caveat applies to us: it is designed for **data
residency, not for distrusting the customer**; returned scores are accepted at
face value.

**The trust inversion is unsolved in this market.** The only prior art is
decentralized RL (Prime Intellect, Gensyn) and crowdsourcing QC. The cheap proven
answer is **gold tasks**: seed known-answer probes into every batch, and stop
trusting a tenant's self-reported grades when its probes come back passing.

### 9b.6 Design: post-conditions, not a training mode

The question was whether to add an explicit **training mode** with an out-of-band
executor shipped in the SDK. Assessment:

**It would be a real mode, not a fake one** — it changes what runs, what side
effects occur, and what is reported. So it does not fail the §2 test. But it is
not needed, because **the harness parent is already an out-of-band executor**:
`serve()` runs in the parent, the agent runs in a child, and the parent already
collects artifacts after the child exits. Running a post-condition there is a few
lines in `_finalise`, and it is out-of-band with respect to the agent by
construction.

**Recommended shape — the existing two-sided pattern, reused verbatim:**

| | serve configuration (a fact about the deployment) | task metadata (a property of the task) |
|---|---|---|
| completion gate | `gate_flag` | `completion_gate` |
| **post-condition** | **`verify_env=`** — "this deployment points at a disposable instance" | **`post_conditions`** |

Either side missing runs nothing, which is exactly today's pre-gate behaviour.
The declaration matters on its own: running post-conditions against *production*
Jira is itself unwanted — audit noise, rate limits — so the customer must state
that the target is disposable.

The verdict is recorded as an ordinary tool call, so no new concept enters the
contract:

```json
{"name":"__verify.jira","arguments":{"issue":"PROJ-412"},
 "outcome":"failed","status_code":200,
 "error":"status=In Progress, expected Done"}
```

`status_code: 200` with `outcome: failed` is the `charge_card` case exactly — the
request succeeded, the assertion did not, both halves true.

**Options considered and why they lose:**

| option | verdict |
|---|---|
| post-conditions via the two-sided pattern | **recommended** — no mode, no new process, no new contract concept |
| a `verify=` Python callable on `serve()` | more expressive, but Python-only, so it does not serve dsh or OpenClaw |
| a separate executor process | the donor worker again. **One genuine use** — see below |
| a real `mode="training"` flag | only if several behaviours must be gated together. They need not be |

**The one argument for a separate executor: eventual consistency.** Jira indexes
asynchronously, so a post-condition run immediately after the agent exits may read
stale state. A delayed re-check cannot happen inside the rollout's lease. If that
proves to matter in practice, it is the reason to build an executor — and not
before.

---

## 10. Risks

**Cordis v4 is an unstable rewrite with no documentation.** `cordis@4.0.0-rc.8`.
`waterfall` is *new* in v4; `reusable` and `reactive` were **removed**; scopes
became `Fiber`. `cordis.js.org` is a dead js.org 302 stub — `/en/` and `/guide/`
both 404 — and the repo README is one line. **§6 was read from source, not docs.**

**dsh is `0.1.1-rc.2`**, a self-described developer preview. Pin exact versions,
never a dist-tag.

**First-class means CI must prove it.** We fixed a gap where
`test_zero_instrumentation.py` — the proof of the headline claim — never ran.
Claiming dsh or OpenClaw support without a Node CI job running a real rollout
recreates that failure in a launch README. Note OpenClaw supports Linux x64/arm64
and macOS 14+ arm64; **no Windows**.

**Naming, confirmed in the wild.** PyPI `deepseek-harness` is **not** DeepSeek's —
it is an unrelated project by another author; DeepSeek ships as
`deepseek-harness-sdk`. PyPI `openclaw` is likewise an unrelated `cmdop` CLI. Two
of the largest agent projects on GitHub lost their obvious PyPI names to
squatters. Argument for claiming `percepteye` now.

---

## 11. Sequence

1. **Trajectory reader crash-tolerance** (§1). Real bug regardless of anything else;
   blocks everything that writes the trajectory from outside Python.
2. **`fw.TASK` and `endpoint_env=`** (§4). Two general SDK features with tests.
3. **`percepteye-dsh` L1** — capture only: `tools/result`, the five-case
   classifier, contract test, inert-without-env.
4. **CI** — pinned dsh, Node job, end-to-end rollout against a fake control plane.
5. **OpenClaw capture plugin** — routing is already done; this is the remaining
   half, and it is the second instance of the §6 mechanism.
6. **`fw.attach()`** (§3) — the production-side surface. Independently the highest
   commercial value item, and it depends on none of 3–5.
7. **PydanticAI adapter** — after §9's questions are answered.
8. **Post-conditions** (§9b.6) — `verify_env=` plus `post_conditions` in metadata,
   executed by the harness parent and recorded as an ordinary tool call. Small,
   and it is what makes a tool-calling agent gradeable at all.

Items 1, 2 and 6 are independently valuable and carry no third-party risk. A
piecemeal pickup should start there. **6 is the one that changes the product.**

---

## 12. Sources

- `github.com/deepseek-ai/deepseek-harness`, branch **`master`**, @ `b150a55`
- npm: `@deepseek-ai/dsh@0.1.1-rc.2`, `dsh-tools`, `dsh-headless`, `dsh-llm`,
  `dsh-llm-pi-ai`, `dsh-llm-deepseek`, `dsh-base`, `cordis@4.0.0-rc.8`,
  `@cordisjs/core@3.18.1`, `openclaw@2026.7.1-2`, `acpx@0.13.1`
- PyPI: `deepseek-harness-sdk@0.1.1rc1` (official), `pydantic-ai@2.33.0`
- ACP: `agentclientprotocol/agent-client-protocol`, `schema/v1/schema.json`,
  `schema/v1/meta.json`
- Claw-SWE-Bench: arXiv 2606.12344
- This repo: `outcomes.py:180-210`, `runner.py` `_attach_trajectory` /
  `_attach_stream_tool_calls`, `adapters/adk.py:132`, `adapters/langchain.py:128`,
  `contract.py`, `schema/flywheel-1.json`, `tests/test_oss_boundary.py:24-28`,
  commits `684392c` and `74feb63`, `docs/nemoclaw-sandbox.md`
