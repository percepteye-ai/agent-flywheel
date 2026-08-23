# Plan: PydanticAI adapter + DeepSeek Harness plugin

**Status: parked.** Written 2026-08-22. Nothing here is implemented. No code
has been written against any of it.

Everything marked **[verified]** was read from primary source — npm tarballs,
a shallow clone of the upstream repo, the GitHub API, or this repository.
Everything marked **[open]** is not yet established and must not be assumed.

---

## 1. Why these two need different mechanisms

| | PydanticAI | DeepSeek Harness (dsh) |
|---|---|---|
| What it is | Python library, importable `Agent` objects | TypeScript/Node CLI on Cordis |
| Official Python package | `pydantic-ai` 2.33.0, py≥3.10 **[verified]** | `deepseek-harness-sdk` — a *driver* SDK **[verified]** |
| Runs as | in-process object | child process |
| Capture mechanism | **in-process adapter** | **in-agent plugin** |
| Lands in | `src/percepteye_agent_flywheel/adapters/pydantic_ai.py` | a separate npm package |

This introduces a **third capture mechanism** to the product, and it
generalizes past dsh:

| mechanism | for | ships as |
|---|---|---|
| in-process adapter | Python frameworks — LangChain, LangGraph, ADK, PydanticAI | `adapters/*.py` |
| stdout event stream | CLI agents that already log — prime-agent, OTel GenAI | a `StreamFormat` entry |
| **in-agent plugin** | frameworks with a plugin system — dsh | a package we publish |

The first two are **pull**: we monkeypatch, or we scrape text meant for humans.
The third is **push** — the agent hands us the data through a public extension
point, and the platform guarantees containment rather than us enforcing it by
hand.

### The ledger is already the plugin ABI

No new protocol is needed. This repository already ships one, and the README
documents a **shell one-liner** as a conforming client:

```sh
D="$PERCEPTEYE_LEDGER_DIR/$PERCEPTEYE_ROLLOUT_ID"; mkdir -p "$D"
printf '{"name":"deploy","arguments":{},"outcome":"ok","status_code":200}\n' >> "$D/tool_calls.jsonl"
```

Two environment variables and newline-delimited JSON. No Python, no SDK, no
network. A dsh plugin is therefore the *second implementation of a proven ABI*,
not a new integration surface. The reusable deliverable is to promote the
ledger from an implementation detail to a documented, versioned wire format —
it is already half-specified as `ToolCallRecord` in `schema/flywheel-1.json`.

---

## 2. Why dsh is worth the effort

**[verified]** via the GitHub API and a shallow clone of `master` @ `b150a55`:

| | |
|---|---|
| `deepseek-ai/deepseek-harness` | **185,522 ★**, MIT, created 2026-08-13 |
| Repo description | *"DeepSeek Harness: Everything is a Plugin."* |
| Repos tagged `dsh-plugin` | **10,762** |
| Topics on the repo | `ai-agents`, `cordis`, `dsh`, `dsh-plugin` |

`CONTRIBUTING.md:15` and `README.md:42` both instruct plugin authors to tag
their repository with the `dsh-plugin` topic "for discoverability". That is a
distribution channel the adapter mechanism has no equivalent of.

---

## 3. BLOCKER — fix before anything else

`OutcomeLedger.read()` is not crash-tolerant. `src/percepteye_agent_flywheel/outcomes.py:195`:

```python
for line in fh:
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)          # <- no guard
```

`_attach_ledger` (`runner.py`) calls it unguarded, **unlike**
`_attach_stream_tool_calls`, which is explicitly wrapped with the comment
*"never fail a rollout over telemetry"*. That asymmetry is the bug.

**Failure mode.** A `SIGKILL` on deadline truncates the final line mid-write →
`json.JSONDecodeError` → propagates out of `run_one` → the rollout is reported
as `flywheel_crash`. **Every** tool call is lost, not just the partial one, and
a completed rollout's work is discarded.

Today this is unlikely: `record_tool_call` holds a lock and writes short lines
from the same process. **The moment an external Node process appends to the
ledger under a kill signal it becomes probable.**

**Fix.** Skip malformed lines, count them, and surface the count rather than
swallowing it silently. Absence must stay distinguishable from zero: a file
that existed but yielded no parseable records is not the same as no file.

---

## 4. Two general SDK features these require

Both were chosen deliberately over dsh-specific hacks.

### 4.1 `fw.TASK` — argv templating

dsh headless takes the task as a **positional argv argument**; the SDK
currently delivers it on stdin and in `PERCEPTEYE_TURN_INPUT`, neither of which
reaches it.

```python
fw.serve(["dsh", "--profile", "percepteye", fw.TASK], agent_id=..., input_shape="text")
```

A module-level sentinel rather than a `"{task}"` string, because a sentinel
cannot collide with a literal argument and can be validated at startup —
consistent with the house rule that the dangerous shape should be
unconstructible.

Semantics, all checked at startup:

- **No `TASK` in argv → behaviour unchanged.** Fully backwards compatible.
- **`TASK` present → the task is delivered by argv, and stdin gets nothing.**
  Otherwise an agent reading both processes the task twice. Declared, not
  guessed — same posture as `input_shape`.
- `PERCEPTEYE_TURN_INPUT` stays set either way; it is inert unless read.

Two implementation notes:

- `serve()` builds its registration label with `" ".join(runner._command)`,
  which will raise on a non-string sentinel. This is the same class of bug the
  code comments record having shipped once before. Render it as `<task>`.
- **Security:** argv is visible to other users via `ps`; stdin is not. This is
  a real trade-off and belongs in the docs beside the feature, not discovered
  later.

### 4.2 `endpoint_env=` — configurable endpoint variable names

**[verified]** dsh reads **no** `OPENAI_BASE_URL` — grepped across the whole
repo; it does not exist. The SDK currently sets `OPENAI_BASE_URL` /
`OPENAI_API_KEY` in the child, so **out of the box dsh would ignore us, use its
own credentials, and send every completion to DeepSeek unobserved.** That is
precisely the failure `examples/prime_agent/README.md` records living through
("we simply owned none of the completion wire").

```python
fw.serve(..., endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"))
```

Names declared, never guessed. Useful well beyond dsh.

---

## 5. dsh LLM routing — **[verified] viable**

Sources: npm tarballs `@deepseek-ai/dsh@0.1.1-rc.2`, `dsh-llm@0.0.1-rc.1`,
`dsh-llm-pi-ai`, `dsh-llm-deepseek`, `dsh-headless`, `dsh-base`; repo @ `b150a55`.

Note: `@deepseek-ai/dsh-core` and `@deepseek-ai/dsh-config` **do not exist**
(404 on the registry).

| lever | mechanism |
|---|---|
| Base URL | `DEEPSEEK_BASE_URL` — `llm-deepseek/src/index.ts:185,360` |
| API key | `DEEPSEEK_API_KEY` — `DEFAULT_API_KEY_ENV`, `index.ts:79` |
| Model id | **not settable by env** — needs `--patch` or `settings.yaml` |

`DEEPSEEK_BASE_URL` is on dsh's **bootstrap-only** list
(`packages/boot/app-boot/src/index.ts:93-116`): only the inherited process
environment may supply it, and a `.env` file naming it is rejected at load.
Credential precedence is documented as **inherited process environment always
wins**. That is exactly our spawn model.

**Wire format** (`adapter.ts:607`): `fetch(\`${baseURL}/chat/completions\`)`
with `authorization: Bearer <key>`. Point `DEEPSEEK_BASE_URL` at
`https://host/v1`. `WireRequest` hardcodes `stream: true` and
`stream_options: { include_usage: true }` — **the gateway must serve SSE**
(already true per the prime-agent notes).

**Model id** needs `--patch <path>`, which is repeatable, applied after the
profile layer, and supports `!!js process.env.X` interpolation:

```yaml
- id: agent-default-model
  config:
    provider: deepseek-official
    model: !!js process.env.PERCEPTEYE_MODEL_ID
```

`--dump-config` prints the composed tree for verification. Caveat from
`app-boot/README.md`: an id-targeted patch **replaces the named entry's whole
`config`** — restate unchanged fields.

---

## 6. The plugin: `percepteye-dsh`

One npm package, **zero runtime dependencies**, mounting as an ordinary Cordis
plugin.

### 6.1 Subscribe to exactly one event

**[verified]** from `packages/core/tools/src/index.ts:135-206`. Six `tools/*`
events exist; five are unusable for observation.

| event | `@mode` | use it? |
|---|---|---|
| `tools/pre-execute` | **waterfall** | **never** — can `deny` / `ask` |
| `tools/execute` | **waterfall** | **never** — around-dispatch wrapper |
| `tools/post-execute` | **waterfall** | **never** — can replace or **block** the result |
| `tools/code-dispatch-log` | **waterfall** | **never** — rewrites the logged copy |
| **`tools/result`** | **emit** | **yes — the only one** |
| `tools/change` | emit | n/a (registry churn, deliberately unfiltered) |

`tools/result` is documented as: *"Observe the frozen, lossless-JSON final
outcome. Listener failures are contained. Scope-filtered dispatch: keyed by
`exec.agent`."*

That last clause gives **multi-agent attribution for free** — the same
`agent_name` field the ADK and LangGraph adapters work to populate.

**Payload [verified]:** `ToolExecution` carries `callId`, `rootCallId`, `name`,
`arguments`, `agent?`, `parent?`, `token`, `signal`. `ToolExecutionResult` is a
discriminated union on `isError`, where `ToolExecutionSuccess` has
`value?: never` on the failure side and vice versa.

### 6.2 Outcome mapping — the crux

The naive `isError` reading is wrong, and getting it wrong reintroduces the
exact defect this SDK exists to prevent. dsh's bash tool documents: *"non-zero
exits are reported, not errored; only infrastructure failures surface as
`isError`."* **So `isError: false` means "the dispatch completed", not "the
operation succeeded."** A flat `isError:false → ok` scores every failed
`npm test`, every non-zero `grep`, every rejected `git push` as a success.

| case | → | why |
|---|---|---|
| `isError: true` | `failed` | `ToolExecutionFailure` never carries a value; the registry normalizes throws into it. A real framework-issued verdict. |
| `isError: true`, cancellation/timeout code | **`unknown`** | Cancellation arrives as `isError:true` *even when the body ran and its side effect landed*. `failed` would assert a failure nobody observed — the mirror image of the sin, equally forbidden. |
| `isError: false`, `bash` / `run_code` / `terminal_run` | parse exit code; **`unknown`** if unreadable | These carry the true verdict in-band. Never fall back to `ok`. An exit code is **not** an HTTP status — leave `status_code` null or `ToolCallOutcome.__post_init__` raises `ContractViolation`. |
| `isError: false`, verified tools (`read`, `write`, `edit`, `glob`, `grep`, `todo_write`) | `ok` | No in-band status channel exists, so failure has nowhere to go but the failure variant. **This is where the training signal comes from.** |
| everything else — MCP, third-party, subagents | **`unknown`** | An MCP server returning `{"error": ...}` as ordinary content is the swallowed-error case verbatim, invisible from here. |

A tool moves from `unknown` to `ok` by a fixture plus a contract test, never by
assumption. Config may only widen the verified list; it can never override the
`isError: true` case.

### 6.3 Safety rules

- **Inert when `PERCEPTEYE_LEDGER_DIR` / `PERCEPTEYE_ROLLOUT_ID` are absent** —
  register no listener at all, so the plugin is always safe to leave installed.
- **No tool events → write nothing**, never an empty file. `None` and `[]` must
  stay distinguishable end to end.
- **One synchronous `write(2)` on a single `O_APPEND` fd per record**, built
  fully in memory first. That is what stops concurrent tools interleaving.
- **`try/catch` internally anyway.** dsh guarantees containment; we do not rely
  on it, exactly as `adk.py` does not rely on ADK.
- Never write to stdout/stderr — stdout is the agent's `final_text`.
- **Structural mirror** of dsh's payload types instead of
  `import type` from `@deepseek-ai/dsh-tools`, plus a type-level contract test
  pinning the mirror against the real package. Upstream drift becomes one red
  test in one file rather than silent data loss.

### 6.4 Distribution

**[verified]** A package is a dsh plugin **iff `package.json` declares
`dsh.bundle.patch`**. There is no enforced name convention — not
`dsh-plugin-*`, not `@scope/dsh-*`. `dsh-*` is de-facto community habit;
`dsh-plugin` is a **GitHub topic**, not a package prefix.

- Dev loop, no publishing needed: `--patch` overlay with an absolute path.
- Install: `dsh plugin --profile <name> add <package>` — a thin pnpm forwarder,
  so local paths, tarballs and `github:` specs all work.
- Tag the repo `dsh-plugin`.

---

## 7. Risks

**Cordis v4 is an unstable rewrite with no documentation.** `cordis@4.0.0-rc.8`.
`waterfall` is *new* in v4; `reusable` and `reactive` were **removed**; scopes
became `Fiber`; `Config` moved to Standard Schema. `cordis.js.org` is a dead
js.org 302 stub — `/en/` and `/guide/` both 404 — and the repo README is one
line. **Everything in §6 was read from source, not docs.** Mitigation: pin
exact versions; let the contract test be the alarm.

**dsh itself is `0.1.1-rc.2`**, a self-described developer preview. Pin the
exact version in CI, never a dist-tag.

**First-class means CI must prove it.** We just fixed a gap where
`test_zero_instrumentation.py` — the proof of the headline claim — never ran.
Claiming dsh support without a Node CI job running a real rollout recreates
that exact failure in a launch README. dsh supports Linux x64/arm64 and macOS
14+ arm64; **no Windows**.

**Naming, confirmed in the wild.** PyPI `deepseek-harness` is **not**
DeepSeek's — it is an unrelated project by another author. DeepSeek ships as
`deepseek-harness-sdk`. A 185k-star project lost its obvious PyPI name to a
squatter. Argument for claiming `percepteye` now.

---

## 8. PydanticAI — **[open]**

The research agent did not return; nothing below is established.

Planned shape: a real adapter mirroring `adk.py` / `langchain.py` — no
module-scope framework import, `bind()` returns an unbind, errors contained,
`RoutingNotInstallable` when named explicitly. Plus a `_probe_pydantic_ai` in
`introspection.py`, with the constraint that PydanticAI supports **dynamic
system prompts**, which must be reported as unread rather than invoked.

Three questions that must be answered before writing code:

1. Which interception point fires on **every** tool call and exposes failure —
   `FunctionToolResultEvent`, a `WrapperToolset`, or instrumentation hooks?
2. What is genuinely **unknowable**? Where do `ModelRetry` and
   `RetryPromptPart` sit, and is there a `handle_tool_error` equivalent that
   swallows exceptions?
3. **Does PydanticAI honour `OPENAI_BASE_URL` / `OPENAI_API_KEY` from the
   environment?** Decisive: if yes, routing needs *zero* adapter code, because
   the SDK already sets both in the child.

---

## 9. Sequence

1. **Ledger reader crash-tolerance** (§3). Real bug regardless of dsh; blocks
   everything that writes the ledger from outside Python.
2. **`fw.TASK` argv templating** and **`endpoint_env=`** (§4). Two general SDK
   features with tests.
3. **`percepteye-dsh` L1** — capture only: `tools/result`, the five-case
   classifier, contract test, inert-without-env.
4. **CI** — pinned dsh, Node job, end-to-end rollout against a fake control
   plane.
5. **L2 routing** — `--patch` generation for the model id, once L1 is proven.
6. **PydanticAI adapter** — after §8's questions are answered.

Items 1 and 2 are independently valuable and carry no dsh risk. If this is
picked up piecemeal, do those first.

---

## 10. Sources

- Repo: `github.com/deepseek-ai/deepseek-harness`, default branch **`master`**,
  shallow clone @ `b150a55` (2026-08-21)
- npm: `@deepseek-ai/dsh@0.1.1-rc.2`, `dsh-tools@0.0.1-rc.1`,
  `dsh-headless@0.0.1-rc.1`, `dsh-llm@0.0.1-rc.1`, `dsh-llm-pi-ai`,
  `dsh-llm-deepseek`, `dsh-base`, `cordis@4.0.0-rc.8`, `@cordisjs/core@3.18.1`
- PyPI: `deepseek-harness-sdk@0.1.1rc1` (official), `pydantic-ai@2.33.0`
- Event contract: `packages/core/tools/src/index.ts:135-206`, cross-checked
  against published `lib/types/index.d.ts`
- This repo: `outcomes.py:180-210`, `runner.py` `_attach_ledger` /
  `_attach_stream_tool_calls`, `contract.py`, `schema/flywheel-1.json`
