# prime-agent through the Flywheel

Status: **routing verified; the streaming blocker below is RESOLVED in the
gateway (SSE passthrough + capture) — runnable once that gateway is
deployed.**

This is an **optional example adapter**, not SDK default behavior. prime-agent
is a TypeScript binary, so it is served through the generic **command
entrypoint** with explicit `PrimeHarnessSnapshot` and
`PrimeAgentOutputAdapter` objects — no Percepteye code inside the agent.

## Required fixed-harness configuration

The OPSD example refuses to start without three explicit paths:

```bash
export PERCEPTEYE_PRIME_HARNESS_SOURCE=/absolute/path/to/harness
export PRIME_AGENT_KERNEL_VENV=/absolute/path/to/kernel-venv
export PRIME_AGENT_PACKAGE_ROOT=/absolute/path/to/prime-agent
```

`PERCEPTEYE_PRIME_HARNESS_SOURCE` names the harness directory itself (normally
`~/.prime/agent/harness`), not the whole agent directory. An empty directory is
a valid, explicit empty harness. At startup the flywheel snapshots its exact
paths and bytes once. Every discovery/rollout child receives a private HOME and
`PRIME_AGENT_CODING_AGENT_DIR` containing a read-only copy, and the copy is
attested after exit. Each private agent directory also starts with fixed,
read-only settings that disable auto-refine, auto-compaction, and telemetry
writes. The command disables implicit skills, prompt templates, context files,
and extension discovery; only the explicit `route.ts` extension remains. Any
harness or fixed-settings mutation excludes that rollout.

Every eligible result carries the framework-neutral
`agent_fingerprint.execution_sha256` (one lowercase 64-hex SHA-256). It is a
domain-separated canonical digest over the exact discovery observation, the
command and serve configuration, the SDK/contract version, and opaque
adapter-attested component hashes. OPSD consumes only this generic key; it does
not know that this example uses Prime or a Python kernel.

For diagnostics, `agent_fingerprint.harness_sha256` still names only the
snapshotted continual-harness tree and `agent_fingerprint.kernel_sha256` names
only the stored kernel venv. The Prime adapter also folds the fixed settings,
the explicit `route.ts`, `package.json`, and complete installed `dist/` tree
under `PRIME_AGENT_PACKAGE_ROOT` into the generic execution digest and checks
their metadata seals around each rollout. `prime-agent` on `PATH` must resolve
inside that package root. The model weights remain a separate gateway-derived
policy binding. Native Node/dynamic-library bytes outside the package root and
kernel venv remain host provenance; this digest does not claim otherwise.

`PRIME_AGENT_KERNEL_VENV` must point to a pre-provisioned Prime kernel runtime.
It is shared runtime code, not continual-harness state. Making it explicit
prevents four isolated children from each bootstrapping a separate runtime
under their private HOME. The SDK hashes its complete stored tree (plus direct
regular-file targets and in-root directory aliases such as the common Linux
`lib64 -> lib`) once as
`agent_fingerprint.kernel_sha256`, disables Python bytecode writes, and checks a
tamper-evident metadata seal before and after every child. Any package, file, link, or mode
mutation excludes the current rollout and refuses every later launch, even if
the writer restores the original bytes. This is fail-closed detection, not an
OS permission boundary against another process running as the same user. Use a dedicated, fully provisioned
venv; do not install or upgrade packages while a run is active. External
dynamic libraries loaded by its interpreter remain host-runtime provenance,
not part of this venv digest.

The example starts one run-owned Prime daemon under a separate scrubbed HOME
and empty run-owned working directory, then explicitly shuts it down before
removing the snapshot. The daemon receives no ambient project settings and no
customer, provider, or Percepteye credential. Prime's client-owned session
protocol supplies each worker with its private rollout environment, avoiding a
supervisor whose live state belongs to the first disposable child.

The example reads `PERCEPTEYE_MODEL_ID`, `PERCEPTEYE_CONTEXT_WINDOW`, and
`PERCEPTEYE_MAX_TOKENS` in the parent, then the Prime adapter supplies matching
`PRIME_AGENT_FLYWHEEL_*` child variables and attests their digest. Core scrubs
the inherited `PERCEPTEYE_` namespace and has no Prime route branch.

## What is verified

Routing works. A probe stood up an OpenAI-compatible server, set
`OPENAI_BASE_URL` and `OPENAI_API_KEY` exactly as `build_child_env` does, and
the request arrived:

```
gateway hits: 1
  path: /v1/chat/completions
  auth: Bearer perk_single_use_probe
```

Note prime-agent does **not** read `OPENAI_BASE_URL` itself — `route.ts` reads
it and hands it to `registerProvider`. Setting the variable alone does nothing
and the agent would quietly use its own provider, which is the silent-capture
failure this whole design exists to prevent.

## What blocked it (RESOLVED)

**prime-agent requests `stream: true`, and the gateway used to refuse to
stream.** Measured request keys: `['max_completion_tokens', 'messages',
'model', 'store', 'stream', 'stream_options', 'tools']`. The old gateway
forced `out["stream"] = False`, so the agent got a shape it could not parse —
the assistant message came back with `content: []` and zero usage. prime-agent
has no compat flag to disable streaming on its side.

The gateway now honours `stream: true`: the client sees the upstream's SSE
bytes verbatim, the gateway reassembles the same bytes into the capture the
non-streaming path would have produced, and persists once the stream ends.
Two consequences worth knowing when reading captures:

* Persisting happens at stream END (logprobs / finish reason / usage are not
  known earlier), so a process death mid-stream loses that call's record.
* A stream that ends early is recorded as **truncated**, not dropped — a
  partial completion must never score as a whole one.

Upstream 4xx/5xx after the 200 SSE headers arrives as an SSE error frame, so
the access log alone will not show upstream rejections — read the capture.

## Two traps the integration closes

**`isError` lies.** In `--mode json`, `tool_execution_end.isError` is `false`
even when the IPython cell raised — it is hardcoded `false` whenever `execute()`
returns without throwing (`agent-loop.ts:890`). The truth is in
`result.details.status == "error"`. Reading the obvious field would record
failures as successes, which is precisely the fabrication cell this contract
refuses. The optional Prime adapter therefore treats that structured status as stronger
evidence than either false `isError` flag.

**Not every completion is a rollout token.** Auto-refine and compaction make
their own model calls through the same provider. They cross the gateway and are
not task tokens, so they must be excluded or they pollute the training wire.
The example launches with `--no-session` (no persisted harness to refine), its
route extension cancels `session_before_compact`, and the runner marks any
stream that still reports refinement or compaction as untrainable.

Prime emits `compaction_start` before the auxiliary call and
`compaction_end` afterward, so an interrupted compaction is still observable.
Refinement exposes only `refine_complete` / `refine_failed`, not a
`refine_start`; an interrupted planning-only refinement has no JSON event to
parse. That is why disabling auto-refine and skills is a required prevention,
not merely an optimization. Post-run seals additionally catch any harness or
kernel write, including a write whose original bytes were restored.

## The identity problem

The Continual Harness rewrites its own prompts, memories and skills via
`/refine`. If it refines mid-group, the replicas of one prompt are no longer the
same agent — and group-relative advantage assumes they are. This example
uses both controls: `--no-session` removes local persistence, while the
run-scoped snapshot isolates and attests the global harness. A refinement write
that still gets through can invalidate only its own child, never a later
replica, and the affected rollout is excluded.

## Better routing than route.ts

Prime Intellect's own RL harness writes a per-rollout `models.json` into a
per-trace agent dir and launches with `--provider`, setting
`PRIME_AGENT_CODING_AGENT_DIR` and `HOME` to that directory. That is worth
adopting over the extension: it isolates the harness state per rollout, which
is also the fix for the identity problem above.
