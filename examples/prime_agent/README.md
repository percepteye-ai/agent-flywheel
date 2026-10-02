# prime-agent through the Flywheel

Status: **routing, streaming capture, execution attestation, and run-owned
daemon cleanup are implemented. A launch still depends on a healthy control
plane and a serving endpoint with the model's native tool parser enabled.**

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

One complete launch looks like this. Keep the model id stable as `policy`
across the base and candidate arms: the per-rollout gateway binds that logical
name to the exact policy under measurement.

```bash
cd /absolute/path/to/agent-flywheel
install -d -m 700 /absolute/path/to/empty-prime-harness
test -z "$(ls -A /absolute/path/to/empty-prime-harness)"
# Qualified macOS host: Node 24 must precede Homebrew's unpinned default Node.
export PATH=/opt/homebrew/opt/node@24/bin:/opt/homebrew/bin:/usr/bin:/bin
export PERCEPTEYE_CONTROL_PLANE_URL='https://control-plane.example/api/flywheel/v1'
export PERCEPTEYE_API_KEY="$(cat /absolute/path/to/prime-agent.key)"
export PERCEPTEYE_AGENT_ID='prime-agent'
export PERCEPTEYE_AGENT_MODE=training
export PERCEPTEYE_PRIME_HARNESS_SOURCE=/absolute/path/to/empty-prime-harness
export PRIME_AGENT_KERNEL_VENV=/absolute/path/to/kernel-venv
export PRIME_AGENT_PACKAGE_ROOT=/absolute/path/to/prime-agent
export PRIME_WORKSPACE=/absolute/path/to/dedicated-rollout-workspace
export PRIME_CONCURRENCY=4
export PRIME_MAX_TURNS=15
export PERCEPTEYE_MODEL_ID=policy
export PERCEPTEYE_CONTEXT_WINDOW=65536
export PERCEPTEYE_MAX_TOKENS=4096
node --version       # require v24.x on this qualified host
prime-agent --version
exec .venv/bin/python examples/prime_agent/serve_prime.py
```

`prime-agent` must resolve through its installed package, not through an
unattested debug wrapper. The example resolves the launcher symlink and refuses
it unless its target is below `PRIME_AGENT_PACKAGE_ROOT`; otherwise a wrapper
could change execution without moving the OPSD execution identity. The
launcher's `#!/usr/bin/env node` also makes the order of `PATH` material, so pin
and probe the intended Node runtime before starting the worker. Before polling,
the example runs the resolved launcher's `--version` and `--help` under its
scrubbed daemon environment and refuses a broken Node runtime or a Prime build
that lacks any rollout flag this adapter uses; it then runs that Prime's own
kernel check against the sealed kernel (below).

`PERCEPTEYE_PRIME_HARNESS_SOURCE` names the harness directory itself (normally
`~/.prime/agent/harness`), not the whole agent directory. An empty directory is
a valid, explicit empty harness. At startup the flywheel snapshots its exact
paths and bytes once. Every discovery/rollout child receives a private HOME and
`PRIME_AGENT_CODING_AGENT_DIR` containing a read-only copy, and the copy is
attested after exit. Each private agent directory also starts with fixed,
read-only settings that disable auto-refine, auto-compaction, and telemetry
writes, and that cap how long one model call may take. That cap is not
cosmetic: Prime's provider client defaults to a 600000ms request timeout with
retries on top, and no command-line flag reaches it — `--autonomous-timeout-ms`
is evaluated between turns, so it does not fire while a turn is blocked in the
call. Without the cap, a gateway that accepts the connection and never answers
holds a rollout until the SDK kills it on its deadline, with nothing in
`inference_calls` and nothing to say why. The command disables implicit skills,
prompt templates, context files, and extension discovery; only the explicit
`route.ts` extension remains. Any harness or fixed-settings mutation excludes
that rollout.

Every eligible result carries the framework-neutral
`agent_fingerprint.execution_sha256` (one lowercase 64-hex SHA-256). The same
framework-neutral digest is sent as `agent_execution_sha256` on the
authenticated rollout-worker registration, allowing the control plane to
freeze the exact worker before dispatch and then reconcile it against every
observed trajectory. It is a
domain-separated canonical digest over the exact discovery observation, the
command and serve configuration, the SDK/contract version, and opaque
adapter-attested component hashes. OPSD consumes only this generic key; it does
not know that this example uses Prime or a Python kernel.

**A sealed workload is bound to the digest it was prepared against, so changing
any input to the digest means re-preparing it.** The fixed settings, `route.ts`,
the harness tree, the route configuration, the kernel venv and the installed
`dist/` are all inputs; edit any of them and the restarted agent registers a
different `agent_execution_sha256`. Work already prepared under the previous one
stays invisible to it — the control plane makes a sealed rollout claimable only
by the exact frozen digest, and rejects a report whose fingerprint does not
match with `410 lease_lost`. The symptom is indistinguishable from an agent that
cannot run at all: it registers, long-polls, prints `claim_empty` forever, and
every rollout is reaped `status=abandoned` / `exclusion_reason=deadline_exceeded`.
Re-prepare every affected workload after the restart.

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

Provision it once with the installed prime-agent's own kernel bootstrap, with
`PRIME_AGENT_KERNEL_PYTHON` unset so that the bootstrap builds the venv rather
than checking an existing interpreter. It needs `uv`, and the network on a cold
uv cache:

```bash
env -u PRIME_AGENT_KERNEL_PYTHON PRIME_AGENT_KERNEL_VENV=/absolute/path/to/kernel-venv \
  node "$PRIME_AGENT_PACKAGE_ROOT/dist/core/kernel/bootstrap-cli.js"
```

Prime receives the venv as `PRIME_AGENT_KERNEL_PYTHON=<venv>/bin/python`, the
variable prime-agent documents for using an existing Python environment instead
of auto-bootstrapping one. Prime uses that interpreter as it is: it never reads
the venv's bootstrap marker and never deletes or rebuilds it, so a venv that a
different prime-agent version provisioned keeps running, under the
`kernel_sha256` that names it. (Handed only `PRIME_AGENT_KERNEL_VENV`, Prime
deletes and rebuilds in place any venv its private readiness marker does not
recognise, at the first session -- which the seal can only report as a mutated
kernel, on a rollout already lost.)

What Prime still does, at every session start, is check the interpreter and
refuse one that lacks `ipykernel`, a current prime-agent-runtime (`rlm`), or one
of the default packages that release installs. A refused kernel does not stop
the session: it completes, and each `ipython` call is answered with the
refusal instead of running. So the example refuses such a kernel before
serving, for the reason Prime gives. The snapshot refuses at startup a venv
whose `bin/python` cannot import `ipykernel` and `rlm`, the two packages
Prime's docs name for that variable, and the runtime preflight then runs the
installed prime-agent's own check (`dist/core/kernel/bootstrap-cli.js`, which
only checks when `PRIME_AGENT_KERNEL_PYTHON` is set) against the sealed
interpreter under the daemon environment. Re-provision if that check refuses
after a prime-agent upgrade. Prime's check starts one interpreter per default
package each time a session starts its kernel: a few seconds per rollout, where
its marker check took well under one.

That startup check is not the last word. Prime repeats it in every rollout,
counts a package whose interpreter merely failed to spawn as missing, and
writes nothing anyone re-checks, so a kernel the preflight accepted can still
be refused mid-run. The output adapter therefore excludes, as
`substrate_unavailable`, any rollout in which a tool call came back as Prime's
kernel-setup refusal (`PRIME_AGENT_KERNEL_PYTHON points to a Python missing`
or `Failed to set up the Python kernel runtime`), however many of its other
calls ran: the policy never ran that code, and is not graded as if it had. A
call counts as refused only when its error result also carries none of the
execution details (`status`, `durationMs`) the ipython tool reports for a cell
the kernel ran, so a cell that prints those words and then raises stays the
policy's own outcome.

The serving shell's Python configuration does not reach the kernel. Prime's
own processes are Node and pass their environment on to the kernel, so every
environment the adapter configures -- the kernel probe's, the daemon's, each
rollout child's and discovery's -- has `PYTHONPATH`, `PYTHONHOME`,
`PYTHONSTARTUP`, `IPYTHONDIR`, and macOS's `PYTHONEXECUTABLE` and
`__PYVENV_LAUNCHER__` removed, not emptied: an import path, Python home,
startup file, IPython profile, or another venv's interpreter path (on macOS
the interpreter starts in the venv beside the path either names) from outside
the venv can neither satisfy the probe's imports nor run code in the kernel
under `kernel_sha256`.

The example starts one run-owned Prime daemon under a separate scrubbed HOME
and empty run-owned working directory, then explicitly shuts it down before
removing the snapshot. The daemon receives no ambient project settings and no
customer, provider, or Percepteye credential. Prime's client-owned session
protocol supplies each worker with its private rollout environment, avoiding a
supervisor whose live state belongs to the first disposable child.

SIGTERM, SIGHUP, and Ctrl-C are converted into the same cleanup path. Repeated
stop signals are ignored while live rollouts drain and teardown completes, so
they cannot race removal of the attested snapshot. SIGKILL and machine failure
cannot be caught; an orphaned
`prime-agent` process alone is therefore not evidence that rollout polling is
alive. The worker is the Python `serve_prime.py` process. Each run uses a unique
Unix socket, so an older orphan cannot evict or serve a later run.

A Unix socket path cannot exceed 103 bytes on macOS, and Prime binds one per
session worker below `TMPDIR`. The run's socket and the `TMPDIR` the adapter
gives the daemon and every child are therefore the system temp directory when
the paths fit, and `/tmp` when a long `TMPDIR` would push them past the limit;
without that, the daemon exits at startup or every session times out on
`create`. A daemon that does exit at startup is refused with its own stderr.

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

## Why the explicit route remains

Prime Intellect's own RL harness writes a per-rollout `models.json` into a
per-trace agent directory. This adapter now provides the same per-rollout
`PRIME_AGENT_CODING_AGENT_DIR` and `HOME` isolation, while `route.ts` supplies
the one fact a static `models.json` cannot: the gateway URL and single-use key
that are different for every claimed rollout. Both the route bytes and their
fixed configuration are part of the execution identity.
