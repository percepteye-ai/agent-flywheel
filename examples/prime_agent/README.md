# prime-agent through the Flywheel

Status: **routing verified, blocked on gateway streaming.**

prime-agent is a TypeScript binary, so it is served through the **command
entrypoint** — no Python integration, no Percepteye code inside it.

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

## What blocks it

**prime-agent requests `stream: true` and our gateway refuses to stream.**
Measured request keys: `['max_completion_tokens', 'messages', 'model', 'store',
'stream', 'stream_options', 'tools']`. `inference.py:178` sets
`out["stream"] = False`, so the agent gets a shape it cannot parse — the
assistant message comes back with `content: []` and zero usage.

There is no compat flag to disable streaming; the documented list covers
`supportsStore`, `supportsUsageInStreaming`, `maxTokensField` and similar, and
none of them turn it off.

So the gateway needs SSE support: pass the stream through to the client while
reassembling it server-side for capture. Until then prime-agent runs and
produces nothing.

## Two traps for whoever writes the event parser

**`isError` lies.** In `--mode json`, `tool_execution_end.isError` is `false`
even when the IPython cell raised — it is hardcoded `false` whenever `execute()`
returns without throwing (`agent-loop.ts:890`). The truth is in
`result.details.status == "error"`. Reading the obvious field would record
failures as successes, which is precisely the fabrication cell this contract
refuses.

**Not every completion is a rollout token.** Auto-refine and compaction make
their own model calls through the same provider. They cross the gateway and are
not task tokens, so they must be excluded or they pollute the training wire.

## The identity problem

The Continual Harness rewrites its own prompts, memories and skills via
`/refine`. If it refines mid-group, the replicas of one prompt are no longer the
same agent — and group-relative advantage assumes they are. Pin
`PRIME_AGENT_CODING_AGENT_DIR` per group, or disable refinement during rollouts.

## Better routing than route.ts

Prime Intellect's own RL harness writes a per-rollout `models.json` into a
per-trace agent dir and launches with `--provider`, setting
`PRIME_AGENT_CODING_AGENT_DIR` and `HOME` to that directory. That is worth
adopting over the extension: it isolates the harness state per rollout, which
is also the fix for the identity problem above.
