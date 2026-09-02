# goose

Serving [goose](https://github.com/aaif-goose/goose) rollouts — a Rust agent
with a desktop app, a CLI and an API, now under the Agentic AI Foundation.

goose is served through the **command entrypoint**: it is a compiled binary, not
an importable Python object, so there is no protocol to implement and nothing is
added to goose itself.

## The two halves

**Routing** needs no extension. goose's openai provider resolves its endpoint
from the environment, and the names it reads are the ones `build_child_env`
already sets per rollout — so `serve_goose.py` is the whole integration.

**Tool outcomes** come from the hook plugin in
[`packages/percepteye-agent-flywheel-hooks/`](../../packages/percepteye-agent-flywheel-hooks/).
Install it once; it is inert unless `PERCEPTEYE_TRAJECTORY_DIR` is set, which the
SDK sets per rollout.

`goose plugin install` takes a git URL, so a local checkout is installed by
copying it into a plugins directory (`~/.agents/plugins/` for user scope,
`<project>/.agents/plugins/` for project scope):

```bash
cp -R ../../packages/percepteye-agent-flywheel-hooks ~/.agents/plugins/percepteye-agent-flywheel
PERCEPTEYE_CONTROL_PLANE_URL=... PERCEPTEYE_API_KEY=... python serve_goose.py
```

## Two ways goose silently routes past the gateway

Both are closed in `serve_goose.py`, and both are worth understanding because
each produces a rollout that looks perfect and teaches nothing.

1. **`OPENAI_HOST` outranks `OPENAI_BASE_URL`.** goose resolves the host in the
   order *`OPENAI_HOST` env → `OPENAI_BASE_URL` env → `OPENAI_HOST` config*. The
   SDK sets `OPENAI_BASE_URL` last so it beats anything inherited, but that only
   settles ties among names the SDK knows — `OPENAI_HOST` is not in
   `isolation.UNROUTABLE_PROVIDER_VARS`, so an operator with it exported sends
   every completion to their own host while the control plane believes it is
   sampling the policy under training.

2. **`GOOSE_PROVIDER` selects the provider at all.** Routing the openai provider
   buys nothing if the operator's config selects `anthropic` or `claude-code`.

`serve_goose.py` scrubs both in the parent and passes `--provider openai`
explicitly.

## One way rollouts silently hang

`GOOSE_MODE` defaults to `auto` (auto-approve tool calls), but the *effective*
value is the operator's persisted config. Someone who once chose "Ask before
every tool call" has `approve` on disk, and every headless rollout then blocks on
a prompt nobody is there to answer until the episode budget kills it. Set
explicitly, not left to a saved preference.

## Bounds worth revisiting

`--max-turns` defaults to **1000**, which is not a bound. `serve_goose.py` sets
25 as a starting value — raise it if rollouts are being cut off before they
produce the evidence the reward reads, which is a wasted rollout rather than
cost control.

Rollouts run with a real shell in `GOOSE_WORKSPACE` (default
`~/.percepteye/goose-workspace`), deliberately outside any checkout. `isolation`
is a process boundary and an environment scrub; it says nothing about the
filesystem.
