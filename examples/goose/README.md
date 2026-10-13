# goose

Serving [goose](https://github.com/aaif-goose/goose) rollouts — a Rust agent
with a desktop app, a CLI and an API, now under the Agentic AI Foundation.

goose is served through the **command entrypoint**: it is a compiled binary, not
an importable Python object, so there is no protocol to implement and nothing is
added to goose itself.

> [!IMPORTANT]
> **This is the capacity-contribution path, and it is not the only one.** Here the
> SDK drives goose: it claims a rollout, spawns the binary, and reports the
> outcome. That fits an operator who wants to lend the flywheel capacity.
>
> If instead you *run goose yourself*, you do not need this file or any worker
> process: `pip install agent-flywheel`, `agent-flywheel install-hooks`, and you
> start goose exactly as before. In training mode (the default), with
> `PERCEPTEYE_CONTROL_PLANE_URL` and `PERCEPTEYE_API_KEY` set, each session
> claims one practice task when a run has one for this agent
> (`PERCEPTEYE_AGENT_ID`, default `goose`), finds it in its working directory
> as `PERCEPTEYE_TASK.md`, and is reported as that task's result. With
> `PERCEPTEYE_AGENT_MODE=production` the hooks claim nothing and report
> nothing, key or no key: tool outcomes are recorded locally and goose keeps
> its own endpoint. `agent-flywheel status` prints which applies.

## The two halves

**Routing** needs no extension. goose's openai provider resolves its endpoint
from the environment, and the names it reads are the ones `build_child_env`
already sets per rollout — so on this path `serve_goose.py` carries the routing.

**Tool outcomes** come from the hook plugin, which **ships in this wheel**. There
is no second install and no external repository to clone: `agent-flywheel
install-hooks` renders it, pointed at this virtualenv's interpreter. Inside a
rollout it records into the `PERCEPTEYE_TRAJECTORY_DIR` the SDK sets; outside
one it records locally under `$PERCEPTEYE_HOME/turns/<session>` (default
`~/.percepteye`), and `PERCEPTEYE_CAPTURE=0` turns recording off.
`serve_goose.py` refuses to start without it, because `reports_tool_calls=True`
is an assertion only the plugin makes true.

```bash
agent-flywheel install-hooks
PERCEPTEYE_CONTROL_PLANE_URL=... PERCEPTEYE_API_KEY=... python serve_goose.py
```

## Two ways goose silently routes past the gateway

Both are closed in `serve_goose.py`, and both are worth understanding because
each produces a rollout that looks perfect and teaches nothing.

1. **`OPENAI_HOST` outranks `OPENAI_BASE_URL`.** Read from
   `crates/goose/src/providers/openai_def.rs::resolve_base_url`, the order is
   *`OPENAI_HOST` env → `OPENAI_BASE_URL` env → `OPENAI_BASE_URL` config →
   `OPENAI_HOST` config → `https://api.openai.com`*. The
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
