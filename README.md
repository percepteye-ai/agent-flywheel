# PerceptEye-Agent-Flywheel

**Run your own agent against Percepteye rollouts.** Outbound-only, one dependency, MIT.

Your agent stays yours. You do not rewrite it, port it, or hand it over. You point its LLM
client at an endpoint we supply, implement or wrap three small methods, and run one sidecar
process. We supply the tasks and the scoring; you supply the agent.

```bash
pip install percepteye-agent-flywheel
```

```python
import percepteye_agent_flywheel as ph

ph.serve(
    "my_package.agent:build",          # importable, so each rollout gets a fresh child
    agent_id="my-coding-agent",
    input_shape="text",
    control_plane_url="https://api.percepteye.com/api/flywheel/v1",
)
```

That is the whole integration. The flywheel dials out, claims a rollout, runs your agent
against it, and reports the result. There is no listening socket, no inbound firewall rule,
no public DNS, and no TLS to terminate on your side.

---

## What this package actually does

| | |
|---|---|
| **Claims work** | Long-polls for rollouts. Leased, with redelivery — a crashed worker's rollout comes back rather than vanishing. |
| **Isolates each rollout** | One child process per rollout. Its own endpoint, its own single-use credential, its own ledger. |
| **Routes the completion wire** | Your agent's LLM calls go to a per-rollout endpoint, which is how the work is attributed and scored. |
| **Records tool outcomes** | `record_tool_call(...)` — the one thing worth instrumenting by hand. See below. |
| **Reports honestly** | Typed exclusions for crashes and expired credentials, so a substrate fault is never scored as a bad decision by your agent. |

---

## The one thing worth reading carefully

When your tool wrapper cannot tell whether a call succeeded, **say so**:

```python
from percepteye_agent_flywheel import record_tool_call

record_tool_call(
    "charge_card",
    {"amount": 4200},
    outcome="unknown",     # "ok" | "failed" | "unknown" — required, no default
    status_code=None,      # required keyword, may be None
)
```

`outcome="unknown"` is a first-class, supported answer. It is recorded as unknown, and the
checks that depend on a per-call outcome abstain on it. It is never rounded up to `"ok"`.

This matters more than it looks. Consider a tool configured to swallow its own exceptions and
return a string that reads like a normal result — a very common configuration. The HTTP status
is 200. The tool failed. If the status alone were treated as the outcome, that failed call
would be scored as a successful one, and an agent trained on that data learns to believe it.

So the type enforces it: `outcome` and `status_code` are both required and non-defaultable —
you cannot record half an outcome by forgetting a keyword — and when the outcome is `unknown`,
any status is discarded, from the field *and* from the result body. **A status may confirm or
contradict an observed outcome. It may never manufacture one.**

The corollary is worth stating too: an application-level permission denial that the API
returns as HTTP 200 is `outcome="failed", status_code=200`. Both halves are true and both are
recorded.

---

## Reporting results

Return a string and we will use it as the final output. Return a `RolloutOutput` when you have
more to say:

```python
from percepteye_agent_flywheel import RolloutOutput

def build():
    class Agent:
        def invoke(self, task: str) -> RolloutOutput:
            ...
            return RolloutOutput(
                final_text=answer,
                artifacts={"workspace": "workspace.tar.gz"},
                reward=my_own_score,        # advisory; recorded, not used to train
            )
    return Agent()
```

Two details that are easy to miss:

- **`tool_calls=None` and `tool_calls=[]` mean different things.** `None` is "I did not report
  tool calls". `[]` is "I observed zero tool calls" — a positive claim. The default is `None`,
  so an agent that reports nothing never accidentally asserts zero.
- **If you cannot classify every executed call, report none of them.** A partial list is
  indistinguishable from a short one. Either emit the unclassifiable call with
  `outcome="unknown"`, or set `tool_calls=None` for the whole rollout.

Your own `reward` scalar is recorded against the rollout and is useful for filtering. It is
not what scores the rollout — our compiled checks are.

---

## Isolation

`isolation="subprocess"` is the default, and it is the only mode with a real timeout: Python
cannot interrupt a synchronous call in another thread, so in-process a hung agent means we stop
*waiting* while it keeps running. A child process can actually be killed.

The trade-off is that a child cannot receive a live Python object, so subprocess isolation
needs an entrypoint it can import — `"my_package.agent:build"`. Passing an already-constructed
object is supported with `isolation="inprocess"`, which is capped at `concurrency=1`, because
concurrent in-process rollouts share one set of environment variables and would race for the
per-rollout endpoint.

Both of those are checked when `serve()` starts, not discovered mid-run.

### What the child process inherits

Your agent needs your credentials — that is the entire point of running it in your own
process — so your environment passes through. Ours does not:

- No variable in the `PERCEPTEYE_` namespace is inherited. The long-lived control-plane key
  stays in the parent and never reaches code the model can influence.
- Exactly two are set for the child: the rollout id and the ledger directory.
- The per-rollout endpoint and its single-use key are set last, so they win over anything
  inherited.

If you want the child to carry only what you name, pass an explicit allowlist.

---

## Entrypoints

Probed in order: `invoke`, `run`, `execute`, `chat`, `__call__`. Pass `method="..."` to name
one. A plain function is its own entrypoint.

Resolution happens once, at startup, and raises there — never mid-rollout. A rollout is one
member of a comparison group, so a rollout that fails to emit because a method name was wrong
does not merely lose itself; it distorts the scores of the rollouts it was supposed to be
compared against.

`input_shape` is declared, never guessed:

| `input_shape` | your entrypoint receives |
|---|---|
| `"text"` | `"the task description"` |
| `"messages"` | `[{"role": "user", "content": "..."}]` |
| `"dict"` | `{"input": "..."}` |

### Your agent can be a process

If it isn't Python — a compiled binary, a Node CLI, anything with a command line — pass an
argv list instead of an importable path. There is no protocol to implement, so an existing
CLI needs no changes:

```python
ph.serve(
    ["opencode", "run"],
    agent_id="coding-agent",
    input_shape="text",
)
```

The task arrives two ways, because CLIs differ and guessing wrongly is silent: on **stdin**,
and as `PERCEPTEYE_TURN_INPUT`. Its **stdout** is the final text. Its exit code is a result —
non-zero is excluded from training, because a crashed run says nothing about the policy.

Outcomes come from the same ledger, which a shell can append to:

```sh
D="$PERCEPTEYE_LEDGER_DIR/$PERCEPTEYE_ROLLOUT_ID"; mkdir -p "$D"
printf '{"name":"deploy","arguments":{},"outcome":"ok","status_code":200}\n' >> "$D/tool_calls.jsonl"
```

The command runs in its own **process group**, and that group is what gets killed on timeout —
a real CLI agent spawns freely, and killing only the process we launched leaves its children
holding the output pipe open.

---

## Framework adapters

**They are on by default, and they are how you get capture with no instrumentation.**
`adapters="auto"` binds every built-in whose framework is installed, so a LangChain,
LangGraph, or Google ADK agent reports per-call outcomes with no Percepteye code in it at
all — not one import, not one `record_tool_call`.

| Adapter | Covers | Outcomes | Routes LLM calls |
|---|---|---|---|
| `adk` | Google ADK | `failed` on raise **and** on `{"error": ...}`; `unknown` otherwise | no — the child environment routes |
| `langchain` | LangChain, LangGraph | `failed` on raise or an explicit `ToolMessage` status; `unknown` otherwise | only when you pass `llm=` |

```python
ph.serve("my_package.agent:build", agent_id="my-agent", input_shape="text")
# adapters="auto" is the default
```

Adapters are selected by **name**, because an adapter cannot cross a process boundary and
the default isolation runs your agent in a child. `adapters=[]` binds none. Naming one
explicitly — `adapters=["langchain"]` — is a stronger statement: you are asserting that is
how your calls are captured, so an adapter that cannot bind raises rather than degrading.

`record_tool_call` remains available and is still the ground truth. Reach for it when either
is true:

- **Your framework's own configuration beats the environment.** A client constructed as
  `ChatOpenAI(base_url=...)` in your code sends every completion to that URL regardless of
  what we set at spawn. The rollout completes, at your provider, and looks identical to a
  clean one from our side. An adapter can check and correct it — without one, we can't.
- **Its outcome interface saves you real instrumentation**, so you don't hand-annotate
  every tool.

```python
ph.serve(
    "my_package.agent:build",
    agent_id="my-agent",
    input_shape="text",
    adapters=["langchain"],
)
```

Adapters ship inside this package and add **no dependencies** — none imports its framework
until you bind it, so an absent framework simply fails that adapter's probe. Binding has
two outcomes and no third: it installs, or it raises `RoutingNotInstallable` at startup
with the reason and a remedy. There is no degraded-routing state, because the failure
destination is a live third party and the rollout still completes.

The `adk` adapter installs by wrapping `PluginManager.__init__` — the chokepoint every ADK
runner goes through, including both the deprecated `Runner(plugins=[...])` argument and the
current `app=` route. You construct your `Runner` exactly as you always have. ADK is the
better-fitting framework of the two, because it separates a tool that **raised** from one
that **returned**, so this adapter reports real `failed` verdicts rather than the defensive
`unknown` LangChain mostly limits us to.

Three things it deliberately never does, because ADK's `_run_callbacks` stops at the first
plugin returning a non-`None` value and uses that as the result: return from
`before_tool_callback` (your tool would never run), from `after_tool_callback` (your tool's
real result would be replaced), or from `on_tool_error_callback` (your exception would be
silently swallowed). It also catches everything internally — ADK re-raises a plugin's
exception as a `RuntimeError` into your agent, and a bug in our observer must cost us the
observation and nothing else.

The `langchain` adapter covers LangChain and LangGraph. Worth knowing what it does with
`on_tool_end`: that callback does **not** mean the tool succeeded. A tool with
`handle_tool_error=True` catches its own exception and returns a string that reads like a
normal result, and LangChain returns raw content rather than a `ToolMessage` whenever
there's no tool call id to attach. Both look like success. The adapter reports them as
`unknown` — which is exactly why it's worth having.

**This is the honest cost of zero instrumentation.** A tool that raises is recorded as
`failed`, a `ToolMessage` with an explicit status is recorded as what it says, and everything
else is `unknown`. Unknown outcomes lower the grade — they do not exclude the rollout — so
capture with no code gets you an honest floor, and one `record_tool_call` at the point where
you actually know the answer gets you the rest.

In a multi-agent graph each call is attributed to the agent that made it, read from the
framework's own run metadata — ADK exposes `ToolContext.agent_name` directly, LangGraph
stamps the node it is executing. A call nothing names stays unattributed: filling it in
would train a planner on a coder's tool calls.

To ship your own, expose a factory in the `percepteye_agent_flywheel.adapters` entry-point group.

## Configuration

Every environment variable this package reads. There are no feature flags, and no variable
selects behaviour — these are addresses and credentials.

| Variable | Meaning |
|---|---|
| `PERCEPTEYE_CONTROL_PLANE_URL` | Control-plane base URL. |
| `PERCEPTEYE_API_KEY` | Your flywheel key. |
| `PERCEPTEYE_ROLLOUT_ID` | *Set by us* for the child. Read it if you want to tag your own logs. |
| `PERCEPTEYE_LEDGER_DIR` | *Set by us* for the child. Write artifacts under `$PERCEPTEYE_LEDGER_DIR/$PERCEPTEYE_ROLLOUT_ID` and they are registered for you. |
| `PERCEPTEYE_TURN_INPUT` | *Set by us* for a command entrypoint. The task, also on stdin. |

---

## The contract

[`schema/flywheel-1.json`](schema/flywheel-1.json) is the authority. This repository owns it;
server implementations pin against the published schema, never the other way round. Python
types are generated from it, and a TypeScript package will be generated from the same file.

Fields derived from the inference wire — token-level data, sampling parameters, grouping keys,
scoring provenance — are absent from the agent contract by construction. An agent cannot
know them, so it is never asked to assert them.

---

## Development

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

The test suite includes a boundary check that fails the build if the runtime dependency list
grows beyond one entry, or if an internal system name appears anywhere in the package —
including inside an error message, where it would be printed into your logs and tell you
something you cannot act on.

## License

MIT. See [LICENSE](LICENSE).
