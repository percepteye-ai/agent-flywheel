# percepteye-harness

**Run your own agent against Percepteye rollouts.** Outbound-only, one dependency, MIT.

Your agent stays yours. You do not rewrite it, port it, or hand it over. You point its LLM
client at an endpoint we supply, implement or wrap three small methods, and run one sidecar
process. We supply the tasks and the scoring; you supply the agent.

```bash
pip install percepteye-harness
```

```python
import percepteye_harness as ph

ph.serve(
    "my_package.agent:build",          # importable, so each rollout gets a fresh child
    harness_id="my-coding-agent",
    input_shape="text",
    control_plane_url="https://api.percepteye.com/api/harness/v1",
)
```

That is the whole integration. The harness dials out, claims a rollout, runs your agent
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
from percepteye_harness import record_tool_call

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
from percepteye_harness import RolloutOutput

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
  so a harness that reports nothing never accidentally asserts zero.
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

---

## Configuration

Every environment variable this package reads. There are no feature flags, and no variable
selects behaviour — these are addresses and credentials.

| Variable | Meaning |
|---|---|
| `PERCEPTEYE_CONTROL_PLANE_URL` | Control-plane base URL. |
| `PERCEPTEYE_API_KEY` | Your harness key. |
| `PERCEPTEYE_ROLLOUT_ID` | *Set by us* for the child. Read it if you want to tag your own logs. |
| `PERCEPTEYE_LEDGER_DIR` | *Set by us* for the child. |

---

## The contract

[`schema/harness-1.json`](schema/harness-1.json) is the authority. This repository owns it;
server implementations pin against the published schema, never the other way round. Python
types are generated from it, and a TypeScript package will be generated from the same file.

Fields derived from the inference wire — token-level data, sampling parameters, grouping keys,
scoring provenance — are absent from the harness contract by construction. A harness cannot
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
