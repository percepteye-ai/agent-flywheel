# Quickstart

Install the package, add two lines to the process that already runs your agent, and it starts claiming rollouts. **Your agent does not move and does not change.**

## Install

```bash
pip install agent-flywheel
```

One runtime dependency: `httpx`. Python 3.10 → 3.13. Nothing else is needed to integrate.

The `[otel]` extra is available and optional — `pip install "agent-flywheel[otel]"` additionally emits each outcome as a span. It pulls the OpenTelemetry **API** only: this package emits into a `TracerProvider` you already registered and never installs one.

## The integration

```python
import agent_flywheel as fw

fw.serve("my_package.agent:build", agent_id="my-coding-agent", input_shape="text")
```

Two environment variables carry the address and the credential. They are the only two required:

```bash
export PERCEPTEYE_CONTROL_PLANE_URL=https://launch.percepteye.ai/api/flywheel/v1
export PERCEPTEYE_API_KEY=pefw_...        # free: sign in at https://launch.percepteye.ai
```

PerceptEye runs the control plane, and signing in is free and self-serve at [launch.percepteye.ai](https://launch.percepteye.ai). Connect your agent there and you get its key — shown once, and scoped to that one agent — along with the exact control-plane URL to export.

Or skip the copying: `agent-flywheel login --agent-id my-coding-agent` opens the same sign-in with a code filled in, and once you click **Approve** the key comes back to your terminal and is stored for that agent (`~/.percepteye/credentials.json`, readable only by you). `serve()` uses it whenever `PERCEPTEYE_API_KEY` is unset, and when it starts in a terminal with no key at all it offers to do this for you. It never asks in CI or when nobody is at a terminal. See [the key `login` stores](configuration.md#environment-variables).

Both are overridable in code as `control_plane_url=` and `api_key=`, and the argument wins. Everything else `serve()` takes has a default.

`serve()` blocks and returns an `int` — the number of rollouts that completed.

## The entrypoint is `"module:attr"`

Subprocess isolation is the default, and a child process cannot receive a live Python object. So the entrypoint is a string the child can import, and what it names has to build the agent on its own:

| you pass | what happens |
|:--|:--|
| `"my_package.agent:build"` — a zero-argument factory | called; the return value is your agent |
| `"my_package.agent:Agent"` — a class | instantiated with no arguments |
| `"my_package.agent:agent"` — an object that already has `invoke`/`run`/`execute`/`chat` | used as-is |

A class that needs constructor arguments is refused, by name, at startup:

```text
ConfigurationError: 'my_package.agent:Agent' names a class that cannot be
constructed with no arguments (Agent.__init__() missing 1 required positional
argument: 'model'). Point the entrypoint at a zero-argument factory that
builds a configured instance instead.
```

A class is instantiated rather than probed for methods on purpose: probing first would find `Agent.invoke` on the class — an unbound function — and call it with the turn input as `self`.

Once the object exists, the method is probed in this order, first match wins: `invoke`, `run`, `execute`, `chat`, `__call__`. Pass `method="..."` to name one. **Resolution happens once, at startup, and never mid-rollout** — a rollout that fails to emit because a method name was wrong does not merely lose itself, it biases the score of every sibling it was supposed to be compared against.

??? note "Passing an already-constructed object"

    `fw.serve(my_agent, ...)` works, but only under `isolation="inprocess"`, which is capped at `concurrency=1` and has no real timeout — Python cannot interrupt a synchronous call in another thread. Under the default it raises at startup:

    ```text
    ConfigurationError: isolation='subprocess' needs an importable entrypoint
    so the child can rebuild the agent -- pass 'my_package.agent:build' instead
    of a constructed object, or set isolation='inprocess' (concurrency=1).
    ```

## `input_shape` is declared, never guessed

| `input_shape` | your entrypoint receives |
|:--|:--|
| `"text"` | `"the task description"` |
| `"messages"` | `[{"role": "user", "content": "the task description"}]` |
| `"dict"` | `{"input": "the task description"}` |

There is no default and no inference. An earlier SDK guessed the call convention from which method name it happened to find, so `run(str)` and `run(dict)` were indistinguishable until something failed at depth.

## Bound the first run

!!! warning "A rollout is a real run"

    Rollouts execute your agent, in your process tree, on your credentials, against whatever your tools actually talk to. Bound the first one with `max_rollouts=8` and point `artifacts_dir=` somewhere you don't mind it writing.

```python
fw.serve(
    "my_package.agent:build",
    agent_id="my-coding-agent",
    input_shape="text",
    max_rollouts=8,
    artifacts_dir="/tmp/percepteye-first-run",
    on_event=lambda kind, ev: print(kind, ev),
)
```

| argument | default | what it does |
|:--|:--|:--|
| `max_rollouts` | `None` | stop after this many rollouts **complete**, then return. The count is completions, not successes — a crash and a graded pass both increment it. |
| `artifacts_dir` | `./.percepteye-artifacts` | created if absent. Each rollout gets `_percepteye/<rollout_id>/` beneath it, deleted once the result is reported, so a long-running flywheel does not grow without bound. |
| `concurrency` | `1` | rollouts in flight, under the default subprocess isolation. |
| `rollout_timeout_s` | `900.0` | per-rollout wall clock. Subprocess isolation is the only mode with a real timeout, because a child process can actually be killed. |

**`max_rollouts` is the only clean exit `serve()` has.** Nothing in this package installs a SIGINT or SIGTERM handler, there is no environment equivalent, and it cannot be changed on a running process — so without it the loop runs until the process is killed and any in-flight rollout is orphaned rather than abandoned, to be reclaimed by lease expiry. See [configuration](configuration.md) for what stopping a live flywheel actually costs.

## Misconfiguration raises before the first rollout

`serve()` does its whole preflight up front, in this order. Nothing below is discovered mid-run:

| step | raises when |
|:--|:--|
| resolve the mode | `PERCEPTEYE_AGENT_MODE` or `mode=` is a value it cannot parse |
| read the control-plane URL | never: with neither `control_plane_url=` nor `PERCEPTEYE_CONTROL_PLANE_URL` set, it is `https://launch.percepteye.ai/api/flywheel/v1` |
| read the API key | neither `api_key=` nor `PERCEPTEYE_API_KEY` is set |
| create `artifacts_dir` | the path cannot be created |
| `GET /health` | the control plane does not answer, or does not speak `flywheel/1` |
| build the runner | the entrypoint does not import, does not resolve to a callable, or is incompatible with the isolation mode |
| `POST /agents` | registration fails |

One check warns rather than refuses. The gateway speaks the OpenAI completions API, so provider variables it cannot serve — `AZURE_OPENAI_ENDPOINT`, `GOOGLE_API_KEY`, `GROQ_API_KEY` and the rest of `UNROUTABLE_PROVIDER_VARS` — are **named at startup and removed from each rollout's child**. A name on that list is evidence, not proof: your agent may not use that provider at all. But a loud warning beats an agent quietly sampling its live provider unobserved. **Your agent must sample through an OpenAI-compatible client.**

## What you see

**By default, nothing.** A flywheel with no `on_event` callback is silent, and an empty queue and a broken claim route look identical from the outside. Pass the callback and they separate:

```text
rollout_start {'rollout_id': 'r-1'}
rollout_reported {'rollout_id': 'r-1', 'grade': None, 'duplicate': False, 'exclusion': None}
claim_empty {'poll_after_ms': 500}
```

`on_event` is `Callable[[str, dict], None]`, called **synchronously on whichever thread produced the event** — the serving thread for `claim_*`, a pool worker for the per-rollout kinds, the heartbeat thread for `lease_lost` and `heartbeat_failed`. Exceptions it raises are swallowed: an observability hook must not be able to change what it observes. Keep it cheap. The full vocabulary is in [configuration](configuration.md).

Two lines `serve()` prints on its own, both to stdout:

```text
[percepteye] policy: <arm> — <reason> → <model> at <base_url>
[percepteye] mode=production — not claiming rollouts. (set PERCEPTEYE_AGENT_MODE=training to contribute capacity)
```

The first is informational only. A `serve()` rollout samples through the per-rollout gateway URL — that is what attributes and captures the completion — so the policy is reported and deliberately not applied. Production agents call `apply_current_policy()` instead; see [production](production.md).

## Standing the same image up without contributing

!!! note "One switch, both ways"

    ```bash
    PERCEPTEYE_AGENT_MODE=production    # serve() returns 0 immediately
    PERCEPTEYE_AGENT_MODE=training      # contribute capacity (the default when unset)
    ```

    In `production`, `serve()` returns **before opening any connection**, so that deployment needs no control-plane URL and no key — turning training off never requires being configured for it. It returns rather than raising, because a switch that crashes the service it was meant to quiet is not a switch, and it prints the line above, because a silent no-op is the failure mode this package refuses everywhere else. An unrecognized value raises instead of guessing. Full table of accepted spellings and precedence in [configuration](configuration.md).

The switch gates `serve()` — the call that does training work. It does not gate `attach()`, which is the production capture surface.

## Next

- Per-call outcomes, and why `unknown` is a supported answer — [outcomes](outcomes.md)
- LangChain, LangGraph and Google ADK with zero code in your agent — [Python agents](python.md)
- A binary, a Node CLI, anything with a command line — [CLI and other languages](cli.md)
- Picking up the trained model and prompt — [production](production.md)
