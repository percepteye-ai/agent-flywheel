# Quickstart

Connect your agent, let it send us its real conversations, and start improving it. Your agent keeps running where it runs today, and its code does not change.

## What you need

<!-- lang -->
- Python 3.10 to 3.13.
- An agent built with LangChain, LangGraph or Google ADK. These are the frameworks whose model calls the SDK can record, and what the model saw and said is what we learn from. Any other agent that already exports OpenTelemetry traces can [send us those](otlp.md) instead.
<!-- /lang -->
- A free sign-in at [launch.percepteye.ai](https://launch.percepteye.ai).

## Install

<!-- lang -->
```bash
pip install agent-flywheel
```

One runtime dependency, `httpx`.
<!-- /lang -->

## Get your key

[Get your key](dashboard.md#get-your-key) in the dashboard, then put it, and the control-plane URL shown with it, in your agent's environment:

```bash
export PERCEPTEYE_CONTROL_PLANE_URL=https://launch.percepteye.ai/api/flywheel/v1
export PERCEPTEYE_API_KEY=pefw_...
```

<!-- lang -->
The URL is also the SDK's default. Without a key the SDK still records on your disk, and sends nothing.
<!-- /lang -->

## Record its conversations

<!-- lang -->
Call `attach()` once, at startup, and wrap each turn your agent serves:

```python
import agent_flywheel as fw

px = fw.attach(agent_id="support-agent", trajectory_root="/var/lib/percepteye")

def answer(conversation_id: str, user_text: str) -> str:
    with px.turn(conversation_id=conversation_id, input_text=user_text) as turn_id:
        reply = agent.invoke(user_text)          # your agent, unchanged
    px.record_answer(turn_id, reply)             # its answer, as text
    return reply
```

One environment variable carries the credential, and it is the only one required:

```bash
export PERCEPTEYE_API_KEY=pefw_...        # from https://launch.percepteye.ai/onboard/agent-sdk
```

PerceptEye runs the control plane, and signing up is free and self-serve at [launch.percepteye.ai](https://launch.percepteye.ai/onboard/agent-sdk). Name your agent there and you get its key, shown once and scoped to that one agent. The control plane's address defaults to `https://launch.percepteye.ai/api/flywheel/v1`; `PERCEPTEYE_CONTROL_PLANE_URL` points it somewhere else. Without a key, `serve()` raises before the first rollout and names that page.

Or skip the copying: `agent-flywheel login --agent-id my-coding-agent` opens the same sign-in with a code filled in, and once you click **Approve** the key comes back to your terminal and is stored for that agent (`~/.percepteye/credentials.json`, readable only by you). `serve()` uses it whenever `PERCEPTEYE_API_KEY` is unset, and when it starts in a terminal with no key at all it offers to do this for you. It never asks in CI or when nobody is at a terminal. See [the key `login` stores](configuration.md#environment-variables).

## See your first trace

The SDK sends turns in batches, so a turn does not appear the moment it happens.

<!-- lang -->
Every 32 turns it sends the ones that have an answer, and it sends the rest when the process exits. To send now, call `px.flush()`. It checks the permissions again and returns what it did, such as `{'sent': 1, 'skipped': 0, 'failed': 0}`; `'not_permitted': 1` means the permission is not on yet, and `px.upload_refusal` says why.
<!-- /lang -->

Then open [your agent's page](dashboard.md#see-what-arrived) to see it counted. Nothing there? See [Troubleshooting](troubleshooting.md#nothing-arrives-in-the-dashboard).

## Start improving

When conversations have arrived, [start improving](dashboard.md#start-improving). The loop:

1. We learn from your agent's real conversations.
2. Each new version is tested against your agent's current best, on conversations it has never seen.
3. A version that does better comes to you in [Decisions](dashboard.md#approve-the-winner-in-decisions). You approve it or not.
4. The next round starts from the best so far. Run it again whenever new conversations have arrived.

If there is not yet enough to learn from, Start improving says so.

## Run the approved version

<!-- lang -->
Ask which model to run on each turn, and rebuild your client only when the answer changes:

```python
from openai import OpenAI

policy = fw.policy_source(agent_id="support-agent")

snap = policy.current()                 # each turn; a cache read between refreshes
if snap.changed:
    client = OpenAI(base_url=snap.base_url, api_key=snap.api_key)
    model = snap.model
```

The approved prompt is a separate read, and the SDK never applies it for you: `fw.prompt_decision(fw.current_prompt(agent_id="support-agent"))` says whether it may be applied, and if not, why.
<!-- /lang -->

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

- [Production](production.md): what is sent and when, retries, rate limits and personal data.
- [OpenTelemetry push](otlp.md): send the traces your agent already exports.
- [FAQ](faq.md) and [Troubleshooting](troubleshooting.md).
<!-- lang -->
- [Python agents](python.md): every option, and practice tasks your agent can work through with `serve()`.
<!-- /lang -->
