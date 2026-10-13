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

`agent_id` is the name you gave the agent in the dashboard. `attach()` registers the agent, which is how the dashboard sees it connect. Give every turn of one conversation the same `conversation_id`: how a conversation went on is part of what we learn from. Each turn is written under `trajectory_root` first; that directory is yours, and nothing deletes it unless you ask.
<!-- /lang -->

## Allow us to learn from them

Once the dashboard has seen your agent connect, give it [permission](dashboard.md#allow-us-to-learn-from-its-conversations) to learn from its conversations. Until you do, no conversation leaves your machine. You do not need to restart your agent: it asks again before each batch it would send.

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

Until you approve a new version, your agent keeps the one it has. The details are in [Production](production.md#run-the-approved-version).

## Next

- [Production](production.md): what is sent and when, retries, rate limits and personal data.
- [OpenTelemetry push](otlp.md): send the traces your agent already exports.
- [FAQ](faq.md) and [Troubleshooting](troubleshooting.md).
<!-- lang -->
- [Python agents](python.md): every option, and practice tasks your agent can work through with `serve()`.
<!-- /lang -->
