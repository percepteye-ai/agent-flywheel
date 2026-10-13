# FAQ

Short answers. Each links to the page with the detail.

## Your data

### What do you receive?

Only what you switch on.

- **With the first [permission](dashboard.md#allow-us-to-learn-from-its-conversations):** each turn's steps: the user's message, the tool calls and their outcomes, and your agent's answer.
- **With the second:** what the model saw and said: the system prompt, the conversation so far, the tool definitions and every tool result.
<!-- lang -->
- **If your agent works through practice tasks** with `serve()`: each task's result, with its tool calls and their outcomes. While it works on one, its model calls go through our gateway, because that is how the task's model calls are recorded; its tool calls run on your side, on your credentials.
- **A description of your agent,** once at startup, when it runs practice tasks with `serve()`: its system prompt (cut at 20,000 characters), its tools' names, descriptions and schemas, its provider and model name, and its sub-agents. `attach()` sends one only when you pass `describe=` or `execution_entrypoint=`. `PERCEPTEYE_INTROSPECT=0` turns it off.
<!-- /lang -->

Never your tools' source code, and never your credentials.

### What data do you keep, and for how long?

What your agent sends us: its turns, its traces, and the results of practice tasks. Nothing is deleted by default.

- After your organization's retention window, 90 days unless an admin changes it, older content is also copied to an archive.
- An admin can ask us to delete content older than the window: [Choose what we keep](dashboard.md#choose-what-we-keep). The record that the work happened stays; its content is removed. The archive copy is kept. Scheduled deletion is switched on one organization at a time, and the setting says whether it runs for yours yet.
- Each improvement round keeps the conversations it learned from and was tested on, with the round.

<!-- lang -->
Your own copy, under the `trajectory_root` you named, is yours, and nothing deletes it unless you ask. → [Production](production.md#what-stays-on-your-machine)
<!-- /lang -->

### What do you do with personal data?

We scan everything we receive for email addresses, phone numbers, card numbers, US social security numbers and IP addresses. In tool-call records a match is masked. In messages, answers and what the model saw, it is kept as sent, because masking would change the conversation, and that turn is never learned from. The SDK masks nothing on your side. → [Production](production.md#personal-data)

### Does my agent's code leave my machine?

No. Your agent runs where it runs today, on your credentials. What crosses is what the answers above list, and every connection is opened from your side.

### Do I need to open a port, run a container, or provide a GPU?

No. There is no listening socket anywhere in this SDK, so there is no inbound firewall rule, no public DNS record and no certificate. The improvement rounds run on our side.

## Getting better

### How does my agent get better?

We learn from its real conversations. Each new version is tested against your agent's current best on conversations it has never seen, and a version that does better comes to you in [Decisions](dashboard.md#approve-the-winner-in-decisions). The next round starts from the best so far. → [Quickstart](quickstart.md#start-improving)

### Who decides what my users get?

You do. A new version reaches your agent only after a person approves it in Decisions, with an explicit approve or reject. A reply in a chat or a document ("sure, go ahead") is never read as an approval. Until then, your agent keeps the version it has.

### How is this different from an eval platform?

An eval platform measures. This also improves: each round produces a new version of your agent, tested against the current best, for you to approve. Your agent picks it up as an endpoint it already knows how to call.

## Using it

### Do I have to change my agent?

<!-- lang -->
No. The SDK records from the outside: LangChain, LangGraph and Google ADK calls are observed by adapters, with no Agent Flywheel code inside your agent. Your serving code wraps each turn in `px.turn()`. → [Python agents](python.md)
<!-- /lang -->

### What if my agent is built some other way?

<!-- lang -->
If it already exports OpenTelemetry traces with message content, [push them](otlp.md); nothing to install. A command-line agent in any language can run practice tasks through `serve()` with an argv list. → [CLI and other languages](cli.md)
<!-- /lang -->

### What does it install?

<!-- lang -->
One runtime dependency, `httpx`. Adapters import their framework only when they bind, and `pip install "agent-flywheel[otel]"` adds the OpenTelemetry API, never the SDK.
<!-- /lang -->

### How do I stop it?

<!-- lang -->
Stop calling `attach()`, or pass `register=False` to keep capture on your disk only. Switching the [permission](dashboard.md#allow-us-to-learn-from-its-conversations) off stops uploads without touching your code. `PERCEPTEYE_AGENT_MODE=production` makes `serve()` return at once, without opening a connection. → [Configuration](configuration.md)
<!-- /lang -->

### What does `unknown` mean?

A tool call nobody observed succeeding or failing. It is never rounded up to `ok`: a failure counted as a success would teach your agent that the failure was fine. When we learn, an `unknown` call counts neither way.

### Python or Node?

Pick the one that matches where your agent runs: the Python SDK for a Python agent or a command-line agent you launch yourself, the Node plugin for an OpenClaw or DeepSeek Harness agent. Never both for one agent.

### What license is this under?

MIT. Please report vulnerabilities privately to **security@percepteye.ai**.
