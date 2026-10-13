# Production

Your production agent records the conversations it serves, sends them to us once you allow it, and runs the version you approved. This page covers what is sent and when, what happens when a request fails, and what we do with personal data. Setting it up is the [Quickstart](quickstart.md).

## Recording conversations

<!-- lang -->
`attach()` records each turn your agent serves: what the user asked (`input_text=`), every tool call and its outcome, and the answer (`record_answer()`). For a LangChain, LangGraph or Google ADK agent it also records each model call inside the turn: the system prompt, the whole conversation so far, the tool definitions and every tool result. That is what we learn from, and the turn sends it as one [OpenTelemetry GenAI](https://opentelemetry.io/docs/specs/semconv/gen-ai/) trace. An agent on another framework records tool outcomes and the turn's text, but no model calls; to learn from it, [push its OpenTelemetry traces](otlp.md).

The adapters only observe. No model call, response or tool result is changed, and nothing in `turn()` can fail the turn: if capture breaks, the turn still runs.
<!-- /lang -->

Two [permissions](dashboard.md#allow-us-to-learn-from-its-conversations) decide what leaves your machine: the conversations, and their message text. Without the second, what the model saw is not recorded at all, not even on your disk. Learning needs both.

## When turns are sent

<!-- lang -->
Every 32 turns, a short-lived background thread sends the turns that have an answer. When the process exits, the rest go too, including turns that never recorded one. `px.flush()` sends now. A request carries at most 64 turns, and the network is never on your request path.
<!-- /lang -->

A turn that could not be sent stays on disk and goes with a later batch. Each batch carries an idempotency key, so a batch that arrives twice is stored once.

## Retries and rate limits

Every request to the control plane gets up to four attempts, with a 30-second timeout each.

- **`429`** is always retried, after the server's `Retry-After` (at most 8 seconds), or after a short random backoff when there is none.
- **`408`, `500`, `502`, `503` and `504`, and a connection that drops mid-request,** are retried, except for a claim for practice tasks: that is resent only when it never reached us, because a claim that did may already have handed work out.
- **Any other `4xx`** is final; sending it again would get the same answer. A `403` on upload is a refusal, such as a permission switched off or a key that may not carry this agent's conversations: the SDK stops uploading, says why, keeps the turns on disk, and starts again once the permission is back.

The control plane's limits are per organization, per minute: 240 turn uploads (each up to 64 turns), 240 practice-task claims, 600 practice-task results, and 600 [OpenTelemetry pushes](otlp.md). Over a limit, the answer is `429` with `Retry-After`, and the SDK waits as above.

## Personal data

The SDK sends what your agent saw and said as it was; it masks nothing on your side. The control plane scans everything it receives for email addresses, phone numbers, card numbers, US social security numbers and IP addresses before it stores it:

- **In the tool-call records** sent beside each turn, a match is masked, as `[redacted:email]` for example.
- **In the user's message, your agent's answer, and everything the model saw**, a match is kept as sent, because masking would change the conversation. The turn is marked as holding personal data and is never learned from. [Your agent's page](dashboard.md#see-what-arrived) counts these turns.

[Pushed traces](otlp.md) follow the same rule. How long we keep it all is in the [FAQ](faq.md#what-data-do-you-keep-and-for-how-long).

## What stays on your machine

<!-- lang -->
Every turn is a directory under the `trajectory_root` you named, and uploading does not delete it. Nothing does unless you ask: `attach(retention_turns=50_000)` keeps the most recent 50,000 turns, and `px.prune(keep_last=...)` or `px.prune(older_than_s=...)` prunes when you choose. Left alone, it grows by roughly 2.4 GB per million turns, more if your tool outputs are large.

Records are flushed, not fsynced: they survive your process dying, and a machine that loses power can lose the most recent ones.
<!-- /lang -->

## Run the approved version

<!-- lang -->
The SDK hands you the approved model and prompt; applying them is your code's decision.

| call | when | what it returns |
|:--|:--|:--|
| `fw.apply_current_policy(agent_id=...)` | once, at startup | `{"base_url", "api_key", "model"}` for your client, or `{}` |
| `fw.policy_source(agent_id=...)` | `.current()` on each turn | a snapshot; `.changed` is true when you must rebuild your client |
| `fw.prompt_decision(fw.current_prompt(agent_id=...))` | when you load your prompt | `.apply`, `.text`, and `.reason` when it may not be applied |

`policy_source()` refreshes once its answer is about five minutes old, or sooner when the control plane asks, but never more often than every 30 seconds. A refresh is one attempt with a one-second budget, made inside your `current()` call; no thread polls in the background. When a refresh fails, it keeps serving the last answer, so a network blip never moves you back to your old model. Watch `snap.stale_for_s`: it counts the seconds past the refresh interval while refreshes fail.

`prompt_decision()` refuses when there is no approved text, when your own baseline prompt changed after the pair was approved, when the status is not one a person approved, or when the prompt was approved against a different checkpoint. `.text` is `None` on a refusal, never `""`, so applying it cannot blank your prompt.
<!-- /lang -->

Until you approve a new version in [Decisions](dashboard.md#approve-the-winner-in-decisions), your agent keeps the one it has, and you can [try a new version on one agent](dashboard.md#try-a-new-version-on-one-agent) first. An empty answer always means "keep what you run", never "run with nothing".

## When the control plane is unreachable

Your agent keeps serving. Recording carries on to disk, turns wait for a later batch, and your agent keeps the model and prompt it has. Nothing on this page raises into your process because the control plane did not answer.

## Reference

<!-- lang -->
Every parameter of `attach()`, `policy_source()` and `prompt_decision()` is in [Python agents](python.md#production-attach), and every environment variable in [Configuration](configuration.md).
<!-- /lang -->
