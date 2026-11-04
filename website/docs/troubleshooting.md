# Troubleshooting

Find what you see; each answer names the cause and the fix. When the control plane refuses something it says why, and the SDK passes its words on:

<!-- lang -->
read `px.upload_refusal` and `px.model_content_refusal` on the object `attach()` returned, and let your logging show the `agent_flywheel` logger and Python warnings.
<!-- /lang -->

## The dashboard never sees your agent connect

- **The key is not in your agent's environment, or the URL points somewhere else.**
  <!-- lang -->
  `serve()` refuses to start without the key: `an API key is required (or set PERCEPTEYE_API_KEY)`. `attach()` keeps your agent running and records on disk; `px.upload_refusal` names the missing variable. Unset, the URL is the production control plane; with a key set, `agent-flywheel status` prints the URL it uses.
  <!-- /lang -->
- **Your network blocks it.** Every connection is outbound HTTPS to your control-plane URL; allow it.
- **The key was revoked, or is more than 180 days old.** [Replace it](dashboard.md#replace-a-lost-key).

## Nothing arrives in the dashboard

- **Nothing has been sent yet.** Turns go in batches.
  <!-- lang -->
  Every 32 turns, and at exit; `px.flush()` sends now.
  <!-- /lang -->
- **`production capture is not enabled for this agent`** — give the agent [permission](dashboard.md#allow-us-to-learn-from-its-conversations). No restart is needed.
- **`this key is bound to a different flywheel`** — the agent id your code reports and the name the key was made for differ. Use the same id in both, or [get a key](dashboard.md#get-your-key) for this agent.
  <!-- lang -->
  The id is `attach(agent_id=...)`.
  <!-- /lang -->
- **`this agent is not registered`** — the control plane has never heard from this agent, so there is nothing to switch on. Start it once with its key.
  <!-- lang -->
  `attach()` registers it unless you passed `register=False`.
  <!-- /lang -->
- **`this flywheel key does not carry the 'turn.report' scope`** — the key was made before conversations could be uploaded, or without that permission. [Replace it](dashboard.md#replace-a-lost-key).

## Conversations arrive, but there is nothing to learn from

- **No message text was sent.** Learning needs the second [permission](dashboard.md#allow-us-to-learn-from-its-conversations), and the model calls themselves.
  <!-- lang -->
  The SDK records model calls for LangChain, LangGraph and Google ADK agents only; for any other agent, [push its OpenTelemetry traces](otlp.md). `px.may_capture_model_content` is the live answer, and `px.model_content_refusal` says why it is `False`.
  <!-- /lang -->
- **`this agent's production turns are attributed to no Product`** — the agent must belong to exactly one agent or product in your dashboard. Add it there once; if two declare it, remove one.
- **Your agent's page says they are kept but not learned from:**
  - *they contain personal data*: see [Personal data](production.md#personal-data).
  - *they hold no final answer*:
    <!-- lang -->
    call `px.record_answer(turn_id, text)` for each turn.
    <!-- /lang -->
- **Start improving says `flywheel_capture_has_no_decision_points`.** What has arrived holds no decision to learn from yet. Let more conversations arrive.

## Pushed traces are refused

- **`403 production_capture_not_enabled`** or **`403 model_content_capture_not_enabled`** — give the agent [permission](dashboard.md#allow-us-to-learn-from-its-conversations); both of its parts must be on. The rest of the message says which is off.
- **`415`** — send `application/x-protobuf` or `application/json`, and gzip or no compression.
- **`400`** — the body is not an OTLP trace export. A trace file belongs in an upload, not a push.
- **Pushes succeed but the agent's page counts nothing to learn from** — the spans carry no message content. Switch it on in your instrumentation.

Every status is in [OpenTelemetry push](otlp.md#what-the-endpoint-answers).

## Requests fail with `429`

Your organization is over a [rate limit](production.md#retries-and-rate-limits), or is already sending as much at once as it may. The SDK waits and retries on its own. If it keeps happening, send fewer, larger requests: let turns batch, and give a Collector's exporter a batch processor. A Collector that sends several requests at once gets `429` for each one past the first while it is read, and retries them a second later; `sending_queue: num_consumers: 1` on our exporter avoids that.

## Your agent is not running the approved version

- **It has not been approved yet, or is not ready to serve yet.** Approving happens in [Decisions](dashboard.md#approve-the-winner-in-decisions). Until a version is approved and serving, the answer is "keep what you run".
<!-- lang -->
- **`apply_current_policy()` returned `{}`.** It returns `{}` for every reason it cannot act, and logs which on the `agent_flywheel` logger: nothing approved, no key, an unreachable control plane, or an address that is a stored file (`gs://`, `s3://`, `file://`) rather than a running endpoint.
- **`policy_source()` still serves the old model.** It refreshes about every five minutes. A `snap.stale_for_s` above zero means refreshes are failing and it is serving the last answer it had.
- **The prompt did not change.** The SDK never applies a prompt for you. `fw.prompt_decision(...).reason` says why a prompt may not be applied.
<!-- /lang -->

## The SDK will not start

<!-- lang -->
`serve()` checks everything before its first practice task and raises `ConfigurationError` naming the cause: a mode it cannot read (set `PERCEPTEYE_AGENT_MODE` to `training` or `production`), a missing key, an entrypoint that does not import, or a control plane that does not answer. `attach()` never refuses to start: it captures on disk and says what it could not do.
<!-- /lang -->
