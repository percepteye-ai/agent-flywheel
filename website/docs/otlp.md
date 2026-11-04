# OpenTelemetry push

If your agent already exports OpenTelemetry traces of its model calls, in the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) and with message content, you can send them to us as well. We learn from the conversations they record exactly as we do from the conversations this SDK sends. Many instrumentations record message content only when you switch it on; without it there is nothing to learn from.

## A second destination

We are one more place your traces go, not a replacement for the place they go now.

- **With an OpenTelemetry Collector**, you add us as one more exporter. Your current destination keeps receiving everything, unchanged.
- **Without a Collector**, setting your agent's own `OTEL_EXPORTER_OTLP_TRACES_*` variables *moves* its traces to us. An exporter has one destination, so they stop going where they went before. To keep both, run a Collector, or add a second exporter in your code.

If your traces already go to an observability provider, we can also read them from there, with nothing to change in your agent: [connect your observability provider](dashboard.md#connect-your-observability-provider).

## Create the agent and its key

[Send us its OpenTelemetry traces](dashboard.md#send-us-its-opentelemetry-traces) in the dashboard. That names the agent, gives it its key, and registers it. Give it [permission](dashboard.md#allow-us-to-learn-from-its-conversations) to learn from its conversations: until then, every push is refused with `403` and nothing in it is read.

The endpoint is per agent, under your control-plane URL:

```
POST $PERCEPTEYE_CONTROL_PLANE_URL/agents/<agent_id>/otlp/v1/traces
```

The dashboard shows it ready to copy. Send the agent's own key as a Bearer token; a key made for another agent is refused.

## With a Collector

Add an exporter for us, and add it to the traces pipeline beside the exporter you already have. Set `PERCEPTEYE_CONTROL_PLANE_URL` and `PERCEPTEYE_API_KEY` in the Collector's environment.

```yaml
exporters:
  otlphttp/percepteye:
    traces_endpoint: ${env:PERCEPTEYE_CONTROL_PLANE_URL}/agents/support-agent/otlp/v1/traces
    headers:
      Authorization: "Bearer ${env:PERCEPTEYE_API_KEY}"

service:
  pipelines:
    traces:
      receivers: [otlp]                                     # as you have them
      processors: [batch]                                   # as you have them
      exporters: [otlp/your-backend, otlphttp/percepteye]   # yours, and ours
```

Keep your own exporter in the list: a pipeline sends only to the exporters it names. The endpoint and the key both belong to one agent, so each agent needs an exporter of its own.

## Without a Collector

Set these in your agent's environment. Its traces then go to us, and no longer to where they went before; metrics and logs are not affected.

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="$PERCEPTEYE_CONTROL_PLANE_URL/agents/support-agent/otlp/v1/traces"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer%20$PERCEPTEYE_API_KEY"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
```

`%20` is the space: OpenTelemetry reads these headers URL-encoded.

## See it arrive

[Your agent's page](dashboard.md#see-what-arrived) counts the spans it has pushed, and how many we keep but do not learn from, and why. Then [start improving](dashboard.md#start-improving).

## What the endpoint answers

- **A request** is an OTLP `ExportTraceServiceRequest`, as `application/x-protobuf` or `application/json`, optionally with `Content-Encoding: gzip`. There is no size limit: each request is written to disk as it arrives and read in bounded memory.
- **`200`** carries an `ExportTraceServiceResponse` in the same content type. Its `partial_success` counts the spans not kept, and says why.
- **`429`** when your organization is over 600 requests a minute or already has a request being read, and **`503`** when the control plane is busy, each with `Retry-After`. The OTLP specification asks every exporter to retry these, and the Collector does.
- **`408`** when a request's body arrives too slowly, and **`507`** when the control plane is short of disk space. Neither is the request's fault, but OTLP exporters do not retry them, so that batch is lost.
- **`400`, `403` and `415`** for a request that will never be taken: a body that is not OTLP; permissions that are off; a content type or `Content-Encoding` it does not read. Sending it again does not help.

## What we keep

- **Personal data is kept, and never learned from.** The message content of every span is scanned for personal data. A request whose spans hold any is kept, marked, and excluded from learning, and `partial_success` says so, with no span rejected.
- **Practice tasks do not come this way.** Spans that carry this SDK's own rollout marker (`percepteye.rollout.id`) are not kept, and are counted as rejected.
- **Send each conversation one way.** Pushed spans and the turns this SDK sends are both kept, so a conversation sent both ways is learned from twice.
- **The agent must belong to exactly one agent or product in your dashboard.** One added in the dashboard does. A push for an agent that two of them declare, or none, is refused with `403`.

## Without the dashboard

The push endpoint never registers an agent. To register one through the API instead, so that its permissions can be switched on:

```bash
curl -X POST "$PERCEPTEYE_CONTROL_PLANE_URL/agents" \
  -H "Authorization: Bearer $PERCEPTEYE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "support-agent", "mode": "production", "discovered_agent": null}'
```

`"discovered_agent": null` records that the agent describes nothing about itself, rather than that you switched that description off.
