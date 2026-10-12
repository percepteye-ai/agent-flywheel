# Push OpenTelemetry spans

If your agent already records its model calls as OpenTelemetry spans — the [GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/), with message content — you can point its OTLP exporter, or your OpenTelemetry Collector, at the control plane instead of installing this SDK. The control plane learns from the conversations those spans record as they arrive, exactly as it does from the turns this SDK uploads, and each improvement of your agent starts from them.

**When to push instead of using the SDK.** Push when the agent is already instrumented with OpenTelemetry, or runs where this SDK cannot be installed, and what you want is for its production traffic to drive the improvement loop. Use the SDK when you also want rollouts — practice tasks written for your agent, which it works through and which are graded — because spans only carry what your agent already did. Send each conversation one way only: pushed spans and uploaded turns are both kept, so the same traffic sent both ways is learned from twice.

**The endpoint** is per agent, under your control plane's URL (the `PERCEPTEYE_CONTROL_PLANE_URL` the dashboard shows when you add the agent):

```
POST $PERCEPTEYE_CONTROL_PLANE_URL/agents/<agent_id>/otlp/v1/traces
```

In the dashboard, add the agent with **Add → An agent → It sends OpenTelemetry**: one click names it, gives it its key and registers it, then its **Permissions** step opens, then its full endpoint is shown, ready to copy (and again on the agent's own page, its **Agent** tab). With the OpenTelemetry SDK's exporter:

```bash
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="$PERCEPTEYE_CONTROL_PLANE_URL/agents/support-agent/otlp/v1/traces"
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Bearer%20$PERCEPTEYE_API_KEY"
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf
```

`%20` is the space: OpenTelemetry reads these headers URL-encoded. If the process exports only traces, `OTEL_EXPORTER_OTLP_ENDPOINT` set to the same URL without `/v1/traces` works too; the exporter appends it. From a Collector, add an exporter to the traces pipeline you already run:

```yaml
exporters:
  otlphttp/percepteye:
    traces_endpoint: ${env:PERCEPTEYE_CONTROL_PLANE_URL}/agents/support-agent/otlp/v1/traces
    headers:
      Authorization: "Bearer ${env:PERCEPTEYE_API_KEY}"
service:
  pipelines:
    traces:
      exporters: [otlphttp/percepteye]
```

**Auth** is the agent's own key (`pefw_…`), sent as a Bearer token: the key you get when you add the agent in the dashboard (**Add → An agent**). A key bound to a different agent is refused.

**Consent comes first, and nothing is read without it.** The agent has to be known to the control plane, and both capture switches have to be on in the **Permissions** step of adding it: its conversations, and their message text. Until both are on, every push is answered `403` and nothing in it is read. The model-content tier is granted only while exactly one active Product declares the agent, as for the SDK. The dashboard's **It sends OpenTelemetry** path registers the agent for you; the push endpoint never registers an agent itself. Through the API instead, an agent that never runs this SDK registers once, so that its Permissions can be switched on:

```bash
curl -X POST "$PERCEPTEYE_CONTROL_PLANE_URL/agents" \
  -H "Authorization: Bearer $PERCEPTEYE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "support-agent", "mode": "production", "discovered_agent": null}'
```

`"discovered_agent": null` records that the agent describes nothing about itself, as a production process does, rather than that you switched that description off.

**What the endpoint takes and answers**, per the OTLP/HTTP specification:

- an `ExportTraceServiceRequest` as `application/x-protobuf` or `application/json`, optionally with `Content-Encoding: gzip`;
- `200` with an `ExportTraceServiceResponse` in the same content type, whose `partial_success` counts the spans not kept and says why;
- `429` when your organization's share of the endpoint is busy or over its rate, and `503` when the control plane is, each with `Retry-After`; the OpenTelemetry exporters and the Collector retry these on their own;
- `400`, `403` and `415` for a request it will never take: retrying it does not help.

There is no size cap. Each request is spooled to disk and read in bounded memory, and one this host's disk cannot hold is answered `507` with a `google.rpc.Status` before it is read. Batches, as the OpenTelemetry SDK's batch processor and the Collector send by default, are learned from soonest.

**Personal data is kept, and never learned from.** The message content of every span is scanned for personal data. A request whose spans hold any is kept, recorded and excluded from training, and the response's `partial_success` message says so (with no span rejected). Spans that carry this SDK's own rollout marker (`percepteye.rollout.id`) are not kept, and are counted as rejected: rollouts reach the control plane through the SDK.

**Seeing it arrive.** The agent's **Agent** tab in the dashboard counts the spans it has pushed, beside the turns the SDK uploaded, and how many of each are kept but not learned from, and why.
