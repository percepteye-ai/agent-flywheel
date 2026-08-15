# Running the flywheel inside a NemoClaw sandbox

NemoClaw runs OpenClaw inside NVIDIA OpenShell. The SDK runs **in the
sandbox too** — it has to, because that is the only place it can talk to the
agent process and see its traffic.

The sandbox is **default-deny egress, scoped per binary**, so nothing below
is optional: without the policy entries the SDK cannot claim work and the
agent cannot reach the model it is supposed to be sampling.

## The one that will bite you

The baseline policy already permits OpenClaw to reach NVIDIA cloud:

```yaml
nvidia:
  endpoints:
    - host: integrate.api.nvidia.com
      port: 443
  binaries:
    - { path: /usr/local/bin/openclaw }
```

So a rollout that inherits `NVIDIA_API_KEY` **can and will** sample
build.nvidia.com — permitted by policy — while the control plane believes it
is sampling the policy under training. Off-policy data, a meaningless
importance ratio, and nothing raising.

The SDK strips `NVIDIA_API_KEY` / `NVIDIA_NIM_API_KEY` from every rollout's
environment and reports them at startup for exactly this reason. **Do not
add them back through `extra=`.** The sandbox permits the wrong destination;
removing the credential is what stops it being used.

## Policy fragment

Add to `nemoclaw-blueprint/policies/openclaw-sandbox.yaml`, substituting your
control-plane host:

```yaml
network_policies:
  percepteye:
    name: percepteye
    endpoints:
      # Claiming work, reporting results, AND the per-rollout inference
      # gateway — one host, the rollout id is in the PATH
      # (/api/flywheel/v1/llm/v1/rollouts/{id}), so a single entry covers
      # every rollout and no dynamic policy update is needed.
      - host: mc.example.com
        port: 443
    binaries:
      # The SDK: claim / report / heartbeat.
      - { path: /usr/bin/python3 }
      # OpenClaw itself: it must reach the gateway to sample the policy.
      # Without this line the agent is routed correctly and then BLOCKED,
      # which at least fails loudly.
      - { path: /usr/local/bin/openclaw }
```

## Filesystem

`read_write` is `/sandbox`, `/tmp`, `/dev/null`. The SDK's artifacts
directory defaults to `$CWD/.percepteye-artifacts`, which may be read-only —
point it somewhere writable:

```python
fw.serve("openclaw", agent_id="my-agent", input_shape="text",
         artifacts_dir="/sandbox/.percepteye-artifacts")
```

The per-rollout working directory lives under it, and a CLI agent's output
files ARE its artifacts, so a read-only choice here silently collects
nothing.

## Entrypoint

OpenClaw is a command, not an importable Python object, so this is the
command path — the same one prime-agent uses. No adapter is required to
*execute* it; the adapters that exist are for frameworks whose own
configuration beats the environment.
