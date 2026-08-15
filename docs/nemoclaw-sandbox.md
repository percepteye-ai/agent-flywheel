# Running the flywheel with OpenClaw / NemoClaw

OpenClaw is a command, not an importable Python object, so this is the
**command entrypoint** — the same path prime-agent uses. No adapter, and no
SDK code: the integration is one config file and a network policy.

NemoClaw runs OpenClaw inside NVIDIA OpenShell. The SDK runs **in the sandbox
too** — it is the only place it can reach the agent process and see its
traffic — and the sandbox is default-deny egress scoped per binary, so the
policy entries below are not optional.

Everything here was verified against `openclaw@2026.3.11` (2026-08-15) with a
clean install and an isolated `HOME`; the observed output is quoted.

## 1. Routing — the config recipe

**The environment alone does not route OpenClaw.** Its chat provider's base
URL comes from its own config, not from `OPENAI_BASE_URL`:

```
grep -rn "process.env.OPENAI_BASE_URL" node_modules/openclaw/dist/   →  0 hits
```

(The only `OPENAI_BASE_URL` occurrences in the bundle are text-to-speech.)

But OpenClaw's config substitutes environment variables **into any string
value** — its own docstring: *"Environment variable substitution for config
values… `${VAR_NAME}` syntax in string values, substituted at config load
time."* So the config can reference exactly the variables the SDK already
sets per rollout, and nothing has to be rewritten between rollouts:

```json
{
  "models": {
    "providers": {
      "percepteye": {
        "baseUrl": "${OPENAI_BASE_URL}",
        "apiKey": "${OPENAI_API_KEY}",
        "models": [{ "id": "policy", "name": "policy" }]
      }
    }
  }
}
```

Write it once to `~/.openclaw/openclaw.json` (`openclaw config file` prints
the active path). `build_child_env` sets both variables for every rollout, so
this resolves per rollout with no file write and no concurrency problem:

```
$ OPENAI_BASE_URL=https://mc.example.com/api/flywheel/v1/llm/v1/rollouts/rl_abc \
  openclaw config get models.providers.percepteye.baseUrl
https://mc.example.com/api/flywheel/v1/llm/v1/rollouts/rl_abc
```

Two details that cost time to find: the nesting is
**`models.providers.<name>`** — a top-level `providers` key is rejected
(`Unrecognized key: "providers"`) — and a provider entry **requires a
`models` array**.

### It fails closed outside a rollout

With the variables absent, the provider becomes unavailable rather than
falling back to `https://api.openai.com/v1`:

```
$ env -u OPENAI_BASE_URL -u OPENAI_API_KEY openclaw config get …baseUrl
missing env var "OPENAI_BASE_URL" at models.providers.percepteye.baseUrl
  — feature using this value will be unavailable
```

That is the property worth having: an OpenClaw started outside a rollout
cannot quietly sample the wrong model.

## 2. The trap: the sandbox PERMITS the wrong destination

The baseline policy already lets OpenClaw reach NVIDIA cloud:

```yaml
nvidia:
  endpoints:
    - host: integrate.api.nvidia.com
      port: 443
  binaries:
    - { path: /usr/local/bin/openclaw }
```

So a rollout that inherits `NVIDIA_API_KEY` **can and will** sample
build.nvidia.com — allowed by policy — while the control plane believes it is
sampling the policy under training. Off-policy data, a meaningless importance
ratio, and nothing raising.

The SDK strips `NVIDIA_API_KEY` / `NVIDIA_NIM_API_KEY` from every rollout's
environment and names them at startup for exactly this reason. **Do not add
them back through `extra=`.** The sandbox permits the wrong destination;
removing the credential is what stops it being used.

## 3. Network policy

Add to `nemoclaw-blueprint/policies/openclaw-sandbox.yaml`, substituting your
control-plane host:

```yaml
network_policies:
  percepteye:
    name: percepteye
    endpoints:
      # Claiming work, reporting results, AND the per-rollout inference
      # gateway — one host, because the rollout id is in the PATH
      # (/api/flywheel/v1/llm/v1/rollouts/{id}). One static entry covers
      # every rollout; no dynamic policy update needed.
      - host: mc.example.com
        port: 443
    binaries:
      # The SDK: claim / report / heartbeat.
      - { path: /usr/bin/python3 }
      # OpenClaw itself, or it is routed correctly and then BLOCKED.
      - { path: /usr/local/bin/openclaw }
```

Both paths are what the image produces: `node:22-slim` + `apt-get install
python3` gives `/usr/bin/python3`, and `npm install -g openclaw` gives
`/usr/local/bin/openclaw`.

## 4. Filesystem

`read_write` is `/sandbox`, `/tmp`, `/dev/null`. The SDK's artifacts
directory defaults to `$CWD/.percepteye-artifacts`, which may be read-only —
point it somewhere writable:

```python
fw.serve("openclaw", agent_id="my-agent", input_shape="text",
         artifacts_dir="/sandbox/.percepteye-artifacts")
```

The per-rollout working directory lives under it, and a CLI agent's output
files ARE its artifacts, so a read-only choice here silently collects
nothing and grades an absence the agent never caused.

Installing the SDK in that image needs `pip3 install --break-system-packages`
(or a venv), the same way the image installs PyYAML.
