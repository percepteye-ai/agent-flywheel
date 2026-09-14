# FAQ

Short answers. Each one links to the page that carries the detail.

## Deployment and security

### Does my agent leave my infrastructure?

**Your code and credentials never do.** `serve()` rebuilds your agent from an importable factory in a child process on the same machine, in your environment, on your credentials; what crosses the boundary is the rollout report and — once, at startup — the agent description. Never your tool source and never your credentials.

    **Your end users' turns cross only if you turn that on.** `attach()` captures production turns to disk always, and uploads them only when an operator has enabled capture for that agent — off by default, precisely because those turns are your users' own messages and the answers you gave them. `Attachment.may_upload` is the live answer. → [Production](production.md)

!!! note "The one thing that does route out"
    During a **training** rollout your agent samples through PerceptEye's per-rollout gateway, because that gateway is what attributes and captures the completion. Completions go out; tool calls stay local and run on your credentials. `attach()` binds its adapters with **no endpoint at all** and cannot re-point a byte of production traffic; `apply_current_policy()` only hands the endpoint back to your own client to use, and touches the process environment only if you ask for it with `set_env=True`. → [Production](production.md)

### Do I need to open a port?

**No — there is no listening socket anywhere in this package.** Every lane is opened from your side, so there is no inbound firewall rule, no public DNS record and no TLS certificate to obtain. Checkable from the repo: `grep -rnE "\.listen\(|HTTPServer|uvicorn|fastapi|flask" src/` exits 1. → [Configuration](configuration.md)

### Do I need a GPU?

**No.** No container to build, no cluster to run, no GPU to provision — the control plane holds those. Your side runs the agent process you already run, plus one HTTP client.

### Does this send my system prompt anywhere?

**Yes, once at startup, unless you turn it off.** Agent discovery sends the *shape* of the constructed agent: system prompt truncated at 20,000 characters (a callable prompt is never invoked, only reported as not-observed), tool names, descriptions and JSON Schemas, provider and model name, and sub-agents to depth 4. Opt out with `introspect=False` or `PERCEPTEYE_INTROSPECT=0` — the cost is generic tasks, since task generation is the thing that reads it. → [Configuration](configuration.md)

## Integration

### Do I have to rewrite my agent?

**Zero lines change inside your agent.** The integration is one import and one call, in a separate serving script; the agent module stays free of any reference to Agent Flywheel. [`tests/test_zero_instrumentation.py`](https://github.com/percepteye-ai/agent-flywheel/blob/main/tests/test_zero_instrumentation.py) is what holds that. → [Quickstart](quickstart.md) · [Python agents](python.md)

### What if my agent isn't Python?

Pass an **argv list** instead of an import path. There is no protocol to implement, so an existing CLI needs no changes at all:

```python
fw.serve(["opencode", "run"], agent_id="coding-agent", input_shape="text")
```

The task arrives on **stdin** and as `PERCEPTEYE_TURN_INPUT`; ordinary **stdout** is final text, while a recognised JSONL stream yields its terminal assistant message. Mark an argv position with `fw.TASK` if your CLI takes the task as an argument. → [CLI and other languages](cli.md)

### Can a CLI report per-tool outcomes?

**Usually yes, and usually for free** — read off the event stream your CLI already prints for its own UI.

| vocabulary | events | you get it if |
|:--|:--|:--|
| `otel-genai` | `gen_ai.tool.start` / `gen_ai.tool.end` | your agent already logs OpenTelemetry GenAI events as JSON lines |
| `prime-agent` | `tool_execution_start` / `tool_execution_end` | you run prime-agent under `--mode json` |

The first row belongs to no vendor, so an agent instrumented for observability needs no work. Failing both, append a line per call to `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`. A stream carrying no tool events leaves `tool_calls` as `None` — never `[]`. → [CLI and other languages](cli.md)

### Is it really one dependency?

**Yes: `httpx>=0.27`, and nothing else.** `tests/test_oss_boundary.py::test_exactly_one_runtime_dependency` and the `clean-install` CI job both fail if the list grows — the test reads the list out of `pyproject.toml`, the CI job reads it back out of the installed metadata. The Node plugin has **zero** — Node builtins only. `otel`, `dev` and `adapters-test` are extras, never runtime dependencies. → [Node harnesses](node.md)

## Running it

### How do I stop contributing capacity?

One environment variable, so the same deployed image does either without a code edit:

```bash
PERCEPTEYE_AGENT_MODE=production    # serve() returns 0 immediately. No claim, no connection.
PERCEPTEYE_AGENT_MODE=training      # contribute capacity (the default when unset)
```

In `production` the call returns **before opening any connection**, so that deployment needs no control-plane URL and no key. It returns rather than raising, and prints one line naming the mode it took. `prod` and `serving` also mean don't; `train`, `rollout`, `rollouts`, `fine-tuning`, `finetuning` and `improving` also mean contribute. A value it cannot parse raises instead of guessing.

!!! warning "Read at startup, and it gates `serve()` only"
    Flipping the variable does not stop a loop already in flight — `max_rollouts=N` is the only clean exit, and nothing in the package catches SIGINT or SIGTERM. It does not gate `attach()`, which *is* the production surface; to stop that, don't call it, or pass `register=False`. → [Configuration](configuration.md)

### Can I run more than one rollout at a time?

**Yes — `concurrency=N`, under the default `isolation="subprocess"`.** Each rollout gets its own child process, its own environment and its own single-use credential, and the runner never claims more than it still intends to run. `isolation="inprocess"` is capped at 1 and raises `ConfigurationError` above it, because concurrent in-process rollouts share one set of environment variables and would silently cross-attribute each other's completions.

### What happens when a rollout crashes or a credential expires?

It comes back as a **typed exclusion**, so a substrate fault is never scored as a bad decision by your agent.

| what happened | `exclusion_reason` |
|:--|:--|
| your Python entrypoint raised instead of returning | `entrypoint_error` |
| a command agent ran and exited non-zero | `agent_exit_nonzero` |
| the rollout blew `rollout_timeout_s` (default 900s) and was killed | `deadline_exceeded` |
| the lease lapsed while the rollout was running | `credential_expired` |
| the SDK's own child died, or returned an unparseable result | `flywheel_crash` |

The first two are split deliberately: one is retryable and untrainable, the other is a real attempt whose failure is exactly what training must see. **Every rollout is reported, carrying its exclusion block** — even one that produced nothing, because whether a typed reason means retry, exclude or train is the control plane's call and it needs the reason to make it with. → [The contract](contract.md)

??? note "Every reason in the enum"
    `flywheel_crash` · `substrate_unavailable` · `credential_expired` · `deadline_exceeded` · `entrypoint_error` · `agent_exit_nonzero` · `cancelled` · `capacity`. The list is declared in [`schema/flywheel-1.json`](https://github.com/percepteye-ai/agent-flywheel/blob/main/schema/flywheel-1.json) and mirrored in `contract.py`; the suite asserts the two agree.

### What is `substrate_unavailable`?

The exclusion for a rollout where **every** tool call was an observed failure — the agent could not act on its environment, so this is a broken environment rather than a low score. It is strict on purpose:

- every call must be `failed`. One `unknown` or one `ok` disqualifies it, because `unknown` is a silence, not a failure.
- at least one call must have been made. A rollout that called no tool has told you nothing about the substrate.
- partial failure is left alone — some calls failing is what agents ordinarily do, and is exactly what you want to train on.
- it never overwrites an exclusion already set; a crash or a dead credential knows more.

→ [Outcomes](outcomes.md)

### What does `unknown` do to my score?

**Nothing — it abstains.** Verifiers that need a per-call outcome skip the call rather than guess, and `unknown` is **never rounded up to `ok`**. Any status is discarded with it, from the `status_code` field *and* from the result body, so a server-side scorer cannot recover a status by parsing the envelope and mistake an unobserved call for a landed one. → [Outcomes](outcomes.md)

## Everything else

### How is this different from an eval platform?

**An eval platform measures. This closes the loop.** The output is not a score sheet: training produces a model and a co-optimized system prompt, and your production agent picks them up as an endpoint through `apply_current_policy()` or `policy_source()`. Two other differences follow from running *beside* your agent rather than in place of it — you never author the task set, because tasks are generated from the tool surface discovery read from inside your process, and outcomes are observed at the call site rather than parsed back out of a transcript. → [How the loop closes](how-it-works.md)

### What license is this under?

**MIT.** Please report vulnerabilities privately to **security@percepteye.ai** rather than in a public issue.
