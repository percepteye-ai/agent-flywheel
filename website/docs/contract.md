# The contract

The wire between an agent you run and a control plane you don't. It is one JSON Schema file and a small number of rules about absence.

## One file is the authority

[`schema/flywheel-1.json`](https://github.com/percepteye-ai/agent-flywheel/blob/main/schema/flywheel-1.json) is the contract, and this repository owns it.

| | |
|:--|:--|
| `$id` | `https://schemas.percepteye.ai/flywheel/1.json` |
| `x-contract-version` | `flywheel/1` |
| Python constant | `contract.CONTRACT_VERSION == "flywheel/1"` |
| Types declared | `ToolCallRecord` · `ProductionTurn` · `ProductionTurnBatch` · `Exclusion` · `RolloutRequest` · `RolloutOutput` · `ClaimResponse` |

**Server implementations pin against the published schema; the schema is never generated from a server type.** The direction matters. Generate the schema from a server and the contract becomes whatever that server currently does — a field renamed in a handler silently redefines the wire for every customer-run agent already deployed against it. Pinning the other way means a server change that breaks the contract fails on the server.

`RolloutOutput` and `ToolCallRecord` are marked **FLYWHEEL-AUTHORED**: an agent asserts them. `RolloutRequest` is **PERCEPTEYE-AUTHORED**: the control plane asserts it.

`ProductionTurnBatch` is the closed body sent to `POST /turns`. Its optional
`task_id` is an opaque customer correlation candidate. It is evidence only:
the field does not claim that a turn succeeded, identify the sampled policy,
carry generated advice, or authorize a training row. Those decisions remain
control-plane responsibilities. Every production identifier is an exact
1–255-character string without edge whitespace or C0/C1 controls. `turn_id`
and `agent_id` additionally forbid path separators and the `.` / `..` path
segments because they cross storage or URL-path boundaries. A batch contains
at most 64 turns; the SDK splits a larger requested flush into conforming
chunks.

## The Python types are hand-written to match

[`contract.py`](https://github.com/percepteye-ai/agent-flywheel/blob/main/src/agent_flywheel/contract.py) is not generated from the schema. It is written by hand, because the schema cannot express the rules that matter most — that `status_code` has no default, that an `unknown` outcome discards a status from the *body* as well as the field, that a weak `entity_ids` key is refused by name. Those are constructor behavior, not shape.

Hand-written means drift is possible, so the suite asserts agreement. Three tests in `tests/test_contract.py`:

| test | what it compares | direction |
|:--|:--|:--|
| `test_output_emits_only_declared_contract_keys` | keys `RolloutOutput.to_wire()` can emit vs. schema `properties` | code ⊆ schema |
| `test_every_exclusion_reason_the_code_knows_is_declared_by_the_schema` | `ExclusionReason` vs. the `reason` enum | exact list equality, **including order** |
| `test_rollout_request_and_the_schema_declare_the_same_keys` | dataclass fields vs. schema `properties` | both directions |

**Both directions, because checking one is how the drift survived.** `lease_expires_ts` was parsed by the client and used to pace heartbeats while the published schema declared `additionalProperties: false` without it — so a conforming server could not send the field the client needed, and a server that sent it violated the schema this repo calls authoritative. `agent_exit_nonzero` went the other way: added to the Python enum, emitted by the runner, asserted by four test modules, while the published enum still listed seven reasons.

!!! note "Order is part of the assertion"
    The exclusion test asserts list equality, not set equality. An extra reason in the schema is vocabulary the control plane owns, but the order carries the same editorial meaning in both files, so a silent divergence in either direction is worth catching.

## What an agent may assert, and what it cannot

`RolloutOutput` is a closed dataclass and the schema sets `additionalProperties: false`. The permitted vocabulary is stated positively — eleven keys, and that is the complete set.

| field | type | default | on the wire |
|:--|:--|:--|:--|
| `final_text` | `str` | required | always |
| `tool_calls` | `list[ToolCallOutcome] \| None` | `None` | always, **including `null`** |
| `tool_calls_omitted_count` | `int` | `0` | always; non-zero forbids a training grade |
| `artifacts` | `dict[str, str \| dict]` | `{}` | always |
| `events` | `list[dict]` | `[]` | always |
| `timing` | `dict` | `{}` | always |
| `llm_call_count` | `int \| None` | `None` | only when set; pinned/on-policy use requires an exact nonnegative count matching the gateway |
| `reward` | `float \| None` | `None` | only when set; **advisory only** |
| `success` | `bool \| None` | `None` | only when set |
| `agent_fingerprint` | `dict \| None` | `None` | only when set |
| `exclusion_reason` + `exclusion_detail` | see below | `None` | emitted as one nested `exclusion` object |

The last row is the one place the Python shape and the wire shape differ on purpose: two flat fields in the dataclass, one nested block on the wire.

**Anything derived from the inference wire is absent by construction, not by filtering.** The SDK is semi-trusted. An agent cannot assert a field it has no way to know, so the type simply has nowhere to put it:

| PerceptEye-authored | why an agent cannot know it |
|:--|:--|
| the grade | computed by the compiled suite after the report; it comes back on the report ack |
| the gateway's completion count | counted at the gateway. `llm_call_count` is what the agent *believes* it issued, and a mismatch means traffic escaped the wire |
| rendered-prompt sha, tool-schema digest, sampling params, tokenizer | observed at the gateway. For OPSD, `agent_fingerprint.execution_sha256` is SDK-authored from canonical registration/discovery facts plus opaque adapter-attested components; unreadable or lossy discovery yields no affirmative digest |

Gateway policy identity and the SDK execution identity are recorded separately and never conflated. The latter is still client-side attestation, not an authenticated host identity. `to_wire()` also raises `ContractViolation` if it would ever emit a key outside `WIRE_KEYS` — a guard against a future edit, not a runtime filter.

!!! warning "A client-side check binds only honest clients"
    Every rule on this page is enforced in the SDK *and* independently at intake. The dataclass makes the dangerous shape hard to build by accident; it is the server that makes it impossible to submit.

## Absence is never collapsed into zero

The single rule the rest of the wire follows: **not reported and reported-as-nothing are different, and the wire keeps them different.**

| `tool_calls` | means |
|:--|:--|
| `null` | I did not report tool calls. The default. |
| `[]` | I observed zero tool calls. **A positive assertion.** |
| `[...]` | I observed exactly these. **Never a partial list.** |

The default is `None` so an agent that reports nothing never accidentally asserts zero. And a list is all-or-nothing: if any executed call cannot be classified, emit it with `outcome="unknown"` or set the whole rollout to `null` — a truncated list is indistinguishable from a short one and corrupts every set-reasoning verifier.

```python
RolloutOutput(final_text="hi").to_wire()["tool_calls"]                  # None
RolloutOutput(final_text="hi", tool_calls=[]).to_wire()["tool_calls"]   # []
```

The same rule runs through `ToolCallRecord`:

| field | `null` means | it does **not** mean |
|:--|:--|:--|
| `status_code` | no status was observed | the call had no status |
| `agent_name` | unattributed | one agent made everything |
| `tool_call_id` | the framework exposed no id | the call did not happen — it counts, it just cannot be placed more precisely than "somewhere in this rollout" |
| `entity_ids` absent or `{}` | no identity evidence — an abstention | the write failed |

`status_code` is a **required key with a nullable value**, and it is forced to `null` whenever `outcome` is `unknown`. That pair — `unknown` beside a 200 — is the exact shape that scores a failed call as a landed mutation, so the type refuses to carry it and scrubs machine-readable status out of the result body too. The tri-state itself is on [outcomes](outcomes.md); the wire's job is only to keep an unobserved call unrecoverable as a successful one.

## RolloutRequest

One unit of work, handed to the agent by the control plane.

| field | required | notes |
|:--|:--:|:--|
| `rollout_id` | ✓ | minted by the control plane **before** the agent runs anything. The attribution unit, never a per-process session id |
| `turn_input` | ✓ | the task |
| `llm_url` | ✓ | per-rollout, unguessable, single-use, expiring. A shared or process-global URL is not an acceptable fallback — it destroys per-rollout attribution and collides across concurrent rollouts |
| `api_key` | ✓ | single-use, for that gateway |
| `artifacts_dir` | ✓ | where the rollout's files go |
| `deadline_ts` | ✓ | authoritative. The timeout is computed from this, never from a local env var |
| `lease_expires_ts` | | absolute epoch seconds at which the **granted** lease expires |
| `lease_token` | | |
| `metadata` | | free-form; carries `completion_gate` for a [CLI agent](cli.md) |

**`deadline_ts` and `lease_expires_ts` are separate numbers for separate jobs.** The deadline bounds the work and does not move. The lease is liveness, and a heartbeat renews it. `seconds_remaining(now)` reads the deadline only, and moves for exactly one reason: time passing.

The lease expiry is read **once, at claim**, and paces heartbeats at a third of the remaining grant for the life of the rollout — measured from what the server *granted*, not what the client requested. A server that clamps the request and does not report the clamp is exactly how a heartbeat ends up slower than the lease it exists to keep alive; when the field is absent the client falls back to the lease it asked for, which is wrong in precisely that case.

??? note "There is deliberately no `environment` key"
    A previous SDK wrote server-supplied values into the running process's `os.environ`. It was both dangerous and useless — the LLM client had already read its `base_url` at construction. The only configuration channel into an agent is subprocess environment at spawn: configuration crosses into a fresh child, never into a running parent.

    `RolloutRequest.from_wire()` raises `ContractViolation` if the key reappears, so its return is an error rather than a surprise.

## Exclusions: eight reasons, six emitted

An exclusion is the typed **"do not train on this"** signal. It is not a low reward — a substrate fault is not a policy failure, and collapsing the two teaches a policy from a broken environment.

| reason | emitted by this SDK | when |
|:--|:--:|:--|
| `flywheel_crash` | ✓ | **our** fault. An isolation child exited non-zero, returned an unparseable result, or received no payload; also the reason sent to `abandon` when `run_one` itself raises |
| `substrate_unavailable` | ✓ | every observed tool call failed — see below |
| `credential_expired` | ✓ | the lease lapsed mid-rollout and the heartbeat thread killed the child |
| `deadline_exceeded` | ✓ | the entrypoint, child or command blew its budget — or the deadline had already passed at claim |
| `entrypoint_error` | ✓ | the Python entrypoint raised. Its class is "nothing usable ran", so the work is worth retrying elsewhere |
| `agent_exit_nonzero` | ✓ | a command entrypoint **ran** and exited non-zero |
| `cancelled` | — | **declared vocabulary only.** The control plane's word for work it withdrew |
| `capacity` | — | **declared vocabulary only.** The control plane's word for work it could not place |

The last two are in the schema and in `ExclusionReason` because a conforming server may send them and a reader must accept them. Nothing in this package emits either one.

**`entrypoint_error` and `agent_exit_nonzero` are separate because they want opposite handling.** One is retryable and untrainable; the other is a real attempt whose failure is exactly what training must see. `Popen` raises when a binary cannot be started at all, so a non-zero exit means the agent executed and chose it — a crash after doing work, or a completion gate it could not satisfy. Sharing one label sent a whole afternoon's triage at the process launcher while the detail string read "quality gate still failing".

```python
run_rollout_as_command(["sh", "-c", "echo partial; exit 3"], ...)
# exclusion_reason  -> "agent_exit_nonzero"
# exclusion_detail  -> "exit 3: ..."
# final_text        -> "partial"      ← the work survives the failure
```

## Evidence is collected before the report, on every path

There is exactly one artifact-collection call and exactly one report call in the serve loop, in that order:

```text
run_one()  →  _collect_artifacts()  →  _mark_substrate_unavailable()  →  transport.report()
```

**Collection is not conditional on a clean exit.** An agent that exhausted its turn budget or its clock still wrote files, and for a coding or REPL agent those files *are* what the reward reads. Collecting only on the tidy path meant a rollout could write a correct solution and have it discarded — measured on run `516b7a55`, where a correct `top_ips.sh` sat in the trajectory while the row carried `artifacts={}`.

**Every rollout is reported, whatever happened to it**, and the report carries its exclusion block. The SDK used to abandon when an exclusion carried no work product. That is a decision about *trainability*, and it is not the SDK's to make: abandon suppresses the record, so the control plane could not see the rollouts that went worst and never received the reason it would have judged them by. Whether a typed reason means retry, exclude or train is the control plane's call, and it now has the reason to make it with.

The same reasoning applies one level down. Artifact **registration** is a separate network call to the same server that may have just expired the rollout, so the bytes are written into the report *before* registration is attempted — losing an index entry must not lose the deliverable.

## `substrate_unavailable` is deliberately strict

A rollout whose every tool call errored is not a low-scoring attempt. It is an attempt that never got to happen. On 2026-08-18 an agent overwrote IPython's own `interactiveshell.py`, every `ipython` call died at import, and a full day of runs reported 0% grounded while reading as "the model cannot use tools."

Three conditions, all required:

| rule | why |
|:--|:--|
| Every call must be an **observed failure** | `unknown` is not failure. It is the honest answer when the framework exposed no verdict, and an agent that reports none must behave exactly as it did before |
| **At least one** call must have been made | a rollout that never called a tool has told us nothing about the substrate |
| Partial failure is left alone | some calls failing is ordinary agent behavior, and is exactly what we want to train on |

| `tool_calls` | result |
|:--|:--|
| `[failed, failed]` | `substrate_unavailable` |
| `[failed]` | `substrate_unavailable` |
| `[failed, ok]` | *(none)* |
| `[failed, unknown]` | *(none)* |
| `[unknown, unknown]` | *(none)* |
| `[]` or `None` | *(none)* |

It **never overwrites an exclusion already set** — a crash or a dead credential knows more than this rule does. Its detail names the tools that failed:

```text
all 2 tool call(s) failed — bash, ipython. The agent could not act on its
environment, so this is a substrate fault rather than a low score: do not
train on it.
```

---

Related: [outcomes](outcomes.md) for the tri-state and why `unknown` exists · [configuration](configuration.md) for isolation, timeouts and environment variables · [production](production.md) for the policy side of the wire.
