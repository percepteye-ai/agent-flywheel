# Outcomes: `ok`, `failed`, `unknown`

Rewards in agent RL are usually computed from transcripts, and **transcripts lie about tool outcomes.** Two shapes cause most of it:

- **A tool that swallows its own exception.** A LangChain tool with `handle_tool_error=True` catches the raise and returns a string that reads like a normal result. `on_tool_end` fires either way. Nothing in the transcript separates it from a success.
- **A permission denial your API returns as HTTP 200.** The body explains the refusal; the status says fine. Score the status and a policy learns the write landed.

*Fuzzing RLVR Verifiers* ([arXiv:2606.01066](https://arxiv.org/abs/2606.01066)) measured false-positive rates of **0.869 on JSON tool calls** for buggy verifiers, against 0.000 for strict ones. That is the cost of the guess.

So this SDK adds a third value and makes the dangerous shape **unconstructible** — not documented, not linted. `contract.py` refuses to build it; see [The contract](contract.md).

## The three values

| value | what it asserts | `status_code` | on the wire, server-side |
|:--|:--|:--|:--|
| `ok` | an observed success | int or `null` | `true` |
| `failed` | an observed failure | int or `null` | `false` |
| `unknown` | **nothing observed a verdict** | forced to `null` | `null` — outcome-dependent verifiers abstain |

The tri-state mapping is declared in `schema/flywheel-1.json`, which this repository owns — see [The contract](contract.md). `Outcome` is `Literal["ok", "failed", "unknown"]` and there is no fourth value; anything else raises `ContractViolation` with a message that names the alternative:

```text
ToolCallOutcome.outcome must be one of ('ok', 'failed', 'unknown'), got 'probably'.
There is no default: report what you observed, or report 'unknown'.
```

**A status is not an outcome.** `status_code` is only meaningful when you actually observed one on the wire, so every capture path in this package passes `status_code=None` — including for `ok`. LangChain surfaces no HTTP status at `on_tool_end`, and inventing one from the payload is how an unobserved call becomes a landed one.

## Both fields are required, and neither has a default

`outcome` and `status_code` are non-defaultable on `ToolCallOutcome` and keyword-only-and-required on `record_tool_call`. **You cannot record half an outcome by forgetting a keyword** — the interpreter stops you before the contract has to.

```python
>>> ToolCallOutcome(name="POST /issue", arguments={}, outcome="ok")
TypeError: ToolCallOutcome.__init__() missing 1 required positional argument: 'status_code'
```

The permission-denial case, stated plainly — **both halves are true, both recorded**:

```python
record_tool_call("charge_card", {"amount": 4200},
                 outcome="failed", status_code=200, error="app_auth_error")
```

```json
{"name": "charge_card", "arguments": {"amount": 4200},
 "outcome": "failed", "status_code": 200, "error": "app_auth_error"}
```

A status may **confirm or contradict** an observed outcome. It may never **manufacture** one.

## `unknown` is legal, and it is never rounded up

`unknown` is the correct answer whenever nothing exposed a real verdict. It is first-class: supported, expected, and reported as-is. It is never promoted to `ok` at capture, in the trajectory, or on the wire.

An agent with **zero instrumentation** and no adapter that can classify its calls reports `unknown` for all of them — that is the honest floor, and `tests/test_zero_instrumentation.py` holds it there:

```json
"tool_calls": [{"name": "lookup", "arguments": {"q": "count to three"},
                "outcome": "unknown", "status_code": null,
                "output": "found: count to three", "latency_ms": 0.199}]
```

## `unknown` discards status from the field *and* from the body

Withholding `status_code` alone is not enough. A scorer that recovers a status by parsing the result envelope will find it there and read an unobserved call as a landed one. **Absence of evidence must not be recoverable as evidence of success through a side channel.**

So `__post_init__` forces `status_code = None` and scrubs the body, and both happen at construction — there is no path that skips them:

```python
tc = ToolCallOutcome(
    name="POST /issue", arguments={}, outcome="unknown", status_code=200,
    output={"status_code": 201, "id": "10001", "note": "created with HTTP 201"},
)
tc.status_code            # None
tc.output                 # {"id": "10001", "note": "created with"}
```

The human-readable content survives — the trajectory still reads the same to a judge. It simply carries no status a scorer can mistake for an outcome.

??? note "What the scrub actually removes"

    `_scrub_status` runs when and only when `outcome == "unknown"`.

    | input | treatment |
    |:--|:--|
    | `dict` | keys `status_code`, `status`, `http_status`, `code`, `statusCode` dropped; every value recursed |
    | `str` that parses as JSON | parsed, scrubbed, re-serialized — `'{"status_code":200,"id":"PROJ-1"}'` → `'{"id": "PROJ-1"}'` |
    | `str` that does not | `\bHTTP\s*\d{3}\b` removed — `"Request returned HTTP 200 OK"` → `"Request returned  OK"` |
    | `list` | each element recursed |

    The schema enforces the same rule independently: `outcome: "unknown"` implies `status_code: {"type": "null"}`, so a hand-rolled client that skips the Python types still fails validation.

## What `unknown` costs, and what it does not

**It lowers the grade. It does not exclude the rollout.**

`unknown` is not an exclusion reason — the vocabulary is `flywheel_crash`, `substrate_unavailable`, `credential_expired`, `deadline_exceeded`, `entrypoint_error`, `agent_exit_nonzero`, `cancelled`, and `capacity`, and none of them is about not knowing. A rollout full of `unknown` calls is reported and graded like any other; the verifiers that grade *"did this call land"* have nothing to read and correctly abstain, so the reward surface available to it is materially smaller.

The one automatic exclusion that reads tool calls is deliberately strict about this. `_mark_substrate_unavailable` fires only when **every** call is an *observed* `failed` and at least one call was made:

- `unknown` is not failure, so a single `unknown` keeps the rollout in.
- Partial failure is left alone — some calls failing is ordinary agent behavior and is exactly what training wants.
- It never overwrites an exclusion already set; a crash or a dead credential knows more.

`Trajectory.unknown_count()` is the local number if you want to watch it.

## `record_tool_call`

The always-available ground truth. Adapters are best-effort acceleration over this function, never a substitute for it.

```python
def record_tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    *,
    outcome: Outcome,
    status_code: int | None,
    output: Any = None,
    error: str | None = None,
    error_class: str | None = None,
    latency_ms: float | None = None,
    agent_name: str | None = None,
    tool_call_id: str | None = None,
    entity_ids: dict[str, str] | None = None,
) -> None: ...
```

| argument | notes |
|:--|:--|
| `outcome` · `status_code` | keyword-only, **required**, no default |
| `error` · `error_class` | free text and exception class name |
| `agent_name` | which agent in a multi-agent system made the call. `None` is *unattributed*, **not** a claim that one agent made everything |
| `tool_call_id` | the id the **model** assigned — LangChain `ToolMessage.tool_call_id`, ADK `function_call_id`, OpenAI `tool_calls[].id`. The join key back to the completion that emitted it: no clocks, survives concurrent tools |
| `entity_ids` | the identity a write produced — see below |

**Outside a rollout it is a no-op**, so the call can sit in a production code path unconditionally.

!!! warning "Recording with no destination warns, once"
    Inside a rollout with no trajectory directory configured, the outcome is **discarded** and a `RuntimeWarning` says so. Silently dropping it would make capture look like it worked while producing an empty tool-call list — which reads downstream as "this agent made no tool calls" rather than "we failed to look."

Records land as one JSON line per call, appended and flushed, in `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`. **A file rather than a list, because the default isolation spawns a child process per rollout** — in-memory state and context variables do not cross that boundary, and a crash mid-rollout leaves every call up to that point recorded rather than nothing.

## `tool_calls`: `None` is not `[]`

Absence is never collapsed into zero.

| value | meaning |
|:--|:--|
| `None` | I did not report tool calls |
| `[]` | I observed zero tool calls — a **positive claim** |

`RolloutOutput.tool_calls` defaults to `None`, so an agent that reports nothing cannot accidentally assert zero. `Trajectory.read()` preserves the same distinction: no `tool_calls.jsonl` at all returns `None`. A line the reader cannot parse — a truncated final write is the *expected* case after a deadline kill — costs that line and nothing else, and is counted into `tool_calls_omitted_count` rather than dropped. A call we know executed and cannot read is not a call that did not happen.

## `entity_ids`: which entity the write produced

A tool outcome says whether a call **worked**. It cannot say whether the change **survived** — a create that returns 201 and then silently fails to persist is, from inside the process, identical to one that landed. Only reading the resource back settles it, and a read-back needs one thing: **which entity to look for.**

Because the SDK runs beside your agent rather than in place of it, the trajectory can carry that. **It carries the identity, never the body:**

- The body is customer production data. Persisting it in a trajectory turns a capture library into a data processor.
- The identity is a handful of opaque strings, and it is nearly all a read-back needs.

Identity is resolved at the moment of capture and the body is free to be dropped. Extract, then discard.

```python
record_tool_call("POST /rest/api/2/issue", {"summary": "..."},
                 outcome="unknown", status_code=None,
                 output={"id": "10001", "key": "ENG-7", "status_code": 201})
```

```json
{"name": "POST /rest/api/2/issue", "arguments": {"summary": "..."},
 "outcome": "unknown", "status_code": null,
 "output": {"id": "10001", "key": "ENG-7"},
 "entity_ids": {"id": "10001", "key": "ENG-7"}}
```

Pass `entity_ids=` when you know it; **a declared identity always wins**, and extraction from `output` is the fallback. Values are strings, filed under the field they came from — per-field filing is not cosmetic, it is what lets a later read-back compare a candidate's `key` to the write's `key` and never to the write's `id`.

### Which fields count

| | fields | why |
|:--|:--|:--|
| **accepted** | `id`, `key`, `uuid`, `sid`, `gid` | high-entropy; they identify one entity |
| **refused by name** | `name`, `number`, `title`, `label`, `summary`, `description` | a created resource routinely echoes a human `name` ("Story 1") or a positional `number` (7) that a **different** entity shares by coincidence |

Refused, not discouraged: `entity_ids={"name": "Story 1"}` raises `ContractViolation`. Matching on a low-entropy field confirms the wrong resource — a false positive dressed as a verified success, **strictly worse than no evidence at all.**

### How extraction decides

| rule | behavior |
|:--|:--|
| JSON string body | parsed, then walked |
| prose | `{}` — an id parsed out of free text is a guess |
| depth | bounded at 3, so an envelope (`{"data": {...}}`) is reached and a deep nested object's id is not |
| two matches, different depths | shallowest wins — the entity's own id sits at the top of its response |
| two matches, same depth, disagreeing | **dropped as ambiguous.** An ambiguous identity is not an identity |
| `true`, `null`, `""`, `"   "` | never an identity (`bool` is an `int` in Python, so it is guarded explicitly) |
| anything unreadable | `{}`. Extraction never raises — it runs on the customer's own request path |

### Identity is recorded even when the outcome is `unknown`

!!! note "This is the point of the whole field"
    `entity_ids` is deliberately **not** scrubbed for an `unknown` outcome, unlike `output`. The scrub exists so an unobserved call cannot have a status recovered from its body and be read as landed. An identity claims nothing about the outcome — it names an entity so that **something else** can go and observe one. Withholding it would suppress the only evidence capable of turning an honest `unknown` into an honest `ok`.

And **absence is abstention.** Empty `entity_ids` means no identity evidence, so a later read-back declines to conclude anything — the same rule `unknown` follows, one layer down. It is never a failure.

## Where outcomes come from when you write nothing

`record_tool_call` is the floor. Above it, capture is free:

| surface | mapping |
|:--|:--|
| **LangChain · LangGraph** | `ok` on `ToolMessage(status="success")` · `failed` on a raise or `status="error"` · `unknown` otherwise |
| **Google ADK** | `failed` on a raise **and** on `{"error": ...}` · `unknown` otherwise |
| **OTel-GenAI / prime-agent event streams** | error flag `false` → `ok` · `true` or a non-empty string → `failed` · absent, or an empty string, → `unknown` |

Every one of them reports `unknown` rather than guessing, because that is the adapter bar. See [Python agents](python.md) and [CLI and other languages](cli.md); the wire types are on [The contract](contract.md).
