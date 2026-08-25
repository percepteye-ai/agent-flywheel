# Plan: confirming outcomes against the system of record

Status: **phase 1 implemented in this repo.** Phases 2–3 are control-plane work
and are specified here, not built here.

## 1. The gap

A tool outcome says whether a call **worked**. It cannot say whether the change
**survived**.

A create that returns `201` and then silently fails to persist is
indistinguishable, from inside the agent's process, from one that landed. The
tri-state already refuses to round an unobserved call up to `ok`, which stops
the SDK from *fabricating* success — but refusing to guess is not the same as
knowing. Nothing observed from inside the process can close that gap.

The only thing that settles it is reading the resource back afterwards, in the
system that owns it. That requires two things the SDK did not previously have:

1. **Which entity to look for.** The trajectory carried the response body or
   nothing. A body is simultaneously too much — it is customer production data
   — and too little, because nothing downstream knew which of its fields named
   the entity the write produced.
2. **A way to look.** Read-only access to the environment the agent operates
   on, supplied by the customer, and absent by default.

## 2. The insight worth more than the feature

Read-back's highest-value application is **not** double-checking calls that
already reported `ok`. It is **resolving `unknown`**.

| outcome says | read-back says | result |
|---|---|---|
| `ok` | confirmed | high-confidence positive |
| `ok` | absent | a success that did not persist — caught |
| **`unknown`** | **confirmed** | **an episode that abstains today becomes trainable** |
| `unknown` | absent | still `unknown`, unless access can see the resource |
| `failed` | present | partial landing; the agent misread its own result |

Today `unknown` lowers the grade and outcome-dependent checks abstain on it.
Confirmation is the only thing that can *honestly promote* it. So the tri-state
is not half a verdict awaiting a second half — it is a **work queue**, telling
the scorer where a limited read budget will actually change an answer.

That reframing has a consequence for the design: identity must be recorded for
`unknown` calls above all. Suppressing it there would starve precisely the
records confirmation exists to settle.

## 3. Phase 1 — make the trajectory sufficient (DONE, this repo)

`ToolCallOutcome` gained one field, `entity_ids`: the identity a write
produced, filed under the field it came from — `{"id": "10001", "key": "ENG-7"}`.

**Extract, then discard.** Identity is resolved at the moment of capture and
the body is never required downstream. This is the whole reason the field is an
identity rather than a payload: a body in a trajectory turns a capture library
into a data processor, while an identity is a few opaque strings that are
nearly all a later read needs.

Rules, all enforced rather than documented:

- **High-entropy fields only** (`id`, `key`, `uuid`, `sid`, `gid`). Low-entropy
  fields (`name`, `number`, `title`, …) are refused *by name*, including when a
  caller declares them explicitly. A created resource routinely echoes a human
  `name` that a different entity shares by coincidence; matching on it confirms
  the wrong resource, which is a false positive wearing a verified badge and is
  strictly worse than having no evidence.
- **Per-field filing, never a flat set.** This is what lets a later read compare
  a candidate's `key` to the write's `key` and never to the write's `id`.
  Collapsing them is how an entity whose literal `id` equals another's `key`
  string gets falsely confirmed.
- **Shallowest wins; ambiguity is dropped.** Two equally-shallow answers that
  disagree mean we do not know which entity is *the* entity, and the honest
  result of ambiguity is no evidence.
- **Prose yields nothing.** An id parsed out of free text is a guess.
- **Not scrubbed on `unknown`,** unlike `output`. Status is scrubbed there so an
  unobserved call cannot have a verdict recovered from its body through a side
  channel. An identity is not that: it claims nothing about the outcome, it
  names an entity so something else can observe one.
- **Absent means absent.** No identity is an abstention, never a failure.

### Coverage

Identity is resolved inside `record_tool_call`, which every adapter already
routes through, so the framework paths required no adapter change at all. That
is a claim worth a test rather than an argument — "it should already work
because of where I put it" is exactly how the child→parent boundary lost
identity while looking correct.

| capture path | covered | how |
|---|---|---|
| `record_tool_call` | yes | resolved centrally; a declared identity wins |
| LangChain / LangGraph | yes | adapter → `record_tool_call` |
| Google ADK | yes | adapter → `record_tool_call` |
| CLI agent event stream | yes | resolved from the structured result |
| child → parent boundary | yes | carried on the wire |
| an agent writing the trajectory itself | yes | the file format is the interface |
| OpenClaw plugin | yes | a mirror of the Python extractor |
| DeepSeek Harness plugin | yes | resolved from the canonical value |

**dsh was the one gap, and it is now closed.** It had no tool-outcome capture
at all: `fw.TASK` and `endpoint_env=` got it running and routed, neither got
its tool calls out. Its JSON stream carries `toolcall_start` / `toolcall_end`,
but those are MODEL-WIRE events — the model emitting a call and streaming its
arguments — ending at the assembled call and never the result, so a stream
dialect would have produced arguments with no verdict. Capture goes through the
harness's own plugin surface instead, on its emit-only tool-result listener.
Identity comes from the canonical value, which the harness documents as
execution-local and deliberately omitted from durable events — so it exists at
that seam and nowhere later.

## 4. Phase 2 — the credentialed reader (control plane, NOT built)

> [!IMPORTANT]
> **Phase 2 is larger than this section assumes, and the reason is worth
> stating before anyone starts.** This was written as though a credentialed
> reader could be swapped into a running verification path. There is no such
> path on the flywheel side: the read-back gate requires a live rollout object,
> returns early when no base URL is configured, and the tool-mode table has no
> flywheel entry at all. So Phase 2 is *adding confirmation where none runs
> today*, not rebinding an existing executor.
>
> A second correction to the framing above: the reader is already shaped as
> "a base URL plus a credential plus a read-only client", so the confirmation
> engine does not need building. What needs building is the plumbing that
> reaches it, and a genuine second credential — today "read-only" is a
> client-side method block over the *same* secrets as the rollout that did the
> writing, which is not a property a customer-facing promise can rest on.


- Bind a reader to a customer-supplied base URL and **read-only** credential.
  Absent → every later phase is skipped and scoring is exactly what it is today.
- **A capability probe at onboarding, and this is not optional.** A read
  credential scoped to a different project or tenant returns a clean `404`,
  which is indistinguishable from "the write did not persist" — the exact
  mirror of the success-shaped failure this whole effort exists to catch. Before
  any `404` from a resource type is trusted as evidence of absence, the
  credential must demonstrate it can see a known-existing entity of that type.
  If it cannot, that type abstains.
- Budget and rate limits. Verification traffic lands on the customer's own API,
  appears in their audit logs, and at training scale is not negligible.

## 5. Phase 3 — the join (control plane, PARTLY BUILT)

- **~~An explicit `unknown` mapping rule at the boundary.~~ DONE, and it was
  not a rule — it was a live bug.** This item was written as "the two defaults
  compose into a bug if left implicit". They did not compose into a *potential*
  bug; the bug was already there and already being paid, in reward.

  The scoring side decided whether a write "returned a definitive success
  envelope" from the response status and body alone. That was correct while
  every write came from a live HTTP tool whose response was the only evidence
  there was. It is not correct once a capture layer records a first-class
  outcome, and two shapes were being credited:

  | what capture recorded | what the envelope concluded |
  |---|---|
  | `outcome="unknown"` | definitive success |
  | `outcome="unknown"`, a 200 in the body | definitive success |
  | **`outcome="failed"`, `status_code=200`** | **definitive success** |

  The third row is the one that matters most and was not anticipated when this
  document was written. An application-level denial returned as HTTP 200 is the
  *canonical* case the tri-state exists to catch — [the flagship example in the
  README](../README.md#why-unknown-is-a-first-class-tool-outcome). Capture
  recorded it as `failed`, correctly, and scoring read the 200 and called it a
  landed write. The SDK was doing its job and the answer was being discarded one
  layer later.

  The fix consults the observed outcome before the envelope. Two properties
  worth carrying forward:

  - **The guard keys on the outcome alone**, not on "unknown *and* no status".
    The obvious formulation misses a tool that swallows its own error and
    returns a 200-shaped body — precisely the `handle_tool_error=True` case.
  - **Absent is not `None`.** A record that never carried an outcome field — a
    stored dataset predating the concept — recorded no outcome at all, and the
    envelope is still the only evidence it has. Collapsing the two withdraws
    credit from every legacy record, which is a bug in the opposite direction
    and harder to notice, because it presents as scores drifting down rather
    than as anything failing.

  Ten tests pin it; four go red without the guard.
- **A freshness bound.** NOT built. There is currently no gating on the age of an episode
  at scoring time. A read-back run hours later verifies state that humans and
  scheduled jobs have since changed. Either bound the window or discount by
  elapsed time, and record both timestamps either way.
- **Coverage, reported per episode.** NOT built. Confirmation is partial by nature: some
  operations leave a readable resource and some do not. Reporting the fraction
  confirmed keeps the claim honest, and is more persuasive than the claim it
  replaces.

### An open question this document created

Identity is restricted to high-entropy fields (`id`, `key`, `uuid`, `sid`,
`gid`), which is right — a low-entropy value a different entity shares by
coincidence confirms the wrong resource. But a post-condition author may name
any response field as the identity source. One outside that set stops
resolving, and the concern is that scoring then lands on the **0.0 anti-hack
cliff** rather than on `inconclusive` — converting a limitation of *our capture
format* into a scored *policy failure*. Training on that teaches the model its
correct action was wrong, which is worse than not scoring the episode at all.

This is recorded as a question rather than a finding, because it is not yet
established that it is reachable, and because the cliff is an anti-gaming
guard. Two conditions have to hold before touching it:

1. **The agent must have no influence over whether identity resolves.** If a
   policy can cause its own writes to become unresolvable, the cliff is correct
   as it stands and must not move.
2. **The two causes must be distinguishable from the data.** "Unresolvable
   because our format could not carry it" and "unresolvable because the write
   never happened" have to be told apart, or there is nothing to route
   differently and this is not fixable at the scoring layer.

If either fails, the honest outcome is to leave the cliff alone and record why.

## 6. Risks

- **Coverage bias.** Sends, deletes (absence is not proof of deletion), pure
  reads and effects outside the API are not confirmable. If confirmable actions
  earn confident grades while everything else abstains, a policy can drift
  toward the confirmable subset. Worth monitoring as a distribution, not
  assuming away.
- **The verifier can be wrong.** A second signal is not automatically a better
  one. Disagreement between an observed outcome and a read-back should resolve
  toward `unknown` and surface for inspection, never silently pick a winner.
- **Data handling.** Extract-and-discard keeps bodies out of trajectories, which
  is the mitigation, not an afterthought. It should stay that way even when a
  body would be convenient.
- **A changed posture.** The SDK's story is that nothing listens and every
  connection is opened from the customer's side. Confirmation adds egress *into*
  customer infrastructure from ours. That is a different arrangement and should
  be described plainly rather than folded into the existing claim. It is,
  however, a smaller step than it sounds: the agent is already writing to that
  environment, so a read-only credential is the least invasive part of it.

## 7. What changes about the pitch

Today's claim is defensive — *observed, not inferred*: we will not tell you a
tool call succeeded when nobody saw it succeed. True, and a floor.

Confirmation moves it from **honest capture** to **verified outcomes**: the
thing the agent claimed to do actually persisted, checked against the system
that holds it.

That changes the competitive axis rather than the score on the existing one.
Other platforms grade from evidence the run produced — a transcript, a model
wire, or an environment they supply and you port your task into. Grading
against the system the customer already runs, with nothing ported, is available
only to something already inside the process that did the work.

One discipline, or this becomes the overclaim the tri-state exists to prevent:
**coverage is partial and the claim must ship scoped.** Report the confirmed
fraction per episode. It is simultaneously the honest framing and the more
persuasive one.
