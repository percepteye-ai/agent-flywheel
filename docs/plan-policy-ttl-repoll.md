# Scope: TTL and re-poll for `current_policy()`

Status: **phases 1 and 2 BUILT** (`src/agent_flywheel/policy.py`,
`tests/test_policy_source.py`). Phase 3 (`background=True`) is deliberately
unbuilt — per §6 it ships only if inline refresh proves insufficient for
low-traffic agents, and nothing has measured that yet.

Three things this document got wrong, corrected in the build and recorded
here so the next reader does not re-derive them:

* §4.2 ("never blocks on the network after the first call") and §4.3 (inline
  refresh with a 1s timeout) contradict each other. §4.3 is the mechanism;
  §4.2 is the intent, which is really "no UNBOUNDED blocking". The bound is a
  total, not per-attempt, so the refresh client is built with
  `max_attempts=1`.
* §4.6 says "the cache needs a lock". It needs TWO. One lock makes every
  concurrent caller queue behind the in-flight fetch on its way to READ the
  cache — the exact latency this design exists to avoid, and invisible to a
  test that only counts requests.
* §4.1's loop rebuilds from the snapshot on every `changed`. Since 0.2.0 a
  WITHDRAWAL (`withdrawn: true`) is a change to an EMPTY snapshot with
  `snap.withdrawn` set, and the caller goes back to its own configuration
  instead; the current loop is in `policy.py`'s docstring and
  `website/docs/production.md`.

Open question 2 is answered: the handle has no `set_env`. Open question 3 is
answered: damping a flapping control plane is not the client's job. Open
question 4 is answered in the module docstring: call it per turn.

## 1. What is true today

`current_policy()` is a single GET, and `apply_current_policy()` wraps it. Both
are one-shot: they read `/policy/current` once, return kwargs, and hold no
state. Verified in `src/agent_flywheel/__init__.py:125` and
`transport.py:75` — there is no cache, no TTL, no refresh, and nothing that
re-reads.

The documented usage compounds this. The README's example calls
`apply_current_policy()` once and constructs a client from the result:

```python
kw = fw.apply_current_policy(agent_id="my-coding-agent")
if kw:
    client = OpenAI(base_url=kw["base_url"], api_key=kw["api_key"])
```

A client built that way lives for the process. So the policy a production agent
runs is **the policy that was current when that process started**.

## 2. Why this matters, and it is not mainly about promotion

The obvious cost is that a promotion does not reach production until the
customer redeploys — a continual-learning loop whose last mile updates on
deploys.

The larger cost is the other direction. **A regressed candidate can only be
withdrawn by redeploying the customer's production agent.** The control plane
can demote it instantly and nothing happens; every running process keeps
sampling the bad model until someone ships. Delayed promotion is an
inconvenience. Delayed *rollback* is an incident, and it is an incident whose
remedy sits in the customer's deploy pipeline rather than ours.

Re-poll is therefore the transport for rollback first and promotion second.
That ordering should drive the design: the failure mode we are buying insurance
against is "serving something bad for longer than necessary".

## 3. Constraints that are load-bearing

These come from decisions already made in the codebase. A design that breaks
any of them is wrong regardless of its other merits.

1. **Fail open, on every path.** `policy_current()` never raises into the
   caller's startup, because an unreachable control plane must not stop an
   agent from serving its users. Refresh must inherit this — a failing refresh
   is not an error, it is a non-event.
2. **The agent is TOLD its arm and never chooses one** (`transport.py:80`).
   That is what keeps results attributable. Re-poll re-reads an assignment; it
   must introduce no client-side selection, no splitting, no fallback picking.
3. **No new route.** `/policy/current` exists in `flywheel/1`. This is
   client-side behaviour over the route we already have.
4. **No global side effects unless asked.** `set_env=True` is opt-in precisely
   because mutating the process environment is a side effect a caller should
   request. A background thread is the same category of imposition.
5. **One runtime dependency.** Nothing new. `threading` and `time` from the
   standard library, and only if a thread is genuinely warranted.
6. **Never for `serve()`.** The correctness boundary must survive: a donor
   rollout samples through the per-rollout gateway, and pointing it at the
   policy endpoint bypasses capture entirely.

## 4. Design

### 4.1 A handle, because the current shape cannot refresh

The existing functions return a plain dict, which is inherently boot-time: once
the caller has built a client from it, nothing will ever call us again. Adding
a `ttl_s=` parameter to `apply_current_policy()` would therefore do nothing for
the documented usage — the caller still calls it once.

So the new surface is a handle the caller keeps and consults:

```python
policy = fw.policy_source(agent_id="my-coding-agent", ttl_s=300)

# per turn, or per request, or per N requests -- the caller's choice
snap = policy.current()
if snap.changed:                      # rebuild ONLY on an actual change
    client = OpenAI(base_url=snap.base_url, api_key=snap.api_key)
    model = snap.model
```

`snap.changed` is the important field. Most OpenAI-compatible clients hold
connection pools, so rebuilding one per request is a real cost. The handle
tracks the last snapshot it handed out and reports whether this one differs, so
the caller rebuilds on promotion and demotion and at no other time.

`current_policy()` and `apply_current_policy()` stay exactly as they are. They
remain correct for the boot-time case and for anyone who redeploys on
promotion; nothing about this is a migration.

### 4.2 Serve-stale-always

`current()` **never blocks on the network** after the first call. It returns
the cached snapshot immediately and marks the cache for refresh when the TTL
has passed.

The alternative — refresh synchronously when stale — puts control-plane latency
on a user-facing request path, and puts control-plane *availability* there too.
That is unacceptable for a production agent and would violate constraint 1 in
spirit even while honouring it in letter.

The first call may block, because it already does today and there is nothing
cached to serve instead.

### 4.3 Refresh without imposing a thread

Default: refresh happens **inline on the call that observes staleness**, with a
hard timeout (proposed 1.0s) and stale served on timeout. One request pays a
bounded cost; the rest are free. No thread, no scheduler, no imposition.

Opt-in: `policy_source(..., background=True)` spawns one daemon thread that
refreshes on the TTL. Offered because a low-traffic agent may go a long time
between calls, and because some callers would rather pay a thread than have any
request pay latency. Off by default, per constraint 4.

### 4.4 Last-good retention — never demote on failure

**If a refresh fails, keep the last good snapshot indefinitely.**

This is the single most important behaviour in the design and it is not
obvious. The naive reading of "fail open" is to return `{}`, meaning "keep your
current configuration". But a handle that returns `{}` after having had a
policy would cause the caller's `snap.changed` to fire and the client to be
rebuilt against — what, exactly? The base model. A control-plane blip would
silently demote production from the trained model back to the untrained one,
which is a strictly worse failure than serving a slightly stale policy.

So: a failed refresh is a non-event. The snapshot stands, `changed` is false,
and staleness accumulates visibly:

```python
snap.stale_for_s     # 0.0 when fresh; grows while refreshes are failing
snap.last_refresh_ok # False after a failed attempt
```

The caller can alert on that. We do not decide for them.

### 4.5 TTL: server hint wins, client sets the default

The control plane knows its own promotion cadence; the customer does not. So if
the response carries a TTL hint, it wins. `ttl_s=` is the default when the
server says nothing, and there is a floor (proposed 30s) so that neither a
misconfigured client nor a server hint can turn a fleet into a polling storm.

**Jitter is mandatory, not a nicety.** A fleet of replicas that boots together
shares a TTL and will re-poll in lockstep, delivering a synchronised spike to
`/policy/current` every TTL. Proposed ±10% uniform jitter per handle, decided
once at construction.

### 4.6 Concurrency

Production agents are concurrent, so the cache needs a lock and snapshots must
be immutable — a frozen dataclass, returned by value. A caller that holds a
snapshot across an in-flight request must be unaffected by a refresh completing
underneath them. Immutability gives that for free; nothing swaps under anyone.

### 4.7 Observability

`on_change=` callback, fired when a snapshot differs from the last one handed
out. Without it, the model your production agent is running changes silently,
which is not a property anyone should accept in a system whose entire premise
is that the model improves over time. Same shape as `serve(on_event=...)`.

## 5. Non-goals

- **Canary, traffic splitting, arm selection.** These stay server-side, per
  constraint 2. The handle reads an assignment. If the control plane wants 10%
  of processes on a candidate, it says so per caller; the client never rolls
  dice.
- **Automatic rollback logic.** The handle is the transport that makes a
  server-side rollback take effect quickly. It does not decide to roll back.
- **A new route.** None needed.
- **Changing `serve()`.** Untouched.

## 6. Phasing

| phase | contents | why this order |
|---|---|---|
| 1 | `policy_source()`, TTL with jitter and floor, serve-stale, last-good retention, `changed` | the whole rollback benefit lands here |
| 2 | `on_change=`, `stale_for_s`, `last_refresh_ok` | makes a silent flip visible |
| 3 | `background=True` | only if phase 1 proves inline refresh is insufficient for low-traffic agents |

Phase 1 is the deliverable. Phases 2 and 3 are separable and neither blocks it.

## 7. Testing constraints

**The clock must be injectable.** This suite has now been bitten twice by
wall-clock assertions — a heartbeat test that counted beats inside a 2.5s
window and failed on a loaded CI runner, and its sibling with the same flaw.
Both were rewritten to assert the rule where the decision is made rather than
its timing consequence. A TTL feature is nothing *but* timing, so it must be
built with a `now=` seam from the first commit rather than retrofitted.

No test in this feature may call `sleep`. Tests to write:

- TTL not elapsed → no network call at all
- TTL elapsed → exactly one refresh, and concurrent callers do not stampede
- refresh fails → snapshot stands, `changed` is False, `stale_for_s` grows
- refresh returns a different endpoint → `changed` is True exactly once
- refresh returns the same endpoint → `changed` is False
- server hint overrides `ttl_s`; floor overrides both
- jitter stays within bounds and differs across handles
- first call with an unreachable control plane → `{}`-equivalent, no raise
- snapshot held across a refresh is unchanged (immutability)

## 8. Open questions

1. **Does the control plane want to send a TTL hint?** If not, phase 1 ships
   with the client default and the hint is added later without a breaking
   change.
2. **What does `set_env=True` mean on a handle?** Re-exporting
   `OPENAI_BASE_URL` on a timer is a global mutation on a schedule, which is
   worse than the one-shot version. Proposal: the handle does not support
   `set_env`, and the docstring says why.
3. **Minimum dwell time.** If the control plane flaps between two policies, the
   handle faithfully flaps production. Is damping our job or theirs? Leaning
   theirs — the client should not second-guess an assignment — but it deserves
   an explicit decision rather than a default.
4. **How often should a caller call `current()`?** Per turn is the natural unit
   for an agent. This wants a documented recommendation, because a caller who
   calls it once has re-created the problem this scope exists to solve.
