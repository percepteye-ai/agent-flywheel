"""A policy handle your production agent can consult, instead of reading once.

The module is ``policy``, not ``policy_source``: the factory function is the
good public name, and a module of the same name on the package would be
shadowed by it the moment ``__init__`` re-exports the function.

WHY THIS EXISTS, AND IT IS NOT MAINLY ABOUT PROMOTION.

``current_policy()`` is a single GET and ``apply_current_policy()`` wraps it.
Both are one-shot, and the documented usage compounds that: build a client from
the result and the client lives for the process. So the policy a production
agent runs is the policy that was current when that process started.

The obvious cost is that a promotion does not reach production until the
customer redeploys. The larger cost is the other direction. **A regressed
candidate can only be withdrawn by redeploying the customer's production
agent.** The control plane can demote it instantly and nothing happens; every
running process keeps sampling the bad model until someone ships. Delayed
promotion is an inconvenience. Delayed *rollback* is an incident, and it is an
incident whose remedy sits in the customer's deploy pipeline rather than ours.

This handle is therefore the transport for ROLLBACK first and promotion second,
and the design follows from that ordering.

USAGE — call it per turn::

    policy = fw.policy_source(agent_id="my-coding-agent", ttl_s=300)

    while True:                       # per turn, or per request
        snap = policy.current()
        if snap.changed:              # rebuild ONLY on an actual change
            client = OpenAI(base_url=snap.base_url, api_key=snap.api_key)
            model = snap.model
        ...

Per turn is the natural unit for an agent, and it is the recommendation. A
caller who calls ``current()`` once has re-created the problem this class
exists to solve.

WHAT IT WILL NOT DO
-------------------
* **It never chooses an arm.** The agent is TOLD its arm, which is what keeps
  results attributable. No client-side splitting, no fallback picking, no dice.
* **It never damps a flapping control plane.** If the assignment flips, the
  handle flips. Second-guessing an assignment is how a client stops being a
  faithful reader of one, and damping belongs on the side that knows why the
  assignment moved.
* **It has no ``set_env``.** ``apply_current_policy(set_env=True)`` mutates the
  process environment once, at a moment the caller chose. Re-exporting
  ``OPENAI_BASE_URL`` on a timer is that same global mutation on a schedule,
  landing under whatever else the process happens to be doing. Read
  ``snap.base_url``.
* **It is never for** :func:`serve`. A donor rollout samples through the
  per-rollout gateway, which is what attributes and captures the completion.
  Pointing one at the policy endpoint bypasses capture entirely.
"""
from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

__all__ = ["PolicySnapshot", "PolicySource", "policy_source"]

#: Never poll faster than this, whatever the caller asks for or the server
#: hints. A misconfigured ``ttl_s=1`` across a fleet is a denial of service
#: aimed at our own control plane, and the client is the wrong place to
#: discover that.
MIN_TTL_S = 30.0

#: A refresh gets ONE attempt and this long. It runs inline on the request
#: that observed staleness, so it is spending a user's latency; a retry ladder
#: would spend several users'. The next stale observation retries naturally,
#: which is what a cache with last-good retention is for.
REFRESH_TIMEOUT_S = 1.0

#: Fraction of the TTL to jitter, +/-. MANDATORY, not a nicety: a fleet of
#: replicas that boots together shares a TTL and will re-poll in lockstep,
#: delivering a synchronised spike to /policy/current every TTL. Decided once
#: per handle at construction so a handle's cadence is stable.
JITTER_FRACTION = 0.10


@dataclass(frozen=True)
class PolicySnapshot:
    """What to run, and how much to trust it. Immutable BY DESIGN.

    Production agents are concurrent. A caller holding a snapshot across an
    in-flight request must be unaffected by a refresh completing underneath
    them, and frozen-and-returned-by-value gives that for free — nothing swaps
    under anyone.
    """

    #: OpenAI-compatible base URL. Empty when the control plane has nothing
    #: deployed for this agent, or has never answered.
    base_url: str = ""
    api_key: str = ""
    model: str = ""

    #: True on the ONE snapshot that first reports a change, false otherwise.
    #: It is a per-handle EDGE, so build one client per handle: two threads
    #: sharing a handle will not both see the edge, by construction.
    changed: bool = False

    #: 0.0 while refreshes are succeeding; grows while they are failing. This
    #: is the number to alert on — it is the only outward sign that an agent
    #: is running a policy the control plane may have moved on from.
    stale_for_s: float = 0.0

    #: False after a failed refresh attempt. The snapshot still stands.
    last_refresh_ok: bool = True

    def __bool__(self) -> bool:
        """Truthy only when there is something to act on — same meaning as
        ``apply_current_policy()``'s ``{}``: keep your current configuration."""
        return bool(self.base_url and self.api_key)

    def as_kwargs(self) -> dict[str, Any]:
        """Client kwargs, identical in shape to ``apply_current_policy()``.

        Here so a caller migrating from the one-shot function changes where the
        dict comes from and nothing else."""
        if not self:
            return {}
        return {"base_url": self.base_url, "api_key": self.api_key,
                "model": self.model}


class PolicySource:
    """A refreshing view of ``/policy/current``. See the module docstring.

    SERVE-STALE, ALWAYS. After the first call, ``current()`` returns the cached
    snapshot and refreshes are bounded (one attempt, ``REFRESH_TIMEOUT_S``)
    rather than unbounded. Putting control-plane *availability* on a
    user-facing request path is not something a production agent can accept,
    and an unbounded refresh does exactly that. The first call may block,
    because it already does today and there is nothing cached to serve
    instead — and even it fails open.

    LAST-GOOD RETENTION. **A failed refresh is a non-event.** The naive reading
    of "fail open" is to return the empty answer meaning "keep your current
    configuration" — but a handle that returns empty after having HAD a policy
    makes ``changed`` fire and the caller rebuild against the base model. A
    control-plane blip would silently demote production from the trained model
    back to the untrained one, which is strictly worse than serving a slightly
    stale policy. So the snapshot stands, ``changed`` is False, and staleness
    accumulates visibly instead.

    NO STAMPEDE. Exactly one caller refreshes; everyone else is handed the
    cached snapshot immediately rather than queueing behind the lock. Fifty
    concurrent turns cost one request and forty-nine cache reads.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        ttl_s: float = 300.0,
        control_plane_url: str | None = None,
        api_key: str | None = None,
        on_change: Callable[[PolicySnapshot], None] | None = None,
        _now: Callable[[], float] = time.monotonic,
        _fetch: Callable[[], dict[str, Any]] | None = None,
        _rand: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.agent_id = agent_id
        self._now = _now
        self._on_change = on_change
        self._api_key_arg = api_key
        self._url_arg = control_plane_url
        self._base_ttl = max(MIN_TTL_S, float(ttl_s))
        # Decided ONCE, so this handle's cadence is stable and differs from its
        # siblings'. Re-jittering per refresh would smear each handle across
        # the window instead of separating handles from each other, which is
        # the actual problem.
        self._jitter = _rand(-JITTER_FRACTION, JITTER_FRACTION)
        self._server_ttl: float | None = None

        # TWO locks, and the split is load-bearing. ``_refresh_lock`` is held
        # across the network call and answers "am I the one refreshing";
        # ``_state_lock`` guards the cached fields and is held for microseconds.
        # One lock for both would make every concurrent caller queue behind the
        # in-flight fetch on its way to READ the cache — which is exactly the
        # control-plane latency this class exists to keep off the request path,
        # and it looks identical to the correct version if you only count
        # requests.
        self._refresh_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._snap = PolicySnapshot()
        self._last_ok_at: float | None = None
        self._next_refresh_at: float = -1.0     # -1 => never fetched
        self._last_identity: tuple[str, str, str] | None = None
        self._fetch_override = _fetch
        self._client: Any = None

    # ── public surface ────────────────────────────────────────────────────

    def current(self) -> PolicySnapshot:
        """The policy to run right now. Never raises."""
        now = self._now()
        if self._next_refresh_at < 0.0 or now >= self._next_refresh_at:
            self._maybe_refresh(now)
        with self._state_lock:
            snap = self._snap
            if snap.changed:
                # The edge is CONSUMED by whoever observes it, so a second
                # caller does not rebuild a client that is already correct.
                self._snap = replace(snap, changed=False)
            return replace(snap, stale_for_s=self._stale_for(now))

    def close(self) -> None:
        """Release the pooled connection. Idempotent."""
        with self._state_lock:
            client, self._client = self._client, None
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def __enter__(self) -> PolicySource:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def ttl_s(self) -> float:
        """The interval in force, server hint included, jitter applied.

        The control plane knows its own promotion cadence and the customer does
        not, so a server hint wins over ``ttl_s=``; the floor wins over both."""
        base = self._server_ttl if self._server_ttl is not None else self._base_ttl
        return max(MIN_TTL_S, base) * (1.0 + self._jitter)

    # ── internals ─────────────────────────────────────────────────────────

    def _stale_for(self, now: float) -> float:
        if self._last_ok_at is None:
            return 0.0
        overdue = now - self._last_ok_at - self.ttl_s
        return overdue if overdue > 0.0 else 0.0

    def _maybe_refresh(self, now: float) -> None:
        # Non-blocking, as a RACE GUARD only. The thing that actually keeps
        # the other callers off the network call is the schedule being written
        # BEFORE the fetch a few lines down — by the time a second caller gets
        # here, ``_next_refresh_at`` is already in the future and it returns at
        # the re-check. This acquire covers the narrow window where a caller
        # passed the outer check before that write landed. Said out loud
        # because the two are NOT one invariant enforced twice: mutating this
        # line alone changes nothing observable, and a future reader should
        # know which of the two is load-bearing.
        if not self._refresh_lock.acquire(blocking=False):
            return
        try:
            with self._state_lock:
                # Re-check: another thread may have just refreshed.
                if 0.0 <= self._next_refresh_at and now < self._next_refresh_at:
                    return
                # Schedule the next attempt BEFORE fetching, so a failing
                # control plane is retried on the TTL rather than on every
                # single call.
                self._next_refresh_at = now + self.ttl_s
            # OUTSIDE the state lock. This is the network call.
            try:
                body = self._fetch()
            except Exception:
                with self._state_lock:
                    self._snap = replace(self._snap, last_refresh_ok=False)
                return
            with self._state_lock:
                self._absorb(body, now)
        finally:
            self._refresh_lock.release()

    def _absorb(self, body: dict[str, Any], now: float) -> None:
        """Fold a successful response into the cache. Called under
        ``_state_lock``."""
        from . import endpoint_kwargs

        self._last_ok_at = now
        hint = body.get("ttl_s") if isinstance(body, dict) else None
        if isinstance(hint, (int, float)) and not isinstance(hint, bool) and hint > 0:
            self._server_ttl = float(hint)
            self._next_refresh_at = now + self.ttl_s

        kw = endpoint_kwargs(body, api_key=self._api_key_arg)
        if not kw and self._last_identity is not None:
            # The control plane answered, and answered with nothing deployed.
            # That is NOT a failed refresh, and it is also not a reason to tear
            # a working client down: "nothing deployed" means keep your current
            # configuration, exactly as the one-shot function documents.
            # Withdrawal is expressed by deploying the base, not by an absence.
            self._snap = replace(self._snap, last_refresh_ok=True)
            return

        identity = (kw.get("base_url", ""), kw.get("api_key", ""),
                    kw.get("model", ""))
        # The first ACTIONABLE answer is a change, because nothing was handed
        # out before it — without that, the documented `if snap.changed:` usage
        # would never build a client at all. An unusable first answer is not a
        # change: there is nothing for the caller to rebuild against.
        actionable = bool(identity[0] and identity[1])
        changed = identity != self._last_identity and (
            actionable or self._last_identity is not None)
        self._last_identity = identity
        self._snap = PolicySnapshot(
            base_url=identity[0], api_key=identity[1], model=identity[2],
            changed=changed, last_refresh_ok=True,
        )
        if changed and self._on_change is not None:
            try:
                self._on_change(self._snap)
            except Exception:
                pass

    def _fetch(self) -> dict[str, Any]:
        if self._fetch_override is not None:
            return self._fetch_override()
        if self._client is None:
            import os

            from . import ENV_API_KEY, ENV_CONTROL_PLANE_URL
            from ._http import ControlPlaneClient

            url = self._url_arg or os.environ.get(ENV_CONTROL_PLANE_URL)
            key = self._api_key_arg or os.environ.get(ENV_API_KEY)
            if not url or not key:
                return {}
            # max_attempts=1: see REFRESH_TIMEOUT_S. The budget has to be a
            # TOTAL, and a retry ladder inside a "bounded" refresh is how a
            # 1s promise becomes 3s plus backoff.
            self._client = ControlPlaneClient(
                base_url=url, api_key=key,
                timeout=REFRESH_TIMEOUT_S, max_attempts=1,
            )
        from .transport import AttachTransport

        return AttachTransport(self._client, agent_id=self.agent_id).policy_current()


def policy_source(
    *,
    agent_id: str,
    ttl_s: float = 300.0,
    control_plane_url: str | None = None,
    api_key: str | None = None,
    on_change: Callable[[PolicySnapshot], None] | None = None,
) -> PolicySource:
    """A :class:`PolicySource` for this agent. See that class and the module
    docstring; ``ttl_s`` is a default the server may override and the
    :data:`MIN_TTL_S` floor may not."""
    return PolicySource(
        agent_id=agent_id, ttl_s=ttl_s, control_plane_url=control_plane_url,
        api_key=api_key, on_change=on_change,
    )
