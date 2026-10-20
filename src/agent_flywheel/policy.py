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

WHAT ARRIVES. A model PerceptEye serves arrives as the agent's one constant URL
on the control plane's gateway, with ``api_key_env: "PERCEPTEYE_API_KEY"``, so
:func:`~agent_flywheel.endpoint_kwargs` opens it with the agent's own flywheel
key; the serving host's credential stays in the control plane and its address
never reaches the agent. That is the ONLY endpoint the flywheel key opens: one
the control plane names ``PERCEPTEYE_API_KEY`` for, at the control plane's own
origin. A provider endpoint the customer owns is still dialled directly, with
that provider's key -- the one the control plane sent, or the variable it
named -- and never with the flywheel key. Because the gateway sends each
request to the model delivered at that moment, a promotion or a rollback
between served models takes effect without a rebuild -- the answer still moves
``model``, so ``changed`` still says so. A withdrawal is the one change the
caller must act on; see below.

USAGE — call it per turn::

    policy = fw.policy_source(agent_id="my-coding-agent", ttl_s=300)
    client, model = OpenAI(), "my-model"      # your own configuration

    while True:                       # per turn, or per request
        snap = policy.current()
        if snap.changed:              # rebuild ONLY on an actual change
            if snap:                  # a served policy
                client = OpenAI(base_url=snap.base_url, api_key=snap.api_key)
                model = snap.model
            elif snap.withdrawn:      # taken back: your own again
                client, model = OpenAI(), "my-model"
        ...

Per turn is the natural unit for an agent, and it is the recommendation. A
caller who calls ``current()`` once has re-created the problem this class
exists to solve.

A WITHDRAWAL IS A CHANGE TO AN EMPTY SNAPSHOT. When the control plane moves
the agent back to a base arm that has nothing to serve -- no champion, no
upstream -- it answers ``withdrawn: true`` (:func:`is_withdrawn`), and the
handle reverts to the snapshot it was constructed with: empty, meaning your
own configuration, reported once as ``changed``, with ``withdrawn`` set and
one log line naming it. Any other answer without an endpoint changes nothing.

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

import logging
import random
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any
from urllib.parse import urlsplit

__all__ = [
    "ARCHIVE_REF_PREFIXES",
    "PolicySnapshot",
    "PolicySource",
    "is_archive_ref",
    "is_withdrawn",
    "missing_credentials_cause",
    "policy_source",
]


def missing_credentials_cause(
    key: str | None,
    *,
    agent_id: str | None = None,
    control_plane_url: str | None = None,
) -> str | None:
    """Why the control plane cannot be asked, or ``None`` when it can.

    Only the key can be missing. The URL has a default
    (:func:`agent_flywheel.control_plane.resolve_control_plane_url`, the Node
    SDK's rule), so it is resolved, never refused; this used to name an unset
    ``PERCEPTEYE_CONTROL_PLANE_URL`` too, because Python alone had no default.

    ONE answer to a question asked wherever the control plane is dialled:
    ``current_policy``, ``current_prompt`` (and ``apply_current_policy``
    through the first), :meth:`PolicySource._fetch`,
    :func:`~agent_flywheel.attach`, and ``agent-flywheel status``. All
    resolve the same pair, and all used to fail the same way: ``if not url or
    not key: return {}``, with nothing said. That empty answer is
    indistinguishable from the control plane's own "nothing is deployed for
    this agent", which is the answer
    callers are documented to read as "keep your current configuration" -- so a
    production agent with an unset variable ran the untrained model forever
    while every health signal the SDK offers said fine.

    Returns the CAUSE only, never the consequence: each call site knows what it
    could not do and says so itself. A refusal that names the wrong subsystem
    costs more than one that says nothing.

    A caller that knows which agent it speaks for passes ``agent_id``: a key
    can also be STORED for that agent (``agent-flywheel login``,
    :func:`~agent_flywheel.credentials.resolve_api_key`), so the cause then
    says that none is, or that the one stored has expired, and names the
    command that gets one.
    """
    from . import ENV_API_KEY
    from .control_plane import GET_A_KEY

    if key:
        return None
    cause = f"{ENV_API_KEY} is not set"
    if agent_id is None:
        # The remedy rides with the cause: it is the same for every call
        # site, and this line is where a developer who installed the SDK and
        # never signed up first learns that a key exists at all.
        return f"{cause} ({GET_A_KEY})"
    from .control_plane import resolve_control_plane_url
    from .credentials import login_hint, stored_key_refusal

    stored = stored_key_refusal(resolve_control_plane_url(control_plane_url), agent_id)
    return (
        f"{cause} and {stored or f'no key is stored for agent {agent_id!r}'} "
        f"({login_hint(agent_id)}, or {GET_A_KEY})"
    )


#: Prefixes that mean STORED ARTIFACT, not sampling address.
#:
#: The control plane already refuses to hand one of these out as a ``model``,
#: for exactly the reason this refuses to use one: ``model="gs://bucket/run/
#: iter_0"`` is a bucket path, and it 404s on every completion. Re-asserting it
#: here is not a duplicate gate -- it is the same argument
#: :data:`~agent_flywheel.prompt.SERVABLE_PROMPT_STATUSES` is
#: re-asserted on, applied to the other half of the pair: the only thing
#: standing between a control plane that one day widens the filter and a
#: customer's agent pointed at an address that answers nothing.
#:
#: THE SAME SET THE NODE SDK REFUSES ON (``ARCHIVE_REF_PREFIXES`` in
#: ``src/policy.js``). Both halves of an improvement are approved together and
#: both SDKs are supposed to refuse the same bundles; this half was enforced in
#: Node and absent here, while the Python README stated it as a live guarantee.
ARCHIVE_REF_PREFIXES = ("gs://", "s3://", "file://", "/")


def is_archive_ref(model: str) -> bool:
    """Is this ``model`` a stored artifact rather than a servable name?"""
    return model.startswith(ARCHIVE_REF_PREFIXES)


def is_withdrawn(policy: Mapping[str, Any] | None) -> bool:
    """Has the control plane WITHDRAWN the policy it was serving this agent?

    ``withdrawn: true`` (with ``endpoint: null``) is its answer when an agent
    is moved back to its base arm and there is nothing to serve there: no
    champion has been promoted and no upstream is registered. It means run
    your OWN model again, the configuration the agent had before any policy
    reached it.

    That is not what ``endpoint: null`` means on its own. Without the flag a
    null endpoint says nothing new is deployed, keep what you run -- and
    reading the two the same way left an agent pinned on a candidate a human
    had already rolled back, because "keep what you run" was the candidate.

    Strictly ``True``, and asked before any endpoint beside it, because the
    Node SDK's ``modelDecision`` asks ``withdrawn === true`` first and the two
    SDKs refuse the same answers.
    """
    return isinstance(policy, Mapping) and policy.get("withdrawn") is True


def _origin(url: object) -> str | None:
    """``scheme://host[:port]`` of an absolute http(s) URL, or ``None``.

    The question :func:`~agent_flywheel.endpoint_kwargs` asks before it hands
    the flywheel key to an endpoint: is it at the control plane's own origin,
    the one address the key already goes to on every call? The Node SDK's
    ``controlPlaneOrigin`` answers it the same way: the host lowercased, a
    default port dropped, userinfo and path ignored -- so
    ``https://Launch.example:443/x`` and ``https://launch.example/y`` are one
    origin, while ``http://`` for ``https://``, another port, or a suffix like
    ``launch.example.evil`` is another.
    """
    try:
        parts = urlsplit(str(url or "").strip())
        scheme = parts.scheme.lower()
        host = parts.hostname
        port = parts.port
    except ValueError:            # a port that is not a number
        return None
    if scheme not in ("http", "https") or not host:
        return None
    if ":" in host:               # IPv6, which urlsplit returns unbracketed
        host = f"[{host}]"
    default = 443 if scheme == "https" else 80
    return f"{scheme}://{host}" + (f":{port}" if port and port != default else "")


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

#: Emits into whatever logging the customer already configured; installs
#: nothing. See the note on the package logger in ``__init__``.
_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PolicySnapshot:
    """What to run, and how much to trust it. Immutable BY DESIGN.

    Production agents are concurrent. A caller holding a snapshot across an
    in-flight request must be unaffected by a refresh completing underneath
    them, and frozen-and-returned-by-value gives that for free — nothing swaps
    under anyone.
    """

    #: OpenAI-compatible base URL. Empty when the control plane has nothing
    #: deployed for this agent, has never answered, or has withdrawn the
    #: policy it served -- in every case, run your own configuration.
    base_url: str = ""
    api_key: str = ""
    model: str = ""

    #: True on the ONE snapshot that first reports a change, false otherwise.
    #: It is a per-handle EDGE, so build one client per handle: two threads
    #: sharing a handle will not both see the edge, by construction.
    changed: bool = False

    #: 0.0 while refreshes are succeeding; grows while they are failing.
    #:
    #: MEASURED FROM THE LAST SUCCESS, OR FROM WHEN THE HANDLE WAS BUILT if
    #: there has never been one. Both cases are the same fact — "this long
    #: since the control plane last told us anything" — and reporting 0.0 for
    #: the second one made the whole-fleet outage the quietest state there is:
    #: a firewall rule that blocked the control plane from first boot left
    #: every replica reporting a perfectly fresh policy it had never received.
    #:
    #: Alert on this together with ``last_refresh_ok``, not instead of it: this
    #: one says how long, that one says whether the last attempt worked.
    stale_for_s: float = 0.0

    #: Whether the LAST refresh attempt succeeded. False after a failed one --
    #: including one that never reached the network because no control-plane
    #: URL or key is configured. The snapshot still stands either way.
    last_refresh_ok: bool = True

    #: The control plane's own answer to "is what I booted with still
    #: current?", empty when it did not send one.
    #:
    #: DELIBERATELY NOT what ``changed`` is computed from, and the two answer
    #: different questions. ``changed`` asks "must I rebuild my client?" and
    #: moves only when the endpoint does. This moves when ANYTHING about what
    #: to run moves — including the system prompt, which this snapshot does
    #: not carry and a client rebuild would not apply. A fleet holds this
    #: value and cycles a replica when it differs; a turn loop reads
    #: ``changed``. Folding them together would either cycle replicas for
    #: endpoint-only changes or miss prompt changes entirely.
    serving_generation: str = ""

    #: True when this snapshot is empty because the control plane WITHDREW
    #: the policy it served (:func:`is_withdrawn`), until it serves one again:
    #: run your own configuration, by instruction. With ``changed`` it is the
    #: edge to rebuild your own client on. Its own field so a caller asks the
    #: flag rather than inferring a withdrawal from an empty snapshot, which
    #: also describes "nothing was ever delivered".
    withdrawn: bool = False

    def __bool__(self) -> bool:
        """Truthy only when there is something to act on — same meaning as
        ``apply_current_policy()``'s ``{}``. A falsy snapshot is never a
        client to build: on ``changed`` with ``withdrawn``, go back to your
        own configuration."""
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
    accumulates visibly instead. The one answer that does take a served policy
    away is the control plane's explicit ``withdrawn: true``, never an absence
    or a failure.

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
        #: The key ``_fetch`` resolved (argument, variable or stored), once it
        #: has: the endpoint half is handed the same one.
        self._resolved_key: str | None = None
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
        #: When this handle was built, so `stale_for_s` can be measured
        #: from something even when no refresh has ever succeeded.
        self._created_at = _now()
        #: Said once per handle, not once per TTL: a missing variable does
        #: not fix itself, and a line per request would be the noisiest
        #: possible way to report the quietest possible failure.
        self._warned_no_credentials = False
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
        # NEVER-SUCCEEDED IS THE STALEST STATE THERE IS, not the freshest. This
        # returned 0.0 whenever `_last_ok_at` was None, which is the state after
        # any number of failed refreshes if none has ever landed -- so a fleet
        # that could not reach the control plane from its first boot reported
        # `stale_for_s = 0.0` forever and the documented alert never fired.
        since = self._last_ok_at if self._last_ok_at is not None else self._created_at
        overdue = now - since - self.ttl_s
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

        withdrawn = is_withdrawn(body)
        # `{}` for a withdrawal too: endpoint_kwargs refuses one before any
        # endpoint beside it, so the identity below is the empty triple.
        # This handle's key and URL are the FLYWHEEL key and where it goes,
        # never a credential for the endpoint: see `_endpoint_credential`.
        kw = endpoint_kwargs(
            body, flywheel_key=self._resolved_key or self._api_key_arg,
            control_plane_url=self._url_arg,
        )
        if not kw and not withdrawn and self._last_identity is not None:
            # The control plane answered, and answered with nothing deployed.
            # That is NOT a failed refresh, and it is also not a reason to tear
            # a working client down: "nothing deployed" means keep your current
            # configuration, exactly as the one-shot function documents.
            # WITHDRAWAL IS NOT THIS BRANCH. It is said with `withdrawn: true`
            # and falls through below to the empty triple, the snapshot this
            # handle held when it was constructed -- your own configuration --
            # reported as a change. An agent rolled back to a base with no
            # champion used to land here and stay on the candidate.
            # The generation still moves: the control plane's answer DID
            # change, even though the part this snapshot carries did not, and
            # a fleet watching it should still see that.
            self._snap = replace(
                self._snap, last_refresh_ok=True,
                serving_generation=(
                    str(body.get("serving_generation") or "")
                    if isinstance(body, dict) else ""
                ) or self._snap.serving_generation,
            )
            return

        generation = body.get("serving_generation") if isinstance(body, dict) else None
        generation = str(generation) if isinstance(generation, str) else ""
        identity = (kw.get("base_url", ""), kw.get("api_key", ""),
                    kw.get("model", ""))
        # The first ACTIONABLE answer is a change, because nothing was handed
        # out before it — without that, the documented `if snap.changed:` usage
        # would never build a client at all. An unusable first answer is not a
        # change: there is nothing for the caller to rebuild against.
        # A WITHDRAWAL is a change by the second clause once a policy was
        # handed out -- the empty triple differs from it, and the caller must
        # go back to its own configuration -- and not before one was, when
        # the caller is still on that configuration.
        actionable = bool(identity[0] and identity[1])
        changed = identity != self._last_identity and (
            actionable or self._last_identity is not None)
        self._last_identity = identity
        self._snap = PolicySnapshot(
            base_url=identity[0], api_key=identity[1], model=identity[2],
            changed=changed, last_refresh_ok=True,
            serving_generation=generation, withdrawn=withdrawn,
        )
        if changed and withdrawn:
            # SAID ONCE, on the edge: the empty snapshot is otherwise the same
            # value as "never delivered", and an operator watching production
            # change models needs the cause.
            _log.warning(
                "agent-flywheel: the control plane WITHDREW the policy it served "
                "agent_id=%r (%s), so this agent goes back to its own "
                "configuration.", self.agent_id,
                (body.get("reason") if isinstance(body, dict) else None)
                or "withdrawn=true",
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
            from ._http import ControlPlaneClient
            from .control_plane import resolve_control_plane_url
            from .credentials import resolve_api_key
            from .errors import ConfigurationError

            url = resolve_control_plane_url(self._url_arg)
            resolved = resolve_api_key(
                self._api_key_arg, agent_id=self.agent_id, control_plane_url=url,
            )
            key = resolved.value if resolved else ""
            # The endpoint half (``_absorb``) reads the SAME key: one
            # resolution per handle, so the policy read and the served
            # endpoint cannot be handed different credentials.
            self._resolved_key = key or None
            cause = missing_credentials_cause(
                key, agent_id=self.agent_id, control_plane_url=self._url_arg,
            )
            if cause:
                if not self._warned_no_credentials:
                    self._warned_no_credentials = True
                    _log.warning(
                        "agent-flywheel: policy_source(agent_id=%r) will never "
                        "refresh: %s, so the control plane is never asked. "
                        "This agent keeps the model it is on, and no "
                        "promotion -- or rollback -- can reach it.",
                        self.agent_id, cause,
                    )
                # RAISED, not returned as `{}`. An empty body is absorbed as a
                # SUCCESSFUL refresh that found nothing deployed, so this path
                # used to set `last_refresh_ok=True` having never opened a
                # socket -- the one health signal the SDK offers, reporting
                # fine about a handle that cannot work.
                raise ConfigurationError(
                    f"agent-flywheel: {cause}, so the control plane cannot be "
                    f"asked for a policy")
            # max_attempts=1: see REFRESH_TIMEOUT_S. The budget has to be a
            # TOTAL, and a retry ladder inside a "bounded" refresh is how a
            # 1s promise becomes 3s plus backoff.
            self._client = ControlPlaneClient(
                base_url=url, api_key=key,
                timeout=REFRESH_TIMEOUT_S, max_attempts=1,
            )
        from .transport import AttachTransport

        # THE RAISING READ. `policy_current()` fails open to `{}`, which
        # `_absorb` takes as a successful "nothing deployed": an unreachable
        # control plane then reported last_refresh_ok=True and stale_for_s=0.0
        # for as long as it was down -- the documented alert could never fire.
        return AttachTransport(
            self._client, agent_id=self.agent_id).policy_current_or_raise()


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
