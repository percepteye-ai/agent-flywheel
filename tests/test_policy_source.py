"""The handle is the transport for ROLLBACK, and that ordering is the design.

``current_policy()`` is a single GET and the documented usage builds a client
from it once, so the policy a production agent runs is the one that was current
when the process started. Promotion arriving late is an inconvenience. A
regressed candidate that can only be withdrawn by redeploying the customer's
production agent is an incident, and one whose remedy sits in someone else's
deploy pipeline.

NO TEST HERE CALLS ``sleep``. This suite has been bitten twice by wall-clock
assertions — a heartbeat test counting beats inside a 2.5s window, and its
sibling with the same flaw — and a TTL feature is nothing BUT timing, so the
clock is injected from the first commit rather than retrofitted.
"""
from __future__ import annotations

import dataclasses
import threading

import pytest

from agent_flywheel.policy import (
    JITTER_FRACTION,
    MIN_TTL_S,
    PolicySnapshot,
    PolicySource,
)


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


#: The agent's gateway URL at the control plane's own origin (the handles
#: below are built with ``control_plane_url="https://cp.example"``): the one
#: kind of endpoint the flywheel key a ``PERCEPTEYE_API_KEY`` answer names may
#: open. These tests are about the cache, not the credential.
GATEWAY = "https://cp.example/llm/v1/o/org-1/agents/agent-1/v1"


def _body(base_url=GATEWAY, model="m1", key_env="PERCEPTEYE_API_KEY"):
    return {"endpoint": {"base_url": base_url, "model": model,
                         "api_key_env": key_env}}


def _source(fetches, *, clock=None, ttl_s=300.0, on_change=None, jitter=0.0):
    """A handle over a scripted fetch. ``fetches`` entries are bodies or
    exceptions; the last one repeats once exhausted."""
    calls = {"n": 0}

    def _fetch():
        i = min(calls["n"], len(fetches) - 1)
        calls["n"] += 1
        item = fetches[i]
        if isinstance(item, Exception):
            raise item
        return item

    src = PolicySource(
        agent_id="agent-1", ttl_s=ttl_s, api_key="k",
        control_plane_url="https://cp.example", on_change=on_change,
        _now=clock or _Clock(), _fetch=_fetch, _rand=lambda lo, hi: jitter,
    )
    return src, calls


# ── caching ──────────────────────────────────────────────────────────


def test_the_first_call_fetches_and_later_calls_inside_the_TTL_do_not():
    clock = _Clock()
    src, calls = _source([_body()], clock=clock)

    first = src.current()
    assert first.base_url == GATEWAY
    assert calls["n"] == 1

    clock.advance(MIN_TTL_S)          # still inside a 300s TTL
    for _ in range(5):
        src.current()
    assert calls["n"] == 1, "a cached read must cost no network call at all"


def test_the_TTL_elapsing_costs_exactly_one_refresh():
    clock = _Clock()
    src, calls = _source([_body()], clock=clock)
    src.current()
    clock.advance(301.0)
    src.current()
    src.current()
    assert calls["n"] == 2


def test_concurrent_callers_do_not_stampede():
    """Fifty concurrent turns cost ONE request, and the other forty-nine are
    served from cache WITHOUT waiting for it.

    Asserting the request count alone is not enough: a blocking lock also
    produces one request, because the losers re-check the schedule once they
    get in. The property that matters is that they never queue behind someone
    else's network call, so this asserts they RETURN while the refresh is
    still in flight — no sleep, no timing window, just an Event the fetch is
    parked on."""
    clock = _Clock()
    in_fetch = threading.Event()
    release = threading.Event()
    calls = {"n": 0}
    returned = threading.Semaphore(0)

    def _fetch():
        calls["n"] += 1
        in_fetch.set()
        release.wait(timeout=5)   # hold the refresher inside the lock
        return _body()

    src = PolicySource(
        agent_id="a", ttl_s=300.0, api_key="k",
        control_plane_url="https://cp.example",
        _now=clock, _fetch=_fetch, _rand=lambda lo, hi: 0.0,
    )

    def _turn():
        src.current()
        returned.release()

    first = threading.Thread(target=_turn)
    first.start()
    assert in_fetch.wait(timeout=5), "the refresher never reached the fetch"

    others = [threading.Thread(target=_turn) for _ in range(49)]
    for t in others:
        t.start()
    for i in range(49):
        assert returned.acquire(timeout=5), (
            f"only {i} of 49 concurrent callers were served while a refresh "
            f"was in flight — the rest are queued behind it"
        )

    release.set()
    first.join(timeout=5)
    for t in others:
        t.join(timeout=5)
    assert calls["n"] == 1


# ── the change edge ──────────────────────────────────────────────────


def test_the_FIRST_actionable_answer_is_a_change():
    """Without this the documented `if snap.changed:` usage would never build a
    client at all — nothing was handed out before it, so it differs."""
    src, _ = _source([_body()])
    assert src.current().changed is True


def test_an_unusable_first_answer_is_NOT_a_change():
    """There is nothing for the caller to rebuild against, and firing the edge
    would have them construct a client with an empty base URL."""
    src, _ = _source([{}])
    snap = src.current()
    assert snap.changed is False
    assert not snap


def test_a_new_endpoint_reports_changed_EXACTLY_once():
    clock = _Clock()
    src, _ = _source([_body(model="m1"), _body(model="m2")], clock=clock)
    src.current()                      # first answer, edge consumed
    clock.advance(301.0)
    assert src.current().changed is True
    assert src.current().changed is False, (
        "the edge is consumed by whoever observes it — a second caller must "
        "not rebuild a client that is already correct"
    )


def test_the_same_endpoint_is_not_a_change():
    clock = _Clock()
    src, _ = _source([_body(), _body()], clock=clock)
    src.current()
    clock.advance(301.0)
    assert src.current().changed is False


def test_on_change_fires_with_the_new_snapshot():
    seen: list[PolicySnapshot] = []
    clock = _Clock()
    src, _ = _source([_body(model="m1"), _body(model="m2")],
                     clock=clock, on_change=seen.append)
    src.current()
    clock.advance(301.0)
    src.current()
    assert [s.model for s in seen] == ["m1", "m2"]


def test_a_raising_observer_never_breaks_serving():
    clock = _Clock()

    def _boom(_snap):
        raise RuntimeError("customer callback blew up")

    src, _ = _source([_body()], clock=clock, on_change=_boom)
    assert src.current().base_url == GATEWAY


# ── last-good retention: the behaviour the whole design turns on ─────


def test_a_failed_refresh_keeps_the_last_good_policy():
    """The naive reading of "fail open" returns the empty answer meaning "keep
    your current configuration" — but a handle that returns empty after having
    HAD a policy makes `changed` fire and the caller rebuild against the base
    model. A control-plane blip would silently demote production from the
    trained model back to the untrained one."""
    clock = _Clock()
    src, _ = _source([_body(model="m1"), RuntimeError("control plane down")],
                     clock=clock)
    src.current()
    clock.advance(301.0)
    snap = src.current()
    assert snap.model == "m1"
    assert snap.changed is False
    assert snap.last_refresh_ok is False


def test_staleness_accumulates_visibly_while_refreshes_fail():
    clock = _Clock()
    src, _ = _source([_body(), RuntimeError("down")], clock=clock)
    assert src.current().stale_for_s == 0.0
    clock.advance(301.0)
    src.current()                       # first failure
    clock.advance(100.0)
    assert src.current().stale_for_s == pytest.approx(101.0)


def test_a_failing_control_plane_is_retried_on_the_TTL_not_every_call():
    clock = _Clock()
    src, calls = _source([RuntimeError("down")], clock=clock)
    for _ in range(10):
        src.current()
    assert calls["n"] == 1
    clock.advance(301.0)
    src.current()
    assert calls["n"] == 2


def test_NOTHING_DEPLOYED_does_not_withdraw_a_working_policy():
    """An absent endpoint means "keep your current configuration", exactly as
    the one-shot function documents. Withdrawal is expressed by deploying the
    base, not by an absence."""
    clock = _Clock()
    src, _ = _source([_body(model="m1"), {}], clock=clock)
    src.current()
    clock.advance(301.0)
    snap = src.current()
    assert snap.model == "m1"
    assert snap.changed is False
    assert snap.last_refresh_ok is True   # the control plane ANSWERED


def test_an_unreachable_control_plane_on_the_FIRST_call_never_raises():
    src, _ = _source([RuntimeError("no route to host")])
    snap = src.current()
    assert not snap
    assert snap.as_kwargs() == {}
    assert snap.changed is False


# ── TTL: floor, server hint, jitter ──────────────────────────────────


def test_the_floor_beats_a_reckless_caller():
    src, _ = _source([_body()], ttl_s=1.0)
    assert src.ttl_s == MIN_TTL_S


def test_a_server_hint_overrides_the_caller_default():
    clock = _Clock()
    src, _ = _source([{**_body(), "ttl_s": 90.0}], clock=clock)
    src.current()
    assert src.ttl_s == pytest.approx(90.0)


def test_the_floor_overrides_a_server_hint_too():
    """A fleet is one misconfigured hint away from a polling storm aimed at our
    own control plane, and the client is the wrong place to discover that."""
    clock = _Clock()
    src, _ = _source([{**_body(), "ttl_s": 0.5}], clock=clock)
    src.current()
    assert src.ttl_s == MIN_TTL_S


@pytest.mark.parametrize("hint", ["soon", 0, -5, True, None])
def test_a_junk_server_hint_is_ignored(hint):
    clock = _Clock()
    src, _ = _source([{**_body(), "ttl_s": hint}], clock=clock)
    src.current()
    assert src.ttl_s == pytest.approx(300.0)


def test_jitter_is_applied_and_bounded():
    src_lo, _ = _source([_body()], jitter=-JITTER_FRACTION)
    src_hi, _ = _source([_body()], jitter=+JITTER_FRACTION)
    assert src_lo.ttl_s == pytest.approx(300.0 * 0.9)
    assert src_hi.ttl_s == pytest.approx(300.0 * 1.1)
    assert src_lo.ttl_s != src_hi.ttl_s


def test_jitter_is_decided_ONCE_per_handle():
    """Re-jittering per refresh smears one handle across the window; the actual
    problem is separating handles from EACH OTHER, so a handle's cadence must
    be stable."""
    clock = _Clock()
    src, _ = _source([_body()], clock=clock, jitter=0.07)
    before = src.ttl_s
    src.current()
    clock.advance(1000.0)
    src.current()
    assert src.ttl_s == before


# ── immutability ─────────────────────────────────────────────────────


def test_a_snapshot_held_across_a_refresh_is_unchanged():
    """A caller holding a snapshot across an in-flight request must be
    unaffected by a refresh completing underneath them."""
    clock = _Clock()
    src, _ = _source([_body(model="m1"), _body(model="m2")], clock=clock)
    held = src.current()
    clock.advance(301.0)
    src.current()
    assert held.model == "m1"


def test_a_snapshot_cannot_be_mutated():
    src, _ = _source([_body()])
    snap = src.current()
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.model = "something else"   # type: ignore[misc]


def test_as_kwargs_matches_apply_current_policy_s_shape():
    src, _ = _source([_body()])
    kw = src.current().as_kwargs()
    assert set(kw) == {"base_url", "api_key", "model"}


# ── the boundaries the design refuses to cross ───────────────────────


def test_the_handle_has_no_set_env():
    """Mutating the process environment once, at a moment the caller chose, is
    a side effect they asked for. Doing it on a timer is that same global
    mutation landing under whatever else the process is doing."""
    import inspect

    from agent_flywheel import policy as ps

    assert "set_env" not in inspect.signature(ps.policy_source).parameters
    assert "set_env" not in inspect.signature(ps.PolicySource.__init__).parameters


def test_the_handle_never_chooses_an_arm():
    """The agent is TOLD its arm and never picks one; that is what keeps
    results attributable. No splitting, no fallback picking, no dice."""
    import inspect

    from agent_flywheel import policy as ps

    src = inspect.getsource(ps)
    assert "random.choice" not in src
    assert "random.random" not in src
    # random is imported for JITTER ONLY.
    assert src.count("random.") == 1 and "random.uniform" in src


def test_close_is_idempotent_and_the_handle_is_a_context_manager():
    src, _ = _source([_body()])
    with src:
        src.current()
    src.close()
    src.close()


def test_the_handle_and_the_one_shot_function_share_ONE_unpack():
    """A caller who unpacks the policy body slightly differently is a caller
    sampling from something other than what was trained — and that applies at
    least as much between two of OUR surfaces as between two of theirs."""
    import inspect

    import agent_flywheel as fw
    from agent_flywheel import policy as ps

    assert "endpoint_kwargs" in inspect.getsource(fw.apply_current_policy)
    assert "endpoint_kwargs" in inspect.getsource(ps.PolicySource._absorb)


def test_an_endpoint_with_no_usable_credential_is_NOTHING_TO_APPLY():
    """Returning it would produce 401s at the caller's first request rather
    than a legible empty answer here — and the handle must agree with the
    one-shot function about that, since they share the unpack."""
    import agent_flywheel as fw

    body = {"endpoint": {"base_url": "https://a/v1", "model": "m",
                         "api_key_env": "NOT_SET_ANYWHERE_XYZ"}}
    assert fw.endpoint_kwargs(body, api_key=None) == {}

    src = PolicySource(
        agent_id="a", api_key=None, control_plane_url="https://cp",
        _now=_Clock(), _fetch=lambda: body, _rand=lambda lo, hi: 0.0,
    )
    assert not src.current()


# ── two questions, two answers ───────────────────────────────────────


def test_the_snapshot_carries_the_control_plane_s_own_generation():
    src, _ = _source([{**_body(), "serving_generation": "abc123"}])
    assert src.current().serving_generation == "abc123"


def test_a_PROMPT_only_change_moves_the_generation_but_not_changed():
    """The two answer different questions. `changed` asks "must I rebuild my
    client?" and a prompt change does not require that — the snapshot does not
    even carry the prompt. `serving_generation` asks "is what I booted with
    still current?", which a fleet uses to cycle a replica. Folding them
    together would either cycle replicas for endpoint-only changes or miss
    prompt changes entirely."""
    clock = _Clock()
    src, _ = _source([
        {**_body(model="m1"), "serving_generation": "gen-1"},
        {**_body(model="m1"), "serving_generation": "gen-2"},   # prompt moved
    ], clock=clock)
    src.current()
    clock.advance(301.0)
    snap = src.current()
    assert snap.changed is False
    assert snap.serving_generation == "gen-2"


def test_an_endpoint_change_moves_BOTH():
    clock = _Clock()
    src, _ = _source([
        {**_body(model="m1"), "serving_generation": "gen-1"},
        {**_body(model="m2"), "serving_generation": "gen-2"},
    ], clock=clock)
    src.current()
    clock.advance(301.0)
    snap = src.current()
    assert snap.changed is True
    assert snap.serving_generation == "gen-2"


def test_a_missing_generation_is_empty_not_invented():
    src, _ = _source([_body()])
    assert src.current().serving_generation == ""
