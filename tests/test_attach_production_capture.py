"""``attach()`` — capture in the process that serves users.

``serve()`` donates capacity. ``attach()`` donates evidence, from turns the
customer's agent is already serving. Same adapters, same tri-state, same
trajectory file — and, critically, NO rerouting: pointing production traffic at
a new endpoint is ``apply_current_policy``'s job and the customer's decision.

The two tests that matter most here are the ones asserting what ``attach`` does
NOT do: it does not re-point anything, and it does not fail a turn when the
control plane is gone.
"""
from __future__ import annotations

import pytest

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel import record_tool_call
from percepteye_agent_flywheel.errors import ConfigurationError

DEAD = "http://127.0.0.1:9"          # discard port: refuses, fast


@pytest.fixture
def attached(tmp_path):
    """Every attachment made here is unbound afterwards.

    Binding is PROCESS-GLOBAL — it patches the framework — so an attachment
    left open leaks into every test that runs after it. That is not a quirk of
    the tests; it is why `close()` exists and why `Attachment` is a context
    manager. The suite found this the moment these tests were added: seventeen
    adapter tests errored at setup because a binding from here was still live.
    """
    made = []

    def _make(**kw):
        kw.setdefault("register", False)
        px = fw.attach(agent_id="support-agent",
                       trajectory_root=kw.pop("root", str(tmp_path)), **kw)
        made.append(px)
        return px

    yield _make
    for px in made:
        px.close()


# ── the turn scope ──────────────────────────────────────────────────────────

def test_tool_calls_inside_a_turn_land_in_that_turn(attached, tmp_path):
    px = attached()
    with px.turn() as tid:
        record_tool_call("search", {"q": "x"}, outcome="ok", status_code=200)
    calls = px.read(tid)
    assert calls is not None
    assert [(c.name, c.outcome) for c in calls] == [("search", "ok")]


def test_two_turns_do_not_share_a_trajectory(attached, tmp_path):
    """Separation is structural: each turn is its own directory."""
    px = attached()
    with px.turn() as a:
        record_tool_call("first", {}, outcome="ok", status_code=200)
    with px.turn() as b:
        record_tool_call("second", {}, outcome="failed", status_code=500)
    assert [c.name for c in px.read(a) or []] == ["first"]
    assert [c.name for c in px.read(b) or []] == ["second"]


def test_a_turn_that_records_nothing_reads_as_none_not_empty(attached, tmp_path):
    """The absence≠zero rule, one layer up. A turn with no tool calls has not
    asserted that zero occurred."""
    px = attached()
    with px.turn() as tid:
        pass
    assert px.read(tid) is None


def test_the_caller_may_supply_the_turn_id(attached, tmp_path):
    """The join key for an outcome that arrives later than the turn does."""
    px = attached()
    with px.turn(turn_id="t_from_my_app") as tid:
        assert tid == "t_from_my_app"
        record_tool_call("x", {}, outcome="unknown", status_code=None)
    assert len(px.read("t_from_my_app") or []) == 1


def test_the_unknown_outcome_survives_the_production_path(attached, tmp_path):
    """Same contract as everywhere else: unknown is never rounded up, and it
    carries no status even if one is offered."""
    px = attached()
    with px.turn() as tid:
        record_tool_call("charge", {}, outcome="unknown", status_code=200)
    c = (px.read(tid) or [])[0]
    assert (c.outcome, c.status_code) == ("unknown", None)


def test_a_conversation_id_is_recorded_when_given(attached, tmp_path):
    px = attached()
    with px.turn(conversation_id="conv-42") as tid:
        pass
    assert (tmp_path / tid / "conversation_id").read_text() == "conv-42"


# ── what attach must NOT do ─────────────────────────────────────────────────

def test_attach_does_not_reroute_anything(attached, tmp_path):
    """The load-bearing negative. Adapters are bound with NO endpoint, so
    nothing production sends can be re-pointed. Routing belongs to
    apply_current_policy, where the customer chooses it explicitly."""
    before = {k: v for k, v in __import__("os").environ.items()
              if "BASE_URL" in k or "API_KEY" in k}
    px = attached()
    after = {k: v for k, v in __import__("os").environ.items()
             if "BASE_URL" in k or "API_KEY" in k}
    assert before == after
    px.close()


def test_a_dead_control_plane_does_not_stop_the_turn(attached, tmp_path):
    """An unreachable Percepteye must never stop an agent serving its users."""
    px = attached(register=True, control_plane_url=DEAD, api_key="k")
    assert px.policy == {}
    with px.turn() as tid:
        record_tool_call("search", {}, outcome="ok", status_code=200)
    assert len(px.read(tid) or []) == 1


def test_a_turn_still_runs_when_capture_has_nowhere_to_write(attached, tmp_path):
    """A capture bug costs the observation and nothing else."""
    px = attached()
    px.trajectory_root = "/proc/percepteye-cannot-write-here"
    ran = False
    with px.turn():
        ran = True
    assert ran


# ── configuration mistakes surface in front of whoever made them ────────────

def test_a_missing_agent_id_raises(attached, tmp_path):
    with pytest.raises(ConfigurationError):
        fw.attach(agent_id="", trajectory_root=str(tmp_path), register=False)


def test_a_missing_trajectory_root_raises(attached, tmp_path):
    """Not defaulted, because nothing deletes these yet and where a customer's
    data lands is not ours to assume."""
    with pytest.raises(ConfigurationError):
        fw.attach(agent_id="a", trajectory_root="", register=False)


# ── lifecycle ───────────────────────────────────────────────────────────────

def test_close_is_idempotent(attached, tmp_path):
    px = attached()
    px.close()
    px.close()


def test_it_works_as_a_context_manager(attached, tmp_path):
    with attached() as px:
        with px.turn() as tid:
            record_tool_call("x", {}, outcome="ok", status_code=200)
        assert len(px.read(tid) or []) == 1


def test_adapters_bound_is_reported_not_implied(attached, tmp_path):
    """`()` is a real answer. An agent on no supported framework captures
    nothing automatically, and saying so beats implying coverage."""
    px = attached(adapters=[])
    assert px.adapters_bound == ()
