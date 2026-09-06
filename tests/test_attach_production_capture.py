"""``attach()`` — capture in the process that serves users.

``serve()`` does training work. ``attach()`` records what your agent already did, from turns the
customer's agent is already serving. Same adapters, same tri-state, same
trajectory file — and, critically, NO rerouting: pointing production traffic at
a new endpoint is ``apply_current_policy``'s job and the customer's decision.

The two tests that matter most here are the ones asserting what ``attach`` does
NOT do: it does not re-point anything, and it does not fail a turn when the
control plane is gone.
"""
from __future__ import annotations

from contextlib import contextmanager

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


# ── the rollout/production boundary ─────────────────────────────────────────
# `current_rollout_id()` answers "which trajectory am I appending to", and in a
# production turn that is the TURN's id -- so branching on it took the rollout
# branch in production. `current_capture()` is the function that distinguishes
# them, and these pin all three states.


def test_current_capture_names_a_production_turn_a_turn(tmp_path, monkeypatch):
    monkeypatch.delenv("PERCEPTEYE_ROLLOUT_ID", raising=False)
    with fw.attach(agent_id="a", trajectory_root=str(tmp_path),
                   register=False) as px, px.turn() as tid:
        assert fw.current_capture() == ("turn", tid)


def test_current_capture_is_none_outside_any_scope(tmp_path, monkeypatch):
    monkeypatch.delenv("PERCEPTEYE_ROLLOUT_ID", raising=False)
    with fw.attach(agent_id="a", trajectory_root=str(tmp_path),
                   register=False) as px:
        with px.turn():
            pass
        assert fw.current_capture() is None


def test_current_capture_names_a_rollout_a_rollout(monkeypatch):
    """The env var is written only by build_child_env, only for a rollout."""
    monkeypatch.setenv("PERCEPTEYE_ROLLOUT_ID", "rl_9")
    assert fw.current_capture() == ("rollout", "rl_9")


def test_current_rollout_id_still_returns_the_turn_id(tmp_path, monkeypatch):
    """Deliberate, and load-bearing.

    `record_tool_call` reads this to find its destination. Narrowing it to
    rollouts would silently disable production capture -- a naming bug traded
    for a data-loss bug. The boundary lives in `current_capture()` instead.
    """
    monkeypatch.delenv("PERCEPTEYE_ROLLOUT_ID", raising=False)
    with fw.attach(agent_id="a", trajectory_root=str(tmp_path),
                   register=False) as px, px.turn() as tid:
        assert fw.current_rollout_id() == tid


def test_a_production_turn_is_not_reported_as_a_rollout(tmp_path, monkeypatch):
    """The branch a customer actually writes."""
    monkeypatch.delenv("PERCEPTEYE_ROLLOUT_ID", raising=False)
    with fw.attach(agent_id="a", trajectory_root=str(tmp_path),
                   register=False) as px, px.turn():
        kind, _ = fw.current_capture()
        assert kind != "rollout"


def test_current_capture_is_exported():
    assert "current_capture" in fw.__all__


# ── what attach() tells the control plane it is ─────────────────────────────
# Nothing asserted the shape of this payload before, which is how `mode` came
# to carry the same value from both surfaces.


class _RecordingTransport:
    """Captures the registration payload; policy_current always answers."""

    def __init__(self, *, register_raises: Exception | None = None):
        self.registered: dict | None = None
        self._raises = register_raises
        self.policy_calls = 0

    def register(self, spec):
        self.registered = dict(spec)
        if self._raises is not None:
            raise self._raises
        return {}

    def policy_current(self):
        self.policy_calls += 1
        return {"arm": "candidate-3", "endpoint": {"base_url": "https://gw"}}


@contextmanager
def _attach_with(monkeypatch, tmp_path, transport):
    """attach()'s control-plane block against a fake transport, always closed.

    A context manager rather than a plain call because an Attachment that
    outlives a FAILING assertion leaves its adapters bound, and the next test
    to bind trips the suite's own "a previous test left a binding active"
    guard. One wrong regex in here cascaded into 17 errors in another file.
    """
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_x")
    monkeypatch.setattr(
        "percepteye_agent_flywheel.transport.AttachTransport",
        lambda client, agent_id: transport,
    )
    monkeypatch.setattr(
        "percepteye_agent_flywheel._http.ControlPlaneClient.close",
        lambda self: None,
    )
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    try:
        yield px
    finally:
        px.close()


def test_attach_registers_as_production_not_as_a_rollout(monkeypatch, tmp_path):
    """The field exists to tell a training donor from a production reader.

    Both surfaces used to send "rollout", so it answered neither.
    """
    tr = _RecordingTransport()
    with _attach_with(monkeypatch, tmp_path, tr):
        pass
    assert tr.registered is not None, "attach() did not register"
    assert tr.registered["mode"] == "production"
    assert tr.registered["entrypoint"] == "attach"


def test_a_refused_registration_still_yields_the_policy(monkeypatch, tmp_path):
    """The failure mode this change had to avoid.

    One `except` used to cover the register AND the policy read, so anything
    intake disliked about the payload silently demoted production to the
    untrained model. `policy_current` is a bare GET scoped by API key and does
    not depend on a registration having landed, so it must still run.
    """
    tr = _RecordingTransport(register_raises=RuntimeError("HTTP 422: bad mode"))
    with pytest.warns(RuntimeWarning, match="refused this registration"):
        with _attach_with(monkeypatch, tmp_path, tr) as px:
            assert tr.policy_calls == 1, "the policy read was skipped"
            assert px.policy.get("arm") == "candidate-3"


def test_a_refused_registration_is_not_silent(monkeypatch, tmp_path):
    """An empty policy already means two different benign things. A third
    cause that looks identical is one the customer cannot act on.

    Matched on the AGENT ID and the reassurance, not on a named cause. This
    used to match "mode='production'" -- a cause that stopped existing the day
    the control plane learned to accept that mode, at which point the test
    would have been pinning a sentence the code should no longer say.
    """
    tr = _RecordingTransport(register_raises=RuntimeError("HTTP 422"))
    with pytest.warns(RuntimeWarning, match="refused this registration") as rec:
        with _attach_with(monkeypatch, tmp_path, tr):
            pass
    text = str(rec[-1].message)
    assert "agent_id='support'" in text
    assert "unaffected" in text, "the warning must say what still works"


def test_a_refused_registration_still_does_not_fail_the_attach(
    monkeypatch, tmp_path
):
    """Fail-open is the whole posture of this function and it survives."""
    tr = _RecordingTransport(register_raises=RuntimeError("boom"))
    with pytest.warns(RuntimeWarning):
        with _attach_with(monkeypatch, tmp_path, tr) as px, px.turn() as tid:
            assert tid
