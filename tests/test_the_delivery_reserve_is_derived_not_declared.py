"""The time held back to deliver a result must be the time delivering costs.

``REPORT_MARGIN_S = 10.0`` was a literal with no relationship to the work it
reserved for. The client that spends that time defaults to a 30s read timeout
and four attempts, so ONE report's own worst case is ~123.5s -- twelve times
the reserve -- and ``_collect_artifacts`` spends another such envelope PER FILE
before the report is even attempted. The producer of the deadline and the
consumer of the time never referenced each other.

These tests assert the two halves are now ONE number: what the runner reserves
moves when the transport's cost moves, and what the HTTP loop spends is bounded
by the same value. A test asserting ``child_s == 765.5`` would pass equally
against a hardcoded 765.5, so every test here VARIES the inputs.

They also cover the teardown that sits between the two, which had no timeout on
either of its call sites, and the durability rule: the mechanism lives in core
and reaches every harness without an adapter copying anything.
"""
from __future__ import annotations

import pathlib
import socket
import stat
import sys
import threading
import time
import warnings
from typing import ClassVar

import pytest

import agent_flywheel.runner as runner_mod
from agent_flywheel import RolloutRequest
from agent_flywheel._http import (
    ControlPlaneClient,
    default_delivery_envelope_s,
    delivery_envelope_s,
)
from agent_flywheel.contract import RolloutOutput
from agent_flywheel.errors import LeaseLost, TransportError
from agent_flywheel.runner import (
    CALL_CEILING_FRACTION,
    REPORT_MARGIN_S,
    RolloutRunner,
    episode_plan,
    harness_call_ceiling_s,
    run_rollout_as_command,
)
from agent_flywheel.transport import AttachTransport

SRC = pathlib.Path(runner_mod.__file__).parent


def _req(tmp_path, rid="rl_1", seconds=894.0) -> RolloutRequest:
    return RolloutRequest(
        rollout_id=rid, turn_input="x",
        llm_url=f"https://gw.invalid/llm/v1/rollouts/{rid}", api_key="k",
        artifacts_dir=str(tmp_path), deadline_ts=time.time() + seconds,
        lease_token="lt",
    )


class _T:
    """A transport that declares a cost. The number under test."""

    accepts_delivery_deadline = True
    agent_id = "a"
    lease_ms = 1_800_000

    def __init__(self, envelope_s: float | None = None):
        if envelope_s is not None:
            self.delivery_envelope_s = envelope_s
        self.reported: list[str] = []

    def claim(self, n):
        return [], 10

    def heartbeat(self, rollout, *, progress=None):
        return None

    def report(self, rollout, output, *, deadline=None):
        self.reported.append(rollout.rollout_id)
        return {}

    def abandon(self, rollout, *, reason, detail=""):
        return None


def _script(tmp_path, body: str, name="a.sh") -> str:
    p = tmp_path / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _kill_deadline(tmp_path, transport, budget_s: float) -> float:
    """What the runner ACTUALLY kills the child at, read from the real path."""
    runner = RolloutRunner(
        ["sh", _script(tmp_path, "echo hi\n")], transport=transport,
        input_shape="text", artifacts_dir=str(tmp_path),
        rollout_timeout_s=900.0,
    )
    seen: dict = {}
    original = runner_mod.run_rollout_as_command

    def spy(cmd, rollout, *, timeout_s, **kw):
        seen["timeout_s"] = timeout_s
        return original(cmd, rollout, timeout_s=timeout_s, **kw)

    runner_mod.run_rollout_as_command = spy
    try:
        runner.run_one(_req(tmp_path, seconds=budget_s))
    finally:
        runner_mod.run_rollout_as_command = original
    return seen["timeout_s"]


# ── one number, both ends ──────────────────────────────────────────────────


def test_the_child_deadline_moves_with_what_delivery_actually_costs(tmp_path):
    """The assertion that a hardcoded number cannot pass.

    Pinning ``child_s == 765.5`` would hold just as well against a literal, so
    this varies the transport's declared cost and requires the kill deadline to
    follow it. That is the whole difference between a derivation and a guess.
    """
    cheap = _kill_deadline(tmp_path, _T(envelope_s=20.0), 894.0)
    dear = _kill_deadline(tmp_path, _T(envelope_s=200.0), 894.0)

    assert cheap > dear, (
        f"the agent got the same {cheap:.0f}s whether delivery costs 20s or "
        f"200s -- the reserve is not derived from anything"
    )
    assert cheap - dear == pytest.approx(180.0, abs=1.0), (
        "the extra reserve did not come out of the agent's window"
    )
    # And it is the SDK's own arithmetic, not a coincidence.
    assert dear == pytest.approx(
        episode_plan(894.0, rollout_timeout_s=900.0, envelope_s=200.0).child_s,
        abs=1.0,
    )


def test_the_reserve_is_the_clients_own_worst_case_end_to_end(tmp_path):
    """From ``ControlPlaneClient``'s settings through to the child's deadline.

    Every serve-path client takes the module defaults, so this is the number
    production actually runs with -- and varying the CLIENT (not a fake) must
    move the child's window, which is what proves the chain has no gap in it.
    """
    def envelope_for(timeout, attempts):
        client = ControlPlaneClient(
            "https://cp.invalid", "pehk_x", timeout=timeout, max_attempts=attempts
        )
        return AttachTransport(client, "a").delivery_envelope_s

    assert envelope_for(30.0, 4) == default_delivery_envelope_s() == 123.5
    assert envelope_for(10.0, 2) == delivery_envelope_s(10.0, 2) == 20.5
    assert envelope_for(60.0, 4) > envelope_for(30.0, 4), (
        "a slower client must reserve more, or the reserve is not its cost"
    )

    class _Real(_T):
        def __init__(self, timeout, attempts):
            super().__init__()
            self._client = ControlPlaneClient(
                "https://cp.invalid", "pehk_x",
                timeout=timeout, max_attempts=attempts,
            )

        @property
        def delivery_envelope_s(self):
            return self._client.delivery_envelope_s

    patient = _kill_deadline(tmp_path, _Real(30.0, 4), 894.0)
    hasty = _kill_deadline(tmp_path, _Real(5.0, 2), 894.0)
    assert hasty > patient, (
        f"a client that gives up quickly must hand the agent MORE of the "
        f"episode ({hasty:.0f}s vs {patient:.0f}s)"
    )


def test_the_reserve_can_never_swallow_a_short_episode(tmp_path):
    """This deployment mints 300s episodes; the envelope alone is 123.5s.

    An honest reserve that took 41% of a short episode would be its own
    failure, so the fraction cap binds there and the floor binds nowhere.
    """
    short = episode_plan(300.0, rollout_timeout_s=900.0, envelope_s=123.5)
    assert short.delivery_s == 75.0, short
    assert short.child_s == 220.0, short

    tiny = episode_plan(20.0, rollout_timeout_s=900.0, envelope_s=123.5)
    assert tiny.child_s >= 1.0 and tiny.delivery_s > 0.0, tiny

    # The floor still applies to a transport that declares nothing at all.
    assert episode_plan(894.0, rollout_timeout_s=900.0).delivery_s == REPORT_MARGIN_S
    plain = _kill_deadline(tmp_path, _T(), 894.0)          # no envelope declared
    assert plain == pytest.approx(894.0 - REPORT_MARGIN_S - 5.0, abs=1.0), (
        "a transport that predates the envelope must keep the old floor margin"
    )


def test_one_report_cannot_outlast_the_time_reserved_for_it():
    """AGAINST A REAL SOCKET that accepts the connection and never answers.

    This is the shape that started the investigation, one layer up: a gateway
    that accepts and goes silent. ``report`` passed no per-call timeout at all
    -- unlike ``claim``, one line above it -- so the client's full four-attempt,
    ~123.5s envelope ran against a rollout whose reserve was ten seconds.
    """
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    port = server.getsockname()[1]
    held: list[socket.socket] = []
    stop = threading.Event()

    def black_hole():
        server.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except (TimeoutError, OSError):
                continue
            held.append(conn)          # accepted, never answered

    thread = threading.Thread(target=black_hole, daemon=True)
    thread.start()
    try:
        client = ControlPlaneClient(
            f"http://127.0.0.1:{port}", "pehk_x", timeout=30.0, max_attempts=4
        )
        transport = AttachTransport(client, "a")
        rollout = RolloutRequest(
            rollout_id="rl_1", turn_input="x", llm_url="http://x.invalid",
            api_key="k", artifacts_dir="/tmp", deadline_ts=time.time() + 60,
            lease_token="lt",
        )
        started = time.time()
        with pytest.raises(TransportError):
            transport.report(
                rollout, RolloutOutput(final_text="done"),
                deadline=started + 3.0,
            )
        elapsed = time.time() - started
    finally:
        stop.set()
        thread.join(timeout=3)
        for conn in held:
            conn.close()
        server.close()

    assert elapsed < 20.0, (
        f"one report spent {elapsed:.0f}s against a 3s reserve; unbounded it "
        f"is {default_delivery_envelope_s():.0f}s"
    )


def test_no_route_can_be_parked_forever_by_a_gateway_that_goes_silent():
    """MEASURED 2026-09-16, and the single worst thing found in this pass.

    ``request`` passed the caller's ``timeout`` straight to httpx, and httpx
    reads an explicit ``timeout=None`` as NO TIMEOUT AT ALL -- its "use the
    client's value" is a sentinel object, not ``None``. Every route except
    ``claim`` passed ``None``: report, heartbeat, abandon, register_artifact,
    health, register. Against a control plane that accepts the connection and
    never answers, each of those blocked FOREVER.

    A parked heartbeat is the whole incident shape in one call: beats are
    issued serially from ONE thread, so every in-flight lease stops being
    renewed, and the reaper then writes ``abandoned`` / ``deadline_exceeded``
    over all of them with no report ever landing.

    Verified here on ``heartbeat``, which names no timeout of its own and so
    can only be bounded by the client's -- the exact case that was broken.
    """
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    port = server.getsockname()[1]
    held: list[socket.socket] = []
    stop = threading.Event()

    def black_hole():
        server.settimeout(0.5)
        while not stop.is_set():
            try:
                conn, _ = server.accept()
            except (TimeoutError, OSError):
                continue
            held.append(conn)

    thread = threading.Thread(target=black_hole, daemon=True)
    thread.start()
    outcome: dict = {}

    def beat():
        client = ControlPlaneClient(
            f"http://127.0.0.1:{port}", "pehk_x", timeout=2.0, max_attempts=1
        )
        started = time.time()
        try:
            AttachTransport(client, "a").heartbeat(
                RolloutRequest(
                    rollout_id="rl_1", turn_input="x", llm_url="http://x.invalid",
                    api_key="k", artifacts_dir="/tmp",
                    deadline_ts=time.time() + 60, lease_token="lt",
                )
            )
        except TransportError:
            outcome["elapsed"] = time.time() - started

    worker = threading.Thread(target=beat, daemon=True)
    worker.start()
    worker.join(timeout=30.0)
    alive = worker.is_alive()
    stop.set()
    thread.join(timeout=3)
    for conn in held:
        conn.close()
    server.close()

    assert not alive, (
        "a heartbeat against a silent gateway never returned; the one "
        "heartbeat thread is now stopped and EVERY lease this process holds "
        "will be reaped"
    )
    assert outcome["elapsed"] < 20.0, outcome


def test_a_server_supplied_backoff_cannot_park_a_report_for_an_hour():
    """``Retry-After`` was honoured uncapped.

    A proxy -- not necessarily the control plane -- answering ``Retry-After:
    3600`` made one report sleep an hour, three times, inside a path the runner
    believed cost seconds.
    """
    import httpx

    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503, headers={"Retry-After": "3600"}, json={})
        return httpx.Response(200, json={"ok": True})

    client = ControlPlaneClient(
        "https://cp.invalid", "pehk_x",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    started = time.time()
    assert client.request("POST", "/x", json_body={}) == {"ok": True}
    assert time.time() - started < 30.0, "an hour-long Retry-After was honoured"
    assert attempts["n"] == 3


# ── the teardown between the two ───────────────────────────────────────────


_ESCAPES_THE_GROUP = """
import os, sys, time
pid = os.fork()
if pid == 0:
    os.setsid()               # a NEW process group: the kill below misses it
    time.sleep(120)           # while still holding the inherited pipes
    os._exit(0)
sys.stderr.write("POST https://gw.invalid/v1/chat/completions: no response\\n")
sys.stderr.flush()
time.sleep(120)
"""


def test_a_descendant_that_escapes_the_kill_cannot_hang_the_runner(tmp_path):
    """REPRODUCED before the fix: ``communicate()`` after the kill had no timeout.

    ``_kill_tree`` kills one process group, so a descendant that called
    ``setsid`` survives; if it inherited the write end of stdout the follow-up
    ``communicate()`` never returns -- inside the window reserved for
    delivering the result. Measured: >15s and in principle forever, against
    0.001s for an ordinary in-group tree.

    Run on a thread with a join timeout so the unfixed behaviour FAILS rather
    than hanging the suite.
    """
    child = tmp_path / "escapes.py"
    child.write_text(_ESCAPES_THE_GROUP)
    result: dict = {}

    def run():
        result["out"] = run_rollout_as_command(
            [sys.executable, str(child)], _req(tmp_path, seconds=60.0),
            trajectory_dir=str(tmp_path), timeout_s=1.0, teardown_s=2.0,
            env={"PATH": "/usr/bin:/bin"},
        )

    worker = threading.Thread(target=run, daemon=True)
    started = time.time()
    worker.start()
    worker.join(timeout=25.0)
    elapsed = time.time() - started

    assert not worker.is_alive(), (
        f"the runner never came back after killing the child ({elapsed:.0f}s "
        f"and counting) -- an escaped descendant still holds its pipes"
    )
    out = result["out"]
    assert out.exclusion_reason == "deadline_exceeded"
    detail = out.exclusion_detail or ""
    assert "could not be read" in detail and "process group" in detail, (
        f"an unread stderr must not read as a silent agent: {detail!r}"
    )


def test_an_ordinary_child_is_still_drained_and_still_says_why(tmp_path):
    """The control for the test above: bounding teardown must not cost evidence.

    An in-group tree drains immediately, so the child's own account of why it
    stopped moving -- the half that was being thrown away before 2026-09-15 --
    still reaches the report.
    """
    out = run_rollout_as_command(
        ["sh", _script(tmp_path, 'echo "gateway.invalid: no response" >&2\nsleep 60\n')],
        _req(tmp_path, seconds=60.0), trajectory_dir=str(tmp_path),
        timeout_s=1.0, teardown_s=5.0, env={"PATH": "/usr/bin:/bin"},
    )
    assert out.exclusion_reason == "deadline_exceeded"
    assert "gateway.invalid: no response" in (out.exclusion_detail or "")
    assert "could not be read" not in (out.exclusion_detail or "")


def test_losing_the_lease_kills_a_COMMAND_child_too(tmp_path):
    """``_attach_proc`` was called only on the Python-entrypoint path.

    So ``_InFlight.proc`` was ``None`` for prime-agent, goose, incident-lens
    and every other BYOH command agent -- the entire class this product serves
    -- and the heartbeat thread's lease-lost kill was unreachable code for
    them. The comment claiming the guarantee sat on the Python path's Popen.
    """
    class LoseIt(_T):
        def __init__(self):
            super().__init__(envelope_s=5.0)
            self._first = True

        def claim(self, n):
            if self._first:
                self._first = False
                return [_req(tmp_path, "rl_1", seconds=3600.0)], 10
            self.runner.stop()
            return [], 10

        def heartbeat(self, rollout, *, progress=None):
            raise LeaseLost("gone")

    tr = LoseIt()
    runner = RolloutRunner(
        ["sh", _script(tmp_path, "sleep 120\n")], transport=tr,
        input_shape="text", artifacts_dir=str(tmp_path), rollout_timeout_s=120.0,
    )
    tr.runner = runner
    original = RolloutRunner._beat_interval_for
    RolloutRunner._beat_interval_for = lambda self, r: 1.0
    started = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runner.serve_forever(max_rollouts=1)
    finally:
        RolloutRunner._beat_interval_for = original
    elapsed = time.time() - started

    assert elapsed < 30.0, (
        f"the command child ran on for {elapsed:.0f}s against a revoked "
        f"credential; the lease-lost kill never reached it"
    )


# ── durability: one place, and no harness names in it ──────────────────────


def _enclosing_callers(path: pathlib.Path, attr_or_name: str) -> list[str]:
    """Every function in ``path`` that calls ``attr_or_name``, by name.

    Read from the AST rather than by grepping lines, so a call split across
    two lines counts once and a mention in a comment or docstring counts not
    at all.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    callers: list[str] = []

    class Walk(ast.NodeVisitor):
        def __init__(self):
            self.stack: list[str] = []

        def visit_FunctionDef(self, node):
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node):
            func = node.func
            name = (
                func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name == attr_or_name:
                callers.append(self.stack[-1] if self.stack else "<module>")
            self.generic_visit(node)

    Walk().visit(tree)
    return callers


def test_the_budget_arithmetic_has_exactly_one_consumer():
    """A second copy of this arithmetic is how a reserve and its cost drift.

    ``run_one`` is the ONLY place an episode budget becomes a child deadline,
    so both child paths -- and therefore every adapter, including ones that do
    not exist yet -- inherit the same division by construction. The one other
    caller is ``harness_call_ceiling_s``, the adapter-facing translation
    helper: that is an adapter ASKING core, which is the supported direction.
    """
    callers = _enclosing_callers(SRC / "runner.py", "episode_plan")
    assert sorted(callers) == ["harness_call_ceiling_s", "run_one"], (
        f"the episode budget is divided somewhere new: {callers}"
    )
    assert set(_enclosing_callers(SRC / "runner.py", "run_one")) == {
        "_serve_one_inner"
    }, "a second path runs a rollout, and it would divide the budget its own way"


def test_the_report_leaves_through_exactly_one_door():
    """Four terminal paths and one report is how an outcome got lost.

    Every ``transport.report`` call must sit inside ``_report``, which is
    reached only from ``_deliver_record`` -- so spooling, redelivery, the named
    refusal and the brake cannot be bypassed by a future path that simply
    returns, which is precisely what the three dropped paths did.
    """
    callers = set(_enclosing_callers(SRC / "runner.py", "report"))
    assert callers == {"_report"}, (
        f"transport.report is called from {sorted(callers)}; every one of "
        f"those is a path that can drop an outcome"
    )
    assert set(_enclosing_callers(SRC / "runner.py", "_report")) == {
        "_deliver_record"
    }


def test_core_names_no_harness():
    """The mechanism must not know what it is protecting.

    A new harness type -- goose, incident-lens, a customer's own binary --
    inherits the reserve, the bounded teardown, the spool and the redelivery
    without writing a line, because none of that code reads a harness name, a
    binary path, or a settings file. Checked over IDENTIFIERS and IMPORTS, not
    over prose: a comment may name the harness the defect was found on, and
    several deliberately do.
    """
    import ast
    import tokenize

    forbidden = ("prime", "goose", "incident", "netgenius", "openai",
                 "anthropic", "timeoutms", "settings")
    offenders: list[str] = []
    for path in (SRC / "runner.py", SRC / "transport.py", SRC / "_http.py"):
        with tokenize.open(path) as fh:
            for tok in tokenize.generate_tokens(fh.readline):
                if tok.type != tokenize.NAME:
                    continue
                low = tok.string.lower()
                if any(word in low for word in forbidden):
                    offenders.append(f"{path.name}:{tok.start[0]}: {tok.string}")
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", "") or ""
                names = " ".join(a.name for a in node.names)
                if "adapters" in f"{module} {names}".lower():
                    offenders.append(
                        f"{path.name}:{node.lineno}: imports from .adapters"
                    )
    assert not offenders, (
        f"core has learned a harness's vocabulary; a new agent type will not "
        f"inherit this protection for free: {offenders}"
    )


def test_a_brand_new_command_agent_inherits_the_whole_mechanism(tmp_path):
    """No adapter, no snapshot, no settings file: plain argv, like a BYOH agent.

    The durability claim, executed rather than argued. This agent type did not
    exist when the mechanism was written and needed nothing written for it.
    """
    delivered: list[dict] = []

    class Flaky(_T):
        def __init__(self):
            super().__init__(envelope_s=5.0)
            self._batches = [[_req(tmp_path, "rl_new", seconds=894.0)]]
            self._fail = True

        def claim(self, n):
            if self._batches:
                return self._batches.pop(0), 10
            self.runner.stop()
            return [], 10

        def report(self, rollout, output, *, deadline=None):
            if self._fail:
                self._fail = False
                raise TransportError("reset")
            delivered.append(output.to_wire())
            return {}

    tr = Flaky()
    runner = RolloutRunner(
        ["sh", _script(tmp_path, 'echo "novel harness output"\n')],
        transport=tr, input_shape="text", artifacts_dir=str(tmp_path),
        rollout_timeout_s=900.0,
    )
    tr.runner = runner
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        count = runner.serve_forever(max_rollouts=1)

    assert count == 1 and delivered, "a plain argv agent did not get redelivery"
    assert delivered[0]["final_text"] == "novel harness output"


# ── the translation layer, and the number it translates ────────────────────


def test_the_prime_ceiling_is_derived_from_the_runner_not_typed_into_prime():
    """Requirement: the adapter setting must DERIVE from the central value.

    Asserted as an IDENTITY against the runner's own function rather than as a
    literal, so replacing the derivation with the number it currently produces
    fails here.
    """
    import json as _json

    from agent_flywheel.adapters import prime_agent

    sealed = _json.loads(prime_agent._FIXED_SETTINGS)["retry"]["provider"]
    expected_ms = int(
        harness_call_ceiling_s(prime_agent.PRIME_REFERENCE_EPISODE_BUDGET_S)
    ) * 1000
    assert sealed["timeoutMs"] == expected_ms, (
        "the sealed provider ceiling no longer follows the runner's arithmetic"
    )
    # And moving the runner's share moves what an adapter would seal -- the
    # property a hardcoded 300000 could never have.
    with_more_room = harness_call_ceiling_s(
        prime_agent.PRIME_REFERENCE_EPISODE_BUDGET_S, envelope_s=10.0
    )
    assert with_more_room > harness_call_ceiling_s(
        prime_agent.PRIME_REFERENCE_EPISODE_BUDGET_S, envelope_s=400.0
    ), "the harness ceiling does not follow the delivery reserve"
    assert 0.0 < CALL_CEILING_FRACTION < 0.5, (
        "at half or more, one wedged call can consume the episode and still "
        "look like an agent doing slow work"
    )


def test_an_inert_harness_ceiling_is_announced_on_the_first_rollout(tmp_path):
    """A floor written in a comment protects nobody.

    At the 300s budget this deployment has actually minted, a 306s ceiling
    cannot fire inside the episode -- and nothing at runtime said so. Declared
    optionally by the adapter, checked in core, so any harness that declares a
    ceiling gets the same warning and one that declares none is untouched.
    """
    class _Child:
        environment: ClassVar[dict] = {}
        discovery_environment: ClassVar[dict] = {}

    class Declaring:
        """A minimal adapter that declares a ceiling and nothing else."""

        adapter_name = "Declaring"
        worst_case_call_s = 306.0
        execution_components: ClassVar[dict] = {}
        execution_identity_complete = False
        diagnostic_fingerprint: ClassVar[dict] = {}

        def prepare_child(self, label="rollout"):
            return _Child()

        def attest_child(self, child):
            return None

        def release_child(self, child):
            return None

    def warnings_from_a_real_rollout(budget_s: float) -> list[str]:
        # Driven through run_one, NOT by calling the check directly: a check
        # with no call site would otherwise pass this test, which is the exact
        # producer/consumer failure this whole change exists to stop repeating.
        runner = RolloutRunner(
            ["sh", _script(tmp_path, "echo hi\n")],
            transport=_T(envelope_s=123.5), input_shape="text",
            artifacts_dir=str(tmp_path), rollout_timeout_s=900.0,
            execution_snapshot=Declaring(),
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            runner.run_one(_req(tmp_path, seconds=budget_s))
        return [str(w.message) for w in caught]

    short = warnings_from_a_real_rollout(300.0)
    assert any("306s" in m and "cannot fire" in m for m in short), short

    assert warnings_from_a_real_rollout(894.0) == [], (
        "the episode the ceiling was sized for must not be reported as inert"
    )
