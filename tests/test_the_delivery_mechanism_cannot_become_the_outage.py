"""The fix must not become the next cause of the incident it fixed.

THE SHAPE UNDER TEST. The outage being closed was: the flywheel registers, it
claims, every rollout comes back ``abandoned`` / ``deadline_exceeded`` with
zero inference calls, and nothing anywhere says why. The mechanism that closes
it -- an outbox, a drain at the top of the claim loop, a refusal brake, a
heartbeat thread that keeps every lease alive -- introduced four fresh ways to
produce that IDENTICAL signature, three of them from a single bad input:

  * one spooled file the SDK cannot rebuild, raising out of the drain that runs
    as the first statement of the claim loop, so the process never reaches a
    claim -- and the file survives to do it again on the next start;
  * one unexpected exception type out of a customer transport's ``heartbeat``,
    unwinding the ONE thread that renews EVERY lease this process holds;
  * three lease losses in a row tripping a brake built for configuration
    faults, stopping the fleet claiming;
  * and, on that same path, deleting the only copy of work the agent did.

Each test here asserts the property an operator cares about -- THE FLYWHEEL
STILL CLAIMS, AND WORK THAT WAS DONE STILL EXISTS -- rather than that a
particular branch was taken.
"""
from __future__ import annotations

import inspect
import json
import pathlib
import stat
import time
import warnings

from test_an_outcome_is_never_silently_dropped import (
    ScriptedTransport,
    _outbox,
    _req,
    _runner,
    _script,
)

import agent_flywheel
from agent_flywheel.errors import LeaseLost, TransportError
from agent_flywheel.runner import DEFAULT_ROLLOUT_TIMEOUT_S

from .repo_files import repo_files

# ── one bad file in the outbox ─────────────────────────────────────────────


def _plant(tmp_path, rollout_id: str, result: dict) -> pathlib.Path:
    """Put a file in the outbox that ``_read_spool`` ACCEPTS.

    The envelope checks -- schema, rollout id, result-is-a-dict -- all pass.
    Only rebuilding the result's CONTENTS fails, which is the case the drain's
    own validation cannot reach and therefore the one that got through.
    """
    root = _outbox(tmp_path)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{rollout_id}.json"
    path.write_text(json.dumps({
        "schema": "percepteye.outbox/v1",
        "rollout_id": rollout_id,
        "lease_token": "lt_x",
        "deadline_ts": time.time() + 600.0,
        "lease_lost": False,
        "first_seen_ts": time.time(),
        "attempts": 0,
        "last_error": None,
        "result": result,
    }))
    return path


def test_one_unrebuildable_spooled_file_cannot_stop_the_flywheel_claiming(tmp_path):
    """A tool call with no ``name`` raises ``KeyError`` out of the rebuild.

    That rebuild used to sit OUTSIDE ``_deliver_record``'s try, so the KeyError
    escaped the record, escaped the drain, and escaped the first statement of
    the claim loop. The flywheel then claimed nothing -- and because the file
    was never settled, every restart died on the same line. A permanent outage
    whose row-level signature is exactly the one being investigated, caused by
    the machinery meant to end it.
    """
    bad = _plant(tmp_path, "rl_bad", {
        "final_text": "x", "tool_calls": [{"outcome": "ok"}],
    })
    tr = ScriptedTransport([[_req(tmp_path, "rl_good")]])
    events: list[tuple[str, dict]] = []
    runner = _runner(
        tmp_path, tr, ["sh", _script(tmp_path, 'echo "the answer"\n')],
        on_event=lambda name, data: events.append((name, data)),
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        delivered = runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_good"], (
        "one unrebuildable file in the outbox stopped the flywheel claiming: "
        f"reported={tr.reported}. This is the outage, reproduced by the fix."
    )
    assert delivered == 1
    assert not bad.exists(), "the poison file was left to brick the next start"
    assert (bad.parent / "rl_bad.json.unreadable").is_file(), (
        "the unrebuildable result was deleted rather than kept for inspection"
    )
    # The rollout id proves the INNERMOST handler settled it. The outer
    # per-file guard in the drain reports an empty id, so this distinguishes
    # "the record settled itself" from "the backstop caught it".
    named = [d for n, d in events if n == "outcome_unrebuildable"]
    assert [d["rollout_id"] for d in named] == ["rl_bad"], (
        f"the record that could not be rebuilt was not named: {named}"
    )
    assert any("rl_bad" in str(w.message) for w in caught), (
        "nothing was said out loud about the result that could not be rebuilt"
    )


# ── the one thread that renews every lease ─────────────────────────────────


def test_a_heartbeat_transport_that_raises_anything_keeps_every_lease_alive(tmp_path):
    """``FlywheelTransport`` is a Protocol, so heartbeat can raise ANYTHING.

    Only ``LeaseLost`` and ``TransportError`` were caught. Any other exception
    unwound ``_heartbeat_loop``, and that thread is not per-rollout: it renews
    EVERY in-flight lease. One ``ValueError`` therefore stopped lease renewal
    for the whole process, which is brief hypothesis (d) and reproduces the
    incident verbatim -- every lease lapses, the reaper writes
    abandoned/deadline_exceeded over all of them, no report is accepted.

    Asserting a SECOND beat is the point: one beat proves only that the thread
    started, and the thread that died had already sent one.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1", seconds=600.0)]],
        heartbeat_raises=ValueError("customer transport bug"),
    )
    tr.lease_ms = 1_000          # beat interval -> the 1.0s floor
    runner = _runner(
        tmp_path, tr,
        ["sh", _script(tmp_path, 'sleep 4\necho "the answer"\n')],
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=1)

    assert len(tr.heartbeats) >= 2, (
        f"the heartbeat thread died on the first unexpected exception "
        f"({len(tr.heartbeats)} beat(s) sent). Every lease this process holds "
        f"is now unrenewed and will be reaped."
    )
    assert tr.reported == ["rl_1"], (
        "the rollout did not survive a heartbeat transport that raises"
    )


# ── a lost lease is not a configuration fault ──────────────────────────────


def test_a_refused_lease_neither_destroys_the_result_nor_stops_the_fleet(tmp_path):
    """410 is per-rollout. The brake is for faults that repeat forever.

    Only an identity-sealed rollout's report route can answer 410 at all, and
    what it refuses is THAT rollout's lease. Two things used to follow, both
    wrong: the spooled result was unlinked -- destroying the only copy of work
    the customer's agent actually did, on precisely the path where a sealed
    rollout's deferred delivery is refused BY CONSTRUCTION -- and the refusal
    counted toward a brake built for a key missing ``rollout.report`` scope. At
    a limit of three, three unlucky rollouts stopped the fleet claiming.
    """
    reqs = [_req(tmp_path, f"rl_{i}") for i in range(1, 5)]
    tr = ScriptedTransport(
        [[r] for r in reqs],
        verdicts=[LeaseLost("lease_lost")] * 4,
    )
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "the answer"\n')])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        runner.serve_forever(max_rollouts=4)

    assert tr.handed_out == ["rl_1", "rl_2", "rl_3", "rl_4"], (
        f"a per-rollout lease loss stopped the fleet claiming: "
        f"{tr.handed_out}. The brake exists for configuration faults."
    )
    kept = sorted(_outbox(tmp_path).glob("*.refused-lease-lost"))
    assert len(kept) == 4, (
        f"the refused results were destroyed rather than kept: {kept}"
    )
    held = json.loads(kept[0].read_text())
    assert held["result"]["final_text"] == "the answer", (
        "what was kept is not the agent's own result"
    )
    assert held["last_error"], "the kept file does not say why it was refused"
    assert any("kept at" in str(w.message) for w in caught), (
        "the operator was not told where the refused result went"
    )


def test_a_configuration_fault_still_stops_the_fleet(tmp_path):
    """The other half of the predicate, or the brake is merely disabled.

    Loosening 410 must not loosen 403. A key missing ``rollout.report`` scope
    refuses every delivery forever, and claiming into that is what turned one
    misconfiguration into "every rollout since 2026-09-10".
    """
    refusal = TransportError("403 Flywheel key is missing scope(s): rollout.report")
    refusal.status_code = 403
    reqs = [_req(tmp_path, f"rl_{i}") for i in range(1, 5)]
    # Refuses for as long as it is asked. A finite list would run out, the
    # drain's next retry of a HELD result would succeed, and that success
    # resets the brake -- which is correct behaviour and would have made this
    # test assert nothing.
    tr = ScriptedTransport([[r] for r in reqs], verdicts=[refusal] * 500)
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "the answer"\n')])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=4)

    assert len(tr.handed_out) < 4, (
        f"the flywheel claimed into a permanent configuration refusal and took "
        f"on all the work anyway: {tr.handed_out}"
    )


# ── one object, not two copies ─────────────────────────────────────────────


def test_serve_defaults_to_the_runners_own_rollout_timeout(tmp_path):
    """``serve()``'s default must BE the runner's constant, not equal it.

    The adapter that seals a provider ceiling into an agent's frozen identity
    derives it from ``DEFAULT_ROLLOUT_TIMEOUT_S``, while every production
    rollout was bounded by a literal ``900.0`` typed into ``serve()``. Two
    copies: change either and the sealed identity is sized against arithmetic
    the runner no longer performs, with nothing comparing them. ``is`` rather
    than ``==`` so a future literal that happens to match still fails.
    """
    default = inspect.signature(
        agent_flywheel.serve
    ).parameters["rollout_timeout_s"].default
    assert default is DEFAULT_ROLLOUT_TIMEOUT_S, (
        f"serve() carries its own copy of the rollout timeout ({default!r}); "
        f"the runner's constant is {DEFAULT_ROLLOUT_TIMEOUT_S!r}"
    )


def _src_modules() -> list[pathlib.Path]:
    return repo_files(pathlib.Path(agent_flywheel.__file__).parent, suffix=".py")


def test_the_rollout_timeout_is_declared_in_exactly_one_place():
    """A SECOND COPY TURNS THIS RED. The test above only watches ``serve``.

    While it stood alone, ``RolloutRunner.__init__`` -- the signature every
    production rollout is actually bounded by -- still spelled ``900.0``, and
    ``child.py`` spelled it a third time as the fallback the child runs on.
    Checking one known site proves that site and no more; the drift this
    change exists to end is the site nobody thought to check.

    So both halves are asserted from the SOURCE, over every module in the
    package: no parameter named ``rollout_timeout_s`` may default to anything
    but the constant by NAME, and the number itself may appear exactly once --
    in the constant's own assignment. Add a copy anywhere, under any parameter
    name, and this fails naming the file and line.
    """
    import ast

    literal_sites: list[str] = []
    declarations: list[str] = []
    copied_defaults: list[str] = []
    for path in _src_modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in (900, 900.0) \
                    and isinstance(node.value, (int, float)) \
                    and not isinstance(node.value, bool):
                literal_sites.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "DEFAULT_ROLLOUT_TIMEOUT_S"
                for t in node.targets
            ):
                declarations.append(f"{path.name}:{node.lineno}")
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                positional = args.posonlyargs + args.args
                # Defaults are right-aligned against the positional args, so
                # the unfilled head is padded to keep the two lists the same
                # length -- which is also why `strict=True` is safe here.
                filled = [None] * (
                    len(positional) - len(args.defaults)
                ) + list(args.defaults)
                pairs = list(zip(positional, filled, strict=True)) + list(
                    zip(args.kwonlyargs, args.kw_defaults, strict=True)
                )
                for arg, default in pairs:
                    if arg.arg != "rollout_timeout_s" or default is None:
                        continue
                    if not (isinstance(default, ast.Name)
                            and default.id == "DEFAULT_ROLLOUT_TIMEOUT_S"):
                        copied_defaults.append(
                            f"{path.name}:{default.lineno} in {node.name}()"
                        )

    assert len(declarations) == 1, (
        f"the rollout timeout is DECLARED in more than one place: {declarations}"
    )
    assert not copied_defaults, (
        f"these rollout_timeout_s defaults are copies of the value rather than "
        f"the constant DEFAULT_ROLLOUT_TIMEOUT_S: {copied_defaults}"
    )
    assert literal_sites == declarations, (
        f"the number 900 appears at {literal_sites}; it may appear only where "
        f"DEFAULT_ROLLOUT_TIMEOUT_S is declared, which is {declarations}"
    )


# ── what the outbox put on disk ────────────────────────────────────────────


def test_the_spooled_results_are_not_readable_by_anyone_else(tmp_path):
    """The outbox is a sibling of the agent's own working directory.

    The child is handed ``artifacts_dir/_percepteye/<rollout_id>`` as its cwd,
    so the outbox is ``../_outbox`` from a shell the customer's untrusted agent
    controls. 0700/0600 does not stop that agent when it runs as this uid
    (nothing at this layer can), but it does stop a co-tenant, a second service
    account, and a permissive default -- and those are free. What the agent
    cannot read there is asserted separately, below, because a file mode was
    never the answer to that half.
    """
    # Transient, and for as long as it is asked: a held result that the exit
    # drain then delivers would leave nothing on disk to inspect.
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1")]],
        verdicts=[TransportError("503 overloaded")] * 500,
    )
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "the answer"\n')])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=1)

    spooled = sorted(_outbox(tmp_path).glob("*.json"))
    assert len(spooled) == 1, f"nothing was held to check: {spooled}"
    assert stat.S_IMODE(spooled[0].stat().st_mode) == 0o600, (
        f"a held result is readable beyond its owner: "
        f"{oct(stat.S_IMODE(spooled[0].stat().st_mode))}"
    )
    assert stat.S_IMODE(_outbox(tmp_path).stat().st_mode) == 0o700, (
        f"the outbox directory is listable beyond its owner: "
        f"{oct(stat.S_IMODE(_outbox(tmp_path).stat().st_mode))}"
    )


def test_no_bearer_credential_is_written_where_the_agent_can_read_it(tmp_path):
    """THE LEASE TOKEN NEVER REACHES DISK. A file mode cannot hold this line.

    ``lease_token`` authorises reporting and abandoning its rollout. The outbox
    is ``../_outbox`` from the cwd the customer's UNTRUSTED agent is handed and
    the agent runs as this uid, so writing the token there disclosed it to the
    party the control plane's fence exists to exclude -- and put it into any
    bundle that swept the artifacts tree. 0600 stops a co-tenant; it does not
    stop the reader this is about.

    Asserted over the BYTES of everything under ``artifacts_dir``, not over the
    one key we happen to remember: a future field carrying the same secret
    under another name must fail here too.

    BOTH DISPOSITIONS IN ONE RUN, because there are two writers. ``rl_1`` is
    RETIRED (a 410 nothing can fix, written back by ``_settle`` and renamed)
    and ``rl_2`` is HELD for retry (a transient 503, rewritten on every
    attempt). A fix that took the token out of the first write and left it in
    the write-back would pass a test that only looked at one of them.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1")], [_req(tmp_path, "rl_2")]],
        verdicts=[LeaseLost("HTTP 410: rollout worker identity or lease lost")]
        + [TransportError("503 overloaded")] * 500,
    )
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "the answer"\n')])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=2)

    held = sorted(_outbox(tmp_path).glob("*.json"))
    retired = sorted(_outbox(tmp_path).glob("*.refused-lease-lost"))
    assert len(held) == 1 and len(retired) == 1, (
        f"the run did not produce one held and one retired record: "
        f"{sorted(p.name for p in _outbox(tmp_path).iterdir())}"
    )
    assert "the answer" in held[0].read_text(), (
        "the held result is not the one under test"
    )

    leaked = [
        (str(p), secret) for p in pathlib.Path(tmp_path).rglob("*")
        for secret in ("lt_rl_1", "lt_rl_2")
        if p.is_file() and secret.encode() in p.read_bytes()
    ]
    assert not leaked, (
        f"a rollout's lease token is readable by the agent it fences out: "
        f"{leaked}"
    )


def test_the_same_process_still_redelivers_with_the_lease_it_holds(tmp_path):
    """The other half: taking the token off disk must not take it off the WIRE.

    A held result is redelivered by the next drain, seconds later, while the
    lease is still live -- and for an identity-sealed rollout that redelivery
    is refused without a matching token. Keeping the token in memory is what
    makes moving it off disk cost nothing real, so the report this process
    finally sends has to carry the one it was claimed with.
    """
    tr = ScriptedTransport(
        [[_req(tmp_path, "rl_1")]], verdicts=[TransportError("503 overloaded")],
    )
    sent: list[str | None] = []
    inner = tr.report

    def _watched(rollout, output, *, deadline=None):
        sent.append(rollout.lease_token)
        return inner(rollout, output, deadline=deadline)

    tr.report = _watched
    runner = _runner(tmp_path, tr, ["sh", _script(tmp_path, 'echo "the answer"\n')])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runner.serve_forever(max_rollouts=1)

    assert tr.reported == ["rl_1"], f"the retry never landed: {tr.reported}"
    assert sent == ["lt_rl_1", "lt_rl_1"], (
        f"the redelivery lost the lease token it was claimed with: {sent}"
    )
