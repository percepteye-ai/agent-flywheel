"""A claimed rollout that produces nothing must say why, and must not be silent.

The failure this file exists for: on 2026-09-15 prime-agent claimed three
rollouts within half a second of their creation and every one of them expired
with ``inference_calls`` empty and no session in the harness's own logs.

BE PRECISE ABOUT WHAT WAS OBSERVED WHERE, because the two are not the same
record and reading them as one is what makes the next investigation long:

* IN PRODUCTION, on those three rows: ``status=abandoned`` with
  ``exclusion_reason=deadline_exceeded``. Only the control plane's reaper
  writes ``abandoned``, and it writes its own ``exclusion_detail``; a report of
  any kind sets ``status=reported`` instead. So the SDK's report never landed
  on those three, and the step that dropped it is NOT diagnosed by this file.
* LOCALLY, measured the same day: a loopback endpoint that accepts a
  completion and never answers holds the prime-agent child for its entire
  budget, produces no completion, and is killed on the deadline. That is the
  mechanism below, and it is a mechanism this deployment can reach — not a
  transcript of what those three rows say.

Two independent defects, each one alone enough to make a week of it:

1. Nothing bounded the request. Prime routes completions through the bundled
   OpenAI/Anthropic client, whose default request timeout is 600000ms with
   retries on top, and no argv flag reaches it -- ``--autonomous-timeout-ms``
   is evaluated between turns, so it does not fire while a turn is blocked in
   the call. The ceiling lives in the attested settings this adapter writes.
2. The runner READ the child's stderr on the deadline path and threw it away,
   so the one exit that has no other explanation was the only one that carried
   none.

A test that checked the argv was built correctly would have passed every day
of that week, so these assert the two BEHAVIOURS instead: what the runner
reports when it kills a wedged child, and what the attested settings bound.
``test_prime_binary_execution.py`` is the same claim against the real binary.
"""
from __future__ import annotations

import inspect
import json
import pathlib
import stat
import time

import agent_flywheel as fw
from agent_flywheel import RolloutRequest
from agent_flywheel.adapters import prime_agent
from agent_flywheel.runner import (
    episode_plan,
    harness_call_ceiling_s,
    run_rollout_as_command,
)

#: The episode the OPSD queue mints, observed live on 2026-09-15:
#: ``budget_at_creation_s = 894.0``. Read from the adapter, which owns it: it
#: is a fact about what this deployment mints and the sealed ceiling is sized
#: against it, so two copies of the number would be two things to keep in step.
OBSERVED_EPISODE_BUDGET_S = prime_agent.PRIME_REFERENCE_EPISODE_BUDGET_S


def _runner_kill_deadline_s(budget_s: float = OBSERVED_EPISODE_BUDGET_S) -> float:
    """What ``RolloutRunner.run_one`` will actually kill the child at.

    Asked of the runner's own arithmetic rather than restated here. This used
    to re-spell ``min(serve_default, budget - REPORT_MARGIN_S)``, which was a
    second copy of a division that has since stopped being that simple -- the
    reserve is now derived from what the transport says delivery costs, and a
    copy here would have gone quietly stale rather than red.
    """
    serve_default = inspect.signature(fw.serve).parameters["rollout_timeout_s"].default
    from agent_flywheel._http import default_delivery_envelope_s

    return episode_plan(
        budget_s,
        rollout_timeout_s=float(serve_default),
        envelope_s=default_delivery_envelope_s(),
    ).child_s


def _worst_case_wedged_call_s() -> float:
    """The longest one wedged completion can hold a Prime child, in seconds.

    ONE definition, because two tests need it and a second copy of this
    arithmetic is how the ceiling and the episode length it assumes drift
    apart. It also owns the "is there a ceiling at all?" refusal: without this,
    deleting the setting surfaces as ``KeyError: 'retry'`` from whichever test
    happened to touch it first, which names the symptom and not the cause.
    """
    settings = json.loads(prime_agent._FIXED_SETTINGS)
    assert "retry" in settings and "provider" in settings["retry"], (
        "no retry ceiling in the attested settings: Prime then uses its own "
        "600000ms request default, which no argv flag can reach"
    )
    retry = settings["retry"]
    provider = retry["provider"]
    # Counted from maxRetries whatever ``enabled`` says, so turning retries
    # back on without re-deriving the ceiling cannot pass by accident.
    attempts = (provider["maxRetries"] + 1) * (retry["maxRetries"] + 1)
    backoff_ms = 2000 * retry["maxRetries"]  # Prime's documented 2s base delay
    return (provider["timeoutMs"] * attempts + backoff_ms) / 1000.0


def _script(tmp_path, body: str) -> str:
    p = tmp_path / "wedged.sh"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _req(tmp_path) -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_wedged", turn_input="ship it",
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_wedged", api_key="perk_x",
        artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60.0,
        lease_token="lt",
    )


def test_a_rollout_killed_on_its_deadline_reports_what_the_child_said(tmp_path):
    """The kill is not the cause. Whatever the child said before it is.

    The agent_exit_nonzero path has always carried ``stderr[-1500:]``; the
    deadline path read the same variable and dropped it, which is precisely
    why a wedged provider call looked identical to an agent doing slow work.
    """
    body = (
        'echo "POST https://gateway.invalid/v1/chat/completions: no response '
        'after 600000ms" >&2\n'
        "sleep 60\n"
    )
    out = run_rollout_as_command(
        ["sh", _script(tmp_path, body)], _req(tmp_path),
        trajectory_dir=str(tmp_path), timeout_s=1.0,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert out.exclusion_reason == "deadline_exceeded"
    detail = out.exclusion_detail or ""
    # The clock, still.
    assert "exceeded 1s and was killed" in detail
    # AND the cause, which is the half that was missing.
    assert "gateway.invalid" in detail, detail
    assert "no response after 600000ms" in detail, detail


def test_a_silent_child_still_reports_the_clock_and_nothing_invented(tmp_path):
    """A child that says nothing must not gain a fabricated cause."""
    out = run_rollout_as_command(
        ["sh", _script(tmp_path, "sleep 60\n")], _req(tmp_path),
        trajectory_dir=str(tmp_path), timeout_s=1.0,
        env={"PATH": "/usr/bin:/bin"},
    )
    assert out.exclusion_reason == "deadline_exceeded"
    assert out.exclusion_detail == "command exceeded 1s and was killed"


def test_the_attested_settings_bound_one_model_call_inside_the_episode():
    """The ceiling Prime's provider client obeys is a SETTING, not a flag.

    Prime's own default is 600000ms per request with its own retries, so with
    no ceiling here a single hanging completion outlasts any episode this
    platform mints and the agent never gets to report. The envelope below is
    what the harness can spend on one wedged call, worst case.
    """
    worst_case_s = _worst_case_wedged_call_s()
    kill_s = _runner_kill_deadline_s()
    # THE CANONICAL PREDICATE, asked once. This used to spell its own version
    # of "comfortably less than the episode" (`worst_case < kill_s / 2`) while
    # the runner, at runtime, compared the same two numbers a different way --
    # two spellings of one question, which is how they end up disagreeing.
    # ``harness_call_ceiling_s`` is now the only place the question is asked,
    # and the runtime check reads it too.
    assert worst_case_s <= harness_call_ceiling_s(OBSERVED_EPISODE_BUDGET_S), (
        f"one wedged model call can burn {worst_case_s:.0f}s of an episode the "
        f"runner kills at {kill_s:.0f}s, leaving too little to reach terminal "
        f"evidence"
    )
    # And the ceiling really is a fraction of the episode, not most of it. At
    # Prime's own 600000ms default this would be 600s of an 884s episode even
    # before a single retry.
    assert worst_case_s < kill_s / 2, (
        f"{worst_case_s:.0f}s of a {kill_s:.0f}s window is not a bound"
    )

    # The other direction, which matters more. The ceiling bounds the WHOLE
    # streamed response, not time-to-first-byte (measured: a 25s response is
    # cut by a 10s ceiling), so a value near the real generation time throws
    # away finished work -- the worse failure, because it corrupts what the
    # model learns. 4096 max_tokens at a pessimistic 20 tok/s is ~205s.
    timeout_ms = json.loads(prime_agent._FIXED_SETTINGS)["retry"]["provider"]["timeoutMs"]
    assert timeout_ms >= 240_000, (
        "too tight: this ceiling cuts healthy generations, not just wedged ones"
    )


def test_the_episode_length_the_ceiling_stops_working_at_is_written_down():
    """A bound this platform can out-live has to SAY where, in a checked number.

    The ceiling cannot follow the rollout's budget: these bytes are hashed into
    ``prime.fixed_settings`` and frozen as the agent's ``agent_execution_sha256``
    before dispatch, so a per-rollout value would change the agent's identity
    every rollout and make it unclaimable. Given that, the honest thing is to
    name the episode length below which the ceiling is inert -- and to re-derive
    that number HERE rather than trust the comment, because this deployment has
    minted 300s episodes (``serve_prime.py``, run 08e40b36) and at 300s one
    wedged call can still outlast the whole rollout.

    Since 2026-09-16 the floor is also enforced at RUNTIME rather than only
    asserted here: the snapshot declares ``worst_case_call_s`` and the runner
    warns on rollout #1 when the episode it was actually handed is below it.
    This test keeps the written number honest; that check keeps the OPERATOR
    informed, and neither replaces the other.
    """
    worst_case_s = _worst_case_wedged_call_s()

    # Solved by search against the runner's own arithmetic rather than by
    # re-deriving it: the smallest whole-second budget at which one wedged call
    # still fits inside the share of the episode core will allow it.
    floor_s = next(
        b for b in range(1, 3601)
        if harness_call_ceiling_s(float(b)) >= worst_case_s
    )
    assert harness_call_ceiling_s(float(floor_s)) >= worst_case_s
    assert harness_call_ceiling_s(float(floor_s - 1)) < worst_case_s

    source = pathlib.Path(prime_agent.__file__).read_text(encoding="utf-8")
    assert f"{floor_s}s" in source, (
        f"the attested settings are inert below a {floor_s}s episode and "
        f"nothing in {pathlib.Path(prime_agent.__file__).name} says so -- a "
        f"reader sizing the ceiling has no way to know it stopped applying"
    )
    # The snapshot must DECLARE it, or the runtime check has no producer and
    # the floor goes back to being a sentence nobody enforces.
    assert prime_agent.PrimeHarnessSnapshot.worst_case_call_s.fget(
        None
    ) == worst_case_s, (
        "the adapter's declared ceiling and the sealed bytes disagree"
    )


#: The digest of the bytes above, pinned. Changing it is a DEPLOYMENT step, not
#: an edit -- see the test below for what has to happen when it moves.
#:
#: MOVED 2026-09-16, deliberately: the provider ceiling stopped being a literal
#: and is now derived from ``runner.harness_call_ceiling_s``, which changed it
#: from 300000ms to 306000ms. Every OPSD workload prepared against the previous
#: digest must be re-prepared -- that is what this pin exists to force.
SEALED_SETTINGS_SHA256 = "0247ed03ed5b04ff38179b3cb5a581435d50d52589244064dc3172d777c9aff0"


def test_changing_the_sealed_settings_cannot_be_done_silently():
    """Editing these bytes strands every workload already prepared. Say so HERE.

    ``_FIXED_SETTINGS_SHA256`` is ``prime.fixed_settings`` in
    ``execution_components``, and the control plane freezes the agent's whole
    ``agent_execution_sha256`` before dispatch: a sealed rollout is claimable
    only by that exact digest, and a report carrying a different one is refused.
    So an agent restarted with edited settings looks EXACTLY like an agent that
    cannot run -- registered, long-polling, ``claim_empty`` forever, every
    rollout eventually reaped ``deadline_exceeded`` -- which is one of the
    shapes this file exists because of. A README paragraph is not a mechanism
    for that; a pin that turns red on the edit is.
    """
    assert prime_agent._FIXED_SETTINGS_SHA256 == SEALED_SETTINGS_SHA256, (
        "the attested Prime settings changed, which rotates the agent's "
        "agent_execution_sha256. Every OPSD workload already PREPARED against "
        "the old digest becomes unclaimable by the restarted agent and must be "
        "re-prepared. Update this pin deliberately, with that done."
    )


def test_the_settings_prime_rewrites_are_never_the_ones_we_seal():
    """Prime migrates ``retry.maxDelayMs`` into ``retry.provider``.

    This adapter writes settings.json read-only and attests its bytes before
    and after every rollout, so a key Prime rewrites on load would turn every
    rollout into a drift exclusion. Spell the migrated-to name directly.
    """
    retry = json.loads(prime_agent._FIXED_SETTINGS)["retry"]
    assert "maxDelayMs" not in retry
    assert "maxRetryDelayMs" in retry["provider"]
