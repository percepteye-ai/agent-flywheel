"""``PERCEPTEYE_MODE`` — the switch that turns donation on and off.

Before this existed, the only way to stop donating capacity was to delete the
``serve()`` call and redeploy, and the only way to resume was to put it back.
That is a code change to express an operational decision, which is both tedious
and the kind of edit that gets made on the wrong branch at the wrong time.

The switch has to satisfy four things at once, and the tests below are one per
failure it would otherwise allow:

* **Off must not need to be configured.** A production deployment has no
  reason to hold a control-plane URL or key, so ``production`` must return
  before anything asks for one.
* **Off must not be silent.** A ``serve()`` that quietly does nothing is the
  precise failure this package refuses everywhere else -- the operator believes
  capacity is being donated and nothing contradicts them.
* **Off must not crash.** This is a switch for a running service. Raising would
  turn "stop donating" into an outage.
* **A typo must not be guessed.** Both directions of a wrong guess are silent
  and unrecoverable: donating capacity that was meant to be off, or donating
  nothing while the operator believes training is running.
"""
from __future__ import annotations

import pytest

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel.errors import ConfigurationError

CALL = dict(agent_id="a", input_shape="text")


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """No ambient Percepteye configuration, so each test states its own."""
    for name in ("PERCEPTEYE_MODE", "PERCEPTEYE_CONTROL_PLANE_URL",
                 "PERCEPTEYE_API_KEY"):
        monkeypatch.delenv(name, raising=False)


# ── the default is unchanged ────────────────────────────────────────────────


def test_unset_means_training_so_existing_calls_are_unaffected():
    assert fw.current_mode() == "training"


def test_training_still_reaches_the_ordinary_preflight():
    """Proof the switch is not swallowing the run: with mode on, the missing
    credential is still what stops it, exactly as before the switch existed."""
    with pytest.raises(ConfigurationError, match="control_plane_url"):
        fw.serve("m:b", **CALL)


# ── off ─────────────────────────────────────────────────────────────────────


def test_production_needs_no_credentials_and_opens_no_connection(monkeypatch):
    """The whole point: turning training OFF must not require the
    configuration that turning it ON would."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    assert fw.serve("m:b", **CALL) == 0


def test_production_returns_rather_than_raising(monkeypatch):
    """A switch that crashes the service it was meant to quiet is not a
    switch. Zero rollouts is also the literal truth."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    assert fw.serve("m:b", **CALL) == 0


def test_production_says_so_out_loud(monkeypatch, capsys):
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    fw.serve("m:b", **CALL)
    out = capsys.readouterr().out
    assert "mode=production" in out
    assert "not claiming rollouts" in out
    # and names the way back, because the operator reading this line is
    # usually the one who needs to flip it the other way
    assert "PERCEPTEYE_MODE=training" in out


def test_production_does_not_resolve_the_entrypoint(monkeypatch):
    """An entrypoint that does not exist is not an error in production mode:
    the switch is read before anything is imported, so a deployment that never
    trains does not have to carry an importable agent."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    assert fw.serve("no.such.module:nope", **CALL) == 0


# ── the toggle, in both directions, with no code change ─────────────────────


def test_one_variable_flips_a_deployment_both_ways(monkeypatch):
    """The same call site, the same image, two behaviours."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    assert fw.serve("m:b", **CALL) == 0

    monkeypatch.setenv("PERCEPTEYE_MODE", "training")
    with pytest.raises(ConfigurationError):
        fw.serve("m:b", **CALL)          # off the switch, on to real preflight


# ── spellings ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("training", "training"), ("train", "training"),
    ("rollout", "training"), ("rollouts", "training"),
    ("fine-tuning", "training"), ("finetuning", "training"),
    ("improving", "training"),
    ("production", "production"), ("prod", "production"),
    ("serving", "production"),
    ("  Production  ", "production"), ("TRAINING", "training"),
])
def test_recognised_spellings(monkeypatch, raw, expected):
    monkeypatch.setenv("PERCEPTEYE_MODE", raw)
    assert fw.current_mode() == expected


def test_the_legacy_value_still_means_what_it_meant(monkeypatch):
    """``mode="rollout"`` was the only accepted value before this change, and
    it meant "donate". It still does."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "rollout")
    assert fw.current_mode() == "training"
    with pytest.raises(ConfigurationError, match="control_plane_url"):
        fw.serve("m:b", mode="rollout", **CALL)


def test_empty_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("PERCEPTEYE_MODE", "   ")
    assert fw.current_mode() == "training"


# ── a wrong value is never guessed ──────────────────────────────────────────


@pytest.mark.parametrize("typo", ["prodution", "off", "0", "false", "no",
                                  "disabled", "inference", "yes"])
def test_an_unrecognised_mode_raises_rather_than_guessing(monkeypatch, typo):
    """Guessing wrong is silent in BOTH directions.

    Read "off" as production and an operator who meant something else donates
    nothing while believing training runs. Read it as training and one who
    meant to stop keeps donating capacity on their own credentials. Neither is
    visible from the outside, so neither is recoverable.
    """
    monkeypatch.setenv("PERCEPTEYE_MODE", typo)
    with pytest.raises(ConfigurationError, match="is not a mode"):
        fw.current_mode()
    with pytest.raises(ConfigurationError, match="is not a mode"):
        fw.serve("m:b", **CALL)


def test_the_refusal_names_both_modes_and_what_they_do(monkeypatch):
    monkeypatch.setenv("PERCEPTEYE_MODE", "nonsense")
    with pytest.raises(ConfigurationError) as e:
        fw.current_mode()
    msg = str(e.value)
    assert "training" in msg and "production" in msg
    assert "PERCEPTEYE_MODE=" in msg      # names the source it read


def test_an_unrecognised_explicit_mode_names_the_argument(monkeypatch):
    """The message must point at the thing the customer can edit."""
    with pytest.raises(ConfigurationError, match=r"mode='sometimes'"):
        fw.serve("m:b", mode="sometimes", **CALL)


# ── precedence ──────────────────────────────────────────────────────────────


def test_an_explicit_mode_beats_the_environment(monkeypatch):
    monkeypatch.setenv("PERCEPTEYE_MODE", "training")
    assert fw.serve("m:b", mode="production", **CALL) == 0


def test_the_environment_decides_when_nobody_passed_one(monkeypatch):
    """The ordinary case, and the reason the parameter defaults to None rather
    than to a mode: omitting it is what hands the decision to the operator."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    assert fw.serve("m:b", **CALL) == 0


def test_a_hardcoded_mode_says_that_is_why(monkeypatch, capsys):
    """If code overrode the environment, the line an operator reads has to say
    so -- otherwise they flip the variable, nothing changes, and there is
    nothing on stdout explaining it."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "training")
    fw.serve("m:b", mode="production", **CALL)
    assert "remove mode= from this call" in capsys.readouterr().out


# ── the public surface ──────────────────────────────────────────────────────


def test_the_mode_helpers_are_exported():
    assert "current_mode" in fw.__all__
    assert "resolve_mode" in fw.__all__


def test_resolve_mode_is_callable_without_touching_the_environment(monkeypatch):
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    assert fw.resolve_mode("training") == "training"
    assert fw.resolve_mode() == "production"


# ── the empty argument, which used to skip the switch entirely ─────────────


@pytest.mark.parametrize("empty", ["", "   ", "\t"])
def test_an_empty_mode_argument_falls_through_to_the_environment(
    monkeypatch, empty
):
    """``mode=cfg.get("percepteye_mode", "")`` is how a config layer says
    "not set". Honouring it as a value pinned the process to training without
    ever reading PERCEPTEYE_MODE -- so an operator who had flipped the switch
    to stop donating kept donating, on their own credentials, silently. It is
    the one input that reached the donate direction without passing the
    closed-set check."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    assert fw.resolve_mode(empty) == "production"
    assert fw.serve("m:b", mode=empty, **CALL) == 0


def test_an_empty_argument_with_no_environment_is_still_training(monkeypatch):
    assert fw.resolve_mode("") == "training"


# ── an override is announced in BOTH directions ────────────────────────────


def test_code_overriding_the_environment_to_donate_says_so(monkeypatch, capsys):
    """The direction that was silent, and the one an operator cares about.

    Env says stop, code says go. serve() proceeds -- the precedence is correct
    and documented -- but it used to proceed with nothing on stdout, so an
    operator who flipped the variable saw the flywheel keep claiming and had
    no way to learn why.
    """
    monkeypatch.setenv("PERCEPTEYE_MODE", "production")
    with pytest.raises(ConfigurationError):        # stopped by the missing URL
        fw.serve("m:b", mode="training", **CALL)
    out = capsys.readouterr().out
    assert "overriding PERCEPTEYE_MODE=production" in out
    assert "mode=training" in out


def test_an_unreadable_environment_switch_is_reported_even_when_overridden(
    monkeypatch,
):
    """The argument is authoritative, so a typo'd variable must not crash a
    process that was told what to do -- but it must not pass unremarked
    either, or the next operator to rely on that variable is misled."""
    monkeypatch.setenv("PERCEPTEYE_MODE", "prodction")
    with pytest.warns(RuntimeWarning, match="is not a mode"):
        assert fw.serve("m:b", mode="production", **CALL) == 0
