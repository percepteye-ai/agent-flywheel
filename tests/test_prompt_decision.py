"""The prompt predicate, asked directly.

``tests/test_prompt_decision_parity.py`` proves this agrees with the Node SDK,
and it is allowed to skip on a checkout that has no Node SDK beside it. So the
BEHAVIOUR is asserted here, where nothing can skip: a suite whose only coverage
of a rule lives behind a skip has no coverage of it on the machine where it
skipped.
"""
from __future__ import annotations

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel.prompt import SERVABLE_PROMPT_STATUSES, prompt_decision

APPROVED = {
    "name": "support-triage", "version": 3, "sha256": "c" * 64,
    "text": "You triage carefully.", "candidate_kind": "replacement",
    "status": "approved", "model_prompt_checksum": "c" * 64,
    "stale_baseline": False,
}


def _with(**over):
    return {**APPROVED, **over}


def test_an_approved_paired_bundle_applies():
    d = prompt_decision(APPROVED)
    assert d.apply is True
    assert d.text == "You triage carefully."
    assert d.checksum == "c" * 64
    assert d.reason == ""


def test_the_empty_answer_changes_nothing():
    """``{}`` is KEEP WHAT YOU HAVE, and the one thing it must never be is
    "run with no prompt": ``text`` is None, not ``""``, so a caller that
    assigns it cannot blank its own instructions by accident."""
    d = prompt_decision({})
    assert (d.apply, d.text) == (False, None)
    assert "serving no approved prompt" in d.reason


def test_a_stale_baseline_refuses():
    d = prompt_decision(_with(stale_baseline=True))
    assert (d.apply, d.text) == (False, None)
    assert d.reason.startswith("stale_baseline=true:")


def test_a_truthy_stale_baseline_is_NOT_the_flag():
    """Strictly ``True``. Coercing would refuse a bundle the Node SDK applies,
    and the two must not disagree about the same body."""
    assert prompt_decision(_with(stale_baseline=1)).apply is True
    assert prompt_decision(_with(stale_baseline="true")).apply is True


def test_an_unapproved_status_refuses_and_NAMES_the_status():
    d = prompt_decision(_with(status="draft"))
    assert d.apply is False
    assert '"draft"' in d.reason, d.reason
    assert "approved/delivered" in d.reason, d.reason


def test_delivered_serves_too():
    assert prompt_decision(_with(status="delivered")).apply is True
    assert SERVABLE_PROMPT_STATUSES == ("approved", "delivered")


def test_a_non_string_status_refuses_rather_than_being_stringified():
    """``status: 7`` must not become the string ``"7"`` and then be compared;
    it is an absent status, and an absent status is not approved."""
    d = prompt_decision(_with(status=7))
    assert d.apply is False
    assert '""' in d.reason, d.reason


def test_a_mismatched_pair_refuses():
    """The bundle disagrees with itself: the prompt's identity is not the one
    stamped on the checkpoint it was certified against."""
    d = prompt_decision(_with(sha256="a" * 64, model_prompt_checksum="b" * 64))
    assert d.apply is False
    assert d.reason.startswith("the bundle disagrees with itself:")
    assert "aaaaaaaaaaaa…" in d.reason and "bbbbbbbbbbbb…" in d.reason


def test_a_missing_half_of_the_pair_is_CANNOT_PROVE_not_MISMATCH():
    """Refusing on a blank half would refuse every bundle that predates the
    stamp, which is a refusal nobody can act on."""
    assert prompt_decision(_with(model_prompt_checksum="")).apply is True
    assert prompt_decision(_with(sha256="")).apply is True


def test_a_stale_baseline_outranks_a_bad_status():
    """Two causes, one message: the operator must be sent to the one they can
    actually act on, and re-certification is not a status change."""
    d = prompt_decision(_with(status="draft", stale_baseline=True))
    assert d.reason.startswith("stale_baseline=true:")


def test_a_bad_status_outranks_a_mismatched_pair():
    d = prompt_decision(_with(
        status="draft", sha256="a" * 64, model_prompt_checksum="b" * 64))
    assert d.reason.startswith("the served bundle's status is")


def test_every_refusal_names_its_OWN_cause():
    """One message covering four failures would send an operator looking in
    the wrong place three times out of four."""
    reasons = {
        prompt_decision(body).reason
        for body in (
            {},
            _with(stale_baseline=True),
            _with(status="draft"),
            _with(sha256="a" * 64, model_prompt_checksum="b" * 64),
        )
    }
    assert len(reasons) == 4, reasons


def test_the_served_text_is_applied_UNEDITED():
    """Whitespace around an approved system prompt is part of what was
    certified. It decides emptiness on the trimmed copy and applies the
    original."""
    d = prompt_decision(_with(text="\n  Be careful.  \n"))
    assert d.text == "\n  Be careful.  \n"


def test_a_garbled_answer_refuses_instead_of_raising():
    """FAIL OPEN. A control plane that answered with something unexpected must
    cost the improvement and nothing else -- never an exception inside a
    customer's turn."""
    for body in (None, "not a body", [1, 2, 3], 7):
        d = prompt_decision(body)
        assert (d.apply, d.text) == (False, None), body


def test_the_checksum_survives_a_refusal():
    """"Which checkpoint was this pair certified with?" is the first thing an
    operator asks when they read one of these refusals."""
    assert prompt_decision(_with(stale_baseline=True)).checksum == "c" * 64
    assert prompt_decision({"model_prompt_checksum": "c" * 64}).checksum == "c" * 64


def test_it_is_exported_because_a_customer_has_to_call_it():
    """Reachable and obvious, not a private helper the docstring alludes to."""
    assert "prompt_decision" in fw.__all__
    assert "PromptDecision" in fw.__all__
    assert fw.prompt_decision is prompt_decision
