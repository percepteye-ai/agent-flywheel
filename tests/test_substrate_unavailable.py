"""A rollout whose every tool call failed is a substrate fault, not a score.

On 2026-08-18 an agent overwrote IPython's own ``interactiveshell.py`` inside
the kernel venv. Every ``ipython`` call — the only real tool — died at import,
and a full day of runs reported 0% grounded while reading as "the 4B cannot use
tools". The SDK already knew every one of those calls had failed; it had
nowhere to say so, because ``substrate_unavailable`` was declared in the
contract and emitted nowhere.
"""
from agent_flywheel.contract import RolloutOutput, ToolCallOutcome
from agent_flywheel.runner import _mark_substrate_unavailable


def _call(outcome, name="ipython"):
    return ToolCallOutcome(name=name, arguments={}, outcome=outcome,
                           status_code=None)


def test_every_call_failed_is_a_substrate_fault():
    out = RolloutOutput(final_text="", tool_calls=[
        _call("failed"), _call("failed"), _call("failed")])
    _mark_substrate_unavailable(out)
    assert out.exclusion_reason == "substrate_unavailable"
    assert "3 tool call(s) failed" in out.exclusion_detail
    assert "ipython" in out.exclusion_detail


def test_one_success_is_ordinary_agent_behaviour():
    """Partial failure is what we WANT to train on."""
    out = RolloutOutput(final_text="", tool_calls=[
        _call("failed"), _call("ok"), _call("failed")])
    _mark_substrate_unavailable(out)
    assert out.exclusion_reason is None


def test_unknown_is_not_failure():
    """``unknown`` is the honest answer when the framework exposed no verdict.
    An agent that reports none must behave exactly as it did before — rounding
    it up to a substrate fault would quarantine healthy rollouts wholesale."""
    out = RolloutOutput(final_text="", tool_calls=[
        _call("unknown"), _call("unknown")])
    _mark_substrate_unavailable(out)
    assert out.exclusion_reason is None

    mixed = RolloutOutput(final_text="", tool_calls=[
        _call("failed"), _call("unknown")])
    _mark_substrate_unavailable(mixed)
    assert mixed.exclusion_reason is None


def test_a_rollout_that_called_no_tool_says_nothing_about_the_substrate():
    for calls in (None, []):
        out = RolloutOutput(final_text="", tool_calls=calls)
        _mark_substrate_unavailable(out)
        assert out.exclusion_reason is None


def test_an_existing_exclusion_is_never_overwritten():
    """A crash or a dead credential knows more than this does."""
    out = RolloutOutput(final_text="", tool_calls=[_call("failed")],
                        exclusion_reason="credential_expired",
                        exclusion_detail="key rejected")
    _mark_substrate_unavailable(out)
    assert out.exclusion_reason == "credential_expired"
    assert out.exclusion_detail == "key rejected"


def test_the_reason_is_in_the_contract_vocabulary():
    import typing

    from agent_flywheel.contract import ExclusionReason
    assert "substrate_unavailable" in typing.get_args(ExclusionReason)
