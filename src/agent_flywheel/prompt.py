"""MAY THIS SERVED PROMPT BE APPLIED, and if not, WHY NOT.

:func:`current_prompt` hands back what the control plane serves. Deciding
whether it may be applied used to live in that function's DOCSTRING -- three
fields a caller was told to check, in an order that mattered, with no
mechanism behind any of it. A rule written in prose is a rule every caller
re-derives, and the ones who read the docstring quickly re-derive it wrong:
the pair check silently skipped, ``stale_baseline`` never read, a bundle
applied because it had text in it.

So the rule is a function now, and this is it. Pure, exported, and separate
from everything that fetches or logs, because it is the part worth testing
exhaustively.

WORD FOR WORD WITH THE NODE SDK, ON PURPOSE
-------------------------------------------
The Node package answers this same question in its own ``promptDecision``,
because it is the host's prompt seam and applies the text itself. A customer
runs one package or the other and may migrate between them, so two SDKs that
refused different bundles -- or refused the same bundle with different
explanations -- would make "may I apply this?" a question about which language
you happened to install.

The four refusals, their order, and their exact sentences are therefore shared
with that implementation rather than paraphrased from it, and
``tests/test_prompt_decision_parity.py`` drives a table of fixtures through
both and asserts every field agrees. Reword one side and that test goes red;
that is the mechanism, and it is why the wording is not adjusted here for
taste.

WHAT IT STILL DOES NOT DO
-------------------------
It does not apply anything. Swapping a model endpoint is invisible to an
agent's behaviour; swapping its system prompt changes what it DOES, and this
package is a library inside an arbitrary process with no idea where that
process's prompt comes from. The decision is answered here; acting on it stays
the caller's::

    served = fw.current_prompt(agent_id="my-agent")
    decision = fw.prompt_decision(served)
    if decision.apply:
        system_prompt = decision.text
    else:
        log.info("keeping the current prompt: %s", decision.reason)

FAIL-OPEN, AT EVERY BRANCH. Every answer other than "apply" leaves the caller
holding the prompt it already had. Nothing here raises, an empty answer is a
refusal rather than an error, and there is no path through this module that
ends with an agent running on no prompt at all.

A NEWER APPROVED PROMPT, WITHOUT A RESTART
------------------------------------------
Prompt optimization does not stop at the first approval. Mission Control keeps
evolving the prompt from the traces this agent keeps producing -- including
while an operator has not yet decided the last improvement -- and each round
has to beat the candidate already waiting, so every approval is the best
prompt found so far. A process that read its prompt once at start ran the
first approval forever.

So ``/policy/current``, which a :class:`~agent_flywheel.PolicySource` polls
anyway, names WHICH approved prompt is served as ``prompt_generation``, and
``/prompt/current`` returns the same token as ``generation``. Two pure rules
decide what a running handle does with them, and the Node SDK asks the same
two with the same answers (``tests/test_prompt_decision_parity.py``):

* :func:`prompt_needs_refresh` -- re-read the text only when the served token
  differs from the one held. The text is unbounded and the poll is not.
* :func:`refreshed_prompt` -- converge on what a fresh start would apply,
  except that an empty or failed answer never takes an applied prompt away:
  the control plane says "keep what you have" exactly that way.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

#: The delivery states that may serve.
#:
#: The control plane's own boundary, not a second one: it serves a prompt only
#: from a delivery whose status is one of these, and its export path enforces
#: the same set. Re-asserting it here is not a duplicate gate -- it is the only
#: thing standing between a control plane that one day widens the filter and a
#: customer's agent silently running an unapproved prompt.
SERVABLE_PROMPT_STATUSES = ("approved", "delivered")


def _str(value: Any) -> str:
    """A string field, trimmed. Anything that is not a string is absent.

    Mirrors the Node SDK's ``str`` helper exactly, including the part that
    matters: a non-string ``status`` is treated as blank rather than coerced,
    so ``status: 7`` refuses instead of being stringified into a value nobody
    approved.
    """
    return value.strip() if isinstance(value, str) else ""


def _short(value: str) -> str:
    """A checksum, shortened for a log line. Never a claim that it is complete."""
    return f"{value[:12]}…" if value else "none"


@dataclass(frozen=True)
class PromptDecision:
    """The answer, and the reason for it.

    ``reason`` is empty when and only when ``apply`` is true. Every refusal
    names ITS OWN cause: one message covering four different failures would
    send an operator looking in the wrong place three times out of four.
    """

    #: Whether the served text may be applied.
    apply: bool
    #: The text to apply, or ``None``. Never a fallback and never the empty
    #: string -- a caller that assigns this unconditionally would blank its own
    #: prompt on every refusal, which is the one outcome nothing here allows.
    text: str | None
    #: The checkpoint identity this text was certified against, when the
    #: bundle carried one. Populated even on a refusal, because "which
    #: checkpoint was this pair certified with?" is a question an operator
    #: reading a refusal asks.
    checksum: str | None
    #: Why not, in a sentence an operator can act on. Empty when applying.
    reason: str


def prompt_decision(prompt: Mapping[str, Any] | None) -> PromptDecision:
    """THE predicate. Ask this, never one field of it.

    :param prompt: a ``/prompt/current`` body -- what :func:`current_prompt`
        returns, possibly ``{}``. Anything that is not a mapping is treated as
        ``{}``, so a garbled answer refuses rather than raises.
    """
    p: Mapping[str, Any] = prompt if isinstance(prompt, Mapping) else {}
    raw = p.get("text")
    text = raw if isinstance(raw, str) else ""
    checksum = _str(p.get("model_prompt_checksum")) or None

    # WITHDRAWN: a person took the approved prompt back out of service and
    # nothing approved is left. Unlike `{}` below, this is an instruction to
    # stop applying: the prompt the host assembles is what runs now.
    if p.get("withdrawn") is True:
        return PromptDecision(
            apply=False, text=None, checksum=checksum, reason=WITHDRAWN_REASON,
        )

    # `{}` IS AN ANSWER, and it is "keep what you have". Nothing is being
    # optimized for this agent yet, nothing has passed the approval gate, or
    # the control plane could not answer -- and every one of those means the
    # prompt the caller already has stands. It never means "run with no prompt".
    if not text.strip():
        return PromptDecision(
            apply=False, text=None, checksum=checksum,
            reason="the control plane is serving no approved prompt for this "
                   "agent (empty answer); the prompt this host assembles stands",
        )

    # THE BASELINE MOVED UNDER IT. The prompt the customer supplied as the
    # baseline changed after this bundle was approved, so the approved text was
    # shaped against a prompt they no longer ship. The deliverable still stands
    # as a record; applying it is a different question. Re-certification is
    # manual, so this does not clear on its own.
    if p.get("stale_baseline") is True:
        return PromptDecision(
            apply=False, text=None, checksum=checksum,
            reason="stale_baseline=true: this agent's own baseline prompt changed "
                   "after the pair was certified, so the approved text was shaped "
                   "against a prompt this agent no longer ships. Re-certification is "
                   "manual -- ask the control plane to re-certify the pair",
        )

    status = _str(p.get("status"))
    if status not in SERVABLE_PROMPT_STATUSES:
        return PromptDecision(
            apply=False, text=None, checksum=checksum,
            reason=f"the served bundle's status is "
                   f"{json.dumps(status, ensure_ascii=False)}, not "
                   f"one of {'/'.join(SERVABLE_PROMPT_STATUSES)}; only an operator-"
                   "approved bundle is applied",
        )

    # THE PAIR, AS FAR AS THIS SIDE CAN CHECK IT. ``model_prompt_checksum`` is
    # the prompt identity STAMPED ON THE CHECKPOINT at certification; ``sha256``
    # is the optimized prompt's own identity. The control plane refuses to
    # assemble a delivery unless the two are equal, so a served bundle where
    # they differ is not the validated pair, whatever its status says. That is
    # an inconsistency in the deliverable, not a policy call, so it refuses.
    #
    # Checked only when BOTH are present: a blank one means we could not look,
    # which is "cannot prove" rather than a proven mismatch, and refusing on it
    # would refuse every bundle that predates the stamp.
    sha = _str(p.get("sha256"))
    if sha and checksum and sha != checksum:
        return PromptDecision(
            apply=False, text=None, checksum=checksum,
            reason="the bundle disagrees with itself: the prompt's sha256 is "
                   f"{_short(sha)} but the checkpoint it was certified against is "
                   f"stamped {_short(checksum)}. Assembly refuses that pair, so this "
                   "answer is not the thing that was approved",
        )

    # The text as it was served, NOT the trimmed copy the emptiness check was
    # made on. Leading whitespace in an approved system prompt is part of what
    # was certified, and this side does not get to edit it.
    return PromptDecision(apply=True, text=text, checksum=checksum, reason="")


#: Why nothing is applied once a person withdrew the approved prompt. Word for
#: word the Node SDK's ``WITHDRAWN_REASON``.
WITHDRAWN_REASON = (
    "a person withdrew the approved prompt for this agent and none is "
    "approved in its place; the prompt this host assembles stands"
)


#: Why nothing is applied before the control plane has been asked. Word for
#: word the Node SDK's ``PENDING_ANSWER_REASON``.
PENDING_ANSWER_REASON = "the control plane has not been asked yet"

#: The decision a handle holds before its first answer: apply nothing, so the
#: prompt the host assembles stands.
PENDING_PROMPT = PromptDecision(
    apply=False, text=None, checksum=None, reason=PENDING_ANSWER_REASON,
)


def served_prompt_generation(policy: Mapping[str, Any] | None) -> str:
    """Which approved prompt ``/policy/current`` says is served, or ``""``.

    The token ``/prompt/current`` returns beside the text as ``generation``.
    ``""`` when the control plane serves none -- or predates the field, which
    then disables refreshing rather than forcing a read every poll."""
    raw = policy.get("prompt_generation") if isinstance(policy, Mapping) else None
    return _str(raw)


def prompt_needs_refresh(held_generation: str,
                         policy: Mapping[str, Any] | None) -> bool:
    """Should the text be re-read? Only when a prompt is served and it is not
    the one held. Absence is never a reason to read: it is the answer
    "keep what you have"."""
    served = served_prompt_generation(policy)
    return bool(served) and served != _str(held_generation)


def refreshed_prompt(
    held: PromptDecision, held_generation: str,
    policy: Mapping[str, Any] | None, answer: Mapping[str, Any] | None,
) -> tuple[PromptDecision, str]:
    """The decision and generation to hold after a poll. Pure.

    :param held: the decision held now (:data:`PENDING_PROMPT` at first).
    :param held_generation: the generation it was read at, ``""`` for none.
    :param policy: the ``/policy/current`` body just read.
    :param answer: the ``/prompt/current`` body read because
        :func:`prompt_needs_refresh` said so, or ``None`` when not read.

    THE RULE: converge on what a fresh start would apply, with one exception.

    * Nothing served and nothing ever held: the fresh start's answer to an
      empty read, said as such rather than "not asked yet".
    * An answer with text: whatever :func:`prompt_decision` says of it --
      a newer approved prompt applies; a stale or mispaired one is refused
      and the host's own prompt stands, exactly as at start -- held at the
      answer's ``generation`` (or the served one, if it carries none), so the
      same answer is not re-read every poll.
    * A WITHDRAWAL (``{"withdrawn": true}``): what a fresh start would do --
      apply nothing, so the host's own prompt runs -- even after a prompt was
      held. It is the one answer that takes an applied prompt away, because a
      person decided it, and it is never what a failure answers.
    * THE EXCEPTION -- no answer, an empty one, or nothing served after a
      prompt was held: keep what is held. ``/prompt/current`` answers ``{}``
      for "could not answer" too, and an unreachable control plane must never
      blank a running agent's instructions.
    """
    held_gen = _str(held_generation)
    served = served_prompt_generation(policy)
    if not served:
        if held_gen or held.apply:
            return held, held_gen
        return prompt_decision({}), ""
    if isinstance(answer, Mapping):
        if answer.get("withdrawn") is True:
            return (prompt_decision(answer),
                    _str(answer.get("generation")) or served)
        raw = answer.get("text")
        if isinstance(raw, str) and raw.strip():
            return (prompt_decision(answer),
                    _str(answer.get("generation")) or served)
    return held, held_gen
