"""ONE consent predicate, in two languages, proved to agree.

``capture_verdict`` here and ``captureVerdict`` in the Node package answer the
same question -- may this agent's captured production turns be uploaded, and if
not why not. A production turn is an end user's own message and the answer the
agent gave them, so this is the gate with the highest cost of being wrong in
either direction, and the two SDKs disagreed on it: Python asked
``bool(cap.get("turns"))`` and Node asks ``capture.turns === true``. The same
registration payload therefore granted consent to one package and refused it in
the other.

``{"turns": "false"}`` is the case that matters and it is not exotic: a JSON
string where a boolean was expected is what an older intake, a proxy, or a
hand-written fixture produces. It is truthy in Python. Under the old gate a
Python ``attach()`` read that refusal as consent, registered the shutdown hook
and began uploading every 32 turns; the identical payload uploaded nothing on
the Node side, and nothing anywhere said the two had diverged.

THE SKIP TRAP. ``test-full`` fails the build on ANY skip, so a test that always
skips turns the build red while proving nothing. The rule for when this may
skip lives in ``tests/node_sdk.py``: auto-discovery skips, a NAMED
``PERCEPTEYE_NODE_SDK`` that is not there fails. CI names it.
"""
from __future__ import annotations

import json

import pytest

from agent_flywheel import capture_verdict

from .node_sdk import require_node_sdk, run_node

#: Every case both sides have to answer identically. The truthy-but-not-true
#: block is the one this file exists for; the rest pin the surrounding edges so
#: a future "fix" cannot widen the gate to make one of them pass.
CASES: dict[str, object] = {
    # ── the one grant ────────────────────────────────────────────────────
    "turns is exactly true": {"capture": {"turns": True}},
    "true, and a reason sent anyway": {
        "capture": {"turns": True, "reason": "enabled by an operator"}},

    # ── truthy, and NOT consent ──────────────────────────────────────────
    "the string 'false'": {"capture": {"turns": "false"}},
    "the string 'true'": {"capture": {"turns": "true"}},
    "the string 'no'": {"capture": {"turns": "no"}},
    "the number 1": {"capture": {"turns": 1}},
    "a non-empty list": {"capture": {"turns": ["yes"]}},
    "a non-empty object": {"capture": {"turns": {"enabled": True}}},

    # ── falsy, and refused with the server's own sentence when it sent one ─
    "turns is false": {"capture": {"turns": False}},
    "turns is false, with a reason": {
        "capture": {"turns": False,
                    "reason": "an operator has not enabled production capture"}},
    "turns is null": {"capture": {"turns": None}},
    "turns is 0": {"capture": {"turns": 0}},
    "turns absent from the capture block": {"capture": {}},
    # A blank reason is no reason: both sides fall back rather than showing an
    # operator an empty string.
    "a blank reason": {"capture": {"turns": False, "reason": "   "}},
    "a reason that is not a string": {"capture": {"turns": False, "reason": 7}},

    # A padded sentence is the server's sentence: both sides trim it. Node
    # used to pass it through untrimmed while Python stripped it.
    "a padded reason": {"capture": {"turns": False,
                                    "reason": "  scope missing \n"}},

    # ── THE SECOND TIER: model content (system prompt, history, tool results)
    "model content granted": {"capture": {"turns": True, "model_content": True}},
    "model content withheld, with the server's sentence": {
        "capture": {"turns": True, "model_content": False,
                    "model_content_reason": "an operator has not enabled "
                                            "model-content capture "}},
    "model content withheld, no sentence": {
        "capture": {"turns": True, "model_content": False}},
    "model content absent from an older control plane": {
        "capture": {"turns": True}},
    "model content the string 'true'": {
        "capture": {"turns": True, "model_content": "true"}},
    "model content the number 1": {
        "capture": {"turns": True, "model_content": 1}},
    "model content true but turns refused": {
        "capture": {"turns": False, "model_content": True,
                    "reason": "the key lacks the scope"}},
    "model content true but turns absent": {
        "capture": {"model_content": True}},
    "model content withheld, a blank sentence": {
        "capture": {"turns": True, "model_content": False,
                    "model_content_reason": "  "}},
    "model content withheld, a sentence that is not a string": {
        "capture": {"turns": True, "model_content": None,
                    "model_content_reason": ["no"]}},

    # ── the control plane did not answer at all ──────────────────────────
    "no capture block": {},
    "capture is null": {"capture": None},
    "capture is a string": {"capture": "yes"},
    "capture is a list": {"capture": [{"turns": True}]},
    "the body is null": None,
    "the body is a string": "not a body",
    "the body is a list": [1, 2, 3],
}


def _python(body: object) -> dict:
    v = capture_verdict(body)
    # The Node answer is `{mayUpload, reason, answered, mayCaptureModelContent,
    # modelContentReason}`; compared field for field under one spelling so a
    # difference in any field is a failure here rather than a difference in
    # how the two are read.
    return {"mayUpload": v.may_upload, "reason": v.reason,
            "answered": v.answered,
            "mayCaptureModelContent": v.may_capture_model_content,
            "modelContentReason": v.model_content_reason}


@pytest.fixture(scope="module")
def node_answers() -> dict[str, dict]:
    """Every fixture answered by the Node SDK, in ONE child process."""
    sdk, node = require_node_sdk()
    script = (
        "import {captureVerdict} from './src/capture.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const cases = JSON.parse(raw);\n"
        "const out = {};\n"
        "for (const [name, body] of Object.entries(cases)) "
        "out[name] = captureVerdict(body);\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    answers = json.loads(run_node(sdk, node, script, stdin=json.dumps(CASES)))
    assert set(answers) == set(CASES), (
        "the Node side did not answer every fixture: "
        f"{sorted(set(CASES) - set(answers))}")
    return answers


#: Cases where the two languages give the same VERDICT and a different
#: SENTENCE, with the reason, and with `test_the_known_divergences_still_
#: diverge` below to delete the entry the day it stops being true.
#:
#: EMPTY, AND THAT IS THE POINT: every case below is now compared field for
#: field. It held one entry, "capture is a list" -- `typeof [] === "object"`
#: in JavaScript, so Node's `captureVerdict` accepted a JSON ARRAY as a
#: capture block, found no `turns` on it and fell through to "production
#: capture is not enabled for this agent", naming a switch when what actually
#: happened is that the control plane sent something that is not an answer.
#: Node gained the `Array.isArray(capture)` guard this entry asked for, so the
#: entry came out, which is the whole life-cycle this list exists to have.
#: Both sides always refused, so the safety property never depended on it.
_REASON_DIVERGES: set[str] = set()


@pytest.mark.parametrize("name", list(CASES))
def test_both_sdks_reach_the_same_capture_verdict(name, node_answers):
    """Both fields, including the sentence explaining a refusal.

    ``mayUpload`` is the half that protects an end user's message and it is
    compared for EVERY case; the sentence is compared everywhere the two
    languages can express the same answer.
    """
    mine = _python(CASES[name])
    theirs = node_answers[name]
    assert mine["mayCaptureModelContent"] == theirs["mayCaptureModelContent"], (
        f"the two SDKs disagree about whether {name!r} is MODEL-CONTENT "
        f"consent:\n  python: {json.dumps(mine, ensure_ascii=False)}\n"
        f"  node  : {json.dumps(theirs, ensure_ascii=False)}"
    )
    assert mine["mayUpload"] == theirs["mayUpload"], (
        f"the two SDKs disagree about whether {name!r} is consent:\n"
        f"  python: {json.dumps(mine, ensure_ascii=False)}\n"
        f"  node  : {json.dumps(theirs, ensure_ascii=False)}"
    )
    if name in _REASON_DIVERGES:
        return
    assert mine == theirs, (
        f"the two SDKs disagree about {name!r}:\n"
        f"  python: {json.dumps(mine, ensure_ascii=False)}\n"
        f"  node  : {json.dumps(theirs, ensure_ascii=False)}"
    )


def test_the_known_divergences_still_diverge(node_answers):
    """The exception list cleans itself up.

    An entry that no longer diverges is a stale exception hiding real parity,
    so this goes red the day the Node side is fixed and the fix is to delete
    the entry -- which is what makes the list above a mechanism rather than a
    comment.

    An empty list must not make this test VACUOUS, which is what a bare loop
    over an empty set would be -- it would pass just as happily if someone
    emptied the list to silence a real divergence. So the empty case asserts
    the thing that makes emptiness legitimate: that every case agrees in full.
    """
    for name in _REASON_DIVERGES:
        assert _python(CASES[name]) != node_answers[name], (
            f"{name!r} no longer diverges -- remove it from _REASON_DIVERGES")
    if not _REASON_DIVERGES:
        disagreeing = {
            name: {"python": _python(CASES[name]), "node": node_answers[name]}
            for name in CASES
            if _python(CASES[name]) != node_answers[name]
        }
        assert not disagreeing, (
            "_REASON_DIVERGES is empty, which claims total parity, but these "
            f"cases disagree:\n{json.dumps(disagreeing, indent=2, ensure_ascii=False)}")


def test_the_table_reaches_every_branch(node_answers):
    """A table that had drifted to one outcome would still be green."""
    grants = [n for n, a in node_answers.items() if a["mayUpload"]]
    unanswered = [n for n, a in node_answers.items()
                  if not a["mayUpload"] and "did not answer" in (a["reason"] or "")]
    refusals = [n for n, a in node_answers.items()
                if not a["mayUpload"] and n not in unanswered]
    assert grants and unanswered and refusals, (
        f"grants={grants} unanswered={unanswered} refusals={refusals}")


def test_every_refusal_carries_a_sentence():
    """``upload_refusal`` is documented as "the sentence explaining a False".

    It was ``None`` for the two commonest refusals there are -- a control plane
    too old to send the block, and one that sent no reason -- so an operator
    following the README got no cause, no remedy, and no way to tell those two
    apart from a key with the wrong scope.
    """
    for name, body in CASES.items():
        v = capture_verdict(body)
        assert (v.reason is None) == v.may_upload, name
        if not v.may_upload:
            assert v.reason and v.reason.strip(), name
        # The second tier keeps the same promise, and never outruns the first.
        assert (v.model_content_reason is None) == v.may_capture_model_content, name
        if not v.may_capture_model_content:
            assert v.model_content_reason and v.model_content_reason.strip(), name
        assert not v.may_capture_model_content or v.may_upload, name


def test_model_content_is_granted_only_by_the_exact_value_on_a_granted_turn():
    """The second tier's own branches, so a table that drifted to one outcome
    for it cannot stay green: one grant, and every way to be refused."""
    granted = [n for n, b in CASES.items()
               if capture_verdict(b).may_capture_model_content]
    assert granted == ["model content granted"], granted
    withheld = capture_verdict(
        CASES["model content withheld, with the server's sentence"])
    assert withheld.model_content_reason == (
        "an operator has not enabled model-content capture")
    riding = capture_verdict(CASES["model content true but turns refused"])
    assert riding.model_content_reason == riding.reason == "the key lacks the scope"
