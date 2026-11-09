"""ONE prompt predicate, in two languages, proved to agree.

``prompt_decision`` here and ``promptDecision`` in the Node package answer the
same question -- may this served prompt be applied, and if not why not -- for
the same customer. A customer runs one package or the other and may migrate
between them, so if the two disagreed, "may I apply this bundle?" would be a
question about which language you happened to install. Worse in one direction
than the other: the SDK that wrongly says yes puts an unapproved or mispaired
prompt in front of end users.

So the fixtures below are driven through BOTH implementations in one pass and
every field of the answer is compared -- ``apply``, ``text``, ``checksum`` AND
``reason``. The reason strings are shared word for word on purpose, which makes
this test the mechanism behind that claim rather than a comment asserting it.

THE SKIP TRAP. ``test-full`` fails the build on ANY skip, so a test that always
skips turns the build red while proving nothing. The rule for when this may
skip lives in ``tests/node_sdk.py`` and is the same one
``scripts/cross_language_abi_check.sh`` uses: auto-discovery skips, a NAMED
``PERCEPTEYE_NODE_SDK`` that is not there fails. CI names it.
"""
from __future__ import annotations

import dataclasses
import json

import pytest

from agent_flywheel import prompt_decision

from .node_sdk import require_node_sdk, run_node

_SHA = "a" * 64
_OTHER = "b" * 64

#: Every case both sides have to answer identically: the one that applies, each
#: of the four refusals, the ORDER between them when two would fire, and the
#: type edges where two languages most easily drift apart.
CASES: dict[str, object] = {
    # ── withdrawn: a person took the approved prompt back out of service ──
    "withdrawn, none approved in its place": {
        "withdrawn": True, "generation": "w1",
    },
    "withdrawn wins over any text beside it": {
        "withdrawn": True, "text": "t", "status": "approved",
    },
    "withdrawn must be the boolean true": {
        "withdrawn": "true", "text": "t", "status": "approved",
    },
    # ── applies ──────────────────────────────────────────────────────────
    "approved and paired": {
        "name": "support-triage", "version": 3, "sha256": _SHA,
        "text": "You triage carefully.", "candidate_kind": "replacement",
        "status": "approved", "model_prompt_checksum": _SHA,
        "stale_baseline": False,
    },
    "delivered is servable too": {
        "text": "t", "status": "delivered", "sha256": _SHA,
        "model_prompt_checksum": _SHA,
    },
    # CANNOT PROVE is not PROVEN WRONG. Refusing on a blank half would refuse
    # every bundle that predates the stamp.
    "no checksum to compare against": {
        "text": "t", "status": "approved", "sha256": _SHA,
    },
    "no sha to compare with": {
        "text": "t", "status": "approved", "model_prompt_checksum": _SHA,
    },
    "checksum blank after trimming": {
        "text": "t", "status": "approved", "sha256": _SHA,
        "model_prompt_checksum": "   ",
    },
    # The text is applied AS SERVED. Leading whitespace in an approved prompt
    # is part of what was certified, and neither side gets to edit it.
    "surrounding whitespace in the text survives": {
        "text": "\n  You triage carefully.  \n", "status": "approved",
    },
    "status is trimmed, not rejected": {
        "text": "t", "status": "  approved  ",
    },
    # `false` is the ordinary value of this field; only `true` refuses.
    "stale_baseline false": {
        "text": "t", "status": "approved", "stale_baseline": False,
    },

    # ── refusal 1: nothing to apply ──────────────────────────────────────
    "the empty answer": {},
    "no text key at all": {"status": "approved", "sha256": _SHA},
    "text present but blank": {"text": "   \n ", "status": "approved"},
    "text is not a string": {"text": 123, "status": "approved"},
    "text is null": {"text": None, "status": "approved"},
    "the body is null": None,
    "the body is a string": "not a body",
    "the body is a list": [1, 2, 3],
    # An empty answer still reports the checksum it carried, if any.
    "empty text but a checksum": {
        "text": "", "status": "approved", "model_prompt_checksum": _SHA,
    },

    # ── refusal 2: the baseline moved ────────────────────────────────────
    "stale baseline": {
        "text": "t", "status": "approved", "sha256": _SHA,
        "model_prompt_checksum": _SHA, "stale_baseline": True,
    },
    # STRICTLY true. A truthy 1 is not the flag, on either side -- and a side
    # that coerced would refuse a bundle the other applied.
    "stale_baseline is 1, not true": {
        "text": "t", "status": "approved", "stale_baseline": 1,
    },
    "stale_baseline is the STRING 'true'": {
        "text": "t", "status": "approved", "stale_baseline": "true",
    },
    # ORDER: a stale baseline outranks a bad status. Both sides must name the
    # same one of the two causes, or an operator is sent to fix the other.
    "stale baseline AND a bad status": {
        "text": "t", "status": "draft", "stale_baseline": True,
    },

    # ── refusal 3: not an approved status ────────────────────────────────
    "status draft": {"text": "t", "status": "draft"},
    "status missing": {"text": "t"},
    "status blank": {"text": "t", "status": "  "},
    "status is not a string": {"text": "t", "status": 7},
    "status is null": {"text": "t", "status": None},
    "status differs by case": {"text": "t", "status": "Approved"},
    # A quote in the value: both sides render it the same way or the message
    # itself has drifted.
    "status contains a quote": {"text": "t", "status": 'dr"aft'},
    # ORDER: a bad status outranks a mismatched pair.
    "bad status AND a mismatched pair": {
        "text": "t", "status": "draft", "sha256": _SHA,
        "model_prompt_checksum": _OTHER,
    },

    # ── refusal 4: the bundle disagrees with itself ──────────────────────
    "sha does not match the checksum": {
        "text": "t", "status": "approved", "sha256": _SHA,
        "model_prompt_checksum": _OTHER,
    },
    "mismatch after trimming both": {
        "text": "t", "status": "approved", "sha256": f"  {_SHA} ",
        "model_prompt_checksum": f" {_OTHER}  ",
    },
    # Equal only after trimming: that is the same pair, not a mismatch.
    "equal once trimmed": {
        "text": "t", "status": "approved", "sha256": f" {_SHA}",
        "model_prompt_checksum": f"{_SHA} ",
    },
}


def _python(body: object) -> dict:
    return dataclasses.asdict(prompt_decision(body))  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def node_answers() -> dict[str, dict]:
    """Every fixture answered by the Node SDK, in ONE child process.

    One process rather than one per case: the fixtures are a table, and paying
    a node startup per row would make the table something a future author
    trims.
    """
    sdk, node = require_node_sdk()
    script = (
        "import {promptDecision} from './src/policy.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const cases = JSON.parse(raw);\n"
        "const out = {};\n"
        "for (const [name, body] of Object.entries(cases)) "
        "out[name] = promptDecision(body);\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    out = run_node(sdk, node, script, stdin=json.dumps(CASES))
    answers = json.loads(out)
    assert set(answers) == set(CASES), (
        "the Node side did not answer every fixture: "
        f"{sorted(set(CASES) - set(answers))}")
    return answers


@pytest.mark.parametrize("name", list(CASES))
def test_both_sdks_reach_the_same_prompt_decision(name, node_answers):
    """Field for field, including the sentence explaining a refusal."""
    mine = _python(CASES[name])
    theirs = node_answers[name]
    assert mine == theirs, (
        f"the two SDKs disagree about {name!r}:\n"
        f"  python: {json.dumps(mine, ensure_ascii=False, indent=2)}\n"
        f"  node  : {json.dumps(theirs, ensure_ascii=False, indent=2)}"
    )


#: How each refusal opens. Five causes, five openings -- a refusal that named
#: one wrong cause for all five would tell an operator to go looking in the
#: wrong place four times out of five.
CAUSE_OPENINGS = (
    "a person withdrew the approved prompt",
    "the control plane is serving no approved prompt",
    "stale_baseline=true:",
    "the served bundle's status is",
    "the bundle disagrees with itself:",
)


def test_the_table_reaches_every_branch(node_answers):
    """A parity table that had drifted to one refusal would still be green.

    Both implementations have six outcomes -- apply, and five causes. This
    asserts the fixtures reach all five ON THE NODE SIDE, so the coverage claim
    is made against the implementation this file cannot otherwise see.
    """
    assert [n for n, a in node_answers.items() if a["apply"]], (
        "no fixture applies -- the happy path is untested")
    seen = [a["reason"] for a in node_answers.values() if not a["apply"]]
    for opening in CAUSE_OPENINGS:
        assert any(r.startswith(opening) for r in seen), (
            f"no fixture reaches the refusal opening {opening!r}")
    unclassified = [r for r in seen if not r.startswith(CAUSE_OPENINGS)]
    assert not unclassified, (
        "the Node SDK refused with a cause this table does not know:\n"
        + "\n".join(sorted(set(unclassified))))


# ── the refresh rule: when a running agent re-reads, and what it then holds ──

from agent_flywheel.prompt import (  # noqa: E402
    PENDING_PROMPT,
    WITHDRAWN_REASON,
    prompt_needs_refresh,
    refreshed_prompt,
)

_APPLIED = {"apply": True, "text": "OLD", "checksum": _SHA, "reason": ""}
_PENDING = dataclasses.asdict(PENDING_PROMPT)
_NEW = {"text": "NEW", "status": "approved", "sha256": _OTHER,
        "model_prompt_checksum": _OTHER, "generation": "g2"}

#: (held decision, held generation, /policy/current body, /prompt/current
#: answer or None). Each row is one situation a running agent meets.
REFRESH_CASES: dict[str, list] = {
    "same prompt served: nothing to read":
        [_APPLIED, "g1", {"prompt_generation": "g1"}, None],
    "a newer approval is served and read":
        [_APPLIED, "g1", {"prompt_generation": "g2"}, _NEW],
    "a newer approval, its read failed or came back empty":
        [_APPLIED, "g1", {"prompt_generation": "g2"}, {}],
    "a newer approval, the read returned junk":
        [_APPLIED, "g1", {"prompt_generation": "g2"}, "junk"],
    "nothing served any more, after a prompt was held":
        [_APPLIED, "g1", {}, None],
    "nothing served, nothing ever held":
        [_PENDING, "", {"prompt_generation": None}, None],
    "an older control plane with no field":
        [_PENDING, "", {"arm": "base"}, None],
    "first prompt served to a fresh handle":
        [_PENDING, "", {"prompt_generation": "g2"}, _NEW],
    "the newer bundle is stale: refused, held at its generation":
        [_APPLIED, "g1", {"prompt_generation": "g2"},
         {**_NEW, "stale_baseline": True}],
    "an answer without a generation is held at the served one":
        [_APPLIED, "g1", {"prompt_generation": "g2"},
         {k: v for k, v in _NEW.items() if k != "generation"}],
    "the generation is trimmed on both sides":
        [_APPLIED, " g1 ", {"prompt_generation": "g1  "}, None],
    "a non-string generation is no generation":
        [_APPLIED, "g1", {"prompt_generation": 7}, None],
    # A person decided it, so it is the one answer that takes an applied
    # prompt away -- unlike an empty or failed read just above.
    "withdrawn after a prompt was applied: the host's own prompt runs":
        [_APPLIED, "g1", {"prompt_generation": "w1"},
         {"withdrawn": True, "generation": "w1"}],
    "withdrawn before anything was applied":
        [_PENDING, "", {"prompt_generation": "w1"},
         {"withdrawn": True, "generation": "w1"}],
    "a withdrawal without a generation is held at the served one":
        [_APPLIED, "g1", {"prompt_generation": "w1"}, {"withdrawn": True}],
}


def _py_refresh(held, held_gen, policy, answer) -> dict:
    from agent_flywheel import PromptDecision

    decision, gen = refreshed_prompt(
        PromptDecision(**held), held_gen, policy, answer)
    return {"needs": prompt_needs_refresh(held_gen, policy),
            "decision": dataclasses.asdict(decision), "generation": gen}


@pytest.fixture(scope="module")
def node_refresh_answers() -> dict[str, dict]:
    sdk, node = require_node_sdk()
    script = (
        "import {promptNeedsRefresh, refreshedPrompt} from './src/policy.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const cases = JSON.parse(raw);\n"
        "const out = {};\n"
        "for (const [name, [held, gen, policy, answer]] of Object.entries(cases)) {\n"
        "  const [decision, generation] = refreshedPrompt(held, gen, policy, answer);\n"
        "  out[name] = {needs: promptNeedsRefresh(gen, policy), decision, generation};\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    answers = json.loads(run_node(sdk, node, script,
                                  stdin=json.dumps(REFRESH_CASES)))
    assert set(answers) == set(REFRESH_CASES)
    return answers


@pytest.mark.parametrize("name", list(REFRESH_CASES))
def test_both_sdks_refresh_a_running_prompt_the_same_way(
        name, node_refresh_answers):
    """When to re-read, and what to hold after, field for field: a customer
    who migrates SDKs must not find a newer approval reaching one language's
    agents and not the other's, nor an empty read blanking one and not the
    other."""
    mine = _py_refresh(*REFRESH_CASES[name])
    theirs = node_refresh_answers[name]
    assert mine == theirs, (
        f"the two SDKs disagree about {name!r}:\n"
        f"  python: {json.dumps(mine, indent=2)}\n"
        f"  node  : {json.dumps(theirs, indent=2)}")


def test_the_refresh_table_reaches_every_outcome():
    """Adopt a newer one, refuse a stale one, keep on an empty read, keep on
    absence, and the fresh-start empty answer -- each reached at least once."""
    outs = {n: _py_refresh(*c) for n, c in REFRESH_CASES.items()}
    assert any(o["decision"]["text"] == "NEW" for o in outs.values())
    assert any(not o["decision"]["apply"]
               and "stale_baseline=true" in o["decision"]["reason"]
               for o in outs.values())
    assert any(o["needs"] and o["decision"]["text"] == "OLD"
               for o in outs.values())
    assert any(not o["needs"] and o["decision"]["text"] == "OLD"
               for o in outs.values())
    assert any("empty answer" in o["decision"]["reason"] for o in outs.values())
    assert any(o["decision"]["reason"] == WITHDRAWN_REASON
               and not o["decision"]["apply"] and o["decision"]["text"] is None
               and o["generation"] == "w1"
               for o in outs.values())
