"""ONE reading of what an agent's environment states, in two languages.

Six tables, each driven through this SDK and through the Node SDK, and the
two answers compared:

* ``OUTCOME_FILES`` -- raw ``outcome.json`` texts, through
  ``runner._attach_reported_outcome`` and Node's ``takeReportedOutcome``. The
  file is written by the customer's agent in any language; which runner drives
  the agent must not change what the file says.
* ``SEEDS`` -- through ``contract.seeds_to_wire`` and ``seedsToWire``. Python
  once took any ``int`` and no ``float``; Node took any safe integer.
* ``OCCURRED_AT`` -- through the outcome and turn-feedback validators of both.
  Both SDKs REQUIRE the time (the contract they author says so) while the
  control plane's intake accepts a report without one; what is pinned here is
  that the two SDKs refuse the same reports in the same words.
* ``CHECKS`` -- what a report's ``checks`` may be, through
  ``RolloutOutput.to_wire`` and ``rolloutOutputToWire``.
* ``TEXTS`` -- ids and notes holding a lone surrogate, which the control
  plane's parser refuses; both SDKs now refuse them at the call.
* ``EPISODE_IDS`` -- through ``current_rollout_id`` and the Node plugin's
  ``currentRolloutId``: the episode id an outcome is reported against. Node
  once had no producer of it for application code.

The skip rule is ``tests/node_sdk.py``'s.
"""
from __future__ import annotations

import json

import pytest

from agent_flywheel.contract import (
    ContractViolation, RolloutOutput, production_outcome, seeds_to_wire,
    turn_feedback,
)
from agent_flywheel.runner import OUTCOME_FILENAME, _attach_reported_outcome

from .node_sdk import require_node_sdk, run_node

_CHECK = {"id": "row", "status": "abstained"}


def _file(**parts) -> str:
    return json.dumps(parts)


#: name -> the file's text, exactly as an agent might write it.
OUTCOME_FILES: dict[str, str] = {
    "every part, well-formed": _file(
        reward=0.5, success=True, checks=[_CHECK],
        termination={"class": "completed", "detail": "done"},
        seeds={"environment": 7, "sampling": "abc"}),
    "a whole reward": _file(reward=1),
    "a zero reward": _file(reward=0.0),
    "a negative reward": _file(reward=-2.5),
    "a reward spelled as a string": _file(reward="1"),
    "a reward that is a boolean": _file(reward=True),
    "a reward that overflows a float": '{"reward": 1e999}',
    "a reward of NaN": '{"reward": NaN, "success": true}',
    "a reward of Infinity": '{"reward": Infinity, "success": false}',
    "a reward of -Infinity": '{"reward": -Infinity}',
    "a constant's name inside a string": _file(
        reward=1, seeds={"environment": "NaN Infinity"}),
    "an integer reward too large to be a number": (
        '{"reward": 1' + "0" * 400 + ', "success": true}'),
    "a success that is not a boolean": _file(reward=1, success="yes"),
    "a success of zero": _file(success=0),
    "a null reward and success": _file(reward=None, success=None),
    "a whole float seed": '{"seeds": {"environment": 7.0, "sampling": 1e3}}',
    "a negative zero seed": '{"seeds": {"environment": -0.0}}',
    "the largest seed a number carries": (
        '{"seeds": {"environment": 9007199254740991}}'),
    "a seed past what a number carries": (
        '{"seeds": {"environment": 9007199254740992}, "reward": 1}'),
    "a far larger seed": '{"seeds": {"sampling": 1' + "0" * 30 + "}}",
    "a fractional seed": _file(seeds={"environment": 1.5}),
    "a boolean seed": _file(seeds={"environment": True}),
    "seeds that are not an object": _file(seeds=[1, 2], reward=1),
    "seeds with an unknown key": _file(seeds={"environment": 1, "other": 2}),
    "empty seeds": _file(seeds={}),
    "seeds of nulls": _file(seeds={"environment": None, "sampling": None}),
    "a seed string at the limit": _file(seeds={"environment": "s" * 128}),
    "a seed string past the limit": _file(seeds={"environment": "s" * 129}),
    "an astral seed string at the limit": _file(
        seeds={"environment": "\U0001F600" * 128}),
    "an astral seed string past the limit": _file(
        seeds={"environment": "\U0001F600" * 129}),
    "an empty seed string": _file(seeds={"sampling": ""}),
    "more checks than a report carries": _file(checks=[_CHECK] * 203),
    "a malformed row inside the limit": _file(
        reward=1, checks=[_CHECK, {"id": "x", "status": "nonsense"}]),
    "a malformed row past the limit": _file(
        checks=[_CHECK] * 200 + [{"status": "nonsense"}]),
    "a measured check with its value": _file(checks=[
        {"id": "m", "status": "measured", "value": 0.25, "required": True}]),
    "a value on a check that was not measured": _file(checks=[
        {"id": "m", "status": "error", "value": 0.25}]),
    "a check value of NaN": '{"checks": [{"id": "m", "status": "measured", "value": NaN}]}',
    "every field of a check": _file(checks=[
        {"id": "m", "status": "measured", "value": 1, "basis": "b",
         "required": False, "establishes": "state_reached"}]),
    "a check proving something unnamed": _file(checks=[
        {**_CHECK, "establishes": "nonsense"}]),
    "a check with an unknown key": _file(checks=[{**_CHECK, "other": 1}]),
    "two checks with one id": _file(checks=[_CHECK, _CHECK]),
    "no checks": _file(checks=[], reward=1),
    "null checks": _file(checks=None),
    "checks that are not a list": _file(checks=_CHECK, success=True),
    "a check that is not an object": _file(checks=["measured"]),
    "a nonsense termination class": _file(
        termination={"class": "nonsense"}, reward=1),
    "a termination without a class": _file(termination={"detail": "x"}),
    "a termination that is not an object": _file(termination="completed"),
    "a termination detail at the limit": _file(
        termination={"class": "completed", "detail": "d" * 500}),
    "a termination detail past the limit": _file(
        termination={"class": "completed", "detail": "d" * 501}),
    "an unknown top-level key": _file(reward=1, other={"a": 1}),
    "a duplicated key": '{"reward": 1, "reward": 2}',
    "a list": "[1, 2]",
    "a string": '"reward"',
    "a number": "1",
    "null": "null",
    "nothing": "",
    "whitespace": "  \n",
    "a byte-order mark": "﻿" + _file(reward=1),
    "not JSON": "{reward: 1}",
    "a trailing comma": '{"reward": 1,}',
    "a cut-off file": '{"reward": 1, "checks": [',
}

SEEDS: dict[str, object] = {
    "both": {"environment": 7, "sampling": "abc"},
    "one": {"sampling": 3},
    "none": {},
    "nulls": {"environment": None, "sampling": None},
    "zero": {"environment": 0},
    "negative": {"environment": -5},
    "a whole float": {"environment": 7.0},
    "the largest": {"environment": 2**53 - 1},
    "the smallest": {"environment": -(2**53 - 1)},
    "past the largest": {"environment": 2**53},
    "a fraction": {"environment": 0.5},
    "a boolean": {"sampling": False},
    "a list": {"sampling": [1]},
    "an object": {"sampling": {"a": 1}},
    "an empty string": {"environment": ""},
    "a string at the limit": {"environment": "s" * 128},
    "a string past the limit": {"environment": "s" * 129},
    "an astral string at the limit": {"sampling": "\U0001F600" * 128},
    "a padded string": {"sampling": "  a  "},
    "an unknown key": {"environment": 1, "seed": 2},
    "not an object": [1, 2],
    "a string, not an object": "7",
    "null": None,
}

#: name -> the ``occurred_at`` a report states; ``_ABSENT`` leaves it out.
_ABSENT = "<absent>"
OCCURRED_AT: dict[str, object] = {
    "UTC": "2026-11-10T09:00:00Z",
    "an offset": "2026-11-10T09:00:00+05:30",
    "a negative offset": "2026-11-10T09:00:00-08:00",
    "fractional seconds": "2026-11-10T09:00:00.123456Z",
    "a leap day": "2028-02-29T00:00:00Z",
    "absent": _ABSENT,
    "null": None,
    "no offset": "2026-11-10T09:00:00",
    "a date alone": "2026-11-10",
    "a space for the T": "2026-11-10 09:00:00Z",
    "a lower-case z": "2026-11-10T09:00:00z",
    "an offset without its colon": "2026-11-10T09:00:00+0530",
    "a day that does not exist": "2026-02-30T09:00:00Z",
    "a leap day in a common year": "2026-02-29T00:00:00Z",
    "hour 24": "2026-11-10T24:00:00Z",
    "a leap second": "2026-11-10T23:59:60Z",
    "an offset of a day": "2026-11-10T09:00:00+24:00",
    "year zero": "0000-01-01T00:00:00Z",
    "other digits": "２０２６-11-10T09:00:00Z",
    "a trailing newline": "2026-11-10T09:00:00Z\n",
    "epoch seconds": 1794301200,
    "an empty string": "",
}

#: name -> what an agent passes as a report's checks. An empty list states
#: nothing; anything else that is not a list is refused by both -- Python once
#: read ``{}`` and ``""`` as nothing while Node refused them.
CHECKS: dict[str, object] = {
    "absent": None,
    "an empty list": [],
    "an empty object": {},
    "an empty string": "",
    "zero": 0,
    "false": False,
    "one row": [_CHECK],
    "an object holding a row": {"0": _CHECK},
}


_NODE = r"""
import { mkdtempSync, writeFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { takeReportedOutcome } from './src/trajectory.js';
import {
  seedsToWire, productionOutcome, turnFeedback, rolloutOutputToWire,
} from './src/wire.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const { files, seeds, outcomes, feedback, checks, texts } = JSON.parse(raw);
const answer = (fn) => {
  try { return { ok: fn() }; } catch (err) { return { refused: err.message }; }
};
const out = {
  files: {}, seeds: {}, outcomes: {}, feedback: {}, checks: {}, texts: {},
};
const dir = mkdtempSync(join(tmpdir(), 'outcome-parity-'));
try {
  for (const [name, text] of Object.entries(files)) {
    writeFileSync(join(dir, 'outcome.json'), text);
    out.files[name] = takeReportedOutcome(dir);
  }
} finally {
  rmSync(dir, { recursive: true, force: true });
}
for (const [name, value] of Object.entries(seeds)) {
  out.seeds[name] = answer(() => seedsToWire(value));
}
for (const [name, wire] of Object.entries(outcomes)) {
  out.outcomes[name] = answer(() => productionOutcome(wire));
}
for (const [name, wire] of Object.entries(feedback)) {
  out.feedback[name] = answer(() => turnFeedback(wire));
}
for (const [name, [kind, wire]] of Object.entries(texts)) {
  out.texts[name] = answer(() => (
    kind === 'outcome' ? productionOutcome(wire) : turnFeedback(wire)));
}
for (const [name, value] of Object.entries(checks)) {
  out.checks[name] = answer(() => {
    const wire = rolloutOutputToWire({ finalText: 'x', checks: value });
    return { checks: wire.checks ?? null, checks_dropped: wire.checks_dropped ?? null };
  });
}
process.stdout.write(JSON.stringify(out));
"""


def _stating(base: dict, occurred_at: object) -> dict:
    return base if occurred_at == _ABSENT else {**base, "occurred_at": occurred_at}


_OUTCOME = {"outcome_id": "o-1", "outcome": "resolved", "episode_id": "e-1"}
_FEEDBACK = {"feedback_id": "f-1", "signal": "accepted", "episode_id": "e-1",
             "turn_index": 0}
OUTCOMES = {n: _stating(_OUTCOME, v) for n, v in OCCURRED_AT.items()}
FEEDBACK = {n: _stating(_FEEDBACK, v) for n, v in OCCURRED_AT.items()}

_LONE = "case-\ud800"
_PAIRED = "case-\U0001F600"
_AT = {"occurred_at": "2026-11-10T09:00:00Z"}
#: name -> ("outcome" | "feedback", the report). A LONE SURROGATE in an id or
#: a note is not text a JSON parser takes: both SDKs once accepted it and the
#: control plane refused the batch (422). A paired one is an emoji, and kept.
TEXTS: dict[str, tuple[str, dict]] = {
    **{f"a lone surrogate in an outcome's {field}": ("outcome", {
        **_OUTCOME, **_AT, field: _LONE}) for field in (
            "outcome_id", "turn_id", "conversation_id", "episode_id", "note")},
    **{f"a lone surrogate in feedback's {field}": ("feedback", {
        **_FEEDBACK, **_AT, field: _LONE}) for field in (
            "feedback_id", "episode_id")},
    **{f"an emoji in an outcome's {field}": ("outcome", {
        **_OUTCOME, **_AT, field: _PAIRED}) for field in (
            "outcome_id", "turn_id", "conversation_id", "episode_id", "note")},
    **{f"an emoji in feedback's {field}": ("feedback", {
        **_FEEDBACK, **_AT, field: _PAIRED}) for field in (
            "feedback_id", "episode_id")},
}




@pytest.fixture(scope="module")
def node() -> dict[str, dict]:
    sdk, binary = require_node_sdk()
    return json.loads(run_node(sdk, binary, _NODE, stdin=json.dumps({
        "files": OUTCOME_FILES, "seeds": SEEDS, "outcomes": OUTCOMES,
        "feedback": FEEDBACK, "checks": CHECKS, "texts": TEXTS})))


def _python_reads(text: str, tmp_path) -> dict:
    """What this SDK takes from the file, in the Node reader's field names
    and with what was not stated left out."""
    (tmp_path / OUTCOME_FILENAME).write_text(text, encoding="utf-8")
    out = RolloutOutput(final_text="")
    _attach_reported_outcome(out, str(tmp_path))
    stated = {"reward": out.reward, "success": out.success,
              "checks": out.checks, "checksDropped": out.checks_dropped or None,
              "termination": out.termination, "seeds": out.seeds}
    return {k: v for k, v in stated.items() if v is not None}


def _answer(fn, value) -> dict:
    try:
        return {"ok": fn(value)}
    except ContractViolation as exc:
        return {"refused": str(exc)}


@pytest.mark.parametrize("name", sorted(OUTCOME_FILES))
def test_both_sdks_read_the_same_outcome_file(node, name, tmp_path):
    python = _python_reads(OUTCOME_FILES[name], tmp_path)
    # Compared as JSON values: 0.0 and 0 are one number on the wire.
    assert node["files"][name] == python, (
        f"{name}: Node read {node['files'][name]!r}, Python {python!r}")


def test_the_outcome_table_is_not_agreement_on_nothing(node, tmp_path):
    """Both readers answering ``{}`` to everything would pass the table."""
    assert node["files"]["every part, well-formed"] == {
        "reward": 0.5, "success": True, "checks": [_CHECK],
        "termination": {"class": "completed", "detail": "done"},
        "seeds": {"environment": 7, "sampling": "abc"}}
    assert node["files"]["a whole float seed"] == {
        "seeds": {"environment": 7, "sampling": 1000}}
    over = node["files"]["more checks than a report carries"]
    assert len(over["checks"]) == 200 and over["checksDropped"] == 3
    assert node["files"]["a reward of NaN"] == {"success": True}
    assert node["files"]["a seed past what a number carries"] == {"reward": 1}
    assert node["files"]["not JSON"] == {}


@pytest.mark.parametrize("name", sorted(SEEDS))
def test_both_sdks_take_the_same_seeds(node, name):
    python = _answer(seeds_to_wire, SEEDS[name])
    theirs = node["seeds"][name]
    # A refusal is compared as a refusal: the sentence about an unknown key
    # lists it in each language's own spelling.
    assert set(theirs) == set(python), (
        f"{name}: Node {theirs!r}, Python {python!r}")
    assert theirs.get("ok") == python.get("ok")


def test_the_seed_table_refuses_and_accepts(node):
    assert node["seeds"]["a whole float"] == {"ok": {"environment": 7}}
    assert "refused" in node["seeds"]["past the largest"]
    assert node["seeds"]["the largest"] == {"ok": {"environment": 2**53 - 1}}


@pytest.mark.parametrize("name", sorted(OCCURRED_AT))
def test_both_sdks_require_the_same_time_of_an_outcome(node, name):
    python = _answer(production_outcome, OUTCOMES[name])
    assert node["outcomes"][name] == python, (
        f"{name}: Node {node['outcomes'][name]!r}, Python {python!r}")


@pytest.mark.parametrize("name", sorted(OCCURRED_AT))
def test_both_sdks_require_the_same_time_of_turn_feedback(node, name):
    python = _answer(turn_feedback, FEEDBACK[name])
    assert node["feedback"][name] == python, (
        f"{name}: Node {node['feedback'][name]!r}, Python {python!r}")


@pytest.mark.parametrize("kind", ["outcomes", "feedback"])
def test_a_report_without_its_time_is_refused_and_one_with_it_is_not(node, kind):
    for name in ("absent", "null", "no offset", "a date alone"):
        assert "occurred_at must be an RFC 3339 time" in (
            node[kind][name]["refused"]), name
    assert node[kind]["UTC"]["ok"]["occurred_at"] == "2026-11-10T09:00:00Z"
    assert node[kind]["an offset"]["ok"]["occurred_at"] == (
        "2026-11-10T09:00:00+05:30")


#: name -> (the rollout a session is running or None, the session asking,
#: whether that rollout has ended). Python reads its context; Node is asked
#: with the host context the rollout's session is keyed by.
EPISODE_IDS: dict[str, tuple[str | None, str, bool]] = {
    "outside any rollout": (None, "s1", False),
    "inside a rollout": ("rl_mc_41", "s1", False),
    "after the rollout ended": ("rl_mc_41", "s1", True),
    "another session's rollout": ("rl_mc_41", "s2", False),
}

_NODE_EPISODE = r"""
import { register } from "./src/index.js";
const cases = JSON.parse(await new Promise((ok) => {
  let s = ""; process.stdin.on("data", (d) => { s += d; });
  process.stdin.on("end", () => ok(s));
}));
const api = {
  on() {}, logger: { info() {}, warn() {} },
  config: { plugins: { entries: { "agent-flywheel": {
    hooks: { allowConversationAccess: true } } } } },
  session: { workflow: { enqueueNextTurnInjection: async () => ({ enqueued: true }) } },
};
const out = {};
for (const [name, c] of Object.entries(cases.cases)) {
  const plugin = register(api, {
    mode: "training", apiKey: "pefw_test",
    controlPlaneUrl: "http://127.0.0.1:9/api/flywheel/v1",
  }, { PERCEPTEYE_HOME: cases.home, PERCEPTEYE_TRAJECTORY_DIR: "" });
  const [rollout, asking, ended] = c;
  if (rollout) plugin.rollouts.track("s1", { rolloutId: rollout });
  if (ended) plugin.rollouts.inFlight.delete("s1");
  out[name] = plugin.currentRolloutId({ sessionKey: asking });
  plugin.rollouts.inFlight.clear();
}
process.stdout.write(JSON.stringify(out));
"""


def _python_episode(rollout, asking, ended, trajectory_dir) -> str | None:
    import contextvars

    from agent_flywheel.outcomes import _RolloutScope, current_rollout_id

    # A session is a context of its own: Python's rollout lives in the one it
    # was entered in, as Node's lives in the host session it was claimed for.
    sessions = {"s1": contextvars.copy_context(), "s2": contextvars.copy_context()}
    scope = _RolloutScope(rollout, trajectory_dir) if rollout else None
    if scope is not None:
        sessions["s1"].run(scope.__enter__)
        if ended:
            sessions["s1"].run(scope.__exit__, None, None, None)
    return sessions[asking].run(current_rollout_id)


def test_both_sdks_name_the_same_episode(tmp_path, monkeypatch):
    monkeypatch.delenv("PERCEPTEYE_ROLLOUT_ID", raising=False)
    sdk, binary = require_node_sdk()
    theirs = json.loads(run_node(sdk, binary, _NODE_EPISODE, stdin=json.dumps({
        "home": str(tmp_path / "home"), "cases": EPISODE_IDS})))
    for name, (rollout, asking, ended) in EPISODE_IDS.items():
        python = _python_episode(rollout, asking, ended, str(tmp_path))
        assert theirs[name] == python, (
            f"{name}: Node {theirs[name]!r}, Python {python!r}")
    # Not agreement on nothing: inside a rollout both name it.
    assert theirs["inside a rollout"] == "rl_mc_41"


def _python_checks(value: object) -> dict:
    try:
        wire = RolloutOutput(final_text="x", checks=value).to_wire()
    except ContractViolation as exc:
        return {"refused": str(exc)}
    return {"ok": {"checks": wire.get("checks"),
                   "checks_dropped": wire.get("checks_dropped")}}


@pytest.mark.parametrize("name", sorted(CHECKS))
def test_both_sdks_read_the_same_checks(node, name):
    python = _python_checks(CHECKS[name])
    assert node["checks"][name] == python, (
        f"{name}: Node {node['checks'][name]!r}, Python {python!r}")


def test_the_checks_table_refuses_and_accepts(node):
    assert node["checks"]["an empty list"] == {
        "ok": {"checks": None, "checks_dropped": None}}
    assert "refused" in node["checks"]["an empty object"]
    assert node["checks"]["one row"]["ok"]["checks"] == [
        {"id": "row", "status": "abstained"}]


@pytest.mark.parametrize("name", sorted(TEXTS))
def test_both_sdks_refuse_the_same_text(node, name):
    kind, wire = TEXTS[name]
    python = _answer(production_outcome if kind == "outcome" else turn_feedback,
                     wire)
    assert node["texts"][name] == python, (
        f"{name}: Node {node['texts'][name]!r}, Python {python!r}")
    assert ("refused" in python) == name.startswith("a lone surrogate"), name
