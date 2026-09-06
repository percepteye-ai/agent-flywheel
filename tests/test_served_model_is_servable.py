"""A served ``model`` that is a stored artifact is refused, on every path.

THE MISSING HALF OF THE PAIR RULE. A promoted improvement is a (prompt,
checkpoint) pair and both halves are supposed to be refused on the same terms
by both SDKs. Python re-asserted the PROMPT half -- ``SERVABLE_PROMPT_STATUSES``
in ``prompt.py``, with a comment explaining exactly why a client re-asserts a
server-side rule -- and had nothing at all for the model half, while the README
stated it as a live guarantee:

    "An address that is a stored artifact rather than a running endpoint -- a
    bucket or file path -- is refused."

Nothing refused it. ``endpoint_kwargs`` returned ``model="gs://bucket/run/
iter_0"`` verbatim, the caller built ``OpenAI(model=...)`` from it, and every
completion 404'd. Node's ``modelDecision`` refuses the same payload with a named
reason and keeps the champion.

``PolicySource`` made it worse than a one-shot read: ``changed`` fires on the
first actionable answer, so the documented ``if snap.changed:`` loop would tear
down a WORKING client to rebuild against the unservable one -- and ``current()``
consumes the change edge, so it would never rebuild again.

WHY BOTH CONSUMERS ARE DRIVEN HERE. ``endpoint_kwargs`` is the seam, and a test
of the seam alone would pass with the guard moved to one of its two callers.
"""
from __future__ import annotations

import json

import pytest

import percepteye_agent_flywheel as fw
from percepteye_agent_flywheel.policy import PolicySource, is_archive_ref

#: Every prefix the control plane can hand back as an artifact reference, plus
#: the names that must keep working. A guard that refused too much would keep a
#: customer off a model they were promoted onto, which is the same outage from
#: the other side.
ARCHIVE = [
    "gs://percepteye-ckpts/run_x/iter_0",
    "s3://bucket/ckpt",
    "file:///var/lib/ckpt",
    "/mnt/checkpoints/iter_9",
]
SERVABLE = [
    "qwen3-coder-8b",
    "run_x-iter_0",
    "org/model:v3",
    "gpt-4o-mini",
    # NOT an archive ref: the prefix is `gs://`, not the letters `gs`.
    "gs-tuned-model",
]


def _policy(model: str) -> dict:
    return {"endpoint": {"base_url": "https://gpu.example/v1",
                         "model": model, "api_key": "sk-live"}}


@pytest.mark.parametrize("model", ARCHIVE)
def test_endpoint_kwargs_refuses_a_stored_artifact(model):
    """The seam both callers go through."""
    assert fw.endpoint_kwargs(_policy(model)) == {}, (
        f"{model!r} was handed to a caller as a model name; every completion "
        f"against that address 404s")


@pytest.mark.parametrize("model", SERVABLE)
def test_a_servable_name_is_still_applied(model):
    """The guard must not refuse a name an endpoint can actually serve."""
    assert fw.endpoint_kwargs(_policy(model))["model"] == model


@pytest.mark.parametrize("model", ARCHIVE)
def test_policy_source_never_hands_out_a_stored_artifact(model):
    """The SECOND consumer, and the one where the cost compounds.

    Without this the caller tears down a working client on ``changed`` and
    rebuilds against an address that answers nothing -- once, because the edge
    is consumed.
    """
    src = PolicySource(agent_id="a", _fetch=lambda: _policy(model))
    snap = src.current()
    assert snap.model == ""
    assert not snap.changed, (
        "a refused policy still reported a change, so a production caller "
        "would rebuild its client against the address that was refused")
    assert not snap, "a refused policy was truthy, so `if snap:` would apply it"


def test_apply_current_policy_refuses_it_too(monkeypatch):
    """The documented production snippet, end to end."""
    monkeypatch.setattr(fw, "current_policy",
                        lambda **kw: _policy("gs://bucket/run/iter_0"))
    assert fw.apply_current_policy(agent_id="a") == {}


def test_the_refusal_is_said_out_loud(caplog):
    """`{}` is also this function's ordinary "nothing is deployed" answer.

    An operator whose promotion never arrives has to be able to tell those two
    apart, so the refusal names the model and why it was refused.
    """
    with caplog.at_level("WARNING", logger="percepteye_agent_flywheel"):
        fw.endpoint_kwargs(_policy("gs://bucket/run/iter_0"))
    assert any("gs://bucket/run/iter_0" in r.getMessage()
               and "stored-artifact" in r.getMessage()
               for r in caplog.records), caplog.text


def test_the_same_answers_as_the_node_sdk():
    """Both SDKs refuse the same models, so migrating cannot change the answer.

    Node returns a decision object and Python returns kwargs, so what is
    compared is the part that has to agree: WHICH models are applied.
    """
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    script = (
        "import {modelDecision} from './src/policy.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const out = {};\n"
        "for (const m of JSON.parse(raw)) out[m] = modelDecision("
        "{endpoint: {base_url: 'https://gpu.example/v1', model: m}});\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    models = ARCHIVE + SERVABLE
    theirs = json.loads(run_node(sdk, node, script, stdin=json.dumps(models)))
    for model in models:
        assert bool(fw.endpoint_kwargs(_policy(model))) == theirs[model]["apply"], (
            f"the two SDKs disagree about {model!r}: node said "
            f"apply={theirs[model]['apply']} ({theirs[model]['reason']})")


def test_the_prefix_set_matches_the_node_sdks():
    """The set itself, not just this table's rows.

    A prefix added on one side and not the other is a divergence the table
    above would not see until somebody thought to add a row for it.
    """
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    theirs = json.loads(run_node(
        sdk, node,
        "import {ARCHIVE_REF_PREFIXES} from './src/policy.js';\n"
        "process.stdout.write(JSON.stringify(ARCHIVE_REF_PREFIXES));\n"))
    assert sorted(fw.ARCHIVE_REF_PREFIXES) == sorted(theirs)
    assert all(is_archive_ref(p + "x") for p in theirs)
