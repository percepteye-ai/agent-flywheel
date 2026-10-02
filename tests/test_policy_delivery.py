"""The last mile: the agent finds out about a model the customer approved.

Training produced better models for months and none reached a customer's
process. ``serve()`` discarded the registration response and nothing ever
asked what to run, so a trained adapter arrived only when a human edited a
config by hand.

What is delivered is an ENDPOINT, not an artifact URI. ``gs://…`` is not
something an agent can call, so the payload leads with an OpenAI-compatible
``base_url`` + ``model`` and the artifact rides along as provenance.

Every test here is one property: **the control plane must never be able to
stop a customer's agent from serving its users.** Not by being down, not by
erroring, not by being an older build.
"""
from __future__ import annotations

import json
import pathlib

import pytest

import agent_flywheel as fw
from agent_flywheel.transport import AttachTransport


class _Client:
    def __init__(self, answer=None, boom=False):
        self.answer, self.boom, self.calls = answer, boom, []

    def request(self, method, path, **kw):
        self.calls.append((method, path))
        if self.boom:
            raise RuntimeError("control plane unreachable")
        return self.answer

    def close(self):
        pass


def _tr(**kw):
    return AttachTransport(_Client(**kw), agent_id="coding-agent")


def test_the_endpoint_is_what_comes_back():
    """The deliverable an agent can act on."""
    answer = {
        "arm": "candidate",
        "endpoint": {"base_url": "https://gpu.example/v1",
                     "model": "run_x_policy",
                     "api_key_env": "PERCEPTEYE_API_KEY"},
        "reason": "an operator put this agent on the candidate",
    }
    got = _tr(answer=answer).policy_current()
    assert got["endpoint"]["base_url"] == "https://gpu.example/v1"
    assert got["endpoint"]["model"] == "run_x_policy"


def test_it_asks_the_right_route():
    tr = _tr(answer={})
    tr.policy_current()
    assert tr._c.calls == [("GET", "/policy/current")]


def test_an_unreachable_control_plane_does_not_raise():
    """A customer's agent must keep serving its users when we are down. An
    exception here would take their production down with ours."""
    assert _tr(boom=True).policy_current() == {}


def test_a_null_answer_becomes_an_empty_dict():
    """So callers can write ``if pol.get("endpoint")`` without a None check
    that someone will forget."""
    assert _tr(answer=None).policy_current() == {}


def test_current_policy_without_configuration_is_empty_not_an_error(monkeypatch):
    """Called from a customer's own agent with no key or URL set: return
    nothing and let them keep their configuration. Raising would make
    adopting this a risk rather than an improvement."""
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    assert fw.current_policy(agent_id="coding-agent") == {}


def test_current_policy_is_exported():
    """It is the seam a customer's production agent calls — a helper they
    cannot import is one that does not exist."""
    assert "current_policy" in fw.__all__
    assert callable(fw.current_policy)


def test_serve_tolerates_a_transport_without_the_method():
    """``policy_current`` is not on the FlywheelTransport Protocol, so a
    third-party or older transport need not implement it. Calling it
    directly made every such transport crash at startup."""
    class _Old:
        pass

    ask = getattr(_Old(), "policy_current", None)
    assert ask is None          # the shape serve() guards against


# ── NVIDIA NIM (NemoClaw / OpenClaw) must not sample off-policy ───────


def test_nvidia_keys_are_treated_as_unroutable():
    """NemoClaw runs OpenClaw with inference through build.nvidia.com, and
    configures it ONCE at onboarding into the OpenShell provider registry —
    NOT the process environment. So our per-rollout OPENAI_BASE_URL does not
    win, and without this the agent samples NVIDIA cloud while the control
    plane believes it is sampling the policy under training. Every rollout
    would be off-policy and nothing would raise."""
    from agent_flywheel.isolation import UNROUTABLE_PROVIDER_VARS

    assert "NVIDIA_API_KEY" in UNROUTABLE_PROVIDER_VARS
    assert "NVIDIA_NIM_API_KEY" in UNROUTABLE_PROVIDER_VARS


def test_an_nvidia_key_is_reported_at_startup():
    """Visible to the person who made the misconfiguration, rather than
    surfacing a run later as a training-grade exclusion nobody can explain."""
    from agent_flywheel.isolation import unroutable_providers_present

    found = unroutable_providers_present({"NVIDIA_API_KEY": "nvapi-xxx"})
    assert found == ("NVIDIA_API_KEY",)


def test_an_nvidia_key_does_not_reach_the_child():
    """Removing the key is the reliable half — a provider we cannot redirect
    must not be usable from inside the rollout."""
    from agent_flywheel.isolation import build_child_env

    env = build_child_env(
        rollout_id="r1", llm_url="http://gw/v1", api_key="k",
        trajectory_dir="/tmp/l",
        parent_env={"NVIDIA_API_KEY": "nvapi-xxx", "PATH": "/usr/bin"},
    )
    assert "NVIDIA_API_KEY" not in env
    assert env["OPENAI_BASE_URL"] == "http://gw/v1"


# ── what the control plane delivers, and how it is taken back ─────────
#
# Both bodies are the control plane's own answers, vendored byte for byte
# with the Node SDK and the control plane (tests/cross_repo/
# test_fixture_parity.py holds the copies equal). The gateway answer is how
# every model PerceptEye serves arrives: one constant URL for the agent,
# opened with the flywheel key the agent already holds. The withdrawal is the
# answer to a rollback onto a base arm that has no champion to serve.

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def flywheel_key(monkeypatch):
    """The agent's flywheel key, the credential the gateway answer names. Set
    here rather than inherited, so a developer's own key cannot decide a
    test."""
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pefw_x")
    return "pefw_x"


def _answer(name):
    return json.loads((FIXTURES / name).read_bytes())


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _handle(answers, *, on_change=None):
    """A PolicySource fed ``answers`` in order, one per refresh."""
    clock = _Clock()
    queue = list(answers)
    src = fw.PolicySource(
        agent_id="support-agent", on_change=on_change,
        _now=clock, _fetch=lambda: queue.pop(0), _rand=lambda a, b: 0.0,
    )
    return src, clock


def _next(src, clock):
    clock.t += 10_000.0          # past every TTL: this call refreshes
    return src.current()


def _triple(snap):
    return (snap.base_url, snap.api_key, snap.model)


def test_the_gateway_delivery_is_opened_with_the_flywheel_key(monkeypatch, flywheel_key):
    """A platform-served policy names PERCEPTEYE_API_KEY, and the agent's own
    provider keys sitting in the environment must not win. Handing the
    gateway an OpenAI key would 401 every completion of a production agent
    that had just been delivered a better model."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-provider")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-provider")
    body = _answer("policy_current_gateway.json")
    ep = body["endpoint"]
    expected = {"base_url": ep["base_url"], "api_key": flywheel_key, "model": ep["model"]}

    assert fw.endpoint_kwargs(body) == expected

    src, _ = _handle([body])
    snap = src.current()
    assert snap.changed
    assert snap.as_kwargs() == expected


def test_a_withdrawal_after_a_candidate_reverts_to_the_constructor_time_configuration(
        flywheel_key):
    """The rollback that used to never arrive. An agent moved back to a base
    arm with no champion was told `endpoint: null`, which this handle read as
    "keep what you run" -- and what it ran was the candidate a human had just
    rolled back. The withdrawal flag makes it a change, back to the snapshot
    the handle was built with: your own configuration."""
    seen = []
    withdrawn = _answer("policy_current_withdrawn.json")
    src, clock = _handle(
        [_answer("policy_current_gateway.json"), withdrawn], on_change=seen.append)
    assert src.current().base_url

    snap = _next(src, clock)

    built_with = fw.PolicySource(agent_id="support-agent", _fetch=dict).current()
    assert snap.changed
    assert _triple(snap) == _triple(built_with) == ("", "", "")
    assert not snap
    assert snap.as_kwargs() == {}
    assert snap.last_refresh_ok
    assert snap.serving_generation == withdrawn["serving_generation"]
    assert _triple(seen[-1]) == ("", "", "")
    # Reported once: the next read is not another rebuild.
    assert not src.current().changed


def test_a_null_endpoint_without_the_withdrawal_flag_keeps_the_last_configuration(
        flywheel_key):
    """Nothing deployed is NOT a withdrawal. An answer with no endpoint for
    any other reason must not tear a working client down to the untrained
    model."""
    gateway = _answer("policy_current_gateway.json")
    nothing = _answer("policy_current_withdrawn.json")
    del nothing["withdrawn"]
    src, clock = _handle([gateway, nothing])
    first = src.current()
    assert first.base_url == gateway["endpoint"]["base_url"]

    snap = _next(src, clock)

    assert not snap.changed
    assert _triple(snap) == _triple(first)
    assert snap.serving_generation == nothing["serving_generation"]


def test_a_withdrawal_before_any_policy_is_not_a_change(flywheel_key):
    """Nothing was handed out, so the caller is still on its own
    configuration and has nothing to rebuild."""
    w = _answer("policy_current_withdrawn.json")
    src, clock = _handle([w, w])
    assert not src.current().changed
    assert not _next(src, clock).changed


def test_a_second_withdrawal_is_not_a_second_change(flywheel_key):
    w = _answer("policy_current_withdrawn.json")
    src, clock = _handle([_answer("policy_current_gateway.json"), w, w])
    src.current()
    assert _next(src, clock).changed
    assert not _next(src, clock).changed


def test_a_served_policy_after_a_withdrawal_is_delivered_again(flywheel_key):
    gateway = _answer("policy_current_gateway.json")
    src, clock = _handle(
        [gateway, _answer("policy_current_withdrawn.json"), gateway])
    src.current()
    _next(src, clock)

    snap = _next(src, clock)

    assert snap.changed
    assert snap.base_url == gateway["endpoint"]["base_url"]


def test_withdrawn_wins_over_an_endpoint_beside_it(flywheel_key):
    """Both SDKs refuse the same answers: the Node SDK's modelDecision asks
    `withdrawn === true` before it looks at any endpoint, so this one asks it
    too, in the one place a body becomes client kwargs."""
    gateway = _answer("policy_current_gateway.json")
    both = {**gateway, "withdrawn": True}

    assert fw.endpoint_kwargs(both) == {}

    src, clock = _handle([gateway, both])
    src.current()
    snap = _next(src, clock)
    assert snap.changed
    assert _triple(snap) == ("", "", "")


def test_only_a_literal_true_is_a_withdrawal():
    """`withdrawn === true` in Node, so a truthy string is not the flag."""
    assert fw.is_withdrawn({"withdrawn": True})
    for not_it in ({"withdrawn": "true"}, {"withdrawn": 1}, {"withdrawn": False},
                   {}, None, "withdrawn"):
        assert not fw.is_withdrawn(not_it), not_it


def test_an_arm_change_behind_the_same_gateway_url_is_a_change(flywheel_key):
    """Every model PerceptEye serves arrives at the SAME constant gateway URL,
    so a promotion or a rollback between two served models moves only the
    model. Comparing base_url alone would miss every one of them."""
    candidate = _answer("policy_current_gateway.json")
    champion = json.loads(json.dumps(candidate))
    champion["arm"] = "base"
    champion["reason"] = "champion"
    champion["endpoint"]["model"] = "policy_support_agent_0c1d2e3f4a5b6c7d"
    champion["serving_generation"] = "b52d10f8e3c96a47"
    src, clock = _handle([candidate, champion])
    first = src.current()

    snap = _next(src, clock)

    assert snap.base_url == first.base_url
    assert snap.changed
    assert snap.model == "policy_support_agent_0c1d2e3f4a5b6c7d"


def test_the_same_answers_as_the_node_sdk(flywheel_key):
    """Both SDKs ship in one release and apply the same answers, so the
    comments above saying "as in Node" are held by a mechanism: one table,
    driven through Node's modelDecision and through endpoint_kwargs, which
    must agree on WHICH answers are applied."""
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    gateway = _answer("policy_current_gateway.json")
    withdrawn = _answer("policy_current_withdrawn.json")
    nothing = {k: v for k, v in withdrawn.items() if k != "withdrawn"}
    table = {
        "gateway": gateway,
        "withdrawn": withdrawn,
        "nothing deployed": nothing,
        "withdrawn beside an endpoint": {**gateway, "withdrawn": True},
        "withdrawn false": {**gateway, "withdrawn": False},
        "withdrawn as a string": {**gateway, "withdrawn": "true"},
    }
    script = (
        "import {modelDecision} from './src/policy.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const out = {};\n"
        "for (const [k, body] of Object.entries(JSON.parse(raw)))"
        " out[k] = modelDecision(body);\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    theirs = json.loads(run_node(sdk, node, script, stdin=json.dumps(table)))
    for row, body in table.items():
        assert bool(fw.endpoint_kwargs(body)) == theirs[row]["apply"], (
            f"the two SDKs disagree about {row!r}: node said "
            f"apply={theirs[row]['apply']} ({theirs[row]['reason']})")
