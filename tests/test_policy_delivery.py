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

#: Where the agent's flywheel key already goes: the documented control-plane
#: URL, at the gateway fixture's origin. The flywheel key opens an endpoint
#: only at this origin.
CONTROL_PLANE = "https://launch.percepteye.ai/api/flywheel/v1"

#: Every provider variable a test here could read, cleared so a developer's
#: own environment cannot decide one.
_PROVIDER_VARS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "TOGETHER_API_KEY")


@pytest.fixture
def flywheel_key(monkeypatch):
    """The agent's flywheel key, the credential the gateway answer names, and
    the control plane it authenticates to. Set here rather than inherited, so
    a developer's own key cannot decide a test."""
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pefw_x")
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", CONTROL_PLANE)
    for name in _PROVIDER_VARS:
        monkeypatch.delenv(name, raising=False)
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


# ── where the flywheel key may go ─────────────────────────────────────
#
# A `pefw_` key used to be fallback (4) of the credential order, so it went to
# any endpoint the control plane did not give a working key for, and the
# caller's own flywheel key -- the `api_key=` of apply_current_policy() and
# policy_source() -- beat a provider's key as if it were an endpoint
# override. The Node SDK's flywheelKeyDecision already refused all of it.


def _provider(**ep):
    return {"arm": "candidate", "reason": "policy_channel_candidate",
            "endpoint": {"base_url": "https://api.openai.com/v1",
                         "model": "ft:gpt-4o-mini:acme::abc123", **ep}}


def _at(base_url, **ep):
    """The gateway answer with its endpoint moved to ``base_url``; a field
    given as None is left out, as the control plane omits one."""
    body = _answer("policy_current_gateway.json")
    merged = {**body["endpoint"], "base_url": base_url, **ep}
    body["endpoint"] = {k: v for k, v in merged.items() if v is not None}
    return body


#: Each is an answer the flywheel key used to be sent for, and must not be.
_NOT_FOR_THE_FLYWHEEL_KEY = {
    # A provider whose key travels in the body: the control plane OMITS
    # api_key_env because it can name no variable -- never "use yours".
    "provider, no variable named": _provider(),
    # The variable the control plane named is not set in this process.
    "provider, its variable unset": _provider(api_key_env="OPENAI_API_KEY"),
    # The same two at the control plane's own origin, where only the NAME can
    # refuse the key: an absent name is not ours, and neither is another one.
    "control plane's origin, no variable named": _at(
        "https://launch.percepteye.ai/llm/v1/o/x/agents/support-agent/v1",
        api_key_env=None),
    "control plane's origin, another variable unset": _at(
        "https://launch.percepteye.ai/llm/v1/o/x/agents/support-agent/v1",
        api_key_env="OPENAI_API_KEY"),
    # A self-hosted box's raw address, named PERCEPTEYE_API_KEY by a control
    # plane without the gateway projection, exactly as it names the gateway.
    "self-hosted raw address": _at("http://203.0.113.7:8000/v1"),
    # The gateway's path at a look-alike origin.
    "look-alike origin": _at(
        "https://launch.percepteye.ai.evil/llm/v1/o/x/agents/support-agent/v1"),
    # The control plane's host over another scheme.
    "plain http to the control plane's host": _at(
        "http://launch.percepteye.ai/llm/v1/o/x/agents/support-agent/v1"),
    # The control plane's host on another port.
    "another port": _at(
        "https://launch.percepteye.ai:8443/llm/v1/o/x/agents/support-agent/v1"),
}


@pytest.mark.parametrize("row", sorted(_NOT_FOR_THE_FLYWHEEL_KEY))
def test_the_flywheel_key_opens_only_the_control_plane_s_own_origin(
        row, flywheel_key, caplog):
    """The key goes only where the transport already sends it. Each of these
    answers used to return ``api_key='pefw_x'`` for an address that is not
    the control plane's: a provider's API, a GPU box, a look-alike host. Not
    applying them is the legible answer -- the agent keeps the model it is
    on, and the log line says why -- where sending the key handed a
    credential to whoever answers there."""
    body = _NOT_FOR_THE_FLYWHEEL_KEY[row]
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        assert fw.endpoint_kwargs(body) == {}, row
        assert fw.endpoint_kwargs(body, control_plane_url=CONTROL_PLANE) == {}, row
    assert flywheel_key not in caplog.text
    assert "no credential for its endpoint" in caplog.text, caplog.text

    src, _ = _handle([body])
    assert not src.current(), row


def test_the_refusal_names_why_the_key_was_withheld(flywheel_key, caplog):
    """THE PATTERN Q6: `{}` reads as "nothing deployed", so the line names
    the actual cause -- the address, and the one address the key goes to."""
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        fw.endpoint_kwargs(_at("http://203.0.113.7:8000/v1"))
        fw.endpoint_kwargs(_provider())
        fw.endpoint_kwargs(_provider(api_key_env="OPENAI_API_KEY"))
    lines = [r.getMessage() for r in caplog.records]
    assert any("http://203.0.113.7:8000" in m and "https://launch.percepteye.ai"
               in m and "not sent there" in m for m in lines), lines
    assert any("names no variable" in m for m in lines), lines
    assert any("OPENAI_API_KEY" in m and "is not set" in m for m in lines), lines


def test_the_gateway_at_the_same_origin_spelled_differently_gets_the_key(
        flywheel_key):
    """One origin however it is spelled: the host's case and a default port
    are not a different address."""
    body = _at("https://Launch.PerceptEye.ai:443/llm/v1/o/x/agents/support-agent/v1")
    assert fw.endpoint_kwargs(body)["api_key"] == flywheel_key


def test_the_default_control_plane_is_where_the_key_goes(flywheel_key, monkeypatch):
    """With no control-plane URL given, the key's home is the URL every reader
    resolves, the default included, as the Node plugin's resolved
    ``config.controlPlaneUrl`` is: the transport sends the key there on every
    call, so the gateway URL there gets it. Reading the variable alone left an
    agent on the default with no home, and refused it every model we serve."""
    from urllib.parse import urlsplit

    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL")
    body = _answer("policy_current_gateway.json")
    assert (urlsplit(body["endpoint"]["base_url"]).netloc
            == urlsplit(fw.DEFAULT_CONTROL_PLANE_URL).netloc)
    assert fw.endpoint_kwargs(body)["api_key"] == flywheel_key

    # And a handle built with no URL asks the same resolver.
    src = fw.PolicySource(agent_id="support-agent", api_key=flywheel_key,
                          _fetch=lambda: body)
    assert src.current().api_key == flywheel_key


def test_the_default_control_plane_does_not_open_another_address(
        flywheel_key, monkeypatch):
    """The default is a home, not a wildcard: an endpoint named for the key at
    any other address is still refused."""
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL")
    body = _answer("policy_current_gateway.json")
    body["endpoint"]["base_url"] = "http://10.0.0.7:8000/v1"
    assert fw.endpoint_kwargs(body) == {}


def test_the_callers_flywheel_key_is_never_a_provider_key(monkeypatch, flywheel_key):
    """``api_key=`` on policy_source() and apply_current_policy() is the key
    to the CONTROL PLANE. It used to be handed to endpoint_kwargs as the
    caller's endpoint override, which beat both the key the control plane
    sent and the provider's own variable -- so a provider endpoint was opened
    with a `pefw_` key, sending it to the provider and 401ing every request.
    At the gateway it is still the key used."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-provider")
    sent = _provider(api_key_env="OPENAI_API_KEY", api_key="sk-vault")
    named = _provider(api_key_env="OPENAI_API_KEY")
    gateway = _answer("policy_current_gateway.json")

    for body, want in ((sent, "sk-vault"), (named, "sk-provider"),
                       (gateway, "pefw_cp")):
        src = fw.PolicySource(
            agent_id="support-agent", control_plane_url=CONTROL_PLANE,
            api_key="pefw_cp", _fetch=lambda b=body: b, _rand=lambda a, b: 0.0)
        assert src.current().api_key == want, body["endpoint"]["base_url"]

        monkeypatch.setattr(fw, "current_policy", lambda b=body, **kw: b)
        got = fw.apply_current_policy(
            agent_id="support-agent", control_plane_url=CONTROL_PLANE,
            api_key="pefw_cp")
        assert got["api_key"] == want, body["endpoint"]["base_url"]


@pytest.mark.parametrize("env_url, arg_url, applied", [
    (None, CONTROL_PLANE, True),
    ("https://elsewhere.example/api/flywheel/v1", CONTROL_PLANE, True),
    (CONTROL_PLANE, "https://elsewhere.example/api/flywheel/v1", False),
])
def test_the_control_plane_a_caller_names_is_where_the_key_may_go(
        monkeypatch, flywheel_key, env_url, arg_url, applied):
    """The origin asked is the control plane this caller authenticates to,
    so ``control_plane_url=`` wins over the environment exactly as it does
    for the GET itself -- in both callers."""
    if env_url is None:
        monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL")
    else:
        monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", env_url)
    gateway = _answer("policy_current_gateway.json")

    src = fw.PolicySource(
        agent_id="support-agent", control_plane_url=arg_url,
        _fetch=lambda: gateway, _rand=lambda a, b: 0.0)
    assert bool(src.current()) is applied

    monkeypatch.setattr(fw, "current_policy", lambda **kw: gateway)
    got = fw.apply_current_policy(agent_id="support-agent", control_plane_url=arg_url)
    assert bool(got) is applied


def test_an_endpoint_without_a_model_is_not_applied(flywheel_key):
    """An OpenAI client sends ``model`` on every request, and ``""`` names
    nothing; Node's modelDecision reads it as no endpoint, and so does this."""
    body = _answer("policy_current_gateway.json")
    del body["endpoint"]["model"]
    assert fw.endpoint_kwargs(body) == {}
    body["endpoint"]["model"] = ""
    assert fw.endpoint_kwargs(body) == {}
    src, _ = _handle([body])
    assert not src.current()


# ── a withdrawal is a flag, and it is said ────────────────────────────


def test_a_withdrawal_is_flagged_on_the_snapshot(flywheel_key):
    """The caller asks ``snap.withdrawn`` rather than inferring a withdrawal
    from an empty snapshot, which is also what "never delivered" looks like.
    It stays set while the agent runs its own configuration because of it --
    a later "nothing deployed" keeps that configuration, and its reason --
    and clears when a policy is served again."""
    gateway = _answer("policy_current_gateway.json")
    withdrawn = _answer("policy_current_withdrawn.json")
    nothing = {k: v for k, v in withdrawn.items() if k != "withdrawn"}
    src, clock = _handle([gateway, withdrawn, nothing, gateway])

    assert not src.current().withdrawn
    snap = _next(src, clock)
    assert snap.changed and snap.withdrawn and not snap
    assert src.current().withdrawn            # the edge is consumed, not the flag
    assert _next(src, clock).withdrawn
    back = _next(src, clock)
    assert back.changed and back and not back.withdrawn


def test_a_withdrawal_before_any_policy_is_flagged_but_not_a_change(flywheel_key):
    src, _ = _handle([_answer("policy_current_withdrawn.json")])
    snap = src.current()
    assert snap.withdrawn and not snap.changed


def test_a_withdrawal_is_said_once_naming_its_reason(flywheel_key, caplog):
    """THE PATTERN Q6: the empty snapshot and the one-shot `{}` are the same
    values "nothing deployed" produces, so the withdrawal is named -- once
    per change, not once per refresh."""
    w = _answer("policy_current_withdrawn.json")
    src, clock = _handle([_answer("policy_current_gateway.json"), w, w])
    src.current()
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        _next(src, clock)
        _next(src, clock)
    said = [r.getMessage() for r in caplog.records if "WITHDREW" in r.getMessage()]
    assert len(said) == 1, said
    assert w["reason"] in said[0] and "support-agent" in said[0]


def test_the_one_shot_read_names_a_withdrawal(monkeypatch, flywheel_key, caplog):
    w = _answer("policy_current_withdrawn.json")
    monkeypatch.setattr(fw, "current_policy", lambda **kw: w)
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        assert fw.apply_current_policy(agent_id="support-agent") == {}
    assert any("WITHDREW" in r.getMessage() and w["reason"] in r.getMessage()
               for r in caplog.records), caplog.text


def test_the_same_answers_as_the_node_sdk(flywheel_key):
    """Both SDKs ship in one release and must agree on two things for every
    answer: WHICH answers are applied, and WHERE the flywheel key goes. One
    table, driven through Node's modelDecision and through endpoint_kwargs,
    against the same control-plane URL.

    The one designed difference is who supplies a key that is not the
    flywheel key. Node leaves it to the host's own provider entry
    (``credential: null`` and ``apply: true``); Python applies only with a
    credential it holds, from the body or the variable the control plane
    named, and returns ``{}`` otherwise. So: Python never applies an answer
    Node refuses; Python opens the endpoint with the flywheel key exactly
    when Node names the flywheel key; and when Node names it, Python applies.
    """
    from .node_sdk import require_node_sdk, run_node

    sdk, node = require_node_sdk()
    gateway = _answer("policy_current_gateway.json")
    withdrawn = _answer("policy_current_withdrawn.json")
    nothing = {k: v for k, v in withdrawn.items() if k != "withdrawn"}
    no_model = json.loads(json.dumps(gateway))
    del no_model["endpoint"]["model"]
    table = {
        "gateway": gateway,
        "withdrawn": withdrawn,
        "nothing deployed": nothing,
        "withdrawn beside an endpoint": {**gateway, "withdrawn": True},
        "withdrawn false": {**gateway, "withdrawn": False},
        "withdrawn as a string": {**gateway, "withdrawn": "true"},
        "no model": no_model,
        "archive model": _at(gateway["endpoint"]["base_url"],
                             model="gs://bucket/run/iter_0"),
        "same origin spelled differently": _at(
            "https://Launch.PerceptEye.ai:443/llm/v1/o/x/agents/support-agent/v1"),
        "provider, key in the body": _provider(api_key="sk-vault"),
        **_NOT_FOR_THE_FLYWHEEL_KEY,
    }
    script = (
        "import {modelDecision} from './src/policy.js';\n"
        "let raw='';\n"
        "for await (const chunk of process.stdin) raw += chunk;\n"
        "const {table, controlPlaneUrl} = JSON.parse(raw);\n"
        "const out = {};\n"
        "for (const [k, body] of Object.entries(table))"
        " out[k] = modelDecision(body, {controlPlaneUrl});\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    theirs = json.loads(run_node(sdk, node, script, stdin=json.dumps(
        {"table": table, "controlPlaneUrl": CONTROL_PLANE})))
    for row, body in table.items():
        ours = fw.endpoint_kwargs(body, control_plane_url=CONTROL_PLANE)
        node_says = theirs[row]
        said = f"{row!r}: node said {node_says}, python returned {ours}"
        if not node_says["apply"]:
            assert ours == {}, f"python applied what node refuses -- {said}"
        named = node_says.get("credential") == "flywheel_key"
        assert (ours.get("api_key") == flywheel_key) == named, (
            f"the SDKs disagree about where the flywheel key goes -- {said}")
        if named:
            assert ours, f"node names the flywheel key, python refused -- {said}"
    # The table must exercise every branch it claims to.
    assert {r for r, d in theirs.items() if d["apply"] and d["credential"] is None} \
        >= set(_NOT_FOR_THE_FLYWHEEL_KEY)
    assert fw.endpoint_kwargs(table["provider, key in the body"])["api_key"] == "sk-vault"


def _wired(monkeypatch, cp, key):
    """A handle that builds its own client, as in production, and is handed
    ``cp`` for it."""
    monkeypatch.setattr("agent_flywheel._http.ControlPlaneClient", lambda **kw: cp)
    clock = _Clock()
    src = fw.PolicySource(
        agent_id="support-agent", control_plane_url=CONTROL_PLANE,
        api_key=key, _now=clock, _rand=lambda a, b: 0.0)
    return src, clock


def test_an_unreachable_control_plane_is_a_failed_refresh_not_an_empty_answer(
        monkeypatch, flywheel_key):
    """The handle's whole health signal. Its refresh used to go through the
    fail-open read, which turns any failure into ``{}`` -- and ``{}`` is a
    SUCCESSFUL answer of nothing deployed -- so a control plane that was down
    reported last_refresh_ok=True and stale_for_s=0.0 for as long as it was
    down, and a rollback it could not deliver looked delivered."""
    gateway = _answer("policy_current_gateway.json")
    cp = _Client(answer=gateway)
    src, clock = _wired(monkeypatch, cp, flywheel_key)
    first = src.current()
    assert first.changed and first.last_refresh_ok

    cp.boom = True
    snap = _next(src, clock)

    assert not snap.last_refresh_ok
    assert not snap.changed
    assert _triple(snap) == _triple(first)     # last good stands
    assert snap.stale_for_s > 0.0
    assert cp.calls == [("GET", "/policy/current")] * 2


def test_a_control_plane_never_reached_is_stale_from_the_start(monkeypatch, flywheel_key):
    cp = _Client(boom=True)
    src, clock = _wired(monkeypatch, cp, flywheel_key)

    assert not src.current().last_refresh_ok
    snap = _next(src, clock)
    assert not snap.last_refresh_ok
    assert snap.stale_for_s > 0.0
