"""``agent-flywheel login``: a key for one agent, without leaving the terminal.

What each test pins, and the defect it exists to catch:

  * THE DEVICE FLOW against the control plane's own error vocabulary: the
    code and link are shown, the browser opened on the link, the interval
    honoured (``slow_down`` raises it for every later poll), and the key that
    comes back is stored for (control plane, agent) in a file only its owner
    can read. Denied, expired and a control plane that does not offer this
    client each end in a sentence that names the cause, and nothing stored.
  * ONE RULE FOR WHICH KEY: the argument, then ``PERCEPTEYE_API_KEY``, then the
    key stored for THIS agent on THIS control plane -- never another agent's,
    never an expired one, never read when either of the first two is set.
  * A STORED KEY IS NEVER SILENT: the first use says so, once, on stderr.
  * ``serve()`` OFFERS the login only to a person at a terminal: CI, a pipe,
    ``PERCEPTEYE_NO_PROMPT`` are never asked and get the refusal as before;
    a yes carries ``serve()`` on with the key it stored.
  * NO BODY CARRIES A KEY: what a rollout child prints after reading the
    stored file is redacted before it leaves (owner ruling R2b).
"""
from __future__ import annotations

import io
import json
import os
import stat
from typing import ClassVar

import httpx
import pytest

import agent_flywheel as fw
from agent_flywheel import credentials as creds
from agent_flywheel import login as login_mod
from agent_flywheel._http import REDACTED_KEY, ControlPlaneClient
from agent_flywheel.contract import CONTRACT_VERSION
from agent_flywheel.errors import ConfigurationError

CP = "https://cp.test/api/flywheel/v1"
DEVICE = "https://cp.test/api/v2/auth/device"
AGENT = "my-agent"
KEY = "pefw_" + "K" * 43


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A PERCEPTEYE_HOME of this test's own, no ambient key, nothing said."""
    root = tmp_path / "pe-home"
    monkeypatch.setenv("PERCEPTEYE_HOME", str(root))
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.setattr(creds, "_SAID", set())
    return root


class _ControlPlane:
    """The device-login half of a control plane, scripted per test."""

    def __init__(self, polls, *, code_status=200, code_body=None, interval=5):
        self.polls = list(polls)
        self.code_status = code_status
        self.code_body = code_body
        self.interval = interval
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        form = dict(httpx.QueryParams(request.content.decode()))
        if request.url.path.endswith("/code"):
            assert form["client_id"] == "agent-flywheel-python"
            assert form["scope"] == "flywheel-agent"
            if self.code_status != 200:
                return httpx.Response(self.code_status, json=self.code_body or {})
            return httpx.Response(200, json={
                "device_code": "dev-123", "user_code": "BCDF-GHJK",
                "verification_uri": "https://cp.test/device",
                "verification_uri_complete": "https://cp.test/device#user_code=BCDF-GHJK",
                "expires_in": 600, "interval": self.interval,
            })
        assert request.url.path.endswith("/token")
        assert form["device_code"] == "dev-123"
        step = self.polls.pop(0)
        if isinstance(step, Exception):
            raise step
        status, body = step
        return httpx.Response(status, json=body)


def _granted(**over):
    body = {
        "access_token": KEY, "token_type": "Bearer", "expires_in": 180 * 86400,
        "scope": "flywheel-agent", "org_id": "org-1", "org_name": "Acme",
        "agent_id": AGENT, "flywheel_key_id": "key-1",
    }
    body.update(over)
    return 200, body


def _pending():
    return 400, {"error": "authorization_pending", "error_description": "not yet"}


def _run(cp, **kw):
    out = io.StringIO()
    sleeps: list[float] = []
    opened: list[str] = []
    cred = login_mod.device_login(
        kw.pop("agent_id", AGENT), control_plane_url=CP, out=out,
        client=httpx.Client(transport=httpx.MockTransport(cp)),
        sleep=sleeps.append, clock=lambda: 0.0,
        browser=lambda url: opened.append(url) or True, **kw,
    )
    return cred, out.getvalue(), sleeps, opened


# ── the device flow ─────────────────────────────────────────────────


@pytest.mark.parametrize(("url", "endpoint"), [
    (CP, DEVICE),
    ("https://cp.test/api/flywheel/v1/", DEVICE),
    ("https://mc.example.com/prefix/api/flywheel/v1", "https://mc.example.com/prefix/api/v2/auth/device"),
    ("https://mc.example.com/somewhere/else", "https://mc.example.com/api/v2/auth/device"),
    ("https://launch.percepteye.ai/api/flywheel/v1", "https://launch.percepteye.ai/api/v2/auth/device"),
])
def test_the_device_login_is_found_beside_the_flywheel_api(url, endpoint):
    assert login_mod.device_endpoint(url) == endpoint


def test_a_login_shows_the_code_waits_its_interval_and_stores_the_key(home):
    cp = _ControlPlane([
        _pending(),
        (400, {"error": "slow_down", "interval": 10}),
        _pending(),
        _granted(),
    ])
    cred, out, sleeps, opened = _run(cp)

    assert "BCDF-GHJK" in out and AGENT in out
    assert opened == ["https://cp.test/device#user_code=BCDF-GHJK"]
    # 5, 5, then slow_down raised it to 10 for EVERY later poll.
    assert sleeps == [5, 5, 10, 10]
    assert cp.requests[0].url == f"{DEVICE}/code"
    assert cp.requests[0].headers["user-agent"].startswith("agent-flywheel-python/")
    assert dict(httpx.QueryParams(cp.requests[0].content.decode()))["agent_id"] == AGENT

    assert (cred.api_key, cred.agent_id, cred.org_name) == (KEY, AGENT, "Acme")
    path = home / "credentials.json"
    assert f"saved to {path}" in out
    mode = stat.S_IMODE(os.stat(path).st_mode)
    assert mode == 0o600, oct(mode)
    doc = json.loads(path.read_text())
    assert doc["version"] == 1
    (entry,) = doc["credentials"]
    assert entry["control_plane_url"] == CP
    assert entry["agent_id"] == AGENT and entry["api_key"] == KEY
    assert entry["key_id"] == "key-1" and entry["org_id"] == "org-1"
    assert entry["expires_at"].endswith("Z")
    assert not list(home.glob(".credentials.*")), "a temporary file was left behind"


def test_a_poll_that_cannot_reach_the_control_plane_is_retried(home):
    cp = _ControlPlane([httpx.ConnectError("down"), _granted()])
    cred, _, sleeps, _ = _run(cp)
    assert cred.api_key == KEY
    assert sleeps == [5, 10]


@pytest.mark.parametrize(("step", "said"), [
    ((400, {"error": "access_denied"}), "denied"),
    ((400, {"error": "expired_token"}), "expired"),
    ((400, {"error": "invalid_grant",
            "error_description": "the account that approved this login is no "
                                 "longer an admin of the organization it chose"}),
     "no longer an admin"),
])
def test_a_login_that_produces_no_key_says_why_and_stores_nothing(home, step, said):
    with pytest.raises(login_mod.LoginError, match=said):
        _run(_ControlPlane([step]))
    assert not (home / "credentials.json").exists()


def test_a_control_plane_without_this_client_is_named_and_sends_you_to_the_dashboard(home):
    cp = _ControlPlane([], code_status=400, code_body={
        "error": "invalid_client", "error_description": "unknown client_id"})
    with pytest.raises(login_mod.LoginError, match="does not offer device login"):
        _run(cp)


@pytest.mark.parametrize("bad", ["", "a/b", " pad", "..", "x" * 256])
def test_an_agent_id_no_agent_could_register_as_is_refused_before_any_request(home, bad):
    cp = _ControlPlane([])
    with pytest.raises(login_mod.LoginError, match="agent id"):
        _run(cp, agent_id=bad)
    assert cp.requests == []


# ── which key ───────────────────────────────────────────────────────


def _store(home, **over):
    fields = {"control_plane_url": CP, "agent_id": AGENT, "api_key": KEY,
              "expires_at": "2999-01-01T00:00:00Z"}
    fields.update(over)
    return creds.save_stored(creds.StoredCredential(**fields))


def test_the_argument_then_the_variable_then_the_stored_key(home, monkeypatch):
    _store(home)
    arg = creds.resolve_api_key("pefw_arg", agent_id=AGENT, control_plane_url=CP)
    assert (arg.value, arg.source) == ("pefw_arg", "argument")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pefw_env")
    env = creds.resolve_api_key(None, agent_id=AGENT, control_plane_url=CP)
    assert (env.value, env.source) == ("pefw_env", "environment")
    monkeypatch.delenv("PERCEPTEYE_API_KEY")
    stored = creds.resolve_api_key(None, agent_id=AGENT, control_plane_url=CP + "/")
    assert (stored.value, stored.source) == (KEY, "stored")


def test_a_stored_key_opens_only_its_own_agent_on_its_own_control_plane(home):
    _store(home)
    assert creds.resolve_api_key(None, agent_id="another", control_plane_url=CP) is None
    assert creds.resolve_api_key(
        None, agent_id=AGENT, control_plane_url="https://other.test/api/flywheel/v1",
    ) is None
    assert creds.resolve_api_key(None, agent_id=None, control_plane_url=CP) is None


def test_an_expired_stored_key_is_not_used_and_the_cause_says_it_expired(home):
    _store(home, expires_at="2001-01-01T00:00:00Z")
    assert creds.resolve_api_key(None, agent_id=AGENT, control_plane_url=CP) is None
    cause = fw.missing_credentials_cause(None, agent_id=AGENT, control_plane_url=CP)
    assert "expired on 2001-01-01" in cause
    assert f"agent-flywheel login --agent-id {AGENT}" in cause


def test_with_nothing_stored_the_cause_names_the_command_that_gets_a_key(home):
    cause = fw.missing_credentials_cause(None, agent_id=AGENT, control_plane_url=CP)
    assert cause.startswith("PERCEPTEYE_API_KEY is not set")
    assert f"no key is stored for agent {AGENT!r}" in cause
    assert f"agent-flywheel login --agent-id {AGENT}" in cause


def test_logging_in_again_replaces_that_agents_key_and_keeps_the_others(home):
    _store(home, api_key="pefw_old")
    _store(home, agent_id="second", api_key="pefw_second")
    _store(home, api_key=KEY)
    entries = json.loads((home / "credentials.json").read_text())["credentials"]
    assert sorted((e["agent_id"], e["api_key"]) for e in entries) == [
        (AGENT, KEY), ("second", "pefw_second")]
    assert creds.remove_stored(CP, AGENT) is True
    assert creds.remove_stored(CP, AGENT) is False
    second = creds.resolve_api_key(None, agent_id="second", control_plane_url=CP)
    assert second.value == "pefw_second"


def test_a_file_from_a_newer_sdk_is_neither_read_nor_overwritten(home, capsys):
    home.mkdir(parents=True)
    newer = {"version": 2, "credentials": [{"control_plane_url": CP,
                                             "agent_id": AGENT, "api_key": KEY}]}
    (home / "credentials.json").write_text(json.dumps(newer))
    assert creds.resolve_api_key(None, agent_id=AGENT, control_plane_url=CP) is None
    assert "version 2" in capsys.readouterr().err
    with pytest.raises(ConfigurationError, match="Upgrade agent-flywheel"):
        _store(home)
    assert json.loads((home / "credentials.json").read_text()) == newer


def test_a_stored_key_is_announced_once_on_stderr(home, capsys):
    path = _store(home)
    for _ in range(3):
        creds.resolve_api_key(None, agent_id=AGENT, control_plane_url=CP)
    err = capsys.readouterr().err
    assert err.count("using the key stored for agent") == 1, err
    assert str(path) in err and CP in err
    assert f"agent-flywheel logout --agent-id {AGENT}" in err


def test_current_policy_reads_with_the_stored_key(home, monkeypatch):
    _store(home)
    seen: dict = {}

    class _CP:
        def __init__(self, base_url, api_key, **kw):
            seen["key"] = api_key

        def request(self, method, path, **kw):
            return {"endpoint": None}

        def close(self):
            pass

    monkeypatch.setattr("agent_flywheel._http.ControlPlaneClient", _CP)
    fw.current_policy(agent_id=AGENT, control_plane_url=CP)
    assert seen["key"] == KEY


# ── serve() offers it, to a person, only ────────────────────────────


class _Tty(io.StringIO):
    def __init__(self, text="", tty=True):
        super().__init__(text)
        self._tty = tty

    def isatty(self):
        return self._tty


@pytest.mark.parametrize(("env", "stdin_tty", "stderr_tty", "expected"), [
    ({}, True, True, True),
    ({"CI": "true"}, True, True, False),
    ({"CI": "false"}, True, True, True),
    ({"GITHUB_ACTIONS": "true"}, True, True, False),
    ({"BUILDKITE": "1"}, True, True, False),
    ({"PERCEPTEYE_NO_PROMPT": "1"}, True, True, False),
    ({}, False, True, False),
    ({}, True, False, False),
])
def test_only_a_person_at_a_terminal_is_asked(env, stdin_tty, stderr_tty, expected):
    assert login_mod.interactive_terminal(
        environ=env, stdin=_Tty(tty=stdin_tty), stderr=_Tty(tty=stderr_tty),
    ) is expected


@pytest.mark.parametrize(("answer", "logs_in"), [
    ("\n", True), ("y\n", True), ("YES\n", True), ("n\n", False), ("", False),
])
def test_the_offer_logs_in_on_yes_and_enter_only(answer, logs_in):
    calls = []

    def _login(agent_id, **kw):
        calls.append(agent_id)
        return creds.StoredCredential(control_plane_url=CP, agent_id=agent_id, api_key=KEY)

    err = _Tty()
    got = login_mod.offer_login(
        AGENT, control_plane_url=CP, environ={}, stdin=_Tty(answer), stderr=err,
        login=_login,
    )
    assert "Create one now (free)? [Y/n]" in err.getvalue()
    assert (got == KEY) is logs_in and bool(calls) is logs_in


def test_a_failed_login_is_said_and_offers_nothing(home):
    def _login(agent_id, **kw):
        raise login_mod.LoginError("the login was denied in the browser")

    err = _Tty()
    assert login_mod.offer_login(
        AGENT, control_plane_url=CP, environ={}, stdin=_Tty("y\n"), stderr=err,
        login=_login,
    ) is None
    assert "denied in the browser" in err.getvalue()


class _FakeTransport:
    def __init__(self, client, agent_id):
        self.client = client

    def health(self):
        return {"contract_versions_supported": [CONTRACT_VERSION]}

    def register(self, spec):
        return {}

    def claim(self, max_rollouts):
        return [], 0


def _serve(monkeypatch, tmp_path):
    box = {}

    def _factory(client, agent_id):
        box["tr"] = _FakeTransport(client, agent_id)
        return box["tr"]

    monkeypatch.setattr(fw, "AttachTransport", _factory)
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", CP)
    fw.serve(lambda task: "ok", agent_id=AGENT, input_shape="text",
             artifacts_dir=str(tmp_path / "art"), max_rollouts=0, introspect=False,
             isolation="inprocess")
    return box["tr"]


def test_serve_with_no_key_and_no_person_raises_as_before_naming_login(home, monkeypatch, tmp_path):
    monkeypatch.setattr(login_mod, "interactive_terminal", lambda **kw: False)
    asked = []
    monkeypatch.setattr(login_mod, "device_login", lambda *a, **k: asked.append(a))
    with pytest.raises(ConfigurationError, match="an API key is required") as exc:
        _serve(monkeypatch, tmp_path)
    assert f"agent-flywheel login --agent-id {AGENT}" in str(exc.value)
    assert asked == []


def test_serve_carries_on_with_the_key_the_offer_stored(home, monkeypatch, tmp_path):
    def _offer(agent_id, *, control_plane_url):
        assert (agent_id, control_plane_url) == (AGENT, CP)
        return KEY

    monkeypatch.setattr(login_mod, "offer_login", _offer)
    tr = _serve(monkeypatch, tmp_path)
    assert tr.client._secret == KEY


def test_serve_uses_a_stored_key_without_asking(home, monkeypatch, tmp_path):
    _store(home)
    monkeypatch.setattr(login_mod, "offer_login",
                        lambda *a, **k: pytest.fail("asked with a key stored"))
    tr = _serve(monkeypatch, tmp_path)
    assert tr.client._secret == KEY


# ── no body carries a key ───────────────────────────────────────────


def test_a_key_a_rollout_printed_never_leaves_in_a_body():
    sent = []

    def _handler(request):
        sent.append(request)
        return httpx.Response(200, json={})

    env_key = "env-key-of-any-shape-0123456789"
    client = ControlPlaneClient(
        CP, env_key, client=httpx.Client(transport=httpx.MockTransport(_handler)),
    )
    other_agents_key = "pefw_" + "z" * 43
    client.request("POST", "/rollouts/r1/report", json_body={
        "output": f"$ cat ~/.percepteye/credentials.json\n{other_agents_key}",
        "steps": [{"stdout": f"PERCEPTEYE_API_KEY={env_key}"}],
    })
    body = json.loads(sent[0].content)
    assert other_agents_key not in sent[0].content.decode()
    assert env_key not in sent[0].content.decode()
    assert body["output"].endswith(REDACTED_KEY)
    assert body["steps"][0]["stdout"] == f"PERCEPTEYE_API_KEY={REDACTED_KEY}"


# ── the commands ────────────────────────────────────────────────────


def test_login_needs_an_agent_and_logout_forgets_only_locally(home, monkeypatch, capsys):
    from agent_flywheel import cli

    monkeypatch.delenv("PERCEPTEYE_AGENT_ID", raising=False)
    assert cli.main(["login"]) == 2
    assert "--agent-id" in capsys.readouterr().err

    seen = {}

    def _login(agent_id, **kw):
        seen.update(agent_id=agent_id, **kw)
        return _store(home)

    monkeypatch.setattr(login_mod, "device_login", _login)
    assert cli.main(["login", "--agent-id", AGENT, "--no-browser"]) == 0
    assert seen["agent_id"] == AGENT and seen["open_browser"] is False

    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", CP)
    assert cli.main(["logout", "--agent-id", AGENT]) == 0
    out = capsys.readouterr().out
    assert "forgot the key" in out and "revoked" in out
    assert creds.resolve_api_key(None, agent_id=AGENT, control_plane_url=CP) is None
    assert cli.main(["logout", "--agent-id", AGENT]) == 0
    assert "no key is stored" in capsys.readouterr().out


def test_a_refused_login_exits_non_zero_with_the_cause(home, monkeypatch, capsys):
    from agent_flywheel import cli

    def _login(agent_id, **kw):
        raise login_mod.LoginError("the login was denied in the browser; no key was issued")

    monkeypatch.setattr(login_mod, "device_login", _login)
    assert cli.main(["login", "--agent-id", AGENT]) == 1
    assert "denied in the browser" in capsys.readouterr().err


class _RecordingClient:
    """A control-plane client that records the key it was built with and
    answers nothing, so the callers fail open on their own terms."""

    keys: ClassVar[list[str]] = []

    def __init__(self, base_url, api_key, **kw):
        _RecordingClient.keys.append(api_key)

    def request(self, method, path, **kw):
        raise fw.TransportError("unreachable in this test")

    def close(self):
        pass


def test_attach_and_policy_source_read_with_the_stored_key(home, monkeypatch, tmp_path):
    """Every reader that knows its agent asks the one rule, so a production
    agent finds the key ``login`` stored exactly as ``serve()`` does.

    Mutation: put ``attach`` or ``PolicySource._fetch`` back on the variable
    alone (the key list below stays empty)."""
    _store(home)
    _RecordingClient.keys = []
    monkeypatch.setattr("agent_flywheel._http.ControlPlaneClient", _RecordingClient)
    px = fw.attach(agent_id=AGENT, trajectory_root=str(tmp_path / "turns"),
                   control_plane_url=CP, adapters=[])
    assert "is not set" not in (px.upload_refusal or "")
    assert KEY in _RecordingClient.keys, "attach() did not read the stored key"

    _RecordingClient.keys = []
    fw.PolicySource(agent_id=AGENT, control_plane_url=CP).current()
    assert _RecordingClient.keys == [KEY], "PolicySource did not read the stored key"
