"""``agent-flywheel login`` trusts nothing the control plane sends.

What each test pins, and the defect it exists to catch:

  * THE LINK IS CHECKED BY SCHEME (PY-C1): the link the control plane sends
    is handed to the OS's URL opener, so ``file:``, ``smb:``, an app's own
    handler, or one carrying whitespace/control characters is printed with
    the reason and never opened. An ``https`` link on ANOTHER origin is
    opened -- the control plane builds it from its dashboard address, which
    need not be the API host -- and ``http`` only for an ``http`` control
    plane or this machine.
  * NO TERMINAL CONTROL FROM THE SERVER (PY-M9): every server string printed
    has its escapes and control characters removed; ``PERCEPTEYE_NO_PROMPT=0``
    means "do prompt"; a file that cannot be written is a ``LoginError``.
  * THE TOKEN IS VALIDATED (PY-H10): a key for another agent is refused (it
    would replace that agent's stored key); every malformed field is a
    ``LoginError``; the interval is at least 1s; no sleep outlives the code.
"""
from __future__ import annotations

import io
import json

import httpx
import pytest

from agent_flywheel import credentials as creds
from agent_flywheel import login as login_mod

CP = "https://cp.test/api/flywheel/v1"
AGENT = "my-agent"
KEY = "pefw_" + "K" * 43


@pytest.fixture
def home(tmp_path, monkeypatch):
    root = tmp_path / "pe-home"
    monkeypatch.setenv("PERCEPTEYE_HOME", str(root))
    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.setattr(creds, "_SAID", set())
    return root


def _code(**over):
    body = {
        "device_code": "dev-123", "user_code": "BCDF-GHJK",
        "verification_uri": "https://cp.test/device",
        "verification_uri_complete": "https://cp.test/device#user_code=BCDF-GHJK",
        "expires_in": 600, "interval": 5,
    }
    body.update(over)
    return body


def _granted(**over):
    body = {
        "access_token": KEY, "token_type": "Bearer", "expires_in": 3600,
        "org_id": "org-1", "org_name": "Acme", "agent_id": AGENT,
        "flywheel_key_id": "key-1",
    }
    body.update(over)
    return body


def _login(code, polls, *, cp_url=CP, agent_id=AGENT, clock=None, sleeps=None):
    """Run the login against a scripted control plane; (cred or exc, out,
    opened, sleeps)."""
    polls = list(polls)
    sleeps = [] if sleeps is None else sleeps
    opened: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/code"):
            if isinstance(code, httpx.Response):
                return code
            return httpx.Response(200, json=code)
        step = polls.pop(0)
        if isinstance(step, httpx.Response):
            return step
        status, body = step
        return httpx.Response(status, json=body)

    out = io.StringIO()
    try:
        result = login_mod.device_login(
            agent_id, control_plane_url=cp_url, out=out,
            client=httpx.Client(transport=httpx.MockTransport(handler)),
            sleep=sleeps.append, clock=clock or (lambda: 0.0),
            browser=lambda url: opened.append(url) or True,
        )
    except login_mod.LoginError as exc:
        result = exc
    return result, out.getvalue(), opened, sleeps


# ── PY-C1: the link ──────────────────────────────────────────────────


@pytest.mark.parametrize("link", [
    "file:///Applications/Calculator.app",
    "smb://attacker.example/share",
    "javascript:alert(1)",
    "x-apple.systempreferences:com.apple.preference",
    "http://cp.test/device",                     # http to a remote host, https CP
    "https://cp.test/device\n--evil",
    "https://cp.test/de vice",
    "https://cp.test/device\x1b[2J",
])
def test_a_link_that_is_not_a_web_page_is_shown_and_never_opened(home, link):
    cred, out, opened, _ = _login(
        _code(verification_uri_complete=link), [(200, _granted())],
    )
    assert opened == []
    assert "will not open" in out
    assert "\x1b" not in out and "\n--evil" not in out
    assert cred.api_key == KEY  # the login itself still completes


@pytest.mark.parametrize(("link", "cp_url"), [
    # The dashboard origin the control plane builds the link from is not the
    # API's host: an origin check would refuse this real login.
    ("https://launch.percepteye.ai/device#user_code=BCDF-GHJK", "https://api.percepteye.ai/api/flywheel/v1"),
    ("http://mc.internal:8000/device", "http://mc.internal:8000/api/flywheel/v1"),
    ("http://localhost:3000/device", CP),
    ("http://127.0.0.1:3000/device", CP),
    ("http://[::1]:3000/device", CP),
])
def test_a_web_link_is_opened_even_on_another_origin(home, link, cp_url):
    _, out, opened, _ = _login(
        _code(verification_uri_complete=link), [(200, _granted())], cp_url=cp_url,
    )
    assert opened == [link]
    assert f"Opened {link}" in out


# ── PY-M9: what is printed, the prompt switch, the file ──────────────


def test_server_strings_reach_the_terminal_without_control_sequences(home):
    _, out, _, _ = _login(
        _code(user_code="BCDF\x1b[2J\x1b]0;pwned\x07-GHJK\x9b"),
        [(200, _granted(org_name="Acme\x1b[31m\r\x08Evil"))],
    )
    assert "\x1b" not in out and "\x9b" not in out and "\r" not in out and "\x08" not in out
    assert "BCDF-GHJK" in out and "AcmeEvil" in out

    err, _, _, _ = _login(_code(), [(400, {
        "error": "invalid_grant", "error_description": "no\x1b[2J\x1b]8;;http://x\x07 way"})])
    assert isinstance(err, login_mod.LoginError)
    assert "\x1b" not in str(err) and "\x07" not in str(err)
    assert "no way" in str(err)


@pytest.mark.parametrize(("value", "asked"), [
    ("0", True), ("false", True), ("No", True), (" off ", True), ("", True),
    ("1", False), ("true", False), ("yes", False),
])
def test_no_prompt_is_a_boolean(value, asked):
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    assert login_mod.interactive_terminal(
        environ={"PERCEPTEYE_NO_PROMPT": value}, stdin=_Tty(), stderr=_Tty(),
    ) is asked


def test_a_key_that_cannot_be_stored_is_a_login_error(home):
    home.mkdir(parents=True)
    (home / "credentials.json").write_text(json.dumps({"version": 2, "credentials": []}))
    err, _, _, _ = _login(_code(), [(200, _granted())])
    assert isinstance(err, login_mod.LoginError), err
    assert "could not be stored" in str(err) and "version 2" in str(err)


def test_an_offer_whose_store_fails_says_so_and_returns_none(home):
    class _Tty(io.StringIO):
        def isatty(self):
            return True

    def login(*a, **kw):
        raise login_mod.LoginError("the key was issued but could not be stored: disk full")

    stderr = _Tty()
    assert login_mod.offer_login(
        AGENT, control_plane_url=CP, environ={}, stdin=_Tty("y\n"), stderr=stderr,
        login=login,
    ) is None
    assert "could not be stored" in stderr.getvalue()


# ── PY-H10: the token and the timing ─────────────────────────────────


def test_a_key_for_another_agent_is_refused_and_nothing_is_stored(home):
    creds.save_stored(creds.StoredCredential(
        control_plane_url=CP, agent_id="victim", api_key="pefw_victim"))
    err, _, _, _ = _login(_code(), [(200, _granted(agent_id="victim", access_token="pefw_new"))])
    assert isinstance(err, login_mod.LoginError)
    assert "'victim'" in str(err) and "nothing was stored" in str(err)
    (entry,) = json.loads((home / "credentials.json").read_text())["credentials"]
    assert entry["agent_id"] == "victim" and entry["api_key"] == "pefw_victim"


@pytest.mark.parametrize(("code", "polls", "said"), [
    (httpx.Response(200, text="<html>"), [], "not JSON"),
    (httpx.Response(200, json=["x"]), [], "not a JSON object"),
    (_code(device_code=None), [], "'device_code'"),
    ({k: v for k, v in _code().items() if k != "user_code"}, [], "'user_code'"),
    (_code(verification_uri_complete=None, verification_uri=7), [], "'verification_uri'"),
    (_code(interval="soon"), [], "'interval'"),
    (_code(expires_in=True), [], "'expires_in'"),
    (_code(expires_in=-5), [], "'expires_in'"),
    (_code(), [httpx.Response(200, text="not json")], "not JSON"),
    (_code(), [(200, ["k"])], "not a JSON object"),
    (_code(), [(200, _granted(expires_in=True))], "'expires_in'"),
    (_code(), [(200, _granted(expires_in="1d"))], "'expires_in'"),
    (_code(), [(200, _granted(org_name=["x"]))], "'org_name'"),
    (_code(), [(200, _granted(access_token=12345))], "sent no key"),
    (_code(), [(200, _granted(access_token="   "))], "sent no key"),
])
def test_every_malformed_answer_is_a_login_error(home, code, polls, said):
    err, _, _, _ = _login(code, polls)
    assert isinstance(err, login_mod.LoginError), repr(err)
    assert said in str(err)
    assert not (home / "credentials.json").exists()


def test_slow_down_never_drops_the_interval_below_one_second(home):
    _, _, _, sleeps = _login(
        _code(), [(400, {"error": "slow_down", "interval": -30}),
                  (400, {"error": "slow_down", "interval": 0.2}),
                  (400, {"error": "slow_down", "interval": True}),
                  (200, _granted())],
    )
    assert all(s >= 1 for s in sleeps), sleeps


def test_no_wait_outlives_the_code(home):
    now = [0.0]

    def clock():
        return now[0]

    class _Sleeps(list):
        def append(self, s):
            now[0] += s
            super().append(s)

    sleeps = _Sleeps()
    err, _, _, _ = _login(
        _code(expires_in=60),
        [(400, {"error": "slow_down", "interval": 3600})] + [
            (400, {"error": "authorization_pending"})] * 5,
        clock=clock, sleeps=sleeps,
    )
    assert isinstance(err, login_mod.LoginError) and "expired" in str(err)
    assert max(sleeps) <= 60, sleeps
    assert now[0] <= 60, f"slept to {now[0]}s on a 60s code"
