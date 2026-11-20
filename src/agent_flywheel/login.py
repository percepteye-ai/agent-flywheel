"""``agent-flywheel login``: get this agent a key without leaving the terminal.

The control plane's device login (RFC 8628) with the ``flywheel-agent``
scope, owner ruling 2026-10-20:

1. ``POST /api/v2/auth/device/code`` names this SDK (``agent-flywheel-python``)
   and the agent; the control plane answers a short code and a link.
2. The person opens the link -- the browser is opened for them -- signs up or
   signs in, sees which program is asking, for which agent, and the code to
   compare, and clicks Approve. Nothing on the page ever shows a secret.
3. This polls ``POST /api/v2/auth/device/token`` at the interval it is given
   until the approval lands, then receives a ``pefw_`` key bound to that one
   agent and stores it (:mod:`agent_flywheel.credentials`).

:func:`offer_login` is the same flow offered inline by ``serve()`` when it
finds no key, and ONLY in an interactive terminal: a process with no person at
it (CI, a service, a pipe) is never asked a question, and keeps raising the
missing-key message it always has.
"""
from __future__ import annotations

import ipaddress
import os
import re
import sys
import time
import webbrowser
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone
from typing import IO, Any
from urllib.parse import urlsplit, urlunsplit

from ._version import __version__
from .control_plane import resolve_control_plane_url
from .credentials import StoredCredential, save_stored
from .errors import ConfigurationError

__all__ = [
    "CLIENT_ID",
    "SCOPE",
    "LoginError",
    "device_endpoint",
    "device_login",
    "interactive_terminal",
    "offer_login",
]

#: Who this SDK is to the control plane's device login. Its approval page
#: names it ("Agent Flywheel SDK (Python)").
CLIENT_ID = "agent-flywheel-python"
#: The one scope this client may ask for: a key bound to one agent.
SCOPE = "flywheel-agent"
GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"
USER_AGENT = f"agent-flywheel-python/{__version__}"

#: The Flywheel API's own path on a control plane. The device login lives on
#: the same origin under the dashboard's API.
_FLYWHEEL_SUFFIX = "/api/flywheel/v1"
_DEVICE_PATH = "/api/v2/auth/device"

#: RFC 8628 §3.5: ``slow_down`` adds five seconds to the interval.
_SLOW_DOWN_STEP_S = 5
#: A poll that cannot reach the control plane is retried until the code
#: expires; this caps the wait between those retries.
_MAX_RETRY_WAIT_S = 30

#: ANSI escape sequences (CSI, OSC, and two-character ESC forms) and every
#: other C0/C1 control character. A string the control plane sends is printed
#: to the person's terminal; a server (or anything between) must not be able
#: to move the cursor, rewrite the code shown, set the window title, or plant
#: a hyperlink there.
_TERMINAL_CONTROL = re.compile(
    r"\x1b\[[0-?]*[ -/]*[@-~]"          # CSI ... final byte
    r"|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?"  # OSC ... BEL or ST
    r"|\x1b[@-_]"                       # other two-character escapes
    r"|[\x00-\x1f\x7f-\x9f]"            # any control character left
)

#: The hosts an ``http`` login link may point at whatever the control plane's
#: own scheme: the person's own machine.
_LOOPBACK_NAMES = ("localhost",)

#: Set to any of these and a process is a CI job, so nobody is there to answer.
_CI_VARS = (
    "CI", "GITHUB_ACTIONS", "GITLAB_CI", "BUILDKITE", "CIRCLECI",
    "JENKINS_URL", "TF_BUILD", "TEAMCITY_VERSION",
)
ENV_NO_PROMPT = "PERCEPTEYE_NO_PROMPT"


class LoginError(ConfigurationError):
    """The device login did not produce a key; the message says why."""


def device_endpoint(control_plane_url: str) -> str:
    """Where the device login is, for the control plane at ``control_plane_url``.

    The Flywheel API is served under ``/api/flywheel/v1`` on the dashboard's
    origin (``DEFAULT_CONTROL_PLANE_URL``); the device login is
    ``/api/v2/auth/device`` beside it. A URL with any other path is a control
    plane mounted elsewhere, and its origin is asked.
    """
    parts = urlsplit(control_plane_url.strip())
    path = parts.path.rstrip("/")
    if path.endswith(_FLYWHEEL_SUFFIX):
        prefix = path[: -len(_FLYWHEEL_SUFFIX)]
    else:
        prefix = ""
    return urlunsplit((parts.scheme, parts.netloc, prefix + _DEVICE_PATH, "", ""))


def _can_open_browser(environ: Mapping[str, str]) -> bool:
    """Would opening a browser put a page in front of this person?

    Not over SSH (it would open on the far machine, or not at all), and not
    on a Linux with no display, where ``webbrowser`` falls back to a text
    browser that takes over the very terminal the code is printed in.
    """
    if environ.get("SSH_CONNECTION") or environ.get("SSH_TTY"):
        return False
    if sys.platform.startswith("linux") and not (
        environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY")
    ):
        return False
    return True


def _set(value: str | None) -> bool:
    """A boolean environment variable, by the rule every one here follows:
    unset or empty is off, ``0``/``false``/``no``/``off`` (any case) is off,
    anything else is on."""
    value = (value or "").strip().lower()
    return bool(value) and value not in ("0", "false", "no", "off")


def _ci(environ: Mapping[str, str]) -> bool:
    return any(_set(environ.get(name)) for name in _CI_VARS)


def printable(text: object) -> str:
    """``text`` with every terminal control sequence and control character
    removed: what is safe to print of a string the control plane sent."""
    return _TERMINAL_CONTROL.sub("", str(text))


def _loopback(host: str) -> bool:
    if host.lower() in _LOOPBACK_NAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def link_refusal(link: str, control_plane_url: str) -> str | None:
    """Why the login link the control plane sent must not be opened, or
    ``None`` when it may.

    The SCHEME is checked, not the origin: the control plane builds the link
    from its public dashboard address, which need not be the API's host, so
    an origin match would refuse real logins. What a browser hand-off must
    never get is a scheme that opens something other than a web page
    (``file:``, ``smb:``, an app's own handler): ``https`` always; ``http``
    only when the control plane is itself spoken to over ``http`` or the link
    is to this machine. Any whitespace or control character is refused
    outright -- no real link carries one, and a handler may split on it.
    """
    if any(ch.isspace() or not ch.isprintable() for ch in link):
        return "it contains whitespace or a control character"
    try:
        parts = urlsplit(link)
        host = parts.hostname or ""
    except ValueError:
        return "it is not a URL"
    scheme = parts.scheme.lower()
    if scheme not in ("https", "http"):
        return f"its scheme {scheme or '(none)'!r} is not a web page (https)"
    if not host:
        return "it names no host"
    if scheme == "http" and not (
        urlsplit(control_plane_url.strip()).scheme.lower() == "http"
        or _loopback(host)
    ):
        return (
            "it is plain http to another machine while the control plane is "
            "spoken to over https"
        )
    return None


def interactive_terminal(
    *,
    environ: Mapping[str, str] | None = None,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
) -> bool:
    """Is a person at this process, able to answer a question?

    Both ends must be a terminal: the question is written to stderr and the
    answer read from stdin, so a pipe on either means nobody sees or answers
    it. And never in CI, which often allocates a TTY, nor when
    ``PERCEPTEYE_NO_PROMPT`` is on (any value but empty, ``0``, ``false``,
    ``no`` or ``off``).
    """
    env = os.environ if environ is None else environ
    if _set(env.get(ENV_NO_PROMPT)) or _ci(env):
        return False
    stdin = sys.stdin if stdin is None else stdin
    stderr = sys.stderr if stderr is None else stderr
    try:
        return bool(stdin and stdin.isatty() and stderr and stderr.isatty())
    except (AttributeError, ValueError):
        return False


def _oauth_error(resp: Any) -> tuple[str, str]:
    """The response's OAuth ``error`` and its description, both made safe to
    print: every message built from them reaches the person's terminal."""
    try:
        body = resp.json()
    except ValueError:
        body = {}
    if not isinstance(body, dict):
        body = {}
    return (
        printable(body.get("error") or f"http_{resp.status_code}"),
        printable(body.get("error_description") or resp.text[:200] or ""),
    )


def _json_object(resp: Any, what: str) -> dict[str, Any]:
    try:
        body = resp.json()
    except ValueError as exc:
        raise LoginError(f"the control plane's {what} was not JSON") from exc
    if not isinstance(body, dict):
        raise LoginError(f"the control plane's {what} was not a JSON object")
    return body


def _text(body: Mapping[str, Any], name: str, what: str, *, required: bool) -> str | None:
    """A string field of a response, or :class:`LoginError` naming it."""
    value = body.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value:
        raise LoginError(
            f"the control plane's {what} has no usable {name!r} "
            f"(got {printable(repr(value))[:80]})"
        )
    return value


def _seconds(body: Mapping[str, Any], name: str, what: str) -> int | None:
    """A whole number of seconds from a response, ``None`` when absent, or
    :class:`LoginError` for anything that is not a positive number."""
    value = body.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
        raise LoginError(
            f"the control plane's {what} has an unusable {name!r} "
            f"(got {printable(repr(value))[:80]})"
        )
    try:
        return int(value)
    except (OverflowError, ValueError) as exc:  # inf, nan
        raise LoginError(
            f"the control plane's {what} has an unusable {name!r} ({value!r})"
        ) from exc


def _unsupported(control_plane_url: str, detail: str) -> LoginError:
    from . import control_plane as _cp

    signup = getattr(_cp, "SIGN_UP_URL", "https://launch.percepteye.ai/signup")
    return LoginError(
        f"the control plane at {control_plane_url} does not offer device "
        f"login for this SDK ({printable(detail)}). Create a key in the "
        f"dashboard instead ({signup}) and set PERCEPTEYE_API_KEY."
    )


def device_login(
    agent_id: str,
    *,
    control_plane_url: str | None = None,
    open_browser: bool = True,
    out: IO[str] | None = None,
    environ: Mapping[str, str] | None = None,
    client: Any = None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    browser: Callable[[str], bool] | None = None,
) -> StoredCredential:
    """Run the device login for ``agent_id`` and store the key it returns.

    Writes what the person needs to do to ``out`` (stderr by default) and
    returns the stored credential. Raises :class:`LoginError` naming why when
    no key came back (denied, expired, the control plane refused or answered
    something malformed, a key for a different agent, the file could not be
    written), and lets ``KeyboardInterrupt`` through so Ctrl-C cancels.

    Every string the control plane sent is printed with its terminal control
    characters removed, and the link is opened only if :func:`link_refusal`
    allows it; a refused link is printed with the reason and not opened.
    """
    import httpx

    from .contract import _production_agent_identifier

    out = sys.stderr if out is None else out
    if _production_agent_identifier(agent_id) is None:
        raise LoginError(
            f"agent id {agent_id!r} is not one an agent can register as: 1-255 "
            "characters, no control characters, no leading or trailing "
            "whitespace, no '/' or '\\', and not '.' or '..'"
        )
    url = resolve_control_plane_url(control_plane_url, environ)
    base = device_endpoint(url)
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0)
    # On every request, not the client: the approval page shows it (marked
    # unverified), whichever client carries the request.
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    def say(line: str = "") -> None:
        print(line, file=out, flush=True)

    try:
        try:
            resp = http.post(f"{base}/code", headers=headers, data={
                "client_id": CLIENT_ID, "scope": SCOPE, "agent_id": agent_id,
            })
        except httpx.HTTPError as exc:
            raise LoginError(
                f"could not reach the control plane at {base}: {exc}"
            ) from exc
        if resp.status_code != 200:
            error, description = _oauth_error(resp)
            if error in ("invalid_client", "invalid_scope") or resp.status_code == 404:
                raise _unsupported(url, description or error)
            raise LoginError(f"the control plane refused the login: {description or error}")
        started = _json_object(resp, "device code answer")
        device_code = str(_text(started, "device_code", "device code answer", required=True))
        user_code = printable(
            _text(started, "user_code", "device code answer", required=True),
        )
        link = str(
            _text(started, "verification_uri_complete", "device code answer", required=False)
            or _text(started, "verification_uri", "device code answer", required=True)
        )
        interval = max(1, _seconds(started, "interval", "device code answer") or 5)
        expires_in = _seconds(started, "expires_in", "device code answer") or 600
        deadline = clock() + expires_in

        say(f"To connect {agent_id} to PerceptEye, approve this code in your browser:")
        say()
        say(f"    {user_code}")
        say()
        refused = link_refusal(link, url)
        shown = printable(link)
        opened = False
        if refused is None and open_browser and (
            browser is not None
            or _can_open_browser(os.environ if environ is None else environ)
        ):
            try:
                opened = bool((browser or webbrowser.open)(link))
            except Exception:
                opened = False
        if refused is not None:
            say(f"The control plane sent a login link this SDK will not open "
                f"({refused}): {shown}")
        elif opened:
            say(f"Opened {shown}")
        else:
            say(f"Open {shown}")
        say("Sign up or sign in there, check the code matches, and click Approve.")
        say(f"Waiting for approval (the code expires in {expires_in // 60} min; "
            "Ctrl-C to cancel)...")

        def expired() -> LoginError:
            return LoginError("the code expired before it was approved; run login again")

        wait = interval
        while True:
            # Never sleep past the code's life: a long ``slow_down`` or retry
            # wait would otherwise hold the person past an expiry already due.
            remaining = deadline - clock()
            if remaining <= 0:
                raise expired()
            sleep(min(wait, remaining))
            if clock() >= deadline:
                raise expired()
            try:
                polled = http.post(f"{base}/token", headers=headers, data={
                    "grant_type": GRANT_TYPE, "device_code": device_code,
                    "client_id": CLIENT_ID,
                })
            except httpx.HTTPError:
                # Unreachable for a moment is not a refusal: keep the code
                # alive and try again, slower, until it expires.
                wait = min(_MAX_RETRY_WAIT_S, wait * 2)
                continue
            if polled.status_code == 200:
                token = _json_object(polled, "token answer")
                break
            error, description = _oauth_error(polled)
            if error == "authorization_pending":
                wait = interval
                continue
            if error == "slow_down":
                try:
                    body = polled.json()
                except ValueError:
                    body = {}
                given = body.get("interval") if isinstance(body, dict) else None
                if isinstance(given, bool) or not isinstance(given, (int, float)):
                    given = None
                try:
                    interval = max(1, int(given or interval + _SLOW_DOWN_STEP_S))
                except (OverflowError, ValueError):
                    interval += _SLOW_DOWN_STEP_S
                wait = interval
                continue
            if error == "access_denied":
                raise LoginError("the login was denied in the browser; no key was issued")
            if error == "expired_token":
                raise expired()
            if polled.status_code >= 500 or polled.status_code == 429:
                wait = min(_MAX_RETRY_WAIT_S, max(interval, wait * 2))
                continue
            raise LoginError(f"no key was issued: {description or error}")
    finally:
        if owns_client:
            http.close()

    what = "token answer"
    key = token.get("access_token")
    if not isinstance(key, str) or not key.strip():
        raise LoginError("the control plane approved the login but sent no key")
    granted_to = _text(token, "agent_id", what, required=False)
    if granted_to is not None and granted_to != agent_id:
        # Stored under the agent the server NAMED, it would silently replace
        # another agent's key; under the one asked for, it would be a key that
        # opens a different agent. Neither: refuse it.
        raise LoginError(
            f"the control plane issued a key for agent {printable(granted_to)!r}, "
            f"not the {agent_id!r} this login asked for; nothing was stored"
        )
    expires = _seconds(token, "expires_in", what)
    now = datetime.now(timezone.utc).replace(microsecond=0)
    try:
        expires_at = (
            (now + timedelta(seconds=expires)).isoformat().replace("+00:00", "Z")
            if expires is not None else None
        )
    except OverflowError as exc:
        raise LoginError(
            f"the control plane's {what} has an unusable 'expires_in' ({expires})"
        ) from exc
    cred = StoredCredential(
        control_plane_url=url,
        agent_id=agent_id,
        api_key=key,
        org_id=_text(token, "org_id", what, required=False),
        org_name=_text(token, "org_name", what, required=False),
        key_id=_text(token, "flywheel_key_id", what, required=False),
        created_at=now.isoformat().replace("+00:00", "Z"),
        expires_at=expires_at,
        client=USER_AGENT,
    )
    try:
        path = save_stored(cred, environ)
    except LoginError:
        raise
    except (ConfigurationError, OSError) as exc:
        raise LoginError(f"the key was issued but could not be stored: {exc}") from exc
    where = f" in {printable(cred.org_name)}" if cred.org_name else ""
    say(f"Approved: a key for {cred.agent_id}{where}, saved to {path}.")
    return cred


def offer_login(
    agent_id: str,
    *,
    control_plane_url: str,
    environ: Mapping[str, str] | None = None,
    stdin: IO[str] | None = None,
    stderr: IO[str] | None = None,
    login: Callable[..., StoredCredential] = device_login,
) -> str | None:
    """Ask the person at this terminal whether to create a key now; the key,
    or ``None``.

    ``None`` without asking when nobody is there to answer
    (:func:`interactive_terminal`), and ``None`` when they decline, cancel, or
    the login fails -- which it says. The caller then refuses exactly as it
    would have with no offer.
    """
    stdin = sys.stdin if stdin is None else stdin
    stderr = sys.stderr if stderr is None else stderr
    if not interactive_terminal(environ=environ, stdin=stdin, stderr=stderr):
        return None
    stderr.write(
        f"No PerceptEye key found for agent {agent_id!r}. "
        "Create one now (free)? [Y/n] "
    )
    stderr.flush()
    try:
        answer = stdin.readline()
    except (KeyboardInterrupt, EOFError):
        stderr.write("\n")
        return None
    if not answer or answer.strip().lower() not in ("", "y", "yes"):
        return None
    try:
        cred = login(
            agent_id, control_plane_url=control_plane_url, out=stderr,
            environ=environ,
        )
    except KeyboardInterrupt:
        stderr.write("\nLogin cancelled.\n")
        return None
    except LoginError as exc:
        stderr.write(f"agent-flywheel: {exc}\n")
        return None
    return cred.api_key
