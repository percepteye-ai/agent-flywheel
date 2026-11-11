"""Count the policy model calls a rollout makes, from inside the agent's process.

Mission Control holds a rollout out of training unless two independent counts
of its policy calls agree: the one its gateway observed, and the one the agent
side REPORTS (``mission_control/flywheel/grading.py``). An agent that returns a
``RolloutOutput`` without ``llm_call_count`` -- almost every agent, since
nothing in its own code knows the number -- was therefore excluded on every
rollout as ``llm_calls_unobserved``, however clean its traffic (τ³ banking,
2026-11-11: 384 + 29 rollouts, all excluded, observed 3-28 calls each).

The SDK owns this number, not the customer: while a rollout runs, every HTTP
request the process completes to that rollout's policy URL is counted at the
client. It is the client-side half of the reconciliation -- a completion the
gateway failed to capture, or captured twice, now disagrees with it.

Counted: a request whose URL is under the rollout's ``llm_url`` and NOT under
its auxiliary route (``hosted_base_for``: user simulators and judges are not
the policy), that got a response below 400. Clients covered: ``httpx`` (sync
and async; the OpenAI, Anthropic and LiteLLM SDKs ride it) and ``requests``.
A rollout with no counted call reports ``None``, never ``0``: a client this
module does not see would otherwise be reported as "made no calls".
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

from .outcomes import hosted_base_for

__all__ = ["count_policy_calls"]

_lock = threading.Lock()
#: llm_url prefix -> [count, hosted prefix]. Keyed by URL, not by context:
#: an agent's HTTP client may run on threads the rollout's context never
#: reaches, and every rollout has its own gateway URL.
_active: dict[str, list[Any]] = {}
_installed = 0
_originals: list[tuple[Any, str, Any]] = []


def _norm(url: str) -> str:
    return url.rstrip("/") + "/"


def _note(method: Any, url: Any, status: Any) -> None:
    if str(method).upper() != "POST":
        return
    try:
        if int(status) >= 400:
            return
    except (TypeError, ValueError):
        return
    u = str(url)
    with _lock:
        for prefix, slot in _active.items():
            if u.startswith(prefix) and not (slot[1] and u.startswith(slot[1])):
                slot[0] += 1
                return


def _patch(owner: Any, name: str, wrap: Any) -> None:
    orig = getattr(owner, name)
    _originals.append((owner, name, orig))
    setattr(owner, name, wrap(orig))


def _install() -> None:
    try:
        import httpx
    except ImportError:
        httpx = None
    if httpx is not None:
        def _sync(orig: Any) -> Any:
            def send(self: Any, request: Any, *a: Any, **k: Any) -> Any:
                resp = orig(self, request, *a, **k)
                _note(request.method, request.url, resp.status_code)
                return resp
            return send

        def _async(orig: Any) -> Any:
            async def send(self: Any, request: Any, *a: Any, **k: Any) -> Any:
                resp = await orig(self, request, *a, **k)
                _note(request.method, request.url, resp.status_code)
                return resp
            return send

        _patch(httpx.Client, "send", _sync)
        _patch(httpx.AsyncClient, "send", _async)
    try:
        import requests
    except ImportError:
        requests = None
    if requests is not None:
        def _req(orig: Any) -> Any:
            def send(self: Any, request: Any, *a: Any, **k: Any) -> Any:
                resp = orig(self, request, *a, **k)
                _note(request.method, request.url, resp.status_code)
                return resp
            return send

        _patch(requests.Session, "send", _req)


def _uninstall() -> None:
    while _originals:
        owner, name, orig = _originals.pop()
        setattr(owner, name, orig)


class _Count:
    """The live count for one rollout; :attr:`value` is final after exit."""

    def __init__(self, slot: list[Any]) -> None:
        self._slot = slot

    @property
    def value(self) -> int | None:
        with _lock:
            n = self._slot[0]
        return n or None


@contextmanager
def count_policy_calls(llm_url: str | None) -> Iterator[_Count]:
    """Count completed policy requests to ``llm_url`` for the block's length."""
    global _installed
    if not isinstance(llm_url, str) or not llm_url.strip():
        yield _Count([0, None])
        return
    prefix = _norm(llm_url)
    hosted = hosted_base_for(llm_url)
    slot: list[Any] = [0, _norm(hosted) if hosted else None]
    with _lock:
        _active[prefix] = slot
        if _installed == 0:
            _install()
        _installed += 1
    try:
        yield _Count(slot)
    finally:
        with _lock:
            _active.pop(prefix, None)
            _installed -= 1
            if _installed == 0:
                _uninstall()
