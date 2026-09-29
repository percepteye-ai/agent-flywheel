"""The one HTTP client, with retry semantics that read the response.

Against named defects in a previous SDK: its retry helper returned the
response object without reading status, and retried only transport
exceptions -- so a 503 was indistinguishable from success, and its only
recovery from an unexpected error was a blanket five-second sleep inside a
bare ``except`` that also swallowed a real ``TypeError``.
"""
from __future__ import annotations

import json
import random
import time
from collections.abc import Mapping
from typing import Any

import httpx

from ._version import __version__
from .contract import CONTRACT_VERSION
from .errors import LeaseLost, TransportError

__all__ = ["ControlPlaneClient", "default_delivery_envelope_s", "delivery_envelope_s"]

#: Transient. Retry these, honouring Retry-After when present.
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: The client's own defaults, named so the time they can SPEND is derivable by
#: whoever has to RESERVE it. Before this they were literals in a signature and
#: the runner's delivery margin was an unrelated literal ten seconds -- the
#: producer of the deadline and the consumer of the time never met, and the
#: reserve was 12x short of one worst-case report.
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_ATTEMPTS = 4

#: Ceiling on ONE backoff sleep, and therefore on a server-supplied
#: ``Retry-After``. Honouring an uncapped header let a proxy answering
#: ``Retry-After: 3600`` park a report for an hour inside a path the runner
#: believed cost seconds.
_MAX_BACKOFF_S = 8.0

#: A deadline shorter than one attempt must still permit ONE attempt. Clamping
#: an attempt to zero would stop delivering exactly the small-budget rollouts
#: this machinery exists to deliver.
_MIN_ATTEMPT_TIMEOUT_S = 1.0


def delivery_envelope_s(timeout: float, max_attempts: int) -> float:
    """Worst-case wall clock ONE request can take, retries and backoff included.

    Inferred from the retry loop below rather than measured: it is the sum of
    every attempt's read timeout plus every backoff sleep's CAP, so a real
    request is normally far quicker. It is the number a caller must reserve
    if it wants the call to finish, which is the only use it has.
    """
    attempts = max(1, int(max_attempts))
    sleeps = sum(min(_MAX_BACKOFF_S, 0.5 * (2 ** (a - 1))) for a in range(1, attempts))
    return float(timeout) * attempts + sleeps


def default_delivery_envelope_s() -> float:
    """The envelope for a client built with this module's own defaults.

    Every serve-path ``ControlPlaneClient`` takes them, so this is the number
    an identity-sealed adapter setting -- which cannot read a live client --
    must be sized against.
    """
    return delivery_envelope_s(DEFAULT_TIMEOUT_S, DEFAULT_MAX_ATTEMPTS)

#: Failures where the connection was never established, so the server cannot
#: have seen the request. Everything else -- a read timeout above all -- may
#: have been processed, and is only safe to resend if the endpoint is replayable.
_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout)


def _may_have_performed(status: int) -> bool:
    """Might a request answered with this status have been PERFORMED?

    A 4xx is the server refusing it and a 503 the server saying it was not
    handled -- the control plane's own store failures answer 503 and register
    nothing. Any other 5xx may come after the work was done: a 500 raised
    after a commit, a 502 or 504 from a proxy whose upstream finished the
    request and lost the answer. Read by :attr:`TransportError.ambiguous`.
    """
    return status >= 500 and status != 503


_SDK_UA = f"agent-flywheel-python/{__version__}"


class ControlPlaneClient:
    """Thin, synchronous, outbound-only client for the agent control plane."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url:
            raise TransportError("control-plane base URL is required")
        if not api_key:
            # No "dev-key" default. A previous SDK defaulted a missing key to a
            # literal placeholder, so a misconfigured deployment ran happily
            # and reported nothing.
            raise TransportError(
                "an API key is required; refusing to start with no credential"
            )
        self.base_url = base_url.rstrip("/")
        self._max_attempts = max(1, max_attempts)
        #: Kept, not merely handed to httpx. It is half of what one call can
        #: cost, and the runner has to reserve that cost before it spends it.
        self._timeout = float(timeout)
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Contract-Version": CONTRACT_VERSION,
                "X-Percepteye-Sdk": _SDK_UA,
                "Accept": "application/json",
            },
        )

    @property
    def delivery_envelope_s(self) -> float:
        """Worst case for ONE call THROUGH THIS CLIENT, from its own settings.

        The producer half of the runner's delivery reserve. A client built with
        a different timeout or attempt count moves this number, and the runner
        hands the agent correspondingly less or more of the episode -- which is
        what makes the reserve a bound rather than a hopeful constant.
        """
        return delivery_envelope_s(self._timeout, self._max_attempts)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> ControlPlaneClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: Mapping[str, Any] | None = None,
        idempotency_key: str | None = None,
        timeout: float | None = None,
        deadline: float | None = None,
        retry_if_ambiguous: bool = True,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send one request, retrying transient failures.

        ``retry_if_ambiguous=False`` narrows retries to failures where the
        request provably never reached the server. Use it for any call whose
        server side has a side effect and no replay protection: retrying a
        request that MAY have been processed performs it twice.

        ``deadline`` is an ABSOLUTE ``time.time()`` value and bounds the whole
        call -- every attempt's read timeout and every backoff sleep is clamped
        to what is left of it, and no new attempt starts past it. It exists so
        the time a caller RESERVED and the time this loop may SPEND are the
        same number. Without it a report attempted with ten seconds left ran a
        four-attempt, two-minute envelope against a rollout the server was
        about to sweep. One attempt is always permitted, however little is
        left: refusing to try at all loses exactly the results this bounds.
        """
        url = f"{self.base_url}{path}"
        request_headers: dict[str, str] = dict(headers or {})
        if idempotency_key:
            request_headers["Idempotency-Key"] = idempotency_key

        last_err: str | None = None
        #: Whether ANY attempt so far may have been performed -- see
        #: ``TransportError.ambiguous``. Over every attempt, not the last:
        #: a retry that was never sent says nothing about the one before it.
        ambiguous = False
        for attempt in range(1, self._max_attempts + 1):
            # NEVER `None`. httpx reads an explicit ``timeout=None`` as NO
            # TIMEOUT AT ALL -- its "use the client's" value is a sentinel, not
            # None -- so passing the caller's None straight through disabled
            # timeouts on every route that did not name one: report, heartbeat,
            # abandon, register_artifact, health, register. Only `claim` named
            # one. VERIFIED against a loopback socket that accepts and never
            # answers: `timeout=None` never returns, while the same request
            # with the client's own 30s value raises ReadTimeout.
            #
            # That is the shape of the incident this SDK was being read for. A
            # heartbeat parked forever stops the SINGLE heartbeat thread, so
            # EVERY in-flight lease stops being renewed and the control plane's
            # reaper writes abandoned/deadline_exceeded over all of them.
            attempt_timeout = self._timeout if timeout is None else timeout
            if deadline is not None:
                remaining = deadline - time.time()
                if attempt > 1 and remaining <= 0.0:
                    last_err = (
                        f"{last_err or 'no attempt succeeded'}; gave up with "
                        f"{attempt - 1} attempt(s) at the delivery deadline"
                    )
                    break
                attempt_timeout = max(
                    _MIN_ATTEMPT_TIMEOUT_S, min(attempt_timeout, remaining)
                )
            try:
                resp = self._client.request(
                    method, url, json=json_body,
                    headers=request_headers or None,
                    timeout=attempt_timeout,
                )
            except httpx.HTTPError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                ambiguous = ambiguous or not isinstance(exc, _NEVER_SENT)
                if attempt == self._max_attempts or not (
                    retry_if_ambiguous or isinstance(exc, _NEVER_SENT)
                ):
                    break
                time.sleep(_bounded(_backoff(attempt), deadline))
                continue

            if resp.status_code == 410:
                raise LeaseLost(_detail(resp) or "lease lost",
                                ambiguous=ambiguous)
            ambiguous = ambiguous or _may_have_performed(resp.status_code)

            if resp.status_code == 429 or (
                resp.status_code in _RETRY_STATUS and retry_if_ambiguous
            ):
                last_err = f"HTTP {resp.status_code}: {_detail(resp)}"
                if attempt == self._max_attempts:
                    break
                time.sleep(
                    _bounded(_retry_after(resp) or _backoff(attempt), deadline)
                )
                continue

            if resp.status_code >= 400:
                # Terminal. A 4xx that is not 408/429 will not become a 2xx by
                # being sent again; retrying it only delays the error.
                raise TransportError(
                    f"{method} {path} -> HTTP {resp.status_code}: {_detail(resp)}",
                    status_code=resp.status_code,
                    ambiguous=ambiguous,
                )

            if not resp.content:
                return {}
            try:
                body = resp.json()
            except ValueError as exc:
                # PERFORMED: the control plane answered 2xx. What it did is
                # unreadable, not undone.
                raise TransportError(
                    f"{method} {path} -> HTTP {resp.status_code} with unparseable body: {exc}",
                    ambiguous=True,
                ) from exc
            return body if isinstance(body, dict) else {"data": body}

        raise TransportError(
            f"{method} {path} failed after {self._max_attempts} attempts: {last_err}",
            ambiguous=ambiguous,
        )


def _bounded(delay: float, deadline: float | None) -> float:
    """Never sleep past the deadline the caller reserved."""
    if deadline is None:
        return delay
    return max(0.0, min(delay, deadline - time.time()))


def _retry_after(resp: httpx.Response) -> float | None:
    """The server's own backoff, CAPPED at the same ceiling as our jitter.

    Uncapped, a ``Retry-After: 3600`` from a proxy -- not necessarily from the
    control plane at all -- turns one report into an hour of sleeping, three
    times over, inside a call path the runner has reserved seconds for.
    """
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, min(_MAX_BACKOFF_S, float(raw)))
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    """Exponential with full jitter, capped. Jitter matters: a fleet of
    flywheels that all retry on the same schedule reconverges into the spike
    that caused the backoff."""
    return random.uniform(0.0, min(_MAX_BACKOFF_S, 0.5 * (2 ** (attempt - 1))))


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "")[:300]
    if isinstance(body, dict):
        for key in ("detail", "error", "message", "reason"):
            if key in body:
                value = body[key]
                # A STRUCTURED detail is rendered as JSON, never as a Python
                # repr. The control plane answers many
                # refusals with an object -- `{error, cause, index, detail}`
                # for a refused rule, `{error, status, detail}` for an agent
                # attributed to no Product -- and the Node SDK renders the same
                # object with `JSON.stringify`, so this is the spelling both
                # packages share: compact separators, non-ASCII kept.
                text = value if isinstance(value, str) else json.dumps(
                    value, ensure_ascii=False, separators=(",", ":"))
                return text[:300]
    # ANYTHING ELSE IS THE ANSWER'S OWN TEXT -- an object with none of those
    # keys, an array, a bare scalar -- exactly as the Node client keeps it
    # (`cut(String(text))`). It used to be ``str(body)``, a Python repr of what
    # the text parsed to (``{'code': 'x', 'hint': None}``), so the same answer
    # read differently in the two packages and in neither as it was sent.
    return (resp.text or "")[:300]
