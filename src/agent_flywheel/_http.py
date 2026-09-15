"""The one HTTP client, with retry semantics that read the response.

Against named defects in a previous SDK: its retry helper returned the
response object without reading status, and retried only transport
exceptions -- so a 503 was indistinguishable from success, and its only
recovery from an unexpected error was a blanket five-second sleep inside a
bare ``except`` that also swallowed a real ``TypeError``.
"""
from __future__ import annotations

import random
import time
from collections.abc import Mapping
from typing import Any

import httpx

from ._version import __version__
from .contract import CONTRACT_VERSION
from .errors import LeaseLost, TransportError

__all__ = ["ControlPlaneClient"]

#: Transient. Retry these, honouring Retry-After when present.
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})

#: Failures where the connection was never established, so the server cannot
#: have seen the request. Everything else -- a read timeout above all -- may
#: have been processed, and is only safe to resend if the endpoint is replayable.
_NEVER_SENT = (httpx.ConnectError, httpx.ConnectTimeout)

_SDK_UA = f"agent-flywheel-python/{__version__}"


class ControlPlaneClient:
    """Thin, synchronous, outbound-only client for the agent control plane."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 30.0,
        max_attempts: int = 4,
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
        self._client = client or httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Contract-Version": CONTRACT_VERSION,
                "X-Percepteye-Sdk": _SDK_UA,
                "Accept": "application/json",
            },
        )

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
        retry_if_ambiguous: bool = True,
        headers: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Send one request, retrying transient failures.

        ``retry_if_ambiguous=False`` narrows retries to failures where the
        request provably never reached the server. Use it for any call whose
        server side has a side effect and no replay protection: retrying a
        request that MAY have been processed performs it twice.
        """
        url = f"{self.base_url}{path}"
        request_headers: dict[str, str] = dict(headers or {})
        if idempotency_key:
            request_headers["Idempotency-Key"] = idempotency_key

        last_err: str | None = None
        for attempt in range(1, self._max_attempts + 1):
            try:
                resp = self._client.request(
                    method, url, json=json_body,
                    headers=request_headers or None,
                    timeout=timeout,
                )
            except httpx.HTTPError as exc:
                last_err = f"{type(exc).__name__}: {exc}"
                if attempt == self._max_attempts or not (
                    retry_if_ambiguous or isinstance(exc, _NEVER_SENT)
                ):
                    break
                time.sleep(_backoff(attempt))
                continue

            if resp.status_code == 410:
                raise LeaseLost(_detail(resp) or "lease lost")

            if resp.status_code == 429 or (
                resp.status_code in _RETRY_STATUS and retry_if_ambiguous
            ):
                last_err = f"HTTP {resp.status_code}: {_detail(resp)}"
                if attempt == self._max_attempts:
                    break
                time.sleep(_retry_after(resp) or _backoff(attempt))
                continue

            if resp.status_code >= 400:
                # Terminal. A 4xx that is not 408/429 will not become a 2xx by
                # being sent again; retrying it only delays the error.
                raise TransportError(
                    f"{method} {path} -> HTTP {resp.status_code}: {_detail(resp)}"
                )

            if not resp.content:
                return {}
            try:
                body = resp.json()
            except ValueError as exc:
                raise TransportError(
                    f"{method} {path} -> HTTP {resp.status_code} with unparseable body: {exc}"
                ) from exc
            return body if isinstance(body, dict) else {"data": body}

        raise TransportError(
            f"{method} {path} failed after {self._max_attempts} attempts: {last_err}"
        )


def _retry_after(resp: httpx.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return max(0.0, float(raw))
    except ValueError:
        return None


def _backoff(attempt: int) -> float:
    """Exponential with full jitter, capped. Jitter matters: a fleet of
    flywheels that all retry on the same schedule reconverges into the spike
    that caused the backoff."""
    return random.uniform(0.0, min(8.0, 0.5 * (2 ** (attempt - 1))))


def _detail(resp: httpx.Response) -> str:
    try:
        body = resp.json()
    except ValueError:
        return (resp.text or "")[:300]
    if isinstance(body, dict):
        for key in ("detail", "error", "message", "reason"):
            if key in body:
                return str(body[key])[:300]
    return str(body)[:300]
