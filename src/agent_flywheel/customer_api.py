"""Reads of the customer-automation API (customer contract 1.29.0).

The rest of this package speaks the agent contract (``flywheel/1``) under
``/api/flywheel/v1`` with an agent's ``pefw_`` key. The runs list, training
eligibility and evaluation sets are the CUSTOMER contract's, served under
``/api/v2`` on the same origin to a customer-automation ``mck_`` key (the one
``pe login`` or the dashboard's Settings issues). An agent key is refused there,
so this client refuses it before sending anything and says which key it needs.

The Node SDK's ``src/customer-api.js`` is the same surface, name for name
(camelCase there): :data:`CUSTOMER_CONTRACT_VERSION`,
:data:`ENV_CUSTOMER_API_KEY`, :func:`customer_api_base`,
:class:`CustomerRefusal`, :class:`CustomerApiClient` with ``list_runs``,
``training_eligibility``, ``list_eval_sets`` and ``supports_contract``.

A refusal is ``{"detail": Refusal}``. Its ``code`` says why; ``message`` is the
developer's text; ``customer_message`` (1.29.0) is the cause in plain words for
a person, present when it differs from ``message``; ``customer_words`` is the
code's remedy in plain words. :attr:`CustomerRefusal.plain_words` is what to
show a person. The words are the server's: this package keeps no copy of them.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from ._version import __version__
from .control_plane import resolve_control_plane_url
from .errors import ConfigurationError

__all__ = [
    "CUSTOMER_CONTRACT_VERSION",
    "ENV_CUSTOMER_API_KEY",
    "PROMPT_RUN_REFUSAL_CODES",
    "RUN_LIST_REFUSAL_CODES",
    "CustomerApiClient",
    "CustomerRefusal",
    "customer_api_base",
]

#: The customer contract this client was written against.
CUSTOMER_CONTRACT_VERSION = "1.29.0"
_MAJOR = CUSTOMER_CONTRACT_VERSION.split(".", 1)[0]

#: Where the customer-automation key is read from when none is passed.
ENV_CUSTOMER_API_KEY = "PERCEPTEYE_MC_API_KEY"

#: 1.29.0: what a launch may read and how far the runs list reaches.
RUN_LIST_REFUSAL_CODES = ("launch_uri_outside_org", "run_page_too_deep")
#: 1.29.0: why a prompt run cannot start (``prompt_optimization`` says ahead).
PROMPT_RUN_REFUSAL_CODES = (
    "prompt_run_balance_too_small",
    "recorded_prompt_absent",
    "recorded_decisions_too_few",
    "recorded_prompt_mismatch",
)

_FLYWHEEL_SUFFIX = "/api/flywheel/v1"
_CUSTOMER_PREFIX = "/api/v2"


def customer_api_base(control_plane_url: str | None = None) -> str:
    """``/api/v2`` beside the Flywheel API, on the control plane's origin.

    The device login's rule (``login.device_endpoint``): a URL ending in
    ``/api/flywheel/v1`` keeps what precedes it; any other path is a control
    plane mounted elsewhere, and its origin is asked.
    """
    parts = urlsplit(resolve_control_plane_url(control_plane_url).strip())
    path = parts.path.rstrip("/")
    prefix = path[: -len(_FLYWHEEL_SUFFIX)] if path.endswith(_FLYWHEEL_SUFFIX) else ""
    return urlunsplit((parts.scheme, parts.netloc, prefix + _CUSTOMER_PREFIX, "", ""))


class CustomerRefusal(Exception):
    """The customer API refused a request; its fields are the body's own."""

    def __init__(self, status_code: int, body: Any) -> None:
        detail = body.get("detail") if isinstance(body, Mapping) else None
        refusal = detail if isinstance(detail, Mapping) else {}
        self.status_code = status_code
        self.body = body
        self.code: str | None = _text(refusal.get("code"))
        self.message: str = (
            _text(refusal.get("message"))
            or (detail if isinstance(detail, str) else None)
            or f"HTTP {status_code}"
        )
        self.customer_message: str | None = _text(refusal.get("customer_message"))
        self.customer_words: str | None = _text(refusal.get("customer_words"))
        self.remediation: str | None = _text(refusal.get("remediation"))
        self.fault_owner: str | None = _text(refusal.get("fault_owner"))
        retryable = refusal.get("retryable")
        self.retryable: bool | None = retryable if isinstance(retryable, bool) else None
        super().__init__(
            f"HTTP {status_code}" + (f" {self.code}" if self.code else "")
            + f": {self.message}"
        )

    @property
    def plain_words(self) -> str:
        """What to show a person: the cause in plain words, else the remedy in
        plain words, else the message."""
        return self.customer_message or self.customer_words or self.message


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _params(**values: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, value in values.items():
        if value is None:
            continue
        out[name] = ("true" if value else "false") if isinstance(value, bool) else value
    return out


class CustomerApiClient:
    """Synchronous reads of the customer API with a customer-automation key."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        control_plane_url: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        env = os.environ if environ is None else environ
        key = (api_key or env.get(ENV_CUSTOMER_API_KEY) or "").strip()
        if not key:
            raise ConfigurationError(
                f"a customer-automation key is required: pass api_key= or set "
                f"{ENV_CUSTOMER_API_KEY} (run `pe login`, or issue one in the "
                "dashboard under Settings, control-plane keys)")
        if key.startswith("pefw_"):
            raise ConfigurationError(
                "this is an agent key (pefw_), which the customer API refuses; "
                "it needs a customer-automation key (mck_): run `pe login`, or "
                "issue one in the dashboard under Settings, control-plane keys")
        self.base_url = (base_url or customer_api_base(control_plane_url)).rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "X-Percepteye-Sdk": f"agent-flywheel-python/{__version__}",
        }

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> CustomerApiClient:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def _get(self, path: str, params: Mapping[str, Any]) -> Any:
        resp = self._client.get(
            f"{self.base_url}{path}", params=dict(params), headers=self._headers)
        try:
            body = resp.json() if resp.content else None
        except ValueError:
            body = resp.text
        if resp.status_code >= 400:
            raise CustomerRefusal(resp.status_code, body)
        return body

    def supports_contract(self) -> bool:
        """Whether the server speaks this client's contract major (GET /health)."""
        body = self._get("/health", {})
        served = (body or {}).get("customer_contract_versions_supported") or []
        return _MAJOR in [str(v) for v in served]

    def list_runs(
        self,
        *,
        status: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
        started_after: str | None = None,
        started_before: str | None = None,
        product_id: str | None = None,
        include_background: bool | None = None,
        include_prompt_evolutions: bool | None = None,
    ) -> list[dict[str, Any]]:
        """GET /runs. With ``include_prompt_evolutions`` each prompt evolution
        is one row (``prompt_evolution`` set) in place of its arms; every run of
        an evolution names it in ``prompt_evolution_id``. Each run names what it
        was for (``target_id``, ``target_kind``, ``target_name``), and
        ``certifies_run_id`` / ``chain_parent_run_id`` place it in its chain.
        Paging past the server's depth is refused ``run_page_too_deep``: use
        ``started_before``."""
        return self._get("/runs", _params(
            status=status, limit=limit, offset=offset,
            started_after=started_after, started_before=started_before,
            product_id=product_id, include_background=include_background,
            include_prompt_evolutions=include_prompt_evolutions))

    def training_eligibility(
        self, *, agent_id: str | None = None, product_id: str | None = None,
    ) -> dict[str, Any]:
        """GET /training/eligibility. ``prompt_optimization`` says whether
        prompt runs are on, and with ``agent_id`` or ``product_id`` the path
        that agent takes: ``mode`` (``live`` / ``offline`` / ``None``),
        ``can_run``, ``needs`` and ``message``, the ``preparation_run_id`` an
        offline run reads, the ``algorithms``, and ``affordable``."""
        return self._get("/training/eligibility", _params(
            agent_id=agent_id, product_id=product_id))

    def list_eval_sets(
        self,
        *,
        product_id: str | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> Any:
        """GET /eval-sets: a Product's evaluation sets, one page."""
        return self._get("/eval-sets", _params(
            product_id=product_id, limit=limit, offset=offset))
