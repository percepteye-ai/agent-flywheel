"""Reads of the customer-automation API (customer contract 1.51.0).

The rest of this package speaks the agent contract (``flywheel/1``) under
``/api/flywheel/v1`` with an agent's ``pefw_`` key. The runs list, training
eligibility and evaluation sets are the CUSTOMER contract's, served under
``/api/v2`` on the same origin to a customer-automation ``mck_`` key (the one
``pe login`` or the dashboard's Settings issues). An agent key is refused there,
so this client refuses it before sending anything and says which key it needs.

The Node SDK's ``src/customer-api.js`` is the same surface, name for name
(camelCase there): :data:`CUSTOMER_CONTRACT_VERSION`,
:data:`ENV_CUSTOMER_API_KEY`, :func:`customer_api_base`,
:class:`CustomerRefusal`, :class:`CustomerContractTooOld`, the six
``*_REFUSAL_CODES`` groups, :class:`CustomerApiClient` with ``list_runs``,
``training_eligibility``, ``list_eval_sets`` and ``supports_contract``
(``tests/test_customer_api_parity.py`` runs both and requires one answer).

A refusal is ``{"detail": Refusal}``. Its ``code`` says why; ``message`` is the
developer's text; ``customer_message`` (1.29.0) is the cause in plain words for
a person, present when it differs from ``message``; ``customer_words`` is the
code's remedy in plain words. :attr:`CustomerRefusal.plain_words` is what to
show a person. The words are the server's: this package keeps no copy of them.

Every read works against any 1.x server from 1.30.0 on. One argument needs a
newer one: ``training_eligibility(training_location=...)`` needs 1.50.0, and
an older server, which ignores the query and answers for the organization's
own setting, is refused with :class:`CustomerContractTooOld` rather than
believed.
"""
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote, urlsplit, urlunsplit

import httpx

from ._version import __version__
from .control_plane import resolve_control_plane_url
from .errors import ConfigurationError

__all__ = [
    "AGENT_REFUSAL_CODES",
    "ATTACK_REFUSAL_CODES",
    "CUSTOMER_CONTRACT_VERSION",
    "CYBER_REFUSAL_CODES",
    "ENV_CUSTOMER_API_KEY",
    "PROMPT_RUN_REFUSAL_CODES",
    "RUN_LIST_REFUSAL_CODES",
    "TRAINING_REFUSAL_CODES",
    "CustomerApiClient",
    "CustomerContractTooOld",
    "CustomerRefusal",
    "customer_api_base",
]

#: The customer contract this client was written against.
CUSTOMER_CONTRACT_VERSION = "1.51.0"
_MAJOR = CUSTOMER_CONTRACT_VERSION.split(".", 1)[0]

#: Where the customer-automation key is read from when none is passed.
ENV_CUSTOMER_API_KEY = "PERCEPTEYE_MC_API_KEY"

#: 1.29.0: what a launch may read and how far the runs list reaches.
RUN_LIST_REFUSAL_CODES = ("launch_uri_outside_org", "run_page_too_deep")
#: 1.29.0: why a prompt run cannot start (``prompt_optimization`` says ahead).
#: 1.38.0: a prompt-objective hill climb's evolution could not start; 1.39.0:
#: the customer's own model is unreachable (no stand-in is ever used). The
#: three 1.40.0 codes (``prompt_box_*``) are not carried: they are the
#: platform's own prompt-optimization host, the platform's fault, and one names
#: its serving stack, which this package does not publish.
PROMPT_RUN_REFUSAL_CODES = (
    "prompt_run_balance_too_small",
    "recorded_prompt_absent",
    "recorded_decisions_too_few",
    "recorded_prompt_mismatch",
    "prompt_objective_unprepared",
    "prompt_evolution_refused",
    "prompt_model_unreachable",
)
#: 1.30.0: the held-out attack set a promotion is measured on never changes.
#: 1.31.0: the attack gate and the attack measurement refuse by name.
#: 1.34.0: a person's yes that was not shown an unmeasured attack resistance.
ATTACK_REFUSAL_CODES = (
    "attack_set_is_sealed",
    "attack_resistance_measuring",
    "attack_regression_not_accepted",
    "attack_approval_stale",
    "attack_regression_not_stated",
    "measurement_deadline_passed",
    "prompt_changed",
    "hillclimb_not_found",
    "round_not_found",
    "event_kind_is_the_platforms",
    "attack_unmeasured_undisclosed",
)
#: Why a hill climb's cyber choice cannot apply: refused, never downgraded
#: (1.44.0), and a trainer whose rewards cannot compare attacks (1.46.0).
CYBER_REFUSAL_CODES = (
    "cyber_disabled",
    "cyber_training_disabled",
    "cyber_needs_training",
    "cyber_no_training_entry",
    "cyber_lane_has_no_attack_reward",
    "hillclimb_security_unknown",
    "cyber_attack_reward_not_comparable",
)
#: Why a training run is refused or stopped: the GPU host's prompt fit, the
#: recipe tokenizer and the teacher gate's evidence (1.32.0); a diverged or
#: unhealthy train (1.36.0, 1.37.0; ``CustomerRefusal.detail`` carries the
#: evidence); a pool that cannot train the objective (1.46.0); a chosen model
#: that cannot be trained (1.49.0); a base model named by no usable id, a
#: destination that cannot train the algorithm, a checkpoint another
#: organization's runs produced (1.51.0, raised before it published them).
TRAINING_REFUSAL_CODES = (
    "opsd_prompts_exceed_host",
    "opsd_recipe_tokenizer_hub_unavailable",
    "opsd_recipe_tokenizer_refused",
    "opsd_recipe_tokenizer_platform_fault",
    "opsd_teacher_preflight_no_evidence_decisions",
    "training_diverged_at_lr",
    "training_unhealthy",
    "training_pool_cannot_train_objective",
    "opsd_chosen_model_not_trainable",
    "hillclimb_base_model_invalid",
    "base_model_unresolvable",
    "destination_algorithm_unsupported",
    "provider_checkpoint_not_owned",
)
#: Why a round with your agent is refused before anything is spent (1.51.0;
#: Mission Control raised each before it published them): its "Allow practice
#: runs" switch is off, no ``serve()`` process runs it with
#: ``PERCEPTEYE_AGENT_MODE=training``, its model-content capture is off for a
#: round read from its capture, it gives practice nothing to write tasks
#: from, or the product does not declare the agent the organization
#: registered (or declares one that cannot serve).
AGENT_REFUSAL_CODES = (
    "practice_rollouts_not_enabled",
    "practice_needs_a_training_process",
    "model_content_capture_not_enabled",
    "flywheel_no_task_source",
    "flywheel_product_undeclared",
    "invalid_flywheel_spec",
)

_FLYWHEEL_SUFFIX = "/api/flywheel/v1"
_CUSTOMER_PREFIX = "/api/v2"


def customer_api_base(
    control_plane_url: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> str:
    """``/api/v2`` beside the Flywheel API, on the control plane's origin.

    The device login's rule (``login.device_endpoint``): a URL ending in
    ``/api/flywheel/v1`` keeps what precedes it; any other path is a control
    plane mounted elsewhere, and its origin is asked.
    """
    parts = urlsplit(resolve_control_plane_url(control_plane_url, environ).strip())
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
        own = refusal.get("detail")
        #: The refusal's own evidence (``Refusal.detail``), such as
        #: ``waiting_on``, ``host_prompt_fit``, ``cause`` or ``chosen_model``;
        #: ``None`` when it carries none.
        self.detail: dict[str, Any] | None = dict(own) if isinstance(own, Mapping) else None
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


class CustomerContractTooOld(ConfigurationError):
    """The server answered, but on a customer contract older than the one a
    requested argument needs, so its answer is not the one asked for."""


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
        self.base_url = (
            base_url or customer_api_base(control_plane_url, env)).rstrip("/")
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
        if not isinstance(body, Mapping):
            return False
        served = body.get("customer_contract_versions_supported") or []
        if not isinstance(served, (list, tuple)):
            return False
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
        A run also names what it was admitted to optimise (``objective``,
        ``security_mode``, ``benchmark_eval_set_id``, 1.42.0), the model it
        trains (``base_model``, 1.43.0), its trainer (``algorithm``, 1.46.0)
        and the run it belongs to (``owner_run_id``, 1.47.0; ``None`` for a run
        the customer started): one row per run the customer started, the rest
        folded under it. A server older than those versions omits the fields.
        Paging past the server's depth is refused ``run_page_too_deep``: use
        ``started_before``."""
        return self._get("/runs", _params(
            status=status, limit=limit, offset=offset,
            started_after=started_after, started_before=started_before,
            product_id=product_id, include_background=include_background,
            include_prompt_evolutions=include_prompt_evolutions))

    def training_eligibility(
        self, *, agent_id: str | None = None, product_id: str | None = None,
        training_location: str | None = None,
    ) -> dict[str, Any]:
        """GET /training/eligibility. ``prompt_optimization`` says whether
        prompt runs are on, and with ``agent_id`` or ``product_id`` the path
        that agent takes: ``mode`` (``live`` / ``offline`` / ``None``),
        ``can_run``, ``needs`` and ``message``, the ``preparation_run_id`` an
        offline run reads, the ``algorithms``, ``affordable``, and (1.29.0)
        ``model_under_test``: the model a prompt run would test on.
        ``security`` (1.30.0; 1.44.0) is the ``default`` (``off``: cyber is a
        choice), whether it is ``enabled``, per objective whether cyber is
        ``available`` (else its ``refusal``, one of
        :data:`CYBER_REFUSAL_CODES`), and ``measurement_funding``
        (``run_quote``: the attack arms are on the run's quote).
        ``flash_reinforce`` (1.46.0) is that practice lane's ``allowed``,
        ``refusals`` and ``cost_bound``; ``opsd.trainable_models`` (1.49.0)
        the models a hill climb may choose to train.

        ``training_location`` (1.50.0: ``platform_pool`` or ``platform_gpu``;
        the server refuses any other) answers for a run trained there instead
        of at the organization's own setting; ``training_locations`` names the place answered for
        (``answered_for``), the ``default``, and per ``options`` entry whether
        it is ``available`` (else its ``refusal``) and, on our GPUs,
        ``gpu_hosts`` and ``gpu_hosts_free``. A server older than 1.50.0
        ignores the argument and answers for the organization's own setting;
        that answer is refused, :class:`CustomerContractTooOld`, never
        returned as the one asked for."""
        body = self._get("/training/eligibility", _params(
            agent_id=agent_id, product_id=product_id,
            training_location=training_location))
        if training_location is not None:
            locations = body.get("training_locations") if isinstance(body, Mapping) else None
            answered = locations.get("answered_for") if isinstance(locations, Mapping) else None
            if answered != training_location:
                raise CustomerContractTooOld(
                    f"the control plane at {self.base_url} answered "
                    f"/training/eligibility without answering for "
                    f"training_location={training_location!r}: choosing where "
                    "a run trains needs customer contract 1.50.0, and an older "
                    "server answers for the organization's own setting instead. "
                    "Omit training_location, or use a newer Mission Control.")
        return body

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

    def hillclimb_security(self, run_id: str) -> dict[str, Any]:
        """GET /hillclimbs/{run_id}/security (1.30.0): the run's security mode
        and why, and its attack-resistance measurement. ``measurement.state``
        is never a pass by omission: ``not_measured``, ``measuring``,
        ``not_enough_attacks``, ``attacks_unpaired``, ``no_regression`` or
        ``regressed``; its ``readout.benchmarks`` (1.35.0) holds each attack
        benchmark to the same rule. ``base`` and ``checkpoints`` (1.44.0) are
        the base model's and each certified round's held-out attack pass rate;
        ``external_benchmarks``, ``benchmark_coverage`` and ``capability``
        (1.45.0) are unpaired, ungated arms. Each is ``None`` (``checkpoints``
        ``[]``) on a run that did not choose cyber, and absent on an older
        server. A run that is not a hill climb of this organization is
        refused 404 ``hillclimb_not_found``."""
        return self._get(f"/hillclimbs/{quote(str(run_id), safe='')}/security", {})
