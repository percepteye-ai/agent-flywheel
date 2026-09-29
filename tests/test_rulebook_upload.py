"""``rules=``: an agent declares its Product's workflow rules, and nothing
the control plane says about them can stop the agent.

THE SEAM. Producer: ``serve(rules=...)`` and ``attach(rules=...)``, through
``AttachTransport.register_rulebook`` and the real ``ControlPlaneClient``.
Consumer: the control plane's ``POST /agents/{agent_id}/rulebook``, played
here by a fake at the HTTP layer (``httpx.MockTransport``) that answers the
way that route does -- ``registered`` / ``unchanged`` / ``retracted``
receipts, a 409 for an agent no single Product declares, a 422 naming a
refused rule by cause and index. Everything between the call site and the
socket is the real code: no transport or client is faked.

Pinned:

* the rules go once, after the registration, on their own route, with the
  closed body the published schema declares and a key derived from it --
  in the BACKGROUND, so a slow or hung rulebook route holds back neither
  ``attach()``'s return nor ``serve()``'s policy read;
* ``rules=None`` sends nothing, ``[]`` retracts, and the same rules on the
  next start register nothing;
* a refused or undeliverable upload is RETURNED (``rulebook_refusal``, the
  line ``serve()`` prints) in the control plane's own words, never raised,
  and costs neither the registration nor the policy read; "not registered"
  is said only when that is known, and a failure after the rules may have
  landed says it is unknown;
* a value that is not a list of strings refuses at startup, before anything
  connects.

Products and rules are generic and obviously fake.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import fields
from pathlib import Path

import httpx
import pytest
from jsonschema import Draft202012Validator

import agent_flywheel as fw
from agent_flywheel import _http, production
from agent_flywheel.contract import (
    RulebookReceipt,
    RulebookRequest,
    RulebookStatus,
)
from agent_flywheel.errors import ConfigurationError, ContractViolation
from agent_flywheel.transport import AttachTransport, RulebookUpload

BASE = "https://cp.example.invalid/api/flywheel/v1"
AGENT = "widget-helper"
PRODUCT = "prod-widget-desk"
RULE_A = ("When the caller has not said which widget colour they want, ask "
          "for the colour before placing any order.")
RULE_B = ("Before cancelling a plan, restate the renewal date the customer "
          "would lose and wait for them to confirm.")
POLICY = {"arm": "candidate", "reason": "an operator promoted it"}
SCHEMA = json.loads(
    (Path(__file__).resolve().parents[1] / "schema" / "flywheel-1.json")
    .read_text(encoding="utf-8"))


def _refusal(status: int, detail: dict) -> httpx.Response:
    return httpx.Response(status, json={"detail": detail})


UNATTRIBUTED = _refusal(409, {
    "error": "rulebook_unattributed", "status": "ambiguous",
    "detail": "agent 'widget-helper' is attributed to no Product (ambiguous)",
})
INVALID = _refusal(422, {
    "error": "rulebook_invalid", "cause": "rule_too_short", "index": 1,
    "detail": "rule 1 is 9 characters; a rule is at least 24",
})


class ControlPlane:
    """The routes ``serve()`` and ``attach()`` reach, answered the way the
    control plane answers them. The rulebook route keeps the one rule the
    real one does about a re-declaration -- rules equal to the ones in force
    are ``unchanged`` -- so a second start can be asserted on."""

    def __init__(self, *, rulebook_answer=None, rulebook_error=None):
        self.requests: list[httpx.Request] = []
        self.rulebook_answer = rulebook_answer
        self.rulebook_error = rulebook_error
        #: Called before the rulebook route answers: a route that is slow.
        self.rulebook_hold = None
        #: Called when the policy is read, to see what it overtook.
        self.on_policy_read = None
        #: The route REGISTERS the rules, then its answer is lost.
        self.lose_answer = False
        self.in_force: list[str] | None = None
        self.version = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        route = self.route(request)
        if route == "GET /health":
            return httpx.Response(
                200, json={"contract_versions_supported": ["flywheel/1"]})
        if route == "POST /agents":
            return httpx.Response(200, json={
                "agent_id": AGENT, "grade": "advisory", "grade_reasons": [],
                "capture": {"turns": True}})
        if route == "GET /policy/current":
            if self.on_policy_read is not None:
                self.on_policy_read()
            return httpx.Response(200, json=POLICY)
        if route.endswith("/rulebook"):
            if self.rulebook_hold is not None:
                self.rulebook_hold()
            if self.rulebook_error is not None:
                raise self.rulebook_error(
                    "connection refused", request=request)
            if self.lose_answer:
                self.receipt(json.loads(request.content)["rules"])
                raise httpx.ReadTimeout("read timed out", request=request)
            if self.rulebook_answer is not None:
                return self.rulebook_answer
            return httpx.Response(200, json=self.receipt(
                json.loads(request.content)["rules"]))
        return httpx.Response(404, json={"detail": f"no route {route}"})

    def receipt(self, rules: list[str]) -> dict:
        current = self.in_force or []
        if rules == current:
            status = "unchanged"
        else:
            status = "registered" if rules else "retracted"
            self.version += 1
            self.in_force = rules
        return {
            "status": status, "product_id": PRODUCT,
            "version_id": f"v-{self.version}" if self.version else None,
            "version": self.version or None,
            "content_sha256": f"{self.version:064x}" if self.version else None,
            "parent_sha256": None, "rule_count": len(self.in_force or []),
        }

    @staticmethod
    def route(request: httpx.Request) -> str:
        path = request.url.raw_path.decode("ascii")
        return f"{request.method} {path.removeprefix('/api/flywheel/v1')}"

    def routes(self) -> list[str]:
        return [self.route(r) for r in self.requests]

    def rulebook_requests(self) -> list[httpx.Request]:
        return [r for r in self.requests
                if self.route(r).endswith("/rulebook")]


@pytest.fixture
def cp(monkeypatch):
    """A fake control plane behind the REAL client: serve() and attach()
    build their own ``ControlPlaneClient``, and this hands each one an httpx
    client whose socket is the fake. One attempt per call, so a failure is
    asserted without the backoff ladder's sleeps."""
    plane = ControlPlane()
    real = _http.ControlPlaneClient

    def _client(url, key, **kw):
        return real(url, key, max_attempts=1, client=httpx.Client(
            transport=httpx.MockTransport(plane.handler)))

    monkeypatch.setattr(_http, "ControlPlaneClient", _client)
    return plane


def _serve(tmp_path, **kw):
    kw.setdefault("isolation", "inprocess")
    kw.setdefault("introspect", False)
    return fw.serve(
        "tests.fixtures_agent:build", agent_id=AGENT, input_shape="text",
        control_plane_url=BASE, api_key="pefw_test",
        artifacts_dir=str(tmp_path), max_rollouts=0, **kw)


@pytest.fixture
def attached(tmp_path):
    made = []

    def _make(**kw):
        kw.setdefault("control_plane_url", BASE)
        kw.setdefault("api_key", "pefw_test")
        px = fw.attach(agent_id=AGENT, trajectory_root=str(tmp_path),
                       adapters=[], **kw)
        made.append(px)
        return px

    yield _make
    for px in made:
        px.close()


# ── the happy path, through the real transport and client ─────────────────


def test_serve_declares_the_rules_once_after_registering(cp, tmp_path, capsys):
    """Mutations: upload before ``tr.register`` -> the order is wrong (red);
    drop the upload -> no rulebook request (red); send the rules inside the
    registration -> the registration body carries them (red).

    The body is compared as JSON, not bytes: the bytes are httpx's encoder's,
    and the declared floor (``httpx>=0.27``) spaces them differently from
    0.28. The control plane reads JSON, and the key is this package's own
    digest of the body, whichever httpx sent it."""
    assert _serve(tmp_path, rules=[RULE_A, RULE_B]) == 0

    routes = cp.routes()
    assert routes[:2] == ["GET /health", "POST /agents"]
    assert sorted(routes[2:]) == sorted(
        [f"POST /agents/{AGENT}/rulebook", "GET /policy/current"])
    (sent,) = cp.rulebook_requests()
    assert json.loads(sent.content) == {"rules": [RULE_A, RULE_B]}
    assert re.fullmatch(rf"rulebook-{AGENT}-[0-9a-f]{{32}}",
                        sent.headers["Idempotency-Key"])
    registration = json.loads(cp.requests[1].content)
    assert "rules" not in registration
    line = capsys.readouterr().out
    assert (f"[percepteye] rulebook: registered as version 1 — 2 rule(s) "
            f"for Product {PRODUCT}") in line


def test_attach_declares_the_rules_and_keeps_the_receipt(cp, attached):
    """Mutations: drop the upload from attach() -> ``rulebook`` is None
    (red); pass the receipt to the wrong attribute -> red."""
    px = attached(rules=[RULE_A, RULE_B])
    assert px.wait_for_rulebook(5)

    assert cp.routes() == [
        "POST /agents", "GET /policy/current", f"POST /agents/{AGENT}/rulebook",
    ]
    assert px.rulebook_refusal is None
    assert isinstance(px.rulebook, RulebookReceipt)
    assert px.rulebook.status == "registered"
    assert px.rulebook.product_id == PRODUCT
    assert px.rulebook.rule_count == 2
    assert px.policy == POLICY
    assert px.may_upload is True


def test_the_same_rules_on_the_next_start_register_nothing(cp, tmp_path, capsys):
    """A process that declares rules re-declares them on every start; the
    second declaration is answered ``unchanged`` and says so."""
    _serve(tmp_path, rules=[RULE_A])
    _serve(tmp_path, rules=[RULE_A])
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "rulebook" in ln]
    assert lines[1] == (
        f"[percepteye] rulebook: unchanged, already in force as version 1 — "
        f"1 rule(s) for Product {PRODUCT}")
    first, second = cp.rulebook_requests()
    assert first.headers["Idempotency-Key"] == second.headers["Idempotency-Key"]


def test_an_empty_list_retracts(cp, attached, tmp_path, capsys):
    """Through BOTH writers: ``[]`` is a declaration, not an absence.

    Mutations: ``if rules:`` in place of ``is not None`` in serve() or
    attach() -> no retraction is sent (red)."""
    cp.in_force, cp.version = [RULE_A], 1
    _serve(tmp_path, rules=[])
    (sent,) = cp.rulebook_requests()
    assert json.loads(sent.content) == {"rules": []}
    assert (f"[percepteye] rulebook: retracted as version 2 — 0 rule(s) for "
            f"Product {PRODUCT}") in capsys.readouterr().out

    cp.requests.clear()
    cp.in_force, cp.version = [RULE_A], 1
    px = attached(rules=[])
    assert px.wait_for_rulebook(5)
    (sent,) = cp.rulebook_requests()
    assert json.loads(sent.content) == {"rules": []}
    assert px.rulebook.status == "retracted"
    assert px.rulebook.rule_count == 0


def test_rules_none_sends_nothing(cp, tmp_path, attached):
    """``None`` is not ``[]``: an agent that declares no rules must not
    retract the ones the Product's API registered.

    Mutation: treat None as [] -> a retraction is sent (red)."""
    _serve(tmp_path)
    px = attached()
    assert cp.rulebook_requests() == []
    assert px.rulebook is None and px.rulebook_refusal is None


# ── refusals: surfaced in the control plane's words, never raised ─────────


@pytest.mark.parametrize(("answer", "names"), [
    (UNATTRIBUTED, ('HTTP 409', '"error":"rulebook_unattributed"',
                    '"status":"ambiguous"')),
    (INVALID, ('HTTP 422', '"error":"rulebook_invalid"',
               '"cause":"rule_too_short"', '"index":1')),
])
def test_a_refusal_is_the_control_planes_own_words_and_never_raised(
        cp, attached, tmp_path, capsys, answer, names):
    """Mutations: raise from ``register_rulebook`` -> serve() and attach()
    fail (red); one fixed sentence for every refusal -> the cause is absent
    (red); render the structured detail as a Python repr -> the JSON
    spelling both SDKs share is absent (red)."""
    cp.rulebook_answer = answer

    assert _serve(tmp_path, rules=[RULE_A, "too short"]) == 0
    line = capsys.readouterr().out
    assert "[percepteye] rulebook NOT confirmed — this agent's rules were " \
           "not registered: POST /agents/widget-helper/rulebook ->" in line
    for name in names:
        assert name in line, (name, line)

    px = attached(rules=[RULE_A, "too short"])
    assert px.wait_for_rulebook(5)
    assert px.rulebook is None
    assert px.rulebook_refusal.startswith(
        "this agent's rules were not registered: ")
    assert px.rulebook_refusal.endswith(
        "The rulebook in force, if any, is unchanged.")
    for name in names:
        assert name in px.rulebook_refusal
    # The refusal cost the rules and nothing else.
    assert px.policy == POLICY
    assert px.may_upload is True


def test_an_unreachable_rulebook_route_costs_nothing_else(cp, attached, tmp_path,
                                                          capsys):
    """Mutation: remove the ``except`` in ``register_rulebook`` -> serve()
    raises and attach()'s policy read is lost to its outer ``except`` (red)."""
    cp.rulebook_error = httpx.ConnectError

    assert _serve(tmp_path, rules=[RULE_A]) == 0
    assert "GET /policy/current" in cp.routes()
    assert "ConnectError" in capsys.readouterr().out

    px = attached(rules=[RULE_A])
    assert px.policy == POLICY
    assert px.wait_for_rulebook(5)
    assert px.rulebook is None
    assert "ConnectError" in px.rulebook_refusal
    # NEVER SENT, so this one is known: nothing registered.
    assert px.rulebook_refusal.startswith(
        "this agent's rules were not registered: ")


# ── a slow route costs the rules, and nothing else ────────────────────────

PENDING = ("the control plane has not answered the rules passed as rules= "
           "yet; Attachment.wait_for_rulebook() waits for its answer.")


def test_a_hung_rulebook_route_does_not_hold_back_attach(cp, attached):
    """attach() returns the policy it has read, the capture verdict and the
    customer's startup while the rulebook route has not answered; the answer
    lands on the attachment when it comes.

    Mutations: declare inline in attach()'s network block -> attach()
    returns only when the route answers, after the hold (red); leave
    ``rulebook_refusal`` None while pending -> red; ``wait_for_rulebook``
    returning without joining -> the receipt is not there yet (red)."""
    release = threading.Event()
    cp.rulebook_hold = lambda: release.wait(3)

    started = time.monotonic()
    px = attached(rules=[RULE_A])
    took = time.monotonic() - started

    assert took < 2, f"attach() waited {took:.2f}s on the rulebook route"
    assert px.policy == POLICY
    assert px.may_upload is True
    assert px.rulebook is None
    assert px.rulebook_refusal == PENDING
    assert px.wait_for_rulebook(0.05) is False

    release.set()
    assert px.wait_for_rulebook(5) is True
    assert px.rulebook.status == "registered"
    assert px.rulebook_refusal is None


def test_a_hung_rulebook_route_does_not_hold_back_serves_policy_read(
        cp, tmp_path, capsys):
    """serve() reads its policy -- and would claim -- while the rulebook
    route has not answered; the line still prints before serve() returns.

    Mutations: declare inline before the policy read -> the route answers
    before the policy is read (red); drop serve()'s join before closing the
    client -> the line is not printed (red)."""
    policy_read = threading.Event()
    answered_after_the_policy_read = []
    cp.on_policy_read = policy_read.set

    def hold():
        answered_after_the_policy_read.append(policy_read.wait(3))
        # Still answering when serve()'s loop has returned: the join is what
        # lets the line reach the console before the client closes.
        time.sleep(0.3)

    cp.rulebook_hold = hold

    assert _serve(tmp_path, rules=[RULE_A]) == 0

    assert answered_after_the_policy_read == [True]
    assert (f"[percepteye] rulebook: registered as version 1 — 1 rule(s) for "
            f"Product {PRODUCT}") in capsys.readouterr().out


def test_close_gives_an_answer_on_its_way_a_bounded_moment(
        cp, attached, monkeypatch):
    """A process that closes moments after attach() does not kill its rules'
    declaration with the interpreter -- and a route that never answers does
    not hold the close open either.

    Mutations: drop the wait in ``close()`` -> no receipt after it (red);
    wait unbounded -> the second close takes the whole hold (red)."""
    release = threading.Event()
    cp.rulebook_hold = lambda: release.wait(3)
    px = attached(rules=[RULE_A])
    threading.Timer(0.2, release.set).start()
    px.close()
    assert px.rulebook is not None and px.rulebook.status == "registered"

    monkeypatch.setattr(production, "RULEBOOK_SETTLE_GRACE_S", 0.1)
    never = threading.Event()
    cp.rulebook_hold = lambda: never.wait(3)
    px = attached(rules=[RULE_B])
    started = time.monotonic()
    px.close()
    assert time.monotonic() - started < 2
    assert px.rulebook_refusal == PENDING
    never.set()


# ── "not registered" only when that is known ──────────────────────────────


UNKNOWN = (
    "whether this agent's rules were registered is unknown: POST "
    "/agents/widget-helper/rulebook failed after 1 attempts: ReadTimeout: "
    "read timed out. The control plane may have received them before the "
    "failure; declaring the same rules again is safe, because rules equal to "
    "the ones in force are answered unchanged."
)


def test_a_failure_after_the_rules_may_have_landed_is_named_unknown(
        cp, attached, tmp_path, capsys):
    """The control plane registers the rules, then its answer is lost. "Not
    registered, the rulebook in force is unchanged" would be false -- the new
    rules ARE in force -- and would send an operator to believe the old ones
    still are.

    Mutations: ``rulebook_refusal`` ignoring ``ambiguous`` -> "not
    registered" (red); the client not marking a read timeout ambiguous ->
    red; serve()'s line saying "NOT registered" -> red."""
    cp.lose_answer = True

    _serve(tmp_path, rules=[RULE_A])
    assert cp.in_force == [RULE_A]
    out = capsys.readouterr().out
    assert f"[percepteye] rulebook NOT confirmed — {UNKNOWN}" in out
    assert "NOT registered" not in out and "not registered" not in out

    px = attached(rules=[RULE_B])
    assert px.wait_for_rulebook(5)
    assert cp.in_force == [RULE_B]
    assert px.rulebook is None
    assert px.rulebook_refusal == UNKNOWN


def _attempts(*answers):
    """A client whose attempts meet ``answers`` in order: an httpx exception
    class (raised as that failure), or ``(status, body)``."""
    queue = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        answer = queue.pop(0)
        if isinstance(answer, type):
            raise answer("failed", request=request)
        status, body = answer
        return httpx.Response(status, content=body)

    return _http.ControlPlaneClient(
        BASE, "pefw_test", max_attempts=len(answers),
        client=httpx.Client(transport=httpx.MockTransport(handler)))


@pytest.mark.parametrize(("answers", "ambiguous"), [
    ((httpx.ConnectError,), False),
    ((httpx.ConnectTimeout, httpx.ConnectError), False),
    ((httpx.ReadTimeout,), True),
    ((httpx.RemoteProtocolError,), True),
    # OVER EVERY ATTEMPT: the last was never sent, the first may have landed.
    ((httpx.ReadTimeout, httpx.ConnectError), True),
    (((503, b'{"detail":"store unavailable"}'),) * 2, False),
    (((429, b"{}"), (409, b'{"detail":"no"}')), False),
    (((502, b"bad gateway"),), True),
    (((500, b"{}"),), True),
    (((504, b"{}"), (409, b'{"detail":"no"}')), True),
    (((200, b"not json"),), True),
    # The lease refusal carries it too: the first attempt may have landed.
    ((httpx.ReadTimeout, (410, b'{"detail":"lease gone"}')), True),
    (((410, b'{"detail":"lease gone"}'),), False),
])
def test_the_client_says_whether_any_attempt_may_have_been_performed(
        monkeypatch, answers, ambiguous):
    """The client is the one place that saw every attempt, so it is the one
    that says. Mutations: judge the last attempt only -> the timeout-then-
    refused case (red); count a 503 as performed -> red; an unparseable 2xx
    as unperformed -> red; a read timeout as never sent -> red."""
    monkeypatch.setattr(_http, "_backoff", lambda attempt: 0.0)
    with pytest.raises(_http.TransportError) as caught:
        _attempts(*answers).request("POST", "/agents/a/rulebook",
                                    json_body={"rules": []})
    assert caught.value.ambiguous is ambiguous

    receipt, refusal = AttachTransport(
        _attempts(*answers), AGENT).register_rulebook([RULE_A])
    assert receipt is None
    assert refusal.startswith(
        "whether this agent's rules were registered is unknown: "
        if ambiguous else "this agent's rules were not registered: ")


def test_attach_names_why_the_rules_were_never_sent(cp, attached, monkeypatch):
    """Two ways to never ask, two sentences -- not one for both, and not
    None, which would read as 'registered'."""
    px = attached(rules=[RULE_A], register=False)
    assert "register=False" in px.rulebook_refusal
    assert px.rulebook is None

    monkeypatch.delenv("PERCEPTEYE_API_KEY", raising=False)
    px = attached(rules=[RULE_A], api_key=None)
    assert "PERCEPTEYE_API_KEY" in px.rulebook_refusal
    assert cp.requests == []


def test_attach_names_a_client_that_would_not_build(attached, monkeypatch):
    """The third way to never ask: credentials configured, but no client to
    ask with. Named as that -- not as a missing key, and not None.

    Mutation: drop the pre-set sentence for the configured case -> None (red)."""
    def _broken(url, key, **kw):
        raise ValueError("control_plane_url has no scheme")

    monkeypatch.setattr(_http, "ControlPlaneClient", _broken)
    px = attached(rules=[RULE_A])
    assert px.rulebook is None
    assert px.rulebook_refusal == (
        "the control plane client could not be built, so the rules passed as "
        "rules= were never sent to it.")
    assert px.policy == {}


def test_a_receipt_outside_the_contract_is_named_not_guessed():
    """A 2xx means the control plane answered, so 'not registered' would be
    a guess. The receipt that cannot be read is named as that."""
    class _Client:
        def request(self, method, path, **kw):
            return {"status": "registered"}

    receipt, refusal = AttachTransport(_Client(), AGENT).register_rulebook(
        [RULE_A])
    assert receipt is None
    assert "cannot read" in refusal and "unknown" in refusal
    assert "not registered" not in refusal


def test_nothing_a_client_raises_escapes_the_upload():
    """Asserted against the REAL method, so its own ``except`` is what is
    exercised -- not a caller's."""
    class _Broken:
        def request(self, method, path, **kw):
            raise RuntimeError("socket closed")

    upload = AttachTransport(_Broken(), AGENT).register_rulebook([RULE_A])
    assert upload == RulebookUpload(None, (
        "this agent's rules were not registered: socket closed. The rulebook "
        "in force, if any, is unchanged."))


# ── a value that cannot be rules refuses before anything connects ─────────


@pytest.mark.parametrize(("rules", "says"), [
    (RULE_A, "a single string is one rule"),
    ({RULE_A, RULE_B}, "got set"),
    ([RULE_A, 7], "rule 1 must be a string; got number"),
    ([RULE_A, None], "rule 1 must be a string; got null"),
    ({"rules": [RULE_A]}, "one rule per string; got object"),
    ([RULE_A, [RULE_B]], "rule 1 must be a string; got array"),
])
def test_rules_that_are_not_a_list_of_strings_refuse_at_startup(
        cp, attached, tmp_path, rules, says):
    """Mutation: accept any iterable -> a set is sent in whatever order this
    process iterates it (red)."""
    with pytest.raises(ConfigurationError, match=re.escape(says)):
        _serve(tmp_path, rules=rules)
    with pytest.raises(ConfigurationError, match=re.escape(says)):
        attached(rules=rules)
    assert cp.requests == []


def test_the_transport_refuses_a_shape_before_sending():
    sent = []

    class _Client:
        def request(self, *a, **kw):
            sent.append(a)

    with pytest.raises(ContractViolation):
        AttachTransport(_Client(), AGENT).register_rulebook(RULE_A)
    assert sent == []


# ── the key ───────────────────────────────────────────────────────────────


def test_the_key_names_the_agent_and_the_rules():
    keys = []

    class _Client:
        def request(self, method, path, **kw):
            keys.append((path, kw["idempotency_key"]))
            return ControlPlane().receipt(kw["json_body"]["rules"])

    for agent, rules in (("desk (eu)-ü", [RULE_A]),
                         ("desk (eu)-ü", [RULE_A]),
                         ("desk (eu)-ü", [RULE_B]),
                         ("other", [RULE_A])):
        AttachTransport(_Client(), agent).register_rulebook(rules)
    (p1, k1), (_, k2), (_, k3), (_, k4) = keys
    assert p1 == "/agents/desk%20%28eu%29-%C3%BC/rulebook"
    assert k1.startswith("rulebook-desk%20%28eu%29-%C3%BC-")
    assert k1 == k2 and k1 != k3
    assert k1.rsplit("-", 1)[1] == k4.rsplit("-", 1)[1]


# ── the published contract and the hand-written types agree ───────────────


def test_the_request_type_and_the_schema_declare_the_same_shape():
    declared = SCHEMA["$defs"]["RulebookRequest"]
    assert declared["additionalProperties"] is False
    assert {f.name for f in fields(RulebookRequest)} == set(
        declared["properties"]) == set(declared["required"])
    body = RulebookRequest([RULE_A, "¿color? \U0001f4e6"]).to_wire()
    _validator("RulebookRequest").validate(body)


def test_the_receipt_type_and_the_schema_declare_the_same_keys():
    """Both directions, as for ``RolloutRequest``: a field only the schema
    declares is dropped by the client, one only the class declares is one a
    conforming server can never send."""
    declared = SCHEMA["$defs"]["RulebookReceipt"]
    assert declared["additionalProperties"] is False
    names = [f.name for f in fields(RulebookReceipt)]
    assert set(names) == set(declared["properties"]) == set(declared["required"])
    assert list(RulebookStatus.__args__) == (
        declared["properties"]["status"]["enum"])


def test_every_receipt_the_fake_answers_is_one_the_schema_admits():
    """The fake above stands in for the route, so its answers must be ones a
    conforming server may send -- else these tests prove the SDK against a
    shape nothing produces."""
    plane = ControlPlane()
    validator = _validator("RulebookReceipt")
    for rules in ([RULE_A], [RULE_A], [RULE_A, RULE_B], [], []):
        answer = plane.receipt(rules)
        validator.validate(answer)
        assert RulebookReceipt.from_wire(answer).status == answer["status"]


@pytest.mark.parametrize("bad", [
    {"status": "accepted"},
    {"rule_count": -1},
    {"rule_count": True},
    {"rule_count": 1.5},
    {"version": 0},
    {"version": 2.5},
    {"version": True},
    {"version_id": 5},
    {"content_sha256": "ABC"},
    {"parent_sha256": "ABC"},
    {"product_id": ""},
])
def test_a_receipt_the_schema_refuses_is_refused_by_the_type(bad):
    answer = {**ControlPlane().receipt([RULE_A]), **bad}
    assert list(_validator("RulebookReceipt").iter_errors(answer))
    with pytest.raises(ContractViolation):
        RulebookReceipt.from_wire(answer)


def test_an_integral_number_is_the_schemas_integer_and_nodes():
    """``2.0`` is JSON Schema's integer and JavaScript's ``2``: the Node SDK
    reads this answer as a receipt, so this one does too -- and keeps it as
    ``2``, or the line serve() prints would say "version 2.0".

    Mutations: ``type(x) is int`` again -> "cannot read" (red); keep the
    float -> the type and the printed line (red)."""
    answer = {**ControlPlane().receipt([RULE_A]), "version": 2.0,
              "rule_count": 1.0}
    assert not list(_validator("RulebookReceipt").iter_errors(answer))
    receipt = RulebookReceipt.from_wire(answer)
    assert (receipt.version, receipt.rule_count) == (2, 1)
    assert type(receipt.version) is int and type(receipt.rule_count) is int
    assert "as version 2 — 1 rule(s)" in fw._rulebook_line(
        RulebookUpload(receipt, None))


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator({
        "$schema": SCHEMA["$schema"], "$defs": SCHEMA["$defs"],
        "$ref": f"#/$defs/{name}",
    })


# ── a structured refusal reads the same in both SDKs ──────────────────────


def test_a_structured_detail_is_rendered_as_json_not_a_repr():
    """The Node client renders an object detail with ``JSON.stringify``;
    this is the same spelling. A Python repr (single quotes, ``None``) would
    tell a customer something different about the same answer."""
    resp = httpx.Response(422, json={"detail": {
        "error": "rulebook_invalid", "cause": "rule_too_short",
        "index": None, "detail": "règle"}})
    assert _http._detail(resp) == (
        '{"error":"rulebook_invalid","cause":"rule_too_short",'
        '"index":null,"detail":"règle"}')
    assert _http._detail(httpx.Response(403, json={"detail": "no"})) == "no"


@pytest.mark.parametrize("text", [
    '{"code":"gateway_busy","hint":null}',
    '{"code": "spaced", "kept": "as sent"}',
    '["x",1]',
    "null",
    '"just text"',
    "7",
])
def test_an_answer_with_no_detail_key_is_its_own_text(text):
    """What the Node client keeps (``cut(String(text))``): the answer as it
    was sent -- never a Python repr of what it parsed to (``{'code': ...,
    'hint': None}``, ``None``, ``just text`` unquoted).

    Mutation: ``return str(body)[:300]`` again -> red."""
    resp = httpx.Response(409, content=text.encode(),
                          headers={"content-type": "application/json"})
    assert _http._detail(resp) == text
