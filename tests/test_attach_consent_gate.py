"""The consent gate, driven through the REAL ``attach()``.

``tests/test_capture_verdict_parity.py`` proves the predicate. This proves the
call site asks it -- which is the half that was actually wrong: the predicate's
strict form already existed in this same file, three hundred lines up in
``_recheck_capture``, while ``attach()`` asked ``bool(cap.get("turns"))``. A test
of the predicate alone would have passed with that untouched.

Every fixture here is a registration response the control plane could really
send, and the assertion is on the attachment a customer holds: ``may_upload``,
``upload_refusal``, and whether the shutdown drain was armed -- because arming
it IS the decision to send an end user's message and their agent's answer.
"""
from __future__ import annotations

from contextlib import contextmanager

import pytest

import agent_flywheel as fw


class _Transport:
    def __init__(self, registration):
        self._registration = registration

    def register(self, spec):
        return self._registration

    def policy_current(self):
        return {}


@contextmanager
def _attached(monkeypatch, tmp_path, registration, **env):
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pefw_x")
    for k, v in env.items():
        monkeypatch.delenv(k, raising=False) if v is None else monkeypatch.setenv(k, v)
    monkeypatch.setattr(
        "agent_flywheel.transport.AttachTransport",
        lambda client, agent_id: _Transport(registration))
    monkeypatch.setattr(
        "agent_flywheel._http.ControlPlaneClient.close",
        lambda self: None)
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    try:
        yield px
    finally:
        px.close()


#: Truthy in Python, and NOT consent. `{"turns": "false"}` is the one that
#: sends: a JSON string from an older intake or a proxy. Under the old gate
#: `attach()` set may_upload=True, granted the upload, registered the atexit
#: hook and began sending production turns every 32 turns.
NOT_CONSENT = [
    {"capture": {"turns": "false", "reason": "an operator has not enabled it"}},
    {"capture": {"turns": "true"}},
    {"capture": {"turns": "no"}},
    {"capture": {"turns": 1}},
    {"capture": {"turns": ["yes"]}},
    {"capture": {"turns": False}},
    {"capture": {}},
    {},
]


@pytest.mark.parametrize("registration", NOT_CONSENT)
def test_attach_uploads_nothing_without_an_exact_yes(
        monkeypatch, tmp_path, registration):
    with _attached(monkeypatch, tmp_path, registration) as px:
        assert px.may_upload is False, (
            f"{registration} was read as consent to upload end users' messages")
        assert not px._exit_flush_registered, (
            "the shutdown drain was armed without consent, so the turns would "
            "be sent at process exit")


def test_attach_uploads_on_an_exact_yes(monkeypatch, tmp_path):
    """The guard must not refuse the one payload that IS consent."""
    with _attached(monkeypatch, tmp_path, {"capture": {"turns": True}}) as px:
        assert px.may_upload is True
        assert px.upload_refusal is None
        assert px._exit_flush_registered


@pytest.mark.parametrize("registration", NOT_CONSENT)
def test_every_refusal_names_a_cause(monkeypatch, tmp_path, registration):
    """The README calls ``upload_refusal`` "the sentence explaining a False".

    It was ``None`` whenever the response carried no ``capture`` block and
    whenever the server sent no ``reason`` -- the two commonest refusals there
    are -- so an operator was given no cause and no way to tell "not enabled
    yet" from "the control plane never answered".
    """
    with _attached(monkeypatch, tmp_path, registration) as px:
        assert px.upload_refusal and px.upload_refusal.strip(), registration


def test_the_servers_own_sentence_is_preferred(monkeypatch, tmp_path):
    """It names the switch and who can flip it; ours could only paraphrase."""
    reg = {"capture": {"turns": False, "reason": "capture is off for this org"}}
    with _attached(monkeypatch, tmp_path, reg) as px:
        assert px.upload_refusal == "capture is off for this org"


def test_an_unconfigured_control_plane_names_the_variable(monkeypatch, tmp_path):
    """The other way to get ``may_upload=False``, and it had no sentence either.

    Nothing is asked of the control plane at all here, so "capture is not
    enabled for this agent" would be a refusal naming the wrong subsystem.
    """
    monkeypatch.delenv("PERCEPTEYE_CONTROL_PLANE_URL", raising=False)
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pefw_x")
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    try:
        assert px.may_upload is False
        assert "PERCEPTEYE_CONTROL_PLANE_URL" in (px.upload_refusal or "")
    finally:
        px.close()


def test_a_blip_does_not_overwrite_a_specific_refusal(monkeypatch, tmp_path):
    """Nothing is LOST on the degraded re-poll path.

    ``_recheck_capture`` runs on the flush cadence and a control plane that
    cannot answer returns ``{}``. That must not revoke consent -- it does not --
    and it must not replace the sentence the server DID give at startup with
    the generic "did not answer" one, which is the sentence an operator can act
    on least.
    """
    reg = {"capture": {"turns": False,
                       "reason": "this key does not carry the turn-report scope"}}
    with _attached(monkeypatch, tmp_path, reg) as px:
        assert px.upload_refusal == reg["capture"]["reason"]

        class _Blip:
            def capture_current(self, agent_id):
                return {}

        monkeypatch.setattr(
            "agent_flywheel.transport.AttachTransport",
            lambda client, agent_id: _Blip())
        px._recheck_capture()
        assert px.may_upload is False
        assert px.upload_refusal == reg["capture"]["reason"], (
            "an empty poll replaced the server's own explanation with the "
            "generic one")


def test_late_consent_asks_the_same_predicate(monkeypatch, tmp_path):
    """``_recheck_capture`` is the second consumer of the same question.

    It only ever turns capture ON, so the assertion is that a truthy-but-not-
    true answer does not turn it on -- and that a real yes does.
    """
    monkeypatch.setattr(
        "agent_flywheel._http.ControlPlaneClient.close",
        lambda self: None)
    class _Cap:
        def __init__(self, payload):
            self.payload = payload

        def capture_current(self, agent_id):
            return {"capture": self.payload}

    for payload, expected in (({"turns": "true"}, False), ({"turns": True}, True)):
        monkeypatch.setattr(
            "agent_flywheel.transport.AttachTransport",
            lambda client, agent_id, _p=payload: _Cap(_p))
        px = fw.attach(agent_id="support", trajectory_root=str(tmp_path),
                       register=False, control_plane_url="https://cp.invalid",
                       api_key="pefw_x")
        try:
            px._recheck_capture()
            assert px.may_upload is expected, payload
        finally:
            px.close()
