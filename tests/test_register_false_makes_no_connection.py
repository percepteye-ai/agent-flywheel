"""``attach(register=False)`` makes no outbound connection, ever.

It used to skip only the registration: the attachment still held the resolved
control-plane URL and key, so the consent re-read on the flush cadence, a
manual ``flush()``, ``close()``, ``report_outcome`` and ``report_turn_feedback``
all contacted the control plane this caller had opted out of -- and, on a yes,
uploaded the end users' turns.
"""
from __future__ import annotations

import httpx

import agent_flywheel as fw
from agent_flywheel import _http
from agent_flywheel.production import _FLUSH_EVERY


def test_register_false_sends_nothing_for_the_life_of_the_attachment(
        monkeypatch, tmp_path):
    requests: list[str] = []
    built: list[str] = []

    def handler(request):
        requests.append(f"{request.method} {request.url.path}")
        # A yes to every question, so anything that asks would go on to send.
        return httpx.Response(200, json={"capture": {
            "turns": True, "model_content": True}})

    real = _http.ControlPlaneClient

    def client(url, key, **kw):
        built.append(url)
        return real(url, key, client=httpx.Client(
            transport=httpx.MockTransport(handler)))

    monkeypatch.setattr(_http, "ControlPlaneClient", client)
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pefw_" + "k" * 43)

    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path),
                   register=False)
    for n in range(_FLUSH_EVERY + 1):
        with px.turn(turn_id=f"t{n}", input_text="hello",
                     conversation_id="c1") as tid:
            pass
        px.record_answer(tid, "hi")
    assert px.report_outcome("resolved", turn_id="t0") is None
    assert px.report_turn_feedback(
        "accepted", episode_id="rl_e1", turn_index=0) is None
    counts = px.flush(include_unanswered=True)
    px.close()

    assert requests == [] and built == [], (requests, built)
    assert counts.get("sent", 0) == 0
    assert px.may_upload is False
    assert "register=False" in (px.upload_refusal or "")
