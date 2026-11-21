"""``close()`` is bounded in all, not only while it waits for the lock.

It bounded the WAIT for the upload lock and nothing after it: once it held the
lock its flush ran with the client's own envelope -- four 30s attempts per
request -- for every pending batch of turns and outcomes in turn. During an
outage the ``atexit`` hook held a customer's process open for minutes.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

import httpx

import agent_flywheel as fw
from agent_flywheel import _http, production


@contextmanager
def _outage(monkeypatch, tmp_path, slow_s):
    """Registration and the policy read answer; every upload answers 503
    after ``slow_s``. Every request is recorded."""
    seen: list[str] = []

    def handler(request):
        key = f"{request.method} {request.url.path}"
        seen.append(key)
        if key == "POST /agents":
            return httpx.Response(200, json={"capture": {
                "turns": True, "model_content": True}})
        if request.method == "POST":
            time.sleep(slow_s)
            return httpx.Response(503, json={"detail": "down"})
        return httpx.Response(200, json={})

    real = _http.ControlPlaneClient
    monkeypatch.setattr(_http, "ControlPlaneClient", lambda url, key, **kw: real(
        url, key, client=httpx.Client(transport=httpx.MockTransport(handler))))
    monkeypatch.setattr(fw.Attachment, "_flush_soon", lambda self: None)
    monkeypatch.setenv("PERCEPTEYE_CONTROL_PLANE_URL", "https://cp.invalid")
    monkeypatch.setenv("PERCEPTEYE_API_KEY", "pek_x")
    px = fw.attach(agent_id="support", trajectory_root=str(tmp_path))
    yield px, seen


def test_close_spends_at_most_its_budget_during_an_outage(
        monkeypatch, tmp_path):
    monkeypatch.setattr(production, "_CLOSE_FLUSH_BUDGET_S", 1.0)
    with _outage(monkeypatch, tmp_path, slow_s=0.2) as (px, seen):
        for n in range(200):                      # four batches of turns
            with px.turn(turn_id=f"t{n}", input_text="q"):
                pass
            px.record_answer(f"t{n}", "a")
        for n in range(70):                       # two batches of outcomes
            px.report_outcome("resolved", turn_id=f"t{n}")
        started = time.monotonic()
        px.close()
        spent = time.monotonic() - started
    # One budget, plus at most one attempt's minimum that started inside it.
    assert spent < 2.5, f"close() took {spent:.1f}s against a 1s budget"
    assert seen.count("POST /outcomes") == 0, "nothing starts once it is spent"
    # Nothing was lost: every turn and outcome is still on disk.
    assert len([d for d in tmp_path.iterdir() if d.is_dir()]) == 200
    assert len([p for p in tmp_path.iterdir()
                if p.name.startswith(".outcome.")]) == 70


def test_a_deadline_is_only_close_s(monkeypatch, tmp_path):
    """An ordinary flush is not bounded by close()'s budget."""
    with _outage(monkeypatch, tmp_path, slow_s=0.0) as (px, _seen):
        assert px._deadline is None
        px.close()
        assert px._deadline is None
