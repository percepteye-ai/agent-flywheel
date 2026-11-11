"""The SDK reports a rollout's policy-call count; the agent need not.

τ³ banking, 2026-11-11: the agent returned no ``llm_call_count``, so Mission
Control could not reconcile its gateway's observed count and excluded every
rollout as ``llm_calls_unobserved``. Mutation: drop the fallback in
``run_rollout_in_process`` -- the count is None and these tests fail.
"""
from __future__ import annotations

import http.server
import threading
import time

import httpx
import pytest
import requests

from agent_flywheel.contract import RolloutOutput, RolloutRequest
from agent_flywheel.entrypoint import resolve_entrypoint
from agent_flywheel.runner import run_rollout_in_process


class _Gateway(http.server.BaseHTTPRequestHandler):
    def _reply(self, code: int) -> None:
        self.send_response(code)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"{}")

    def do_POST(self) -> None:  # noqa: N802
        n = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(n)
        self._reply(500 if self.path.endswith("/fail") else 200)

    def do_GET(self) -> None:  # noqa: N802
        self._reply(200)

    def log_message(self, *a: object) -> None:
        pass


@pytest.fixture()
def gateway():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Gateway)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _run(agent, llm_url: str, tmp_path) -> RolloutOutput:
    req = RolloutRequest(
        rollout_id="r1", turn_input="go", llm_url=llm_url, api_key="perk_x",
        artifacts_dir=str(tmp_path), deadline_ts=time.time() + 60,
    )
    return run_rollout_in_process(
        resolve_entrypoint(agent, input_shape="text"), req,
        trajectory_dir=str(tmp_path), timeout_s=30.0)


def test_policy_posts_are_counted_and_nothing_else_is(gateway, tmp_path):
    llm = f"{gateway}/llm/v1/rollouts/r1"

    def agent(_text: str) -> str:
        with httpx.Client() as c:
            c.post(f"{llm}/chat/completions", json={})            # policy
            c.get(f"{llm}/models")                                 # not a call
            c.post(f"{llm}/hosted/openrouter/chat/completions", json={})  # aux
            c.post(f"{llm}/fail", json={})                         # 500
            c.post(f"{gateway}/elsewhere/chat/completions", json={})  # other
        requests.post(f"{llm}/chat/completions", json={})          # policy

        async def _a() -> None:
            async with httpx.AsyncClient() as c:
                await c.post(f"{llm}/chat/completions", json={})   # policy
        import asyncio
        asyncio.run(_a())
        return "done"

    assert _run(agent, llm, tmp_path).llm_call_count == 3


def test_the_agents_own_count_wins(gateway, tmp_path):
    llm = f"{gateway}/llm/v1/rollouts/r1"

    def agent(_text: str) -> RolloutOutput:
        httpx.post(f"{llm}/chat/completions", json={})
        return RolloutOutput(final_text="x", llm_call_count=7)

    assert _run(agent, llm, tmp_path).llm_call_count == 7


def test_no_counted_call_reports_none_not_zero(gateway, tmp_path):
    """A client this module cannot see must not be reported as 'no calls'."""
    out = _run(lambda _t: "x", f"{gateway}/llm/v1/rollouts/r1", tmp_path)
    assert out.llm_call_count is None


def test_counting_is_uninstalled_after_the_rollout(gateway, tmp_path):
    before = (httpx.Client.send, httpx.AsyncClient.send, requests.Session.send)
    _run(lambda _t: "x", f"{gateway}/llm/v1/rollouts/r1", tmp_path)
    assert (httpx.Client.send, httpx.AsyncClient.send,
            requests.Session.send) == before
