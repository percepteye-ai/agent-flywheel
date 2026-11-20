"""The goose hook claims a practice task, and reports a session, in TRAINING only.

The hook claimed on ``SessionStart`` and reported on ``Stop`` whenever the
control-plane URL and key were set, and never read ``PERCEPTEYE_AGENT_MODE``. So
a goose install in production mode, with a key, leased a rollout whenever a
practice-task run was minting work for its agent id, dropped the task into the
customer's working directory, and POSTed the customer's real session -- its last
answer and every recorded tool call -- as that rollout's result, past the
per-agent production-capture consent. Every other surface claims only in
training: ``serve()`` returns before opening a connection, and the Node plugin
subscribes no rollout hook.

Driven the way goose drives it -- the shipped script, a JSON event on stdin, in a
subprocess -- against a control plane that records every request it receives, so
"no claim" means no request reached the server, not that a function was skipped.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from agent_flywheel.hosts import RECORDER

ROLLOUT_ID = "r-practice-1"
TASK = "Practice task: list the files in this directory."
ANSWER = "the answer the customer's own user was given"
SESSION = "s1"


def _recorder():
    """The shipped recorder as a module, for its own constants: read rather
    than restated, so a rename there cannot leave this test checking a name
    the hook no longer writes."""
    spec = importlib.util.spec_from_file_location("_rec_under_test", RECORDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REC = _recorder()
TASK_FILENAME = _REC.TASK_FILENAME
CLAIM_FILENAME = _REC.CLAIM_FILENAME


@pytest.fixture
def control_plane():
    """A control plane that hands out one practice task and records everything."""
    seen: list[tuple[str, str, dict]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # http.server's spelling
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = json.loads(raw or b"{}")
            seen.append(("POST", self.path, body))
            if self.path.endswith("/rollouts/claim"):
                reply = {"rollouts": [{
                    "rollout_id": ROLLOUT_ID, "turn_input": TASK,
                    "llm_url": "http://127.0.0.1:9/v1", "api_key": "",
                    "artifacts_dir": "", "deadline_ts": time.time() + 600,
                    "lease_token": "lease-1"}], "poll_after_ms": 1000}
            else:
                reply = {"status": "accepted"}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/api/flywheel/v1", seen
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def host(tmp_path):
    """An isolated goose home and the session's working directory."""
    work = tmp_path / "work"
    work.mkdir()
    return tmp_path, work


def _fire(host, url, mode, event, *, key="pefw_test"):
    root, work = host
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PERCEPTEYE_")
           and k.upper() not in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")}
    env.update({
        # GOOSE_PATH_ROOT keeps SessionStart's reconcile off the developer's
        # own goose config; nothing is installed under it, so it is a no-op.
        "GOOSE_PATH_ROOT": str(root),
        "PERCEPTEYE_HOME": str(root / ".percepteye"),
        "PERCEPTEYE_CONTROL_PLANE_URL": url,
        "PERCEPTEYE_AGENT_MODE": mode,
        "NO_PROXY": "127.0.0.1,localhost",
    })
    if key is not None:
        env["PERCEPTEYE_API_KEY"] = key
    proc = subprocess.run(
        [sys.executable, str(RECORDER)],
        input=json.dumps({"session_id": SESSION, "working_dir": str(work),
                          **event}),
        capture_output=True, text=True, env=env, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "", "stdout on a blocking hook chain can deny the turn"
    return proc


def _session_dir(host):
    return host[0] / ".percepteye" / "turns" / SESSION


def _rollout_requests(seen):
    return [(m, p) for m, p, _ in seen if "/rollouts/" in p]


@pytest.mark.parametrize("mode", ["production", "serving"])
def test_a_production_session_start_claims_nothing(host, control_plane, mode):
    """``serving`` as well as ``production``: the hook asks the SDK's own
    ``resolve_mode``, so every spelling of production is production."""
    url, seen = control_plane
    _fire(host, url, mode, {"event": "SessionStart"})
    assert _rollout_requests(seen) == [], seen
    assert not (host[1] / TASK_FILENAME).exists(), (
        "a practice task was dropped into a production session's working dir")
    assert not (_session_dir(host) / CLAIM_FILENAME).exists()


def test_a_production_stop_reports_nothing_even_with_a_claim_on_disk(
        host, control_plane):
    """The REPORT is gated on its own, not only through the claim.

    A claim record left by an earlier training session in the same directory
    -- one that crashed before its Stop -- must not turn a production session's
    answer into that rollout's result. Without the planted claim this case
    would pass on an ungated report too, because there would be nothing to
    report against.
    """
    url, seen = control_plane
    d = _session_dir(host)
    d.mkdir(parents=True)
    (d / CLAIM_FILENAME).write_text(json.dumps({
        "rollout_id": ROLLOUT_ID, "task": TASK, "lease_token": "lease-1",
        "deadline_ts": time.time() + 600, "llm_url": ""}), encoding="utf-8")
    _fire(host, url, "production", {"event": "PostToolUse",
                                    "tool_name": "shell",
                                    "tool_input": {"command": "ls"}})
    _fire(host, url, "production",
          {"event": "Stop", "last_assistant_message": ANSWER})
    assert _rollout_requests(seen) == [], seen
    # Local capture is unaffected by mode: the answer is still recorded.
    turns = (d / "turns.jsonl").read_text(encoding="utf-8")
    assert ANSWER in turns


def test_an_unreadable_mode_claims_nothing(host, control_plane):
    """A mode the SDK cannot parse is not a mode, and guessing training here
    leases work on the customer's key that they may have meant to stop."""
    url, seen = control_plane
    _fire(host, url, "prodction", {"event": "SessionStart"})
    assert _rollout_requests(seen) == [], seen
    assert not (host[1] / TASK_FILENAME).exists()


def test_training_still_claims_and_reports(host, control_plane):
    """The positive control: without it the cases above would also pass on a
    hook that could not reach the control plane at all."""
    url, seen = control_plane
    _fire(host, url, "training", {"event": "SessionStart"})
    assert _rollout_requests(seen) == [("POST", "/api/flywheel/v1/rollouts/claim")]
    claim_body = seen[-1][2]
    assert claim_body["agent_id"] == "goose" and claim_body["max"] == 1
    assert (host[1] / TASK_FILENAME).read_text(encoding="utf-8") == TASK
    assert (_session_dir(host) / CLAIM_FILENAME).is_file()

    _fire(host, url, "training", {"event": "PostToolUse", "tool_name": "shell",
                                  "tool_input": {"command": "ls"}})
    _fire(host, url, "training",
          {"event": "Stop", "last_assistant_message": ANSWER})
    reports = [(p, b) for m, p, b in seen if p.endswith("/result")]
    assert [p for p, _ in reports] == [
        f"/api/flywheel/v1/rollouts/{ROLLOUT_ID}/result"]
    body = reports[0][1]
    assert body["final_text"] == ANSWER
    assert [c["name"] for c in body["tool_calls"]] == ["shell"]
    assert not (_session_dir(host) / CLAIM_FILENAME).exists(), (
        "a reported claim must be consumed")


# ── PY-H40: the lease token on disk, a key stored by login, a long turn ─────


def test_the_claim_record_is_readable_only_by_its_owner(host, control_plane):
    """``claim.json`` carries the lease token, a bearer credential for the
    rollout: 0600 from the moment it exists, in a 0700 directory, under a
    0700 ``$PERCEPTEYE_HOME`` (which also holds ``credentials.json``)."""
    import stat

    url, _ = control_plane
    _fire(host, url, "training", {"event": "SessionStart"})
    claim = _session_dir(host) / CLAIM_FILENAME
    assert json.loads(claim.read_text(encoding="utf-8"))["lease_token"] == "lease-1"
    for path, want in ((claim, 0o600), (_session_dir(host), 0o700),
                       (host[0] / ".percepteye", 0o700)):
        mode = stat.S_IMODE(os.stat(path).st_mode)
        assert mode == want, f"{path}: {oct(mode)}"


def test_a_key_stored_by_login_is_the_hooks_key_too(host, control_plane):
    """``agent-flywheel login --agent-id goose`` stores a key; the hook used to
    read only ``PERCEPTEYE_API_KEY`` and so claimed nothing with it."""
    from agent_flywheel import credentials as creds

    url, seen = control_plane
    creds.save_stored(creds.StoredCredential(
        control_plane_url=url, agent_id="goose", api_key="pefw_" + "S" * 43,
    ), {"PERCEPTEYE_HOME": str(host[0] / ".percepteye")})
    _fire(host, url, "training", {"event": "SessionStart"}, key=None)
    assert _rollout_requests(seen) == [("POST", "/api/flywheel/v1/rollouts/claim")]


def test_a_long_non_ascii_turn_is_shortened_not_dropped(host, control_plane):
    """A turn record has no ``arguments``; ``_fit`` subscripted them anyway, so
    any turn whose JSON-escaped text passed the one-write cap (non-ASCII text
    over ~1.2k characters) raised KeyError and vanished."""
    url, _ = control_plane
    task = "задача " * 500
    _fire(host, url, "production", {"event": "UserPromptSubmit", "message": task})
    lines = (_session_dir(host) / "turns.jsonl").read_bytes().splitlines()
    assert len(lines) == 1
    assert len(lines[0]) <= _REC._MAX_RECORD_BYTES
    record = json.loads(lines[0])
    assert record["kind"] == "task"
    assert record["text"].startswith(task.strip()[:200])
    assert "truncated" in record["text"]
