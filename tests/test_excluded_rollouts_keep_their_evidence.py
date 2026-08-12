"""A rollout that produced work product is REPORTED, even when excluded.

Measured on run 516b7a55 (2026-08-12): prime-agent wrote a CORRECT
``top_ips.sh``, hit ``maxTurns (12/12)``, exited 1 — and the control-plane row
came back with ``artifacts={}`` while the correct file sat in the local ledger.
Artifacts were collected only on the clean-exit path, and any exclusion
short-circuited straight to ``abandon``.

The distinction that matters is not "did the process exit tidily" but "is there
anything to grade":

* produced files → REPORT it, carrying its exclusion block. ``to_wire`` already
  ships ``exclusion``, so the server records the deliverable AND still knows not
  to train on it. The agent's inability to declare itself finished must not
  decide the score; the hidden test does.
* produced nothing → ABANDON, exactly as before. That is what abandon is for —
  "not a low reward, do not train on this" — and retrying may genuinely help.
"""
from __future__ import annotations

import stat
import time

from percepteye_agent_flywheel.contract import RolloutRequest
from percepteye_agent_flywheel.runner import RolloutRunner


class _T:
    lease_ms = 1_800_000

    def __init__(self):
        self.reported: list = []
        self.abandoned: list = []
        self.artifacts: list[str] = []

    def claim(self, n):
        return [], 10

    def heartbeat(self, r, *, progress=None):
        return None

    def report(self, r, o):
        self.reported.append(o)
        return {}

    def abandon(self, r, *, reason, detail=""):
        self.abandoned.append(reason)

    def register_artifact(self, r, *, name, size_bytes, sha256, **kw):
        self.artifacts.append(name)
        return {}


def _script(tmp_path, body: str) -> str:
    p = tmp_path / "agent.sh"
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _req(seconds=60.0) -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_ev", turn_input="do the task",
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_ev", api_key="perk_x",
        artifacts_dir="rollouts/rl_ev", deadline_ts=time.time() + seconds,
        lease_token="lt",
    )


def _run(tmp_path, body: str, *, timeout_s=30.0, deadline=60.0):
    tr = _T()
    runner = RolloutRunner(["sh", _script(tmp_path, body)], transport=tr,
                           input_shape="text", artifacts_dir=str(tmp_path),
                           rollout_timeout_s=timeout_s)
    runner._serve_one_inner(_req(deadline))
    return tr


# ── the case that lost a correct answer ─────────────────────────────────────


def test_nonzero_exit_with_a_file_is_reported_not_abandoned(tmp_path):
    """prime-agent's shape: writes the deliverable, then exits 1 because it
    ran out of turns without being able to confirm success."""
    tr = _run(tmp_path, 'printf "#!/bin/sh\\nawk 1\\n" > top_ips.sh\nexit 1')
    assert not tr.abandoned, f"abandoned instead of reported: {tr.abandoned}"
    assert len(tr.reported) == 1
    out = tr.reported[0]
    assert "top_ips.sh" in out.artifacts, f"collected: {list(out.artifacts)}"
    assert "top_ips.sh" in tr.artifacts, "artifact was never registered"


def test_the_report_still_carries_the_exclusion(tmp_path):
    """Recording the work must NOT launder a failed run into a clean one —
    the exclusion travels so the server still declines to train on it."""
    tr = _run(tmp_path, 'echo x > out.txt\nexit 1')
    out = tr.reported[0]
    assert out.exclusion_reason == "entrypoint_error"
    assert "exclusion" in out.to_wire()
    assert out.to_wire()["exclusion"]["reason"] == "entrypoint_error"


def test_a_timeout_that_produced_files_is_also_reported(tmp_path):
    """The other budget that kills these rollouts. Same rule: files exist, so
    there is something to grade."""
    tr = _run(tmp_path, 'echo answer > done.txt\nsleep 30\n', timeout_s=1.0)
    assert not tr.abandoned
    out = tr.reported[0]
    assert out.exclusion_reason == "deadline_exceeded"
    assert "done.txt" in out.artifacts


# ── the case abandon exists for ─────────────────────────────────────────────


def test_nonzero_exit_with_nothing_produced_is_still_abandoned(tmp_path):
    """No work product means nothing to grade. Hand it back typed so it can be
    retried and never trained on."""
    tr = _run(tmp_path, 'exit 1')
    assert tr.abandoned == ["entrypoint_error"]
    assert not tr.reported


def test_a_timeout_with_nothing_produced_is_still_abandoned(tmp_path):
    tr = _run(tmp_path, 'sleep 30\n', timeout_s=1.0)
    assert tr.abandoned == ["deadline_exceeded"]
    assert not tr.reported


# ── the untouched happy path ────────────────────────────────────────────────


def test_a_clean_run_reports_with_no_exclusion(tmp_path):
    tr = _run(tmp_path, 'echo hi > f.txt\necho done')
    assert not tr.abandoned
    out = tr.reported[0]
    assert out.exclusion_reason is None
    assert "exclusion" not in out.to_wire()
    assert "f.txt" in out.artifacts
