"""Artifacts carry their bytes, not just proofs about their bytes.

For a coding or REPL agent the FILES are the deliverable. Until now the
report carried name→digest only, so a grader could verify integrity of work
it could never read — and the per-rollout working directory wasn't even the
child's cwd, so a CLI agent's files landed wherever the serving process was
started from and collection found nothing at all.

Two changes, both here under test: the command child's cwd IS the collected
directory, and small UTF-8 artifacts inline their content into the report,
bounded per file and per report. Binary and over-cap files keep the plain
digest-string value — same wire shape as before, named, never dropped.
"""
from __future__ import annotations

import hashlib
import stat
import time

from percepteye_agent_flywheel.contract import RolloutRequest
from percepteye_agent_flywheel.runner import (
    MAX_INLINE_ARTIFACT_BYTES,
    MAX_INLINE_TOTAL_BYTES,
    RolloutRunner,
)


class _T:
    lease_ms = 1_800_000

    def __init__(self):
        self.artifacts: list[str] = []

    def claim(self, n):
        return [], 10

    def heartbeat(self, r, *, progress=None):
        return None

    def report(self, r, o):
        return {}

    def abandon(self, r, *, reason, detail=""):
        pass

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
        rollout_id="rl_art", turn_input="do the task",
        llm_url="https://gw.invalid/llm/v1/rollouts/rl_art", api_key="perk_x",
        artifacts_dir="rollouts/rl_art", deadline_ts=time.time() + seconds,
        lease_token="lt",
    )


def _run(tmp_path, body: str):
    tr = _T()
    runner = RolloutRunner(["sh", _script(tmp_path, body)], transport=tr,
                           input_shape="text", artifacts_dir=str(tmp_path),
                           rollout_timeout_s=30.0)
    # Through the real serve path: run_one + artifact collection + report.
    captured: dict = {}
    _orig_report = tr.report

    def _capture(r, o):
        captured["out"] = o
        return _orig_report(r, o)

    tr.report = _capture
    runner._serve_one_inner(_req())
    return tr, captured["out"]


# ── the cwd IS the collected directory ─────────────────────────────────────


def test_a_file_the_agent_writes_to_its_cwd_is_an_artifact(tmp_path):
    """THE join. The agent writes `./triage.sh`; collection scans the
    per-rollout directory; these are now the same place. Before the cwd pin
    this test fails with zero artifacts — measured, not assumed."""
    tr, out = _run(tmp_path, 'printf "#!/bin/sh\\ngrep ERROR *.log\\n" > triage.sh\necho done')
    assert "triage.sh" in out.artifacts, f"collected: {list(out.artifacts)}"
    assert "triage.sh" in tr.artifacts, "the artifact was never registered"


def test_the_ledger_file_is_never_an_artifact(tmp_path):
    tr, out = _run(tmp_path, 'echo x > kept.txt\necho done')
    assert "tool_calls.jsonl" not in out.artifacts
    assert "kept.txt" in out.artifacts


# ── inline bodies, bounded ──────────────────────────────────────────────────


def test_a_small_text_artifact_inlines_its_content(tmp_path):
    tr, out = _run(tmp_path, 'printf "hello grader" > answer.txt\necho ok')
    rec = out.artifacts["answer.txt"]
    assert isinstance(rec, dict), f"expected inline record, got {rec!r}"
    assert rec["content"] == "hello grader"
    assert rec["size_bytes"] == len("hello grader")
    assert rec["sha256"] == hashlib.sha256(b"hello grader").hexdigest(), (
        "the inline body and the registered digest must describe the same bytes"
    )


def test_a_binary_artifact_stays_a_digest_string(tmp_path):
    """UTF-8 or nothing: base64-ing binaries would quietly quadruple report
    sizes for files no text grader can read anyway."""
    tr, out = _run(tmp_path, 'printf "\\377\\376\\000\\001" > blob.bin\necho ok')
    rec = out.artifacts["blob.bin"]
    assert isinstance(rec, str) and len(rec) == 64, rec


def test_a_file_over_the_per_file_cap_stays_a_digest_string(tmp_path):
    n = MAX_INLINE_ARTIFACT_BYTES + 1
    tr, out = _run(tmp_path, f'head -c {n} /dev/zero | tr "\\0" "a" > big.txt\necho ok')
    rec = out.artifacts["big.txt"]
    assert isinstance(rec, str) and len(rec) == 64


def test_the_total_budget_bounds_a_swarm_of_small_files(tmp_path):
    """Two hundred small files must not become a megabyte report. Files past
    the budget keep the digest-string form — present, named, not inlined."""
    each = 32 * 1024
    count = (MAX_INLINE_TOTAL_BYTES // each) + 3
    body = "\n".join(
        f'head -c {each} /dev/zero | tr "\\0" "x" > f{i:02d}.txt'
        for i in range(count)
    ) + "\necho ok"
    tr, out = _run(tmp_path, body)
    inlined = [n for n, v in out.artifacts.items() if isinstance(v, dict)]
    strings = [n for n, v in out.artifacts.items() if isinstance(v, str)]
    assert len(inlined) == MAX_INLINE_TOTAL_BYTES // each
    assert len(strings) == 3, (
        f"budget did not bound inlining: {len(inlined)} inlined, "
        f"{len(strings)} digest-only of {count}"
    )
    total = sum(v["size_bytes"] for v in out.artifacts.values()
                if isinstance(v, dict))
    assert total <= MAX_INLINE_TOTAL_BYTES


# ── the wire stays legal ────────────────────────────────────────────────────


def test_every_artifact_value_shape_is_schema_legal(tmp_path):
    import json
    import pathlib

    schema = json.loads(
        (pathlib.Path(__file__).resolve().parents[1] / "schema" /
         "flywheel-1.json").read_text()
    )
    allowed_obj_keys = set(
        schema["$defs"]["RolloutOutput"]["properties"]["artifacts"]
        ["additionalProperties"]["anyOf"][1]["properties"]
    )
    tr, out = _run(
        tmp_path,
        'printf "small" > a.txt\nprintf "\\377\\376" > b.bin\necho ok',
    )
    wire = out.to_wire()["artifacts"]
    for name, v in wire.items():
        if isinstance(v, dict):
            assert set(v) == allowed_obj_keys, (
                f"{name}: inline record keys {set(v)} != schema {allowed_obj_keys}"
            )
        else:
            assert isinstance(v, str)
