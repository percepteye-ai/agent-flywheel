"""The two commands a CLI-agent customer runs say what is actually wrong.

``install-hooks`` used to close, unconditionally, with "set PERCEPTEYE_API_KEY
to contribute rollouts" -- so the customer who had just followed step 3 of the
README and exported that key was told to export that key, while
``PERCEPTEYE_CONTROL_PLANE_URL``, which was the thing actually blocking them and
which this package has no default for, went unnamed on both subcommands. The
result is the failure mode this package front-loads its checks to prevent: the
plugin loads, tool outcomes land on disk, no rollout is ever claimed, and
nothing anywhere says why.

``agent-flywheel report`` is here for a different reason: the reporter shipped in
every wheel with NO consumer at all -- one line of code named it, the constant
binding it -- so the tool-failure rate the README leads with was reachable only
by finding the script inside site-packages and running it by hand.
"""
from __future__ import annotations

import os
import time

import pytest

from agent_flywheel.cli import main

URL = "PERCEPTEYE_CONTROL_PLANE_URL"
KEY = "PERCEPTEYE_API_KEY"


@pytest.fixture
def goose_home(monkeypatch, tmp_path):
    monkeypatch.setenv("GOOSE_PATH_ROOT", str(tmp_path))
    monkeypatch.setenv("PERCEPTEYE_HOME", str(tmp_path / ".percepteye"))
    monkeypatch.delenv("PERCEPTEYE_AGENT_MODE", raising=False)
    monkeypatch.delenv("PERCEPTEYE_LLM_BASE_URL", raising=False)
    return tmp_path


@pytest.mark.parametrize(
    ("env", "must_name"),
    [
        ({}, URL),
        ({KEY: "pefw_x"}, URL),
        ({URL: "https://cp.example"}, KEY),
    ],
)
def test_install_hooks_names_the_variable_that_is_missing(
        goose_home, monkeypatch, capsys, env, must_name):
    for k in (URL, KEY):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)

    assert main(["install-hooks"]) == 0
    out = capsys.readouterr().out
    assert "NOT claimed" in out, out
    assert must_name in out, out
    # The one that IS set must not be the one it asks for.
    for k in env:
        assert f"{k} is not set" not in out, out


def test_status_says_it_too(goose_home, monkeypatch, capsys):
    """Both subcommands, because a customer runs whichever one they remember.

    ``status`` documents itself as printing "where traffic will actually go on
    the next run" and said nothing at all about whether rollouts are claimed.
    """
    monkeypatch.delenv(URL, raising=False)
    monkeypatch.setenv(KEY, "pefw_x")
    assert main(["install-hooks"]) == 0
    capsys.readouterr()
    main(["status"])
    out = capsys.readouterr().out
    assert "NOT claimed" in out and URL in out, out


def test_a_configured_install_says_so(goose_home, monkeypatch, capsys):
    monkeypatch.setenv(URL, "https://cp.example/api/flywheel/v1")
    monkeypatch.setenv(KEY, "pefw_x")
    assert main(["install-hooks"]) == 0
    out = capsys.readouterr().out
    assert "rollouts  claimed" in out, out
    assert "NOT claimed" not in out, out


# ── the reporter now has a consumer ────────────────────────────────────────


def test_report_reads_the_most_recent_recorded_session(goose_home, capsys):
    """THE seam that had a producer and no consumer.

    The recorder writes one directory per session under ``PERCEPTEYE_HOME``;
    nothing read them. ``agent-flywheel report`` resolves that root by asking the
    recorder for it rather than rebuilding the join, so the reader cannot end
    up pointed somewhere the writer does not write.
    """
    turns = goose_home / ".percepteye" / "turns"
    for name, ok in (("old-session", "ok"), ("new-session", "failed")):
        d = turns / name
        d.mkdir(parents=True)
        (d / "tool_calls.jsonl").write_text(
            f'{{"name": "bash", "outcome": "{ok}"}}\n', encoding="utf-8")
    # Make the ordering unambiguous rather than relying on creation order.
    os.utime(turns / "old-session" / "tool_calls.jsonl",
             (time.time() - 3600, time.time() - 3600))
    os.utime(turns / "old-session", (time.time() - 3600, time.time() - 3600))

    assert main(["report"]) == 0
    out = capsys.readouterr().out
    assert "new-session" in out, out
    assert "old-session" not in out, out
    assert "Real failure rate: 100%" in out, out


def test_report_on_an_explicit_directory(goose_home, capsys, tmp_path):
    """The header names the file it read, in whichever form the reporter picks.

    The reporter shows the SHORTER of the absolute and the cwd-relative path,
    so which form appears depends on where pytest's cwd sits relative to
    ``tmp_path`` -- a TMPDIR beside the checkout renders ``../tmp/...``. The
    claim is that the header names THIS file, so that is what is checked: the
    shown path, read from the cwd the way a person reading it would, is it.
    """
    d = tmp_path / "traj"
    d.mkdir()
    (d / "tool_calls.jsonl").write_text(
        '{"name": "read", "outcome": "ok"}\n', encoding="utf-8")
    assert main(["report", str(d)]) == 0
    out = capsys.readouterr().out
    header = next(line for line in out.splitlines() if line.strip())
    shown = header.split("·", 1)[1].strip()
    assert os.path.samefile(shown, d / "tool_calls.jsonl"), out
    assert "Real failure rate: 0%" in out, out


def test_report_with_nothing_recorded_says_what_to_do(goose_home, capsys):
    """And it does NOT repeat the old claim that capture needs a directory set.

    That stopped being true when the recorder gained a default location, which
    is the whole no-account promise.
    """
    assert main(["report"]) == 1
    out = capsys.readouterr().out
    assert "No trajectory at" in out, out
    assert "does nothing at all until this is set" not in out, out
