"""An unset variable is a cause, and it used to be reported as an answer.

Every control-plane read in this package resolved a URL and a key and did the
same thing when one was missing: ``if not url or not key: return {}``, silently.
``{}`` is also the control plane's own "nothing is deployed for this agent",
which callers are documented to read as *keep your current configuration* -- so
a production deployment that simply never had ``PERCEPTEYE_CONTROL_PLANE_URL``
in its environment ran the untrained model forever, and:

* ``policy_source().current()`` reported ``last_refresh_ok=True`` having never
  opened a socket -- the one health signal offered, saying fine;
* ``stale_for_s`` stayed ``0.0`` no matter how long that lasted, because it
  returned 0.0 whenever no refresh had EVER succeeded, which is exactly the
  whole-fleet outage;
* ``prompt_decision()`` blamed the control plane for "serving no approved
  prompt", sending an operator to the dashboard to look for an approval that
  was never the problem.

Python also ships no default control-plane URL (Node bakes one in), so this is
not an edge case: it is what a customer who followed the key-only onboarding
had.
"""
from __future__ import annotations

import pytest

import agent_flywheel as fw
from agent_flywheel.policy import PolicySource, missing_credentials_cause

URL = "PERCEPTEYE_CONTROL_PLANE_URL"
KEY = "PERCEPTEYE_API_KEY"


@pytest.fixture
def unset(monkeypatch):
    monkeypatch.delenv(URL, raising=False)
    monkeypatch.delenv(KEY, raising=False)


@pytest.mark.parametrize(
    ("url", "key", "expected"),
    [
        (None, None, f"neither {URL} nor {KEY} is set"),
        (None, "pefw_x", f"{URL} is not set"),
        ("https://cp.example", None, f"{KEY} is not set"),
        ("https://cp.example", "pefw_x", None),
    ],
)
def test_the_cause_names_the_variable_that_is_missing(url, key, expected):
    """One wrong cause for every case is worse than no cause at all.

    ``agent-flywheel install-hooks`` used to close by telling the customer to set
    ``PERCEPTEYE_API_KEY`` -- unconditionally, so the customer who had just
    exported it was told to export it, while the URL that was actually blocking
    them went unnamed.
    """
    assert missing_credentials_cause(url, key) == expected


@pytest.mark.parametrize("call", ["current_policy", "current_prompt"])
def test_the_one_shot_reads_say_which_variable_is_missing(unset, caplog, call):
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        assert getattr(fw, call)(agent_id="a") == {}
    assert any(URL in r.getMessage() and KEY in r.getMessage()
               for r in caplog.records), caplog.text


def test_apply_current_policy_says_it_too(unset, caplog):
    """It reads through ``current_policy``, so the seam covers it."""
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        assert fw.apply_current_policy(agent_id="a") == {}
    assert caplog.records


def test_a_handle_that_can_never_refresh_does_not_report_success(unset, caplog):
    """``last_refresh_ok`` is the health signal; it said True having never run."""
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        snap = PolicySource(agent_id="a").current()
    assert snap.last_refresh_ok is False, (
        "a handle with no control plane configured reported a successful "
        "refresh, so an operator watching this signal sees nothing wrong")
    assert any(URL in r.getMessage() for r in caplog.records), caplog.text


def test_it_is_said_once_per_handle_not_once_per_call(unset, caplog):
    """A missing variable does not fix itself; a line per turn is noise."""
    src = PolicySource(agent_id="a")
    with caplog.at_level("WARNING", logger="agent_flywheel"):
        for _ in range(5):
            src.current()
    assert len([r for r in caplog.records if URL in r.getMessage()]) == 1


def test_staleness_grows_through_a_total_outage():
    """THE never-succeeded case, which was the quietest state there was.

    A fleet behind a firewall rule that blocks the control plane from first
    boot never lands a refresh, so ``_last_ok_at`` stays None. That returned
    0.0 forever -- and ``stale_for_s`` is the number the field's own docstring
    tells a customer to alert on.
    """
    now = [0.0]

    def _boom():
        raise RuntimeError("control plane unreachable")

    src = PolicySource(agent_id="a", ttl_s=300.0, _now=lambda: now[0],
                       _fetch=_boom)
    seen = []
    for t in (0.0, 300.0, 3600.0, 86400.0):
        now[0] = t
        snap = src.current()
        seen.append(snap.stale_for_s)
        assert snap.last_refresh_ok is False

    assert seen[0] == 0.0, "not yet overdue at t=0"
    assert seen == sorted(seen), f"staleness did not grow: {seen}"
    assert seen[-1] > 80_000.0, (
        f"after a day with no successful refresh, stale_for_s was {seen[-1]}; "
        f"the documented alert would never have fired")


def test_staleness_still_measures_from_the_last_success():
    """The ordinary case must keep its meaning: overdue since the last answer."""
    now = [0.0]
    ok = {"n": 0}

    def _fetch():
        ok["n"] += 1
        if ok["n"] > 1:
            raise RuntimeError("gone")
        return {"endpoint": {"base_url": "https://x/v1", "model": "m",
                             "api_key": "sk"}}

    src = PolicySource(agent_id="a", ttl_s=300.0, _now=lambda: now[0],
                       _fetch=_fetch)
    now[0] = 1000.0
    assert src.current().base_url == "https://x/v1"
    now[0] = 1000.0 + 300.0 + 60.0
    snap = src.current()
    assert snap.last_refresh_ok is False
    # ~60s overdue, jitter aside — measured from the SUCCESS at t=1000, not
    # from construction at t=0.
    assert 20.0 < snap.stale_for_s < 100.0, snap.stale_for_s
