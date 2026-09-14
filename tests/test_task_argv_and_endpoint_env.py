"""Two general features for CLI agents, both DECLARED rather than sniffed.

``fw.TASK`` — some CLIs take the task as a positional argument. dsh headless is
one: it reads neither stdin nor an env var for it, so the two channels this SDK
already had reached it not at all. A sentinel marks the position.

``endpoint_env=`` — some agents read their endpoint from names that are not the
OpenAI/Anthropic ones. DeepSeek Harness reads DEEPSEEK_BASE_URL and grep finds
zero hits for OPENAI_BASE_URL in its whole source, so an unmodified integration
would look correct while every completion went to the live provider unobserved.

Both are stated by the caller because neither is knowable from here, and the
failure mode of guessing wrong is silent in both cases.
"""
from __future__ import annotations

import time

from agent_flywheel import TASK
from agent_flywheel.contract import RolloutRequest
from agent_flywheel.isolation import build_child_env
from agent_flywheel.runner import _argv_with_task

ARGV = ["dsh", "--profile", "headless", TASK]


def _req(turn_input="count to three") -> RolloutRequest:
    return RolloutRequest(
        rollout_id="rl_t", turn_input=turn_input,
        llm_url="https://gw.invalid/v1", api_key="k",
        artifacts_dir="rollouts/rl_t", deadline_ts=time.time() + 300,
    )


# ── fw.TASK ─────────────────────────────────────────────────────────────────

def test_the_task_lands_where_the_sentinel_was():
    argv, used = _argv_with_task(list(ARGV), "count to three")
    assert argv == ["dsh", "--profile", "headless", "count to three"]
    assert used is True


def test_an_argv_without_the_sentinel_is_untouched():
    """Backwards compatibility is the whole point: every existing command
    entrypoint keeps stdin and PERCEPTEYE_TURN_INPUT and changes nothing."""
    argv, used = _argv_with_task(["opencode", "run"], "count to three")
    assert argv == ["opencode", "run"]
    assert used is False


def test_a_literal_that_looks_like_a_placeholder_is_left_alone():
    """The reason this is a sentinel and not the string "{task}". A real
    argument containing the placeholder text must survive verbatim."""
    argv, used = _argv_with_task(["agent", "--tmpl", "{task}", TASK], "hi")
    assert argv == ["agent", "--tmpl", "{task}", "hi"]
    assert used is True


def test_the_sentinel_may_appear_more_than_once():
    argv, _ = _argv_with_task([TASK, "--echo", TASK], "x")
    assert argv == ["x", "--echo", "x"]


def test_every_other_argument_is_stringified():
    argv, _ = _argv_with_task(["agent", 42, TASK], "go")
    assert argv == ["agent", "42", "go"]


def test_the_sentinel_renders_as_a_label_not_a_crash():
    """serve() builds its registration label by joining the argv. A non-string
    member raised TypeError there — the same class of bug the code comments
    record shipping once, when reading .name off a command entrypoint failed
    every non-Python agent at registration."""
    assert " ".join(str(a) for a in ARGV) == "dsh --profile headless <task>"


def test_a_task_containing_shell_metacharacters_is_not_interpreted():
    """argv is passed to Popen as a list, never through a shell, so quoting
    and substitution are not hazards here."""
    nasty = 'oops"; rm -rf / #'
    argv, _ = _argv_with_task([*ARGV], nasty)
    assert argv[-1] == nasty


# ── endpoint_env= ───────────────────────────────────────────────────────────

def test_named_variables_carry_the_endpoint():
    env = build_child_env(
        rollout_id="rl_t", llm_url="https://gw/v1", api_key="sk-1",
        trajectory_dir="/tmp/t", parent_env={"PATH": "/usr/bin"},
        endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"),
    )
    assert env["DEEPSEEK_BASE_URL"] == "https://gw/v1"
    assert env["DEEPSEEK_API_KEY"] == "sk-1"


def test_the_defaults_are_kept_alongside():
    """Additive, not a replacement. An agent may read a mixture, and removing
    the defaults buys nothing while risking an agent that used them."""
    env = build_child_env(
        rollout_id="rl_t", llm_url="https://gw/v1", api_key="sk-1",
        trajectory_dir="/tmp/t", parent_env={"PATH": "/usr/bin"},
        endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"),
    )
    assert env["OPENAI_BASE_URL"] == "https://gw/v1"
    assert env["ANTHROPIC_API_KEY"] == "sk-1"


def test_omitting_it_changes_nothing():
    a = build_child_env(rollout_id="r", llm_url="u", api_key="k",
                        trajectory_dir="/t", parent_env={"PATH": "/usr/bin"})
    b = build_child_env(rollout_id="r", llm_url="u", api_key="k",
                        trajectory_dir="/t", parent_env={"PATH": "/usr/bin"},
                        endpoint_env=None)
    assert a == b


def test_the_named_key_is_still_a_percepteye_secret_free_zone():
    """Percepteye secrets never cross by inheritance, and naming an endpoint
    variable must not become a way to smuggle one across."""
    env = build_child_env(
        rollout_id="rl_t", llm_url="https://gw/v1", api_key="sk-1",
        trajectory_dir="/tmp/t",
        parent_env={"PATH": "/usr/bin", "PERCEPTEYE_API_KEY": "long-lived"},
        endpoint_env=("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY"),
    )
    assert env.get("PERCEPTEYE_API_KEY") is None
