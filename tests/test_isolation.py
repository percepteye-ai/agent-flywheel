"""The child's environment. This is a security boundary, not a convenience."""
from __future__ import annotations

from percepteye_flywheel.isolation import build_child_env

PARENT = {
    "PATH": "/usr/bin",
    "HOME": "/home/dev",
    # Ours. The long-lived, org-scoped control-plane credential.
    "PERCEPTEYE_API_KEY": "pehk_live_do_not_leak",
    "PERCEPTEYE_CONTROL_PLANE_URL": "https://api.percepteye.com/api/flywheel/v1",
    # Theirs. Their agent legitimately needs these -- that is the premise of
    # running it in their own process.
    "JIRA_API_TOKEN": "customer-token",
    "DATABASE_URL": "postgres://customer",
    "OPENAI_API_KEY": "sk-customer-own-key",
}


def _env(**kw):
    base = dict(
        rollout_id="rl_abc", llm_url="https://gw.percepteye.com/llm/v1/rollouts/rl_abc",
        api_key="perk_single_use", ledger_dir="/tmp/led", parent_env=PARENT,
    )
    base.update(kw)
    return build_child_env(**base)


def test_control_plane_key_never_reaches_the_child():
    """THE finding. The parent holds a long-lived key with register/claim/
    report/artifact scopes; the child runs customer agent code, frequently
    model-authored code with a shell. Splatting the parent environment hands
    the strongest credential in the system to the least trusted process."""
    env = _env()
    assert "PERCEPTEYE_API_KEY" not in env
    assert "pehk_live_do_not_leak" not in "".join(env.values())


def test_no_percepteye_variable_is_inherited():
    env = _env()
    inherited = {k for k in env if k.startswith("PERCEPTEYE_")}
    assert inherited == {"PERCEPTEYE_ROLLOUT_ID", "PERCEPTEYE_LEDGER_DIR"}


def test_rollout_scoped_variables_are_set():
    env = _env()
    assert env["PERCEPTEYE_ROLLOUT_ID"] == "rl_abc"
    assert env["PERCEPTEYE_LEDGER_DIR"] == "/tmp/led"


def test_routing_wins_over_the_customers_own_key():
    """Their OPENAI_API_KEY is in the parent. For this rollout the completion
    wire must be ours, or the rollout is unattributable and untrainable."""
    env = _env()
    assert env["OPENAI_API_KEY"] == "perk_single_use"
    assert env["OPENAI_BASE_URL"].endswith("/rollouts/rl_abc")
    assert env["OPENAI_API_BASE"] == env["OPENAI_BASE_URL"]


def test_the_customers_own_secrets_still_pass_through():
    """BYOH exists so their agent runs against their real systems. Stripping
    their credentials would defeat the product; stripping ours is the point."""
    env = _env()
    assert env["JIRA_API_TOKEN"] == "customer-token"
    assert env["DATABASE_URL"] == "postgres://customer"


def test_allowlist_mode_drops_everything_unnamed():
    env = _env(allowlist=["JIRA_API_TOKEN"])
    assert env["JIRA_API_TOKEN"] == "customer-token"
    assert "DATABASE_URL" not in env
    assert env["PATH"] == "/usr/bin", "a baseline is still needed to start a process"
    assert "PERCEPTEYE_API_KEY" not in env


def test_extra_cannot_smuggle_our_namespace_back_in():
    env = _env(extra={"PERCEPTEYE_API_KEY": "pehk_sneaky", "MY_VAR": "1"})
    assert "PERCEPTEYE_API_KEY" not in env
    assert env["MY_VAR"] == "1"


# ── providers we cannot route ──────────────────────────────────────────────


def test_a_provider_we_cannot_route_is_removed_from_the_child():
    """Setting OPENAI_BASE_URL does nothing for an agent built on Gemini or
    Azure: those clients read their own names, so the agent sailed past the
    gateway and sampled from the live provider -- producing a trajectory that
    cannot back an on-policy objective while looking exactly like one that
    can."""
    env = build_child_env(
        rollout_id="rl_1", llm_url="https://gw/llm/v1/rollouts/rl_1",
        api_key="perk_x", ledger_dir="/tmp/l",
        parent_env={
            "GOOGLE_API_KEY": "live-gemini",
            "AZURE_OPENAI_ENDPOINT": "https://acme.openai.azure.com",
            "AZURE_OPENAI_API_KEY": "live-azure",
            "GROQ_API_KEY": "live-groq",
            "PATH": "/usr/bin",
        },
    )
    for name in ("GOOGLE_API_KEY", "AZURE_OPENAI_ENDPOINT",
                 "AZURE_OPENAI_API_KEY", "GROQ_API_KEY"):
        assert name not in env, f"{name} would have routed past the gateway"
    assert env["PATH"] == "/usr/bin"


def test_the_customers_own_non_model_credentials_still_reach_their_agent():
    """Removing a MODEL endpoint is not the same as removing their Jira token.
    The model is the one thing this product replaces; everything else is the
    reason the agent runs in their process at all."""
    env = build_child_env(
        rollout_id="rl_1", llm_url="https://gw/llm/v1/rollouts/rl_1",
        api_key="perk_x", ledger_dir="/tmp/l",
        parent_env={"JIRA_API_TOKEN": "theirs", "AWS_ACCESS_KEY_ID": "for-s3"},
    )
    assert env["JIRA_API_TOKEN"] == "theirs"
    # Bedrock's own token goes; the general AWS credential stays, because it is
    # the same one their agent uses for storage.
    assert env["AWS_ACCESS_KEY_ID"] == "for-s3"


def test_the_routing_vars_still_win_over_an_inherited_value():
    env = build_child_env(
        rollout_id="rl_1", llm_url="https://gw/llm/v1/rollouts/rl_1",
        api_key="perk_x", ledger_dir="/tmp/l",
        parent_env={"OPENAI_BASE_URL": "https://api.openai.com/v1",
                    "OPENAI_API_KEY": "sk-live"},
    )
    assert env["OPENAI_BASE_URL"].endswith("/rollouts/rl_1")
    assert env["OPENAI_API_KEY"] == "perk_x"


def test_unroutable_providers_are_reported_for_the_startup_warning():
    from percepteye_flywheel.isolation import unroutable_providers_present

    found = unroutable_providers_present(
        {"GEMINI_API_KEY": "x", "JIRA_API_TOKEN": "y", "GROQ_API_KEY": ""}
    )
    assert found == ("GEMINI_API_KEY",), (
        "an empty value is not configuration, and a Jira token is not a model"
    )
