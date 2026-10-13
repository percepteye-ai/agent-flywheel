"""Serve goose rollouts. No Python integration — it is a Rust binary.

    PERCEPTEYE_CONTROL_PLANE_URL=... PERCEPTEYE_API_KEY=... \
      python examples/goose/serve_goose.py

goose (https://github.com/aaif-goose/goose) is served through the COMMAND
entrypoint, which is the path that exists for exactly this: an agent that is a
process rather than an importable Python object.

Unlike prime-agent, goose needs NO routing extension. Its openai provider
resolves the endpoint from the environment, and the names it reads are the ones
``build_child_env`` already sets per rollout:

    crates/goose/src/providers/openai_def.rs :: resolve_base_url
      1. OPENAI_HOST      env var   (session override, deprecated but honoured)
      2. OPENAI_BASE_URL  env var
      3. OPENAI_BASE_URL  config file
      4. OPENAI_HOST      config file (persisted by `goose configure`)
      5. https://api.openai.com

So step 2 is ours and routing is the argv below plus two scrubs. Note ranks 3
and 4: a config-file OPENAI_BASE_URL beats a config-file OPENAI_HOST, which is
why ``install_hooks`` writes the former.

TOOL OUTCOMES come from the hook plugin, which SHIPS IN THIS WHEEL -- there is
no second install and no external repository. ``agent-flywheel install-hooks``
renders it, pointed at this virtualenv's interpreter; it subscribes to goose's
post-tool hook events and appends to ``$PERCEPTEYE_TRAJECTORY_DIR``. Outside a
rollout it records locally under ``$PERCEPTEYE_HOME/turns/<session>`` instead,
and claims a practice task only in training mode with a key -- see README.md.
Nothing in goose's source is modified.

THIS FILE IS THE CAPACITY-DONATION PATH, not the only one. Here the SDK drives
goose. A customer who runs goose THEMSELVES needs no worker at all -- see
"CLI agents: three commands" in the repository README.

VERIFIED AGAINST THE CHECKOUT, not against the docs — every claim in this file
names the file it was read from.
"""
from __future__ import annotations

import os
import pathlib

import agent_flywheel as fw

#: WHERE ROLLOUTS RUN. Each rollout's child starts with its cwd inside this
#: directory and goose has a real shell — so this must never be a source tree.
#: goose's default mode auto-approves tool calls (see GOOSE_MODE below), which
#: means `rm` needs no confirmation from anyone.
WORKSPACE = pathlib.Path(
    os.environ.get("GOOSE_WORKSPACE")
    or (pathlib.Path.home() / ".percepteye" / "goose-workspace"),
).expanduser()
WORKSPACE.mkdir(parents=True, exist_ok=True)


def _scrub_unroutable_goose_vars() -> list[str]:
    """Remove goose-specific env that outranks or bypasses our routing.

    ``build_child_env`` sets OPENAI_BASE_URL last so it beats anything
    inherited — but "last" only settles ties among names the SDK knows. These
    three are goose-specific, are NOT in ``isolation.UNROUTABLE_PROVIDER_VARS``,
    and each silently wins against us:

    * ``OPENAI_HOST`` is resolved BEFORE OPENAI_BASE_URL (openai.rs, above).
      An operator with it exported — the deprecated-but-honoured spelling, so
      plenty of shells still carry it — sends every completion to their own
      host while the control plane believes it is sampling the policy under
      training. The rollout looks perfect; we own none of the completion wire.
    * ``GOOSE_PROVIDER`` selects the provider at all. Routing the openai
      provider buys nothing if the operator's config selects `anthropic` or
      `claude-code`, so `--provider openai` is passed explicitly below and the
      env var is removed rather than left to argue with it.
    * ``GOOSE_MODEL`` names the model within that provider. `--model` is passed
      explicitly for the same reason, and an inherited value would otherwise
      decide which policy the gateway is asked for.

    Scrubbed in the PARENT, before serve(), because the default isolation
    policy inherits the parent environment.
    """
    removed = []
    for name in ("OPENAI_HOST", "GOOSE_PROVIDER", "GOOSE_MODEL"):
        if os.environ.pop(name, None) is not None:
            removed.append(name)
    return removed


#: Directories goose discovers plugins from: user scope, and project scope
#: relative to the rollout's working directory.
_PLUGIN_ROOTS = (
    pathlib.Path.home() / ".agents" / "plugins",
    WORKSPACE / ".agents" / "plugins",
)


def _find_hook_plugin() -> pathlib.Path | None:
    """Where goose will find the recorder, or None if it will not."""
    for root in _PLUGIN_ROOTS:
        if not root.is_dir():
            continue
        for entry in sorted(root.iterdir()):
            if (entry / "scripts" / "percepteye_record.py").is_file():
                return entry
    return None


#: Auto-approve tool calls. `GooseMode::Auto` is the DEFAULT
#: (crates/goose-provider-types/src/goose_mode.rs), but the effective value is
#: `Config::global().get_goose_mode()` — the operator's persisted config file,
#: not the default. An operator who once chose "Ask before every tool call"
#: has `approve` on disk, and every headless rollout then blocks on a prompt
#: nobody is there to answer until the episode budget kills it. Set explicitly
#: so the rollout does not depend on a developer's saved preference.
os.environ["GOOSE_MODE"] = "auto"

_SCRUBBED = _scrub_unroutable_goose_vars()

ARGV = [
    "goose",
    "run",
    # The task arrives on stdin; `-` is how goose spells "instructions from
    # stdin" (cli.rs :: parse_run_input). The SDK also exports it as
    # PERCEPTEYE_TURN_INPUT, which goose does not read.
    "-i", "-",
    # A NAMED SESSION, and it is `--no-session` that had to go — the two are
    # mutually exclusive ("the argument '--no-session' cannot be used with
    # '--name'"). This is not about sessions at all; it is the only way to stop
    # goose spending a COMPLETION on naming one.
    #
    # goose opens every run by asking the model for a 3-5 word session title.
    # That call goes through the same OPENAI_BASE_URL, so under a rollout it
    # authenticates as the rollout and lands as `inference_calls` seq 0 — and
    # the trainable predicate tests only "does it carry logprobs", which the
    # managed sampler attaches to every completion. So the title became
    # transition 0 of the trajectory, inherited the task text as its input,
    # was emitted as a GRPO row whose target is a session title carrying the
    # episode's advantage sign, and inflated the transition count so every
    # REAL decision trained at adv/(K+1) instead of adv/K. It is also the call
    # the scaffold fingerprint is computed from, so `agent_fingerprint_sha`
    # was sha256 of `{"tools": null}` for every rollout and the 5-tool block
    # never entered it.
    #
    # `maybe_update_name` returns early on `session.user_set_name`
    # (session_manager.rs:592), and `--name` is what sets it
    # (cli.rs:456-470). MEASURED: 2 completions per run before, 1 after, and
    # the survivor is the agent's own turn carrying all 5 tools.
    #
    # The leakage `--no-session` was guarding against does not appear: two
    # consecutive runs under the SAME name each started from 2 messages
    # (system + user), so nothing carries over. What it costs is a row per
    # rollout in goose's sessions DB — litter, where the alternative was a
    # corrupted gradient.
    "--name", os.environ.get("GOOSE_SESSION_NAME", "percepteye-rollout"),
    # Quiet: stdout is the model's final response and nothing else, which is
    # what the SDK records as `final_text`. goose's `stream-json` format was
    # the obvious choice here and is the WRONG one now — tool outcomes arrive
    # through the hook plugin, so buying structure on stdout would only turn
    # the final answer into a JSON blob the reward has to dig through.
    "-q",
    # THE TOOL CATALOGUE IS A PROPERTY OF THE ROLLOUT, not of a developer's
    # saved profile. Same argument as `--provider` below, one layer up.
    #
    # `--no-profile` drops the operator's configured extensions and
    # `--with-builtin` names the ones this flywheel runs with. Without the
    # pair, every rollout's task universe is whatever `goose configure` last
    # wrote — so two machines serving the same agent_id report DIFFERENT
    # catalogues, the control plane keeps the last registration, and workflows
    # get generated naming tools half the fleet does not have.
    #
    # It is also the difference between describing this agent and not. A fresh
    # goose config has NO extensions (`config/extensions.rs` defaults to none;
    # `DEFAULT_EXTENSION` is a struct default, not an auto-enable), so goose
    # sends `tools: []` and registers as `introspection_state=unreadable` —
    # measured, not assumed: 0 tools before this flag, 5 after.
    "--no-profile",
    "--with-builtin", os.environ.get("GOOSE_BUILTINS", "developer"),
    # SELECTING the routed provider, not merely making it available. Without
    # these goose keeps whatever `goose configure` persisted, and the completion
    # wire is not ours — the exact failure prime-agent's route.ts was written
    # to prevent, reached here by a different road.
    "--provider", "openai",
    "--model", os.environ.get("PERCEPTEYE_MODEL_ID", "policy"),
    # The default is 1000 (cli.rs), which is not a bound — one runaway rollout
    # would consume the whole episode budget and report nothing. This is a
    # starting value, not a tuned one: raise it if rollouts are being cut off
    # before they produce the evidence the reward reads.
    "--max-turns", os.environ.get("GOOSE_MAX_TURNS", "25"),
]


#: The agent id this flywheel serves under. Read once so the prompt pull and
#: the registration cannot disagree about whose prompt is being applied.
AGENT_ID = os.environ.get("PERCEPTEYE_AGENT_ID", "goose")


def _trained_prompt_argv() -> list[str]:
    """``["--system", <approved prompt>]``, or ``[]`` to keep goose's own.

    THE SECOND HALF OF TRAINING, and the only one goose can accept. A run
    co-optimizes a model AND a system prompt; ``apply_current_policy`` moves
    the model, and until something applies the prompt the agent improves on one
    axis only. goose takes it on the command line and nowhere else — there is
    no prompt field on a rollout (``RolloutRequest`` deliberately carries no
    configuration channel), so this is resolved once at startup exactly as
    ``serve()`` resolves the policy, and for the same reason: a prompt that
    changes under a running agent is the confound the control plane refuses on
    its own training runs.

    ``--system`` is ADDITIVE — it maps to goose's ``additional_system_prompt``
    (cli.rs), not to a replacement — which is what makes this safe to apply
    without knowing what goose put in its own prompt. The trained text layers
    over goose's tool instructions instead of deleting them.

    ``{}`` from the SDK means KEEP THE PROMPT YOU HAVE: nothing has passed the
    approval gate for this agent, or the control plane could not answer. Both
    return ``[]`` here, and neither is an error — an unreachable control plane
    must never blank a running agent's instructions.

    WHETHER it may be applied is ``fw.prompt_decision``'s answer, not this
    file's. The SDK deliberately does not APPLY the prompt for you, because
    swapping a prompt changes what an agent does and that is the owner's call
    — but the four checks that decide whether it MAY be applied are the SDK's
    to make, and asking them here would be a second answer to the same
    question. Applying the result IS the owner's decision, made once, out
    loud, in the integration that owns this deployment — and printed, because
    a prompt that changed with nothing on stdout to say so is
    indistinguishable from one that did not.
    """
    try:
        prompt = fw.current_prompt(agent_id=AGENT_ID)
    except Exception as exc:  # never fail to serve over a prompt
        print(f"[percepteye] prompt pull failed ({exc}); keeping goose's own.")
        return []
    # THE WHOLE PREDICATE, asked in one call. This used to check that the text
    # was non-blank and warn on `stale_baseline`, which is two of the four
    # questions: a bundle whose status was never approved, or whose prompt
    # identity disagrees with the checkpoint it was certified against, went
    # onto goose's command line anyway. Half a predicate at a call site is the
    # defect `fw.prompt_decision` exists to end, and re-deriving it here would
    # also let this integration answer differently from the SDK it ships with.
    decision = fw.prompt_decision(prompt)
    if not decision.apply:
        print(f"[percepteye] prompt: keeping goose's own — {decision.reason}.")
        return []
    print(
        f"[percepteye] prompt: applying v{prompt.get('version')} "
        f"sha={str(prompt.get('sha256'))[:12]} ({len(decision.text)} chars) "
        f"as --system, ADDITIVE to goose's own. Certified as a PAIR with the "
        f"checkpoint stamped {(decision.checksum or 'none')[:12]}."
    )
    return ["--system", decision.text]


def main() -> int:
    if _SCRUBBED:
        print(f"[percepteye] scrubbed goose routing overrides: {', '.join(_SCRUBBED)}")

    # CHECKED HERE, BEFORE THE FIRST ROLLOUT, because `reports_tool_calls=True`
    # below is an assertion and this is the only thing that makes it true.
    # Without the plugin every rollout still runs and still routes -- the run
    # looks entirely healthy -- while the trajectory stays empty and every
    # verifier that grades "did this call land" abstains for the whole run. A
    # silent downgrade of the reward surface is exactly the failure the SDK
    # front-loads its configuration checks to prevent, so this raises rather
    # than warning.
    # ...but ONLY when this process is actually going to serve. In production
    # mode serve() returns before opening a connection and deliberately does
    # not raise -- "a switch that crashes the service it was meant to quiet is
    # not a switch" -- so the same image deployed with the switch flipped must
    # not die here over a plugin it is never going to use. Asking the SDK
    # rather than re-reading the env var keeps the aliases and the closed-set
    # check in one place.
    plugin = _find_hook_plugin()
    if plugin is None and fw.resolve_mode() == "training":
        searched = "\n      ".join(str(r) for r in _PLUGIN_ROOTS)
        raise SystemExit(
            "[percepteye] the goose hook plugin is not installed, so no tool "
            "outcomes would be recorded.\n"
            f"    Searched:\n      {searched}\n"
            "    Install it with:\n"
            "      agent-flywheel install-hooks"
        )
    if plugin is not None:
        print(f"[percepteye] hook plugin: {plugin}")

    argv = list(ARGV) + _trained_prompt_argv()

    return fw.serve(
        argv,
        agent_id=AGENT_ID,
        input_shape="text",
        # TRUE, and backed by the hook plugin this wheel ships — which main()
        # above refuses to start without, so this assertion cannot outlive the
        # thing that makes it true.
        #
        # goose's own stream-json vocabulary carries NO flat tool events
        # (StreamEvent is message / notification / error / complete), so
        # event_stream.parse_tool_calls returns None against it — measured, not
        # assumed. The calls are there but nested inside `message` content
        # blocks, out of reach of a StreamFormat entry.
        #
        # The plugin sidesteps that entirely by reading goose's hook events
        # instead, where the outcome is goose's OWN verdict: it selects
        # PostToolUse vs PostToolUseFailure with `Ok(r) if r.is_error !=
        # Some(true)` (agents/agent.rs), so `ok` and `failed` here are observed
        # rather than reconstructed. An unrecognised event records `unknown`
        # and is never rounded up.
        reports_tool_calls=True,
        # Match this to the GRPO group width. At 1, the rest of a group sits in
        # the queue burning its own absolute deadline while this process works
        # one rollout.
        concurrency=int(os.environ.get("GOOSE_CONCURRENCY", "4")),
        # goose has no completion-gate option, so none is passed. Leaving
        # gate_flag unset is the supported way to say that; a guessed flag
        # would break the binary on the first rollout.
        artifacts_dir=str(WORKSPACE),
    )


if __name__ == "__main__":
    raise SystemExit(main())
