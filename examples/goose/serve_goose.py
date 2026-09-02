"""Serve goose rollouts. No Python integration — it is a Rust binary.

    PERCEPTEYE_CONTROL_PLANE_URL=... PERCEPTEYE_API_KEY=... \
      python examples/goose/serve_goose.py

goose (https://github.com/aaif-goose/goose) is served through the COMMAND
entrypoint, which is the path that exists for exactly this: an agent that is a
process rather than an importable Python object.

Unlike prime-agent, goose needs NO routing extension. Its openai provider
resolves the endpoint from the environment, and the names it reads are the ones
``build_child_env`` already sets per rollout:

    crates/goose-providers/src/openai.rs :: resolve_base_url
      1. OPENAI_HOST      env var   (session override, deprecated but honoured)
      2. OPENAI_BASE_URL  env var
      3. OPENAI_HOST      config file (persisted by `goose configure`)

So step 2 is ours and routing is the argv below plus two scrubs.

TOOL OUTCOMES come from a separate piece: the hook plugin at
https://github.com/percepteye-ai/agent-flywheel-hooks, which subscribes to goose's
post-tool hook events and appends to ``$PERCEPTEYE_TRAJECTORY_DIR``. Install it
once; it is inert whenever that variable is unset, so the same install is a
no-op outside a rollout. Nothing in goose's source is modified.

VERIFIED AGAINST THE CHECKOUT, not against the docs — every claim in this file
names the file it was read from.
"""
from __future__ import annotations

import os
import pathlib

import percepteye_agent_flywheel as fw

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
    # No session file. These are automated runs and a session per rollout is
    # both leakage between rollouts and litter on disk.
    "--no-session",
    # Quiet: stdout is the model's final response and nothing else, which is
    # what the SDK records as `final_text`. goose's `stream-json` format was
    # the obvious choice here and is the WRONG one now — tool outcomes arrive
    # through the hook plugin, so buying structure on stdout would only turn
    # the final answer into a JSON blob the reward has to dig through.
    "-q",
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
            "      goose plugin install "
            "https://github.com/percepteye-ai/agent-flywheel-hooks"
        )
    if plugin is not None:
        print(f"[percepteye] hook plugin: {plugin}")

    return fw.serve(
        ARGV,
        agent_id=os.environ.get("PERCEPTEYE_AGENT_ID", "goose"),
        input_shape="text",
        # TRUE, and backed by the hook plugin at
        # https://github.com/percepteye-ai/agent-flywheel-hooks — which main()
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
