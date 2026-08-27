"""Serve prime-agent rollouts. No Python integration — it is a TypeScript binary.

    PERCEPTEYE_CONTROL_PLANE_URL=... PERCEPTEYE_API_KEY=... \
      python examples/prime_agent/serve_prime.py

prime-agent is served through the COMMAND entrypoint, which is the path that
exists for exactly this: an agent that is a process rather than an importable
Python object. It needs no Percepteye code, and the only thing added to it is
one extension that points its completions at the per-rollout endpoint.
"""
from __future__ import annotations

import os
import pathlib
import tempfile

import percepteye_agent_flywheel as fw

HERE = pathlib.Path(__file__).resolve().parent

#: WHERE ROLLOUTS RUN. Each rollout's child process is started with its cwd
#: inside this directory, and the agent has a real shell — so this must never
#: be a source tree. Left to the default (``$PWD/.percepteye-artifacts``) and
#: launched from a checkout, rollouts write their task artifacts into the
#: repository and can reach the rest of it with ordinary relative paths: on
#: 2026-08-16 a sweep launched that way left shell scripts beside the package
#: and removed tracked files from the working tree, including the licence and
#: the build config.
#:
#: A dedicated directory outside any checkout is the cheap part of the fix.
#: The complete one is an OS-level sandbox — this SDK's ``isolation`` is a
#: process boundary and an environment scrub, and says nothing about the
#: filesystem. Run untrusted agents accordingly.
WORKSPACE = pathlib.Path(
    os.environ.get("PRIME_WORKSPACE")
    or (pathlib.Path.home() / ".percepteye" / "prime-agent-workspace"),
).expanduser()
WORKSPACE.mkdir(parents=True, exist_ok=True)

#: `-p` is print mode: one task, non-interactive, answer on stdout. The task
#: also arrives on stdin, which prime-agent merges into the prompt — the SDK
#: sends it both ways because CLIs differ and guessing wrongly is silent.
#:
#: The autonomous limits are bounded deliberately. A rollout that runs until it
#: decides to stop is a rollout whose cost we do not control, and the episode
#: budget is the only clock the control plane can enforce from outside.
#:
#: Continuations raised 2 -> 6 on 2026-08-12. At 2, run 08e40b36's rollout
#: rl_dz73eyv died `entrypoint_error: exit 1: Autonomous run stopped before
#: terminal evidence; maxContinuations reached (2/2)` — the cap cut the agent
#: off BEFORE it produced the evidence the reward reads, so the rollout scored
#: nothing and told us nothing. A bound that stops the work short of any
#: observable result is not cost control, it is a wasted rollout. The episode
#: budget still bounds the real cost from outside.
#: ONE daemon socket per harness process, at a path nothing else uses.
#:
#: prime-agent runs a daemon supervisor that owns a registry entry. When a
#: second generation takes that entry over, every agent still bound to the old
#: one dies instantly with "Daemon supervisor generation ... no longer owns its
#: registry entry" — which the SDK reports as ``entrypoint_error`` and the
#: liveness canary reports as "no successful real-provider LLM call". Run
#: 32403f96 lost 16 of 40 rollouts that way, mid-run, after a manual daemon
#: kill had "fixed" it an hour earlier.
#:
#: The default socket is shared by every prime-agent on the box: a previous
#: harness's orphaned daemons, and the operator's own interactive sessions.
#: A path keyed to THIS process shares with none of them, so a stale
#: generation cannot evict us and we cannot evict anyone. Per-ROLLOUT sockets
#: would isolate too, but daemons outlive the agent that started them, so that
#: leaks a daemon per rollout; one per harness run is the same isolation
#: without the accumulation.
#:
#: ``prime-agent --version`` works regardless — it never touches the daemon —
#: so it is not a liveness check for this.
_DAEMON_SOCKET = str(
    pathlib.Path(tempfile.gettempdir()) / f"percepteye-prime-{os.getpid()}.sock"
)

ARGV = [
    "prime-agent",
    "--daemon-socket", _DAEMON_SOCKET,
    "-p",
    "--mode", "json",              # structured events, not prose
    "-e", str(HERE / "route.ts"),  # the routing extension
    # SELECTING the provider the extension registers. Registering one does not
    # use it: without these two flags prime-agent kept its own built-in
    # provider and its own credentials, every completion went somewhere we
    # never saw, and `inference_calls` stayed empty platform-wide — so no
    # trajectory carried an llm_call and the substrate-liveness canary
    # correctly reported "no successful real-provider LLM call" on every run
    # for a week. The rollouts looked fine; we simply owned none of the
    # completion wire, which is the exact failure route.ts warns about in its
    # own header and could not itself prevent.
    "--provider", "percepteye",
    "--model", os.environ.get("PERCEPTEYE_MODEL_ID", "policy"),
    "--autonomous",
    # 12 -> 15 on 2026-08-12. NOT because the agent needs more room to find the
    # answer: on run 516b7a55 it wrote a CORRECT top_ips.sh and kept going,
    # because the grading test is hidden by design and it has no way to know it
    # is finished. Raising this to 30 did not buy completions — it converted
    # `maxTurns` exclusions into `deadline_exceeded` ones against the 300s
    # episode budget, and a second attempt at the same task drifted from a
    # correct solution to a wrong one. 15 is a modest allowance over the point
    # where the work is typically done; the real fix is that the artifact is now
    # graded whether or not the agent terminates tidily.
    "--autonomous-max-turns", os.environ.get("PRIME_MAX_TURNS", "15"),
    "--autonomous-max-continuations", "6",
    # Bounded because the DEFAULT is 300000ms — exactly the 300s episode
    # budget, so one hanging gate consumes the entire rollout and the agent
    # never gets to report. A gate is an existence-and-smoke check; the
    # derived `test -s <file>` form returns instantly, and a model-authored
    # one that runs the deliverable should too. 30s is generous for both and
    # leaves the budget to the actual work.
    "--autonomous-gate-timeout-ms", "30000",
]


def main() -> int:
    return fw.serve(
        ARGV,
        agent_id=os.environ.get("PERCEPTEYE_AGENT_ID", "prime-agent"),
        input_shape="text",
        # TRUE as of 2026-08-12: the event parser landed. prime-agent's
        # `--mode json` stream emits tool_execution_start / tool_execution_end
        # with toolCallId, toolName, args and isError, and the SDK reads them
        # off stdout (event_stream.parse_tool_calls) with no instrumentation in
        # the agent at all. So this is now an assertion we can actually back.
        #
        # It stays honest at the edges rather than by promise: a call that ends
        # without an isError is recorded `unknown`, never rounded up to ok, and
        # a stream carrying no tool events leaves tool_calls None rather than
        # claiming zero calls were made.
        reports_tool_calls=True,
        # Raised 1 -> 4 on 2026-08-12. A GRPO group mints its rollouts at once,
        # each stamped with an ABSOLUTE deadline_at; at concurrency 1 the rest
        # sat in the queue burning that clock while this process worked one of
        # them. Run 08e40b36 lost 5 of 6 rollouts that way — two never claimed
        # at all ("rollout budget elapsed before any agent claimed it") and one
        # was claimed with 55s of its 300s left and killed mid-flight, because
        # the SDK takes min(rollout_timeout_s, seconds_remaining) rather than a
        # fresh clock. Serving concurrency has to match the group width or the
        # group's own deadline is what kills it.
        concurrency=int(os.environ.get("PRIME_CONCURRENCY", "4")),
        # How prime-agent spells its completion gate. `--autonomous` is
        # documented as "continue until gates pass or a limit is reached", so
        # with no gate registered a LIMIT is the only exit that exists — which
        # is why every rollout on runs b6b4388d and b60067db ended on maxTurns
        # or maxTokens and was excluded from training even when its artifact
        # was correct. The gate COMMAND is per task and rides in the rollout
        # metadata; this only says how to spell the option.
        gate_flag="--autonomous-gate",
        # Anchor every rollout's working directory outside any checkout. See
        # WORKSPACE above — without this the cwd defaults to the directory the
        # server happened to be started from.
        artifacts_dir=str(WORKSPACE),
    )


if __name__ == "__main__":
    raise SystemExit(main())
