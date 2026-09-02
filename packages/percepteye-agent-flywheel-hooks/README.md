# percepteye-agent-flywheel-hooks

**See your agent's real tool failure rate** — then turn those same outcomes into a
**self-improving, continual learning loop** for your agents.

Your agent's transcript will not tell you which tool calls actually worked. A tool
that swallows its exception and returns a string reads exactly like one that
succeeded. This plugin records the host's *own* success/failure verdict for every
tool call, so you get the real number — and that same record is the training
evidence a flywheel needs to make the agent measurably better.

One package, two hosts: **goose** (Open Plugins format) and **Gemini CLI**
(extension format). Nothing in the host is modified — no patch, no fork, no
rebuild. It subscribes to post-tool lifecycle events the host already emits.

## See it in 60 seconds

No account, no signup, no network. Set the directory for one run, then look:

```bash
PERCEPTEYE_TRAJECTORY_DIR=./traj goose run -i - --no-session
python3 ~/.agents/plugins/percepteye-agent-flywheel/scripts/percepteye_report.py ./traj
```

```text
  percepteye · traj/tool_calls.jsonl

  26 tool calls
  ──────────────────────────────────────────────
  ok         17  ██████████████████░░░░░░░░░░  65%
  failed      7  ████████░░░░░░░░░░░░░░░░░░░░  27%
  unknown     2  ██░░░░░░░░░░░░░░░░░░░░░░░░░░   8%

  Real failure rate: 29%  (7 of 24 observed calls)
  2 call(s) where nothing observed the outcome — held out of that rate, never counted as success.

  Where the failures are
    text_editor                4 failed /   9 calls    44%
    shell                      3 failed /  15 calls    20%
```

The `unknown` row is the part other tooling does not give you. Calls nobody
observed are held **out** of the failure rate rather than counted as successes —
a rate computed over unobserved calls is the specific lie this design exists to
avoid.

## Then: close the loop

That file is not just a report. It is the trajectory format the
[Percepteye Agent Flywheel](https://docs.percepteye.ai/flywheel) trains on — real
outcomes from your agent's real work, which is what turns "here is your failure
rate" into "here is a policy that fails less." The same install serves both: read
it locally today, donate rollouts when you are ready.

## Why a hook and not a stdout parser

The SDK can already read tool outcomes off a CLI's event stream when that stream
uses a vocabulary it knows (`otel-genai` or `prime-agent`). goose does not:
its `stream-json` output carries `message` / `notification` / `error` /
`complete`, and the tool calls are nested *inside* `message` content blocks
rather than emitted as flat tool events. Measured, not assumed:

```
parse_tool_calls(<goose stream-json>) -> None
```

`None` is the honest answer — "this stream carries no tool-execution events" —
but it means zero reward surface. The hook events carry the outcome directly, so
that is where this reads from.

## What it records

One JSON object per line appended to `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`
— the same wire format as the Node plugin, documented in
[`../percepteye-agent-flywheel/src/trajectory.js`](../percepteye-agent-flywheel/src/trajectory.js).
The format is a *file*, not a function call, which is exactly why a host with no
Python or Node SDK can feed the flywheel by appending a line.

```json
{"name": "developer__shell", "arguments": {"command": "ls -la"}, "outcome": "ok", "status_code": null, "tool_call_id": "call_1"}
{"name": "run_shell_command", "arguments": {"command": "false"}, "outcome": "failed", "status_code": null, "error": "exit status 1"}
```

Records are bounded: a single argument value over 1000 characters is clipped and
says so (`…[truncated, 200000 chars total]`), and a whole line is kept under
7KB. A file-writing tool puts the entire file in its arguments, and this record
is evidence about the *call*, not a second copy of its payload. The cap is also
what keeps each append a single `write()` — goose runs tool calls concurrently,
so oversized records would let two hook processes interleave and corrupt both
lines.

## Outcome mapping

| host | event | outcome | where the verdict comes from |
|:--|:--|:--|:--|
| goose | `PostToolUse` | `ok` | goose's own `Ok(r) if r.is_error != Some(true)` |
| goose | `PostToolUseFailure` | `failed` | the same check, other branch |
| Gemini CLI | `AfterTool` | `failed` iff `tool_response.error` | the payload's own error field |
| either | anything else | `unknown` | nothing observed it |

The first two rows are the point: **goose decides**, in
`crates/goose/src/agents/agent.rs`, which of the two events to fire. This plugin
transcribes that verdict rather than reconstructing one from a transcript.

`unknown` is legal, supported and expected, and is **never rounded up to `ok`**.

## Three properties it keeps

1. **Inert unless `PERCEPTEYE_TRAJECTORY_DIR` is set.** No rollout, no file, no
   behaviour. Leave it installed permanently — it is a no-op in production and a
   recorder during a rollout.
2. **Never fails the agent.** Every path exits 0, including malformed input.
3. **Observe-only, structurally.** It registers for *post*-tool events only,
   which are fire-and-forget on both hosts, and writes nothing to stdout. That
   is a property of which events it subscribes to, not a promise about the code.

## Install

**goose** reads the Open Plugins manifest (`plugin.json`), and tries that format
*before* the Gemini one — so the dual manifest is unambiguous rather than a
conflict.

`goose plugin install` takes a **git repository URL**, not a path, so a local
checkout is installed by copying it into a plugins directory. Plugins found
there are enabled unless a settings file disables them.

```bash
# project scope — applies to runs started from this project
mkdir -p <project>/.agents/plugins
cp -R packages/percepteye-agent-flywheel-hooks <project>/.agents/plugins/percepteye-agent-flywheel

# user scope — applies everywhere
cp -R packages/percepteye-agent-flywheel-hooks ~/.agents/plugins/percepteye-agent-flywheel
```

Once published to a repository, the normal path works:

```bash
goose plugin install https://github.com/<org>/percepteye-agent-flywheel-hooks
```

**Gemini CLI** reads `gemini-extension.json` and the same `hooks/hooks.json`.

Requires `python3` on `PATH`. The recorder is standard library only.

## How one hooks.json serves both

Both hosts read `hooks/hooks.json` at the same path, but name their events
differently (`PostToolUse`/`PostToolUseFailure` vs `AfterTool`) and substitute a
different root variable (`${PLUGIN_ROOT}` vs `${extensionPath}`). The file
declares **all three events**, so each host matches only its own entries — and
therefore only ever sees its own substitution variable. goose documents that it
ignores unknown event names.

## Verified, and not

**Verified against a live goose 1.48.0** built from `aaif-goose/goose`, driven by
a stub OpenAI endpoint so the run needs no credentials. goose executed a real
shell tool, the hook fired, and the plugin recorded it — then the SDK's own
`Trajectory.read()` parsed the result with `malformed=0`:

```
# succeeding command
{"name": "shell", "arguments": {"command": "echo percepteye-hook-probe"}, "outcome": "ok", "status_code": null, "tool_call_id": "call_probe_1"}
# failing command — goose fired PostToolUseFailure instead
{"name": "shell", "arguments": {"command": "cat /definitely/no/such/file-percepteye"}, "outcome": "failed", "status_code": null, "tool_call_id": "call_probe_1"}
```

The two lines differ only in what the tool actually did, which is the property
worth having: the outcome is goose's observation, not a guess read off a
transcript. The same run also confirmed goose honours `OPENAI_BASE_URL`, so the
completion wire is ours.

Also verified, by `tests/test_goose_hook_plugin.py`: all four outcome rows, the
dropped-nameless-call and malformed-input guards, append-not-truncate, silence on
stdout, and the inert-without-env-var property.

**Not verified: the Gemini CLI half.** Its event names and `AfterTool` payload
come from Google's published reference, not from a run — there is no Gemini CLI
on this machine. Whether it *ignores* unknown event names (as goose documents it
does) is undocumented; if it rejects them instead, split `hooks.json` per host.
