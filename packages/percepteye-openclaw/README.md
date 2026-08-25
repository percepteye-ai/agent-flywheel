# percepteye-agent-flywheel — OpenClaw plugin

Records what your OpenClaw agent's tools actually did, as `ok`, `failed` or
**`unknown`**, into the trajectory the PerceptEye flywheel trains on.

**Observes only.** It cannot alter, block, delay or replace a tool result —
that is a property of the hook it uses, not a promise about the code.

```bash
openclaw plugins install percepteye-agent-flywheel
```

Then set one environment variable on the process running your agent:

```bash
export PERCEPTEYE_TRAJECTORY_DIR=/path/to/this/rollout
```

With the variable unset the plugin does nothing at all — no file, no writes, no
overhead. That is what lets you leave it installed permanently and have it
record only while a rollout is running.

## What it writes

One JSON object per executed tool call, appended to
`$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`:

```json
{"name":"exec","arguments":{"command":"exit 3"},"outcome":"failed","status_code":null,"error":"(Command exited with code 3)","error_class":"ToolError","latency_ms":155,"agent_name":"main","tool_call_id":"call_percepteye_1"}
```

That line is real output from a live OpenClaw run, and it is the same format
the Python SDK writes and reads — the file *is* the interface between them.

## Why `unknown` is a real answer

`unknown` means the call executed and the host did not tell us whether it
worked. It is never rounded up to `ok`, because a tool call scored as a success
when nobody said so is training data that teaches the model a failure was fine.

The distinction is not academic here. In a real run OpenClaw reports
`details.status: "completed"` for a command that exited **3** — `completed`
means the process finished, not that it worked. A classifier keying on that
field would grade every failed command as a success. So this plugin reads the
exit code, and where it cannot read one it says `unknown`:

| what we see | outcome |
|---|---|
| host sets `error` | `failed` |
| `exec`/`bash` with exit 0 | `ok` |
| `exec`/`bash` with non-zero exit | `failed` |
| `exec`/`bash`, exit code unreadable | `unknown` |
| `read`, `write`, `edit`, `ls`, `grep`, `find` with no error | `ok` |
| anything else — MCP tools, third-party plugin tools | `unknown` |

The last row is the important one. An MCP server that returns `{"error": ...}`
as ordinary content looks identical to success from here, so the honest answer
is that we did not observe the verdict. Tools move out of that row when a
fixture proves how they report failure — promotion is earned, never assumed.

## Which hook, and why

OpenClaw exposes three places to see a tool result. Only one is both complete
and structurally incapable of changing the run:

| surface | can it alter the result? | |
|---|---|---|
| `registerAgentToolResultMiddleware` | **yes** — its return value replaces what the model reads | rejected |
| `registerAgentEventSubscription` | no, but the payload is split across two events and exec results are capped before dispatch | rejected |
| `api.on("after_tool_call", …)` | no — returns `void` | **used** |

`after_tool_call` carries the whole call in one event — `params`, `result`,
`error`, `durationMs`, `toolCallId`, plus agent and session identity — so the
safer hook is also the more complete one. It costs nothing to be unable to
break your agent.

The rule behind it: capturing evidence must never be able to change the
behaviour being measured. A collector that can corrupt the run it is measuring
is worse than no collector.

`openclaw plugins doctor` reports this plugin as *hook-only*, which is correct
— an observer claims no capability seam, and the manifest declares no
`contracts` because there is none to declare.

## Dependencies

None. `node:fs` and `node:path` only. This runs inside someone else's agent
process, and every dependency added here is a version conflict imposed on a
customer who did not ask for it.

## Tests

```bash
npm test
```

The suite includes payloads captured verbatim from a live OpenClaw run
(`test/host-fixtures.test.js`). If a future OpenClaw changes the shape, those
fail — and someone re-reads the payload, rather than discovering it months
later from bad training data.

## License

MIT
