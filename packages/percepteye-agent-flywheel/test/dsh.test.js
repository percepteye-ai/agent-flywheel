/**
 * The DeepSeek Harness plugin.
 *
 * Fixtures follow dsh's own declared shapes: `tools/result` publishes a
 * discriminated union on `isError`, and the exec tools declare an output
 * schema carrying `kind`, `exitCode`, `signal`, `timedOut` and `aborted`.
 * Those five fields are why this classifier can be honest about cases a
 * cruder reading collapses together.
 */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { DSH_EXEC_TOOLS, DSH_VERDICT_TOOLS, classifyDsh } from "../src/dsh-classify.js";
import { HOOK_NAME, apply, name, recordFor } from "../src/dsh.js";
import { TRAJECTORY_DIR_ENV } from "../src/trajectory.js";

const fg = (extra) => ({ isError: false, value: { kind: "foreground", ...extra } });

describe("classifying a dsh tool result", () => {
  it("believes a positive failure assertion, and keeps its error class", () => {
    const v = classifyDsh("bash", {
      isError: true, error: { message: "permission denied", info: "EPERM" },
    });
    assert.equal(v.outcome, "failed");
    assert.equal(v.error, "permission denied");
    // dsh carries an internal class on `info`; far more useful than a label.
    assert.equal(v.error_class, "EPERM");
  });

  it("scores a non-zero exit as failed although isError is false", () => {
    // The case this exists for: the bash tool renders `[exit code: 3]` into
    // the model-facing TEXT and still settles as a non-error result. The tool
    // call worked; the command did not.
    const v = classifyDsh("bash", fg({ exitCode: 3 }));
    assert.equal(v.outcome, "failed");
    assert.equal(v.error_class, "ExitStatus:3");
  });

  it("scores a zero exit as ok", () => {
    assert.equal(classifyDsh("bash", fg({ exitCode: 0 })).outcome, "ok");
  });

  it("calls a backgrounded command unknown, not ok", () => {
    // `kind: "background"` means LAUNCHED, not completed -- there is no exit
    // code because nothing has exited. `ok` would claim a result for work
    // that is still running.
    const v = classifyDsh("bash", { isError: false, value: { kind: "background", jobId: "j1" } });
    assert.equal(v.outcome, "unknown");
    assert.equal(v.error_class, "StillRunning");
  });

  it("calls an aborted command unknown, not failed", () => {
    // Cancelled by the caller. The command never got to say what it would
    // have done, and that is not the agent making a mistake.
    const v = classifyDsh("bash", fg({ aborted: true, exitCode: null }));
    assert.equal(v.outcome, "unknown");
    assert.equal(v.error_class, "Aborted");
  });

  it("scores a timeout and a signal kill as failures", () => {
    assert.equal(classifyDsh("bash", fg({ timedOut: true, exitCode: null })).error_class, "TimedOut");
    const killed = classifyDsh("bash", fg({ exitCode: null, signal: "SIGKILL" }));
    assert.equal(killed.outcome, "failed");
    assert.equal(killed.error_class, "Signal:SIGKILL");
  });

  it("checks the signal before treating a null exit as unreadable", () => {
    // `exitCode` is null exactly when a signal killed it, so order matters:
    // reversed, every signal kill would grade `unknown`.
    assert.equal(classifyDsh("bash", fg({ exitCode: null, signal: "SIGTERM" })).outcome, "failed");
    assert.equal(classifyDsh("bash", fg({ exitCode: null, signal: null })).outcome, "unknown");
  });

  it("refuses to grade a shell whose exit it cannot read", () => {
    const v = classifyDsh("bash", { isError: false, value: { kind: "foreground" } });
    assert.equal(v.outcome, "unknown");
    assert.equal(v.error_class, "ExitStatusUnreadable");
  });

  it("treats a missing isError as unknown rather than as false", () => {
    assert.equal(classifyDsh("read", { value: "x" }).outcome, "unknown");
  });

  it("grades dsh's own verdict tools as ok, and unproven ones as unknown", () => {
    for (const t of DSH_VERDICT_TOOLS) {
      assert.equal(classifyDsh(t, { isError: false, value: "x" }).outcome, "ok", t);
    }
    // A web search returning zero results is not a failure, and a tool that
    // reports its own failure inside a successful value is invisible here.
    for (const t of ["web_search", "web_fetch", "terminal_send", "mcp__x__y"]) {
      assert.equal(classifyDsh(t, { isError: false, value: {} }).outcome, "unknown", t);
    }
  });

  it("never emits a status_code", () => {
    for (const r of [fg({ exitCode: 127 }), { isError: true, error: {} },
                     { isError: false, value: "x" }]) {
      assert.equal(classifyDsh("bash", r).status_code, null);
    }
  });

  it("tracks the harness's own exec tool names", () => {
    assert.deepEqual([...DSH_EXEC_TOOLS].sort(), ["bash", "pwsh"]);
  });
});

describe("the record built from one execution", () => {
  const EXEC = {
    name: "bash", arguments: { command: "exit 3" },
    callId: "call-1", agent: { id: "sess-7" },
  };

  it("carries the outcome, arguments, agent and call id", () => {
    const r = recordFor(EXEC, fg({ exitCode: 3 }));
    assert.equal(r.name, "bash");
    assert.deepEqual(r.arguments, { command: "exit 3" });
    assert.equal(r.outcome, "failed");
    assert.equal(r.agent_name, "sess-7");
    assert.equal(r.tool_call_id, "call-1");
    assert.equal(r.status_code, null);
  });

  it("resolves identity from the canonical value and drops the value", () => {
    // dsh documents `value` as execution-local and "deliberately omitted from
    // durable events" -- it exists HERE and nowhere later. Identity is taken
    // and the value itself never reaches the trajectory.
    const r = recordFor(
      { ...EXEC, name: "write" },
      { isError: false, value: { id: "10001", key: "ENG-7", secret: "sk-live" } },
    );
    assert.deepEqual(r.entity_ids, { id: "10001", key: "ENG-7" });
    assert.equal(JSON.stringify(r).includes("sk-live"), false);
  });

  it("refuses to record an unnamed execution", () => {
    assert.equal(recordFor({ arguments: {} }, fg({ exitCode: 0 })), null);
    assert.equal(recordFor(null, null), null);
  });

  it("omits attribution rather than inventing it", () => {
    const r = recordFor({ name: "read", arguments: {} }, { isError: false, value: "x" });
    assert.equal("agent_name" in r, false);
    assert.equal("tool_call_id" in r, false);
  });

  it("normalises non-object arguments to an empty object", () => {
    const r = recordFor({ name: "read", arguments: "not-an-object" },
                        { isError: false, value: "x" });
    assert.deepEqual(r.arguments, {});
  });
});

describe("the plugin as Cordis loads it", () => {
  it("exports no default, or Cordis would load the wrong thing", async () => {
    // The loader normalizes with `exports.default ?? exports`. A default here
    // would be handed to the harness INSTEAD of `apply`, and the plugin would
    // fail to load. The OpenClaw entry is a separate module for this reason.
    const mod = await import("../src/dsh.js");
    assert.equal(mod.default, undefined);
    assert.equal(typeof mod.apply, "function");
    assert.equal(mod.name, "percepteye-agent-flywheel");
  });

  it("subscribes to the observer hook and only that", () => {
    const seen = [];
    apply({ on: (hook) => { seen.push(hook); return () => {}; } });
    // NOT tools/execute, tools/post or tools/code-dispatch-log -- every one of
    // those is a waterfall that can change what the model sees.
    assert.deepEqual(seen, ["tools/result"]);
    assert.equal(HOOK_NAME, "tools/result");
  });

  it("returns undefined from the listener, as @mode emit requires", () => {
    let handler;
    apply({ on: (_h, fn) => { handler = fn; return () => {}; } });
    assert.equal(handler({ name: "read", arguments: {} }, { isError: false }), undefined);
  });

  it("does not throw out of the listener on a malformed execution", () => {
    let handler;
    apply({ on: (_h, fn) => { handler = fn; return () => {}; } });
    assert.doesNotThrow(() => handler(undefined, undefined));
    assert.doesNotThrow(() => handler({ get name() { throw new Error("x"); } }, {}));
  });

  it("survives a context without an event surface", () => {
    assert.equal(apply({}), null);
    assert.equal(apply(null), null);
  });

  it("writes a record end to end, and stays inert without a destination", () => {
    const saved = process.env[TRAJECTORY_DIR_ENV];
    const dir = mkdtempSync(join(tmpdir(), "pe-dsh-"));
    try {
      delete process.env[TRAJECTORY_DIR_ENV];
      let handler;
      apply({ on: (_h, fn) => { handler = fn; return () => {}; } });
      handler({ name: "read", arguments: {}, callId: "c0" }, { isError: false, value: "x" });

      process.env[TRAJECTORY_DIR_ENV] = dir;
      handler({ name: "bash", arguments: { command: "exit 3" }, callId: "c1" },
              fg({ exitCode: 3 }));

      const lines = readFileSync(join(dir, "tool_calls.jsonl"), "utf8")
        .split("\n").filter(Boolean).map(JSON.parse);
      // Only the second: the first ran with no destination configured.
      assert.equal(lines.length, 1);
      assert.equal(lines[0].name, "bash");
      assert.equal(lines[0].outcome, "failed");
    } finally {
      if (saved === undefined) delete process.env[TRAJECTORY_DIR_ENV];
      else process.env[TRAJECTORY_DIR_ENV] = saved;
    }
  });

  it("ships the bundle patch that joins the dsh layer stack", () => {
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    assert.equal(pkg.dsh.bundle.patch, "./cordis.patch.yml");
    assert.ok(pkg.files.includes("cordis.patch.yml"), "the patch must be published");
    assert.ok(existsSync(new URL("../cordis.patch.yml", import.meta.url)));
    // The patch names the SUBPATH, because the bare package resolves to the
    // OpenClaw entry and its default export.
    const patch = readFileSync(new URL("../cordis.patch.yml", import.meta.url), "utf8");
    assert.ok(patch.includes("percepteye-agent-flywheel/dsh"), patch);
    assert.equal(pkg.exports["./dsh"], "./src/dsh.js");
  });

  it("keeps the two harness entries apart", () => {
    // One package, two entry modules. The OpenClaw side needs a default and
    // the Cordis side must not have one; sharing a module is impossible.
    const pkg = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
    assert.notEqual(pkg.openclaw.extensions[0], pkg.exports["./dsh"]);
    assert.equal(name, "percepteye-agent-flywheel");
  });
});
