import assert from "node:assert/strict";
import { mkdtempSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, it } from "node:test";
import { createRecorder } from "../src/record.js";
import { createWriter, TRAJECTORY_DIR_ENV } from "../src/trajectory.js";
import plugin, { HOOK_NAME } from "../src/index.js";

function scratch() {
  return mkdtempSync(join(tmpdir(), "pe-oc-"));
}

function linesIn(dir) {
  const path = join(dir, "tool_calls.jsonl");
  if (!existsSync(path)) return null;
  return readFileSync(path, "utf8").split("\n").filter(Boolean).map(JSON.parse);
}

const EVENT = {
  toolName: "bash",
  params: { command: "pytest -q" },
  toolCallId: "call_abc123",
  result: { meta: { exitCode: 1 } },
  durationMs: 812,
};

describe("one event becomes one record", () => {
  it("writes the outcome, the arguments and the join key", () => {
    const dir = scratch();
    const rec = createRecorder({ writer: createWriter({ dir }) });

    assert.equal(rec.handle(EVENT, { agentId: "coder" }), true);

    const [line] = linesIn(dir);
    assert.equal(line.name, "bash");
    assert.deepEqual(line.arguments, { command: "pytest -q" });
    assert.equal(line.outcome, "failed");
    assert.equal(line.error_class, "ExitStatus:1");
    assert.equal(line.tool_call_id, "call_abc123");
    assert.equal(line.agent_name, "coder");
    assert.equal(line.latency_ms, 812);
    assert.equal(line.status_code, null);
  });

  it("appends rather than overwriting", () => {
    const dir = scratch();
    const rec = createRecorder({ writer: createWriter({ dir }) });
    rec.handle({ toolName: "read", params: {} });
    rec.handle({ toolName: "write", params: {} });
    assert.deepEqual(linesIn(dir).map((l) => l.name), ["read", "write"]);
    assert.equal(rec.written, 2);
  });

  it("records an unnamed call not at all, rather than badly", () => {
    // The contract rejects an empty name outright, and nothing useful can be
    // said about a call we cannot attribute to a tool.
    const dir = scratch();
    const rec = createRecorder({ writer: createWriter({ dir }) });
    assert.equal(rec.handle({ params: {} }), false);
    assert.equal(rec.handle(null), false);
    assert.equal(linesIn(dir), null);
  });

  it("omits latency and attribution rather than inventing them", () => {
    const dir = scratch();
    const rec = createRecorder({ writer: createWriter({ dir }) });
    rec.handle({ toolName: "read", params: {} }, {});
    const [line] = linesIn(dir);
    assert.equal("latency_ms" in line, false);
    assert.equal("agent_name" in line, false);
    assert.equal("tool_call_id" in line, false);
  });

  it("normalises missing arguments to an empty object", () => {
    const dir = scratch();
    const rec = createRecorder({ writer: createWriter({ dir }) });
    rec.handle({ toolName: "read" });
    assert.deepEqual(linesIn(dir)[0].arguments, {});
  });
});

describe("staying out of the way", () => {
  const saved = process.env[TRAJECTORY_DIR_ENV];
  afterEach(() => {
    if (saved === undefined) delete process.env[TRAJECTORY_DIR_ENV];
    else process.env[TRAJECTORY_DIR_ENV] = saved;
  });

  it("records nothing at all when no trajectory directory is set", () => {
    // The production case. The same install must be a no-op outside a rollout,
    // which is what lets a customer leave the plugin enabled permanently.
    delete process.env[TRAJECTORY_DIR_ENV];
    const writer = createWriter();
    assert.equal(writer.active, false);
    assert.equal(createRecorder({ writer }).handle(EVENT), false);
  });

  it("starts recording if the directory appears later", () => {
    // Read per call, not cached at construction: the Gateway may be started
    // before the rollout hands it a destination.
    const dir = scratch();
    delete process.env[TRAJECTORY_DIR_ENV];
    const rec = createRecorder({ writer: createWriter() });
    assert.equal(rec.handle(EVENT), false);
    process.env[TRAJECTORY_DIR_ENV] = dir;
    assert.equal(rec.handle(EVENT), true);
    assert.equal(linesIn(dir).length, 1);
  });

  it("survives an unwritable destination without throwing", () => {
    const writer = createWriter({ dir: "/proc/nonexistent-percepteye" });
    let warned = 0;
    const w2 = createWriter({
      dir: "/proc/nonexistent-percepteye",
      onError: () => { warned += 1; },
    });
    assert.doesNotThrow(() => writer.write({ name: "x" }));
    w2.write({ name: "x" });
    w2.write({ name: "x" });
    // Reported once, then quiet: a per-call warning would bury the agent's
    // own output on a full disk.
    assert.equal(warned, 1);
  });
});

describe("the plugin registers the observer hook and only that", () => {
  it("subscribes to after_tool_call", () => {
    const seen = [];
    plugin.register({ on: (name) => seen.push(name) });
    assert.deepEqual(seen, [HOOK_NAME]);
  });

  it("registers no tool-result middleware", () => {
    // The middleware seam can REPLACE a tool result. A recorder must never
    // hold it -- this asserts the boundary rather than trusting review.
    let middleware = 0;
    plugin.register({
      on: () => {},
      registerAgentToolResultMiddleware: () => { middleware += 1; },
    });
    assert.equal(middleware, 0);
  });

  it("declares no contracts, because an observer claims no seam", () => {
    const manifest = JSON.parse(
      readFileSync(new URL("../openclaw.plugin.json", import.meta.url), "utf8"),
    );
    assert.equal("contracts" in manifest, false);
    assert.equal(manifest.id, plugin.id);
    assert.equal(manifest.activation.onStartup, true);
  });

  it("does not throw out of the handler on a malformed event", () => {
    let handler = null;
    plugin.register({ on: (_n, h) => { handler = h; } });
    assert.doesNotThrow(() => handler(undefined, undefined));
    assert.doesNotThrow(() => handler({ toolName: "x", get params() { throw new Error("boom"); } }, {}));
  });
});
