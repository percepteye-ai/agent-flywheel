import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";

import {
  buildRecord,
  createDescriber,
  normalizeTool,
  normalizeTools,
  DISCOVERED_AGENT_FILENAME,
  FRAMEWORK,
  SOURCE_TYPE,
} from "../src/describe.js";
import plugin, { DESCRIBE_HOOK_NAME } from "../src/index.js";

function scratch() {
  return mkdtempSync(join(tmpdir(), "pe-desc-"));
}

function recordIn(dir) {
  const path = join(dir, DISCOVERED_AGENT_FILENAME);
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, "utf8"));
}

/** An `llm_input` event in the shape the host actually emits. */
const EVENT = {
  runId: "run_1",
  sessionId: "sess_1",
  provider: "anthropic",
  model: "claude-opus-5",
  systemPrompt: "You are a triage agent.",
  prompt: "why is bgp down",
  historyMessages: [],
  imagesCount: 0,
  tools: [
    { name: "glob", label: "Glob", description: "Find files", parameters: { type: "object", properties: { pattern: { type: "string" } } } },
    { name: "read", description: "Read a file", parameters: { type: "object", properties: {} } },
  ],
};

// ── the translation that decides whether the catalogue survives ────────────
//
// The consumer reads `input_schema` and drops any tool whose `name` is falsy. These
// tests assert the SHAPE THE CONSUMER READS, not that some normalisation ran.

describe("tool normalisation meets the consumer's exact shape", () => {
  it("renames OpenClaw's `parameters` to `input_schema`", () => {
    const t = normalizeTool({
      name: "glob",
      description: "Find files",
      parameters: { type: "object", properties: { pattern: { type: "string" } } },
    });
    assert.equal(t.name, "glob");
    assert.deepEqual(t.input_schema, {
      type: "object",
      properties: { pattern: { type: "string" } },
    });
    // The key the consumer does NOT read must not be the only place it lives.
    assert.equal(t.parameters, undefined);
    assert.equal(t.original_format, "openclaw_tool");
  });

  it("unwraps the OpenAI function envelope, which otherwise yields no name", () => {
    const t = normalizeTool({
      type: "function",
      function: {
        name: "run_command",
        description: "Run a shell command",
        parameters: { type: "object", properties: { cmd: { type: "string" } } },
      },
    });
    // Forwarded verbatim this tool has NO top-level name, and it is dropped.
    assert.equal(t.name, "run_command");
    assert.deepEqual(t.input_schema.properties, { cmd: { type: "string" } });
    assert.equal(t.original_format, "openai_function");
  });

  it("accepts `inputSchema` and `input_schema` spellings too", () => {
    assert.deepEqual(
      normalizeTool({ name: "a", inputSchema: { type: "object", properties: { x: {} } } }).input_schema,
      { type: "object", properties: { x: {} } },
    );
    assert.deepEqual(
      normalizeTool({ name: "b", input_schema: { type: "object", properties: { y: {} } } }).input_schema,
      { type: "object", properties: { y: {} } },
    );
  });

  it("substitutes the empty object schema rather than emitting a falsy one", () => {
    assert.deepEqual(normalizeTool({ name: "bare" }).input_schema, {
      type: "object",
      properties: {},
    });
  });

  it("drops a nameless tool, so the count we report is the count that survives", () => {
    assert.equal(normalizeTool({ description: "no name" }), null);
    assert.equal(normalizeTool({ name: "   " }), null);
    assert.equal(normalizeTool("not an object"), null);
    assert.deepEqual(
      normalizeTools([{ name: "keep" }, { description: "drop" }]).map((t) => t.name),
      ["keep"],
    );
  });

  it("falls back to `label` for a description but never for a name", () => {
    assert.equal(normalizeTool({ name: "glob", label: "Glob" }).description, "Glob");
  });
});

// ── the absence contract, across a process boundary ────────────────────────

describe("null tools means NOT OBSERVED, [] means observed and empty", () => {
  it("emits null when the host sent no tools key at all", () => {
    assert.equal(normalizeTools(undefined), null);
    const rec = buildRecord({ systemPrompt: "p" }, {});
    assert.equal(rec.tool_definitions, null);
    assert.ok(rec.incomplete.includes("tool_definitions"));
  });

  it("emits [] when the host sent an empty list", () => {
    assert.deepEqual(normalizeTools([]), []);
    const rec = buildRecord({ tools: [], systemPrompt: "p" }, {});
    assert.deepEqual(rec.tool_definitions, []);
    assert.ok(!(rec.incomplete ?? []).includes("tool_definitions"));
  });
});

// ── the wire record ────────────────────────────────────────────────────────

describe("the record matches DiscoveredAgent.to_wire()", () => {
  it("carries prompt, tools, model config and identity", () => {
    const rec = buildRecord(EVENT, { agentId: "triage-agent" });
    assert.equal(rec.agent_name, "triage-agent");
    assert.equal(rec.framework, FRAMEWORK);
    assert.equal(rec.source_type, SOURCE_TYPE);
    assert.equal(rec.role, "standalone");
    assert.equal(rec.system_prompt, EVENT.systemPrompt);
    assert.deepEqual(rec.tool_definitions.map((t) => t.name), ["glob", "read"]);
    assert.deepEqual(rec.discovered_model_config, {
      provider: "anthropic",
      model_name: "claude-opus-5",
    });
  });

  it("keeps framework and source_type inside the varchar(64) columns", () => {
    const rec = buildRecord(EVENT, {});
    assert.ok(rec.framework.length <= 64);
    assert.ok(rec.source_type.length <= 64);
    // `framework` is a LOOKUP KEY: the consumer maps exactly this spelling.
    assert.equal(rec.framework, "openclaw");
  });

  it("names what it could not read", () => {
    const rec = buildRecord({ tools: [{ name: "a" }] }, {});
    assert.deepEqual(rec.incomplete, ["system_prompt"]);
  });
});

// ── Postgres JSONB will refuse these, so we must not send them ─────────────

describe("scrubbing what JSONB cannot store", () => {
  it("strips NUL bytes from prompt, tool names and nested schemas", () => {
    const NUL = String.fromCharCode(0);
    const rec = buildRecord(
      {
        systemPrompt: `soul${NUL}text`,
        tools: [{ name: `gl${NUL}ob`, description: `d${NUL}`, parameters: { type: `obj${NUL}ect` } }],
      },
      { agentId: `ag${NUL}ent` },
    );
    const blob = JSON.stringify(rec);
    assert.ok(!blob.includes("\\u0000"), "a NUL reached the wire record");
    assert.equal(rec.system_prompt, "soultext");
    assert.equal(rec.tool_definitions[0].name, "glob");
    assert.equal(rec.tool_definitions[0].input_schema.type, "object");
  });

  it("strips a LONE surrogate but keeps a valid pair", () => {
    const lone = String.fromCharCode(0xd800);
    const pair = "\u{1F600}"; // a real surrogate pair
    const rec = buildRecord({ systemPrompt: `a${lone}b${pair}` }, {});
    assert.equal(rec.system_prompt, `ab${pair}`);
  });

  it("does not eat ordinary whitespace", () => {
    const rec = buildRecord({ systemPrompt: "you are   a triage agent\n\ttabbed" }, {});
    assert.equal(rec.system_prompt, "you are   a triage agent\n\ttabbed");
  });
});

// ── the write policy ───────────────────────────────────────────────────────

describe("the richest catalogue wins, not the newest", () => {
  it("writes the first event even when the agent has no tools", () => {
    const dir = scratch();
    const d = createDescriber({ dir });
    assert.equal(d.handle({ tools: [], systemPrompt: "p" }, {}), true);
    assert.deepEqual(recordIn(dir).tool_definitions, []);
  });

  it("does NOT let a narrow sub-agent turn overwrite the real catalogue", () => {
    const dir = scratch();
    const d = createDescriber({ dir });

    assert.equal(d.handle(EVENT, { agentId: "a" }), true);
    assert.deepEqual(recordIn(dir).tool_definitions.map((t) => t.name), ["glob", "read"]);

    // A later turn carrying a SUBSET must not shrink what we reported.
    assert.equal(d.handle({ ...EVENT, tools: [{ name: "read" }] }, { agentId: "a" }), false);
    assert.deepEqual(recordIn(dir).tool_definitions.map((t) => t.name), ["glob", "read"]);
  });

  it("upgrades when a later turn carries more tools", () => {
    const dir = scratch();
    const d = createDescriber({ dir });
    assert.equal(d.handle({ ...EVENT, tools: [{ name: "read" }] }, {}), true);
    assert.equal(d.handle(EVENT, {}), true);
    assert.deepEqual(recordIn(dir).tool_definitions.map((t) => t.name), ["glob", "read"]);
  });

  it("is inert with no trajectory dir, and writes no stray file", () => {
    const d = createDescriber({ dir: null });
    const saved = process.env.PERCEPTEYE_TRAJECTORY_DIR;
    delete process.env.PERCEPTEYE_TRAJECTORY_DIR;
    try {
      assert.equal(d.active, false);
      assert.equal(d.handle(EVENT, {}), false);
    } finally {
      if (saved !== undefined) process.env.PERCEPTEYE_TRAJECTORY_DIR = saved;
    }
  });
});

// ── the wiring, not just the function ──────────────────────────────────────
//
// The recurring failure in this workspace is a correct function that nothing
// calls. Assert the plugin actually subscribes the hook.

describe("the plugin registers the describer", () => {
  it("subscribes llm_input and writes on it", () => {
    const dir = scratch();
    const handlers = new Map();
    const api = { on: (name, fn) => handlers.set(name, fn), logger: { warn() {} } };

    plugin.register(api);

    assert.ok(handlers.has(DESCRIBE_HOOK_NAME), "llm_input was never subscribed");
    assert.ok(handlers.has("after_tool_call"), "the recorder hook was lost");

    const saved = process.env.PERCEPTEYE_TRAJECTORY_DIR;
    process.env.PERCEPTEYE_TRAJECTORY_DIR = dir;
    try {
      handlers.get(DESCRIBE_HOOK_NAME)(EVENT, { agentId: "triage-agent" });
      const rec = recordIn(dir);
      assert.ok(rec, "no discovered_agent.json was written");
      assert.deepEqual(rec.tool_definitions.map((t) => t.name), ["glob", "read"]);
    } finally {
      if (saved === undefined) delete process.env.PERCEPTEYE_TRAJECTORY_DIR;
      else process.env.PERCEPTEYE_TRAJECTORY_DIR = saved;
    }
  });

  it("a malformed event costs the description, never the run", () => {
    const handlers = new Map();
    plugin.register({ on: (n, f) => handlers.set(n, f), logger: { warn() {} } });
    assert.doesNotThrow(() => handlers.get(DESCRIBE_HOOK_NAME)(null, null));
  });
});
