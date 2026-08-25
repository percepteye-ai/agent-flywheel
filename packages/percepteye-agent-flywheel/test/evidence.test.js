/**
 * Identity evidence, and its agreement with the Python half.
 *
 * These two extractors are one contract in two languages. If they disagree on
 * what counts as identity, the same write produces different evidence
 * depending on which harness ran it -- and the divergence surfaces months
 * later as a read-back confirming the wrong entity.
 */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { IDENTITY_FIELDS, WEAK_ID_FIELDS, extractEntityIds } from "../src/evidence.js";
import { createRecorder } from "../src/record.js";
import { createWriter } from "../src/trajectory.js";

describe("extracting the identity a write produced", () => {
  it("files each value under the field it came from", () => {
    // Not a flat set: per-field filing is what lets a read-back compare a
    // candidate's `key` to the write's `key` and never to the write's `id`.
    assert.deepEqual(extractEntityIds({ id: "10001", key: "PROJ-42" }),
      { id: "10001", key: "PROJ-42" });
  });

  it("parses a JSON string body", () => {
    assert.deepEqual(extractEntityIds('{"id": 7}'), { id: "7" });
  });

  it("finds an enveloped identity", () => {
    assert.deepEqual(extractEntityIds({ data: { uuid: "abc-123" } }),
      { uuid: "abc-123" });
  });

  it("yields nothing for prose rather than guessing", () => {
    assert.deepEqual(extractEntityIds("created ticket PROJ-42"), {});
  });

  it("takes the shallowest occurrence", () => {
    assert.deepEqual(extractEntityIds({ id: "a", items: [{ id: "b" }] }),
      { id: "a" });
  });

  it("drops an ambiguous field instead of picking between", () => {
    assert.deepEqual(extractEntityIds({ xs: [{ id: "a" }, { id: "b" }] }), {});
  });

  it("refuses every low-entropy field", () => {
    // A created resource routinely echoes a human name a DIFFERENT entity
    // shares by coincidence. Matching on it confirms the wrong resource.
    for (const weak of WEAK_ID_FIELDS) {
      assert.deepEqual(extractEntityIds({ [weak]: "Story 1" }), {}, weak);
    }
  });

  it("does not treat a boolean as an identity", () => {
    assert.deepEqual(extractEntityIds({ id: true }), {});
  });

  it("does not reach a deeply buried id", () => {
    assert.deepEqual(extractEntityIds({ a: { b: { c: { d: { id: "x" } } } } }), {});
  });

  it("never throws", () => {
    const hostile = { get id() { throw new Error("boom"); } };
    for (const body of [hostile, null, undefined, 3, Symbol("s")]) {
      assert.equal(typeof extractEntityIds(body), "object");
    }
  });

  it("agrees with the Python field set", () => {
    // Mirrors percepteye_agent_flywheel.evidence.IDENTITY_FIELDS exactly.
    assert.deepEqual([...IDENTITY_FIELDS].sort(),
      ["gid", "id", "key", "sid", "uuid"]);
  });
});

describe("the recorder carries identity, not the body", () => {
  const dir = () => mkdtempSync(join(tmpdir(), "pe-ev-"));
  const lineIn = (d) =>
    JSON.parse(readFileSync(join(d, "tool_calls.jsonl"), "utf8").trim());

  it("records the identity a write produced", () => {
    const d = dir();
    createRecorder({ writer: createWriter({ dir: d }) }).handle({
      toolName: "exec", params: { command: "create" },
      toolCallId: "c1", result: { id: "10001", key: "ENG-7" },
    });
    assert.deepEqual(lineIn(d).entity_ids, { id: "10001", key: "ENG-7" });
  });

  it("never writes the response body", () => {
    // The whole reason identity is resolved at capture: the body is customer
    // production data and must not land in the trajectory.
    const d = dir();
    createRecorder({ writer: createWriter({ dir: d }) }).handle({
      toolName: "exec", params: {},
      result: { id: "1", secret: "sk-live-do-not-store", pii: "a@b.com" },
    });
    const raw = readFileSync(join(d, "tool_calls.jsonl"), "utf8");
    assert.equal(raw.includes("sk-live-do-not-store"), false);
    assert.equal(raw.includes("a@b.com"), false);
    assert.equal(lineIn(d).entity_ids.id, "1");
  });

  it("keeps identity on an unknown outcome", () => {
    // The point of the feature. An identity claims nothing about whether the
    // call worked -- it names an entity so something else can observe one, and
    // that is the only evidence able to settle an `unknown` honestly.
    const d = dir();
    createRecorder({ writer: createWriter({ dir: d }) }).handle({
      toolName: "mcp__jira__create", params: {},
      result: { id: "10001" },
    });
    const line = lineIn(d);
    assert.equal(line.outcome, "unknown");
    assert.deepEqual(line.entity_ids, { id: "10001" });
  });

  it("omits the field entirely when there is no identity", () => {
    const d = dir();
    createRecorder({ writer: createWriter({ dir: d }) }).handle({
      toolName: "read", params: {}, result: "file contents",
    });
    assert.equal("entity_ids" in lineIn(d), false);
  });
});
