/**
 * The consent boundary, and the reader's two load-bearing distinctions.
 */
import { strict as assert } from "node:assert";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { resolveConfig } from "../src/config.js";
import { readTrajectory, TOOL_CALLS_FILENAME } from "../src/trajectory.js";

const dirWith = (lines) => {
  const d = mkdtempSync(join(tmpdir(), "pe-read-"));
  writeFileSync(join(d, TOOL_CALLS_FILENAME), lines.join("\n"));
  return d;
};
const ok = (name) => JSON.stringify({ name, outcome: "ok" });

describe("no key means nothing leaves the machine", () => {
  it("is not contributing without a key", () => {
    // The ordinary first install. Local capture still runs; the network does
    // not. A customer with no key is not misconfigured.
    assert.equal(resolveConfig({}, {}).contributing, false);
  });

  it("contributes once a key resolves, from config OR env", () => {
    assert.equal(resolveConfig({ apiKey: "pefw_a" }, {}).contributing, true);
    assert.equal(resolveConfig({}, { PERCEPTEYE_API_KEY: "pefw_b" }).contributing, true);
  });

  it("prefers config over env, because config is hot-reloaded", () => {
    const c = resolveConfig({ apiKey: "from-config" }, { PERCEPTEYE_API_KEY: "from-env" });
    assert.equal(c.apiKey, "from-config");
  });

  it("treats a blank key as no key", () => {
    assert.equal(resolveConfig({ apiKey: "   " }, {}).contributing, false);
  });

  it("capture is ON unless deliberately disabled", () => {
    // A typo must not silently switch capture off and leave the operator
    // believing it is on.
    assert.equal(resolveConfig({}, {}).captureEnabled, true);
    assert.equal(resolveConfig({}, { PERCEPTEYE_CAPTURE: "typo" }).captureEnabled, true);
    assert.equal(resolveConfig({}, { PERCEPTEYE_CAPTURE: "0" }).captureEnabled, false);
    assert.equal(resolveConfig({ capture: "off" }, {}).captureEnabled, false);
  });
});

describe("absent is not empty", () => {
  it("a missing trajectory reads as null, not []", () => {
    // null means "there was no trajectory"; [] means "I read one and it
    // recorded zero calls". Reporting [] for a missing file would assert the
    // agent made no tool calls when in fact nothing was observed.
    assert.deepEqual(readTrajectory("/nonexistent/dir"), { calls: null, malformed: 0 });
    assert.deepEqual(readTrajectory(null), { calls: null, malformed: 0 });
  });

  it("an empty trajectory file reads as [], a positive assertion", () => {
    assert.deepEqual(readTrajectory(dirWith([])), { calls: [], malformed: 0 });
  });

  it("reads well-formed rows", () => {
    const r = readTrajectory(dirWith([ok("shell"), ok("edit")]));
    assert.equal(r.calls.length, 2);
    assert.equal(r.malformed, 0);
  });
});

describe("corruption is counted, never dropped", () => {
  it("counts unparseable lines instead of throwing", () => {
    const r = readTrajectory(dirWith([ok("a"), "{not json", ok("b")]));
    assert.equal(r.calls.length, 2);
    assert.equal(r.malformed, 1);
  });

  it("counts a row missing the required fields as malformed", () => {
    // The Python reader builds each row with REQUIRED subscripts and counts
    // anything else as malformed; that tally forbids the training grade, so
    // silently skipping such a row would let a partly-unreadable trajectory be
    // graded as a complete one.
    const r = readTrajectory(dirWith([ok("a"), JSON.stringify({ note: "hi" })]));
    assert.equal(r.calls.length, 1);
    assert.equal(r.malformed, 1);
  });

  it("ignores blank lines without counting them as corruption", () => {
    const r = readTrajectory(dirWith([ok("a"), "", "  ", ok("b")]));
    assert.equal(r.calls.length, 2);
    assert.equal(r.malformed, 0);
  });
});
