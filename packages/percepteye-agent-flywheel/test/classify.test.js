import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { classify, readExit } from "../src/classify.js";

describe("the tri-state is not 'no error field means success'", () => {
  it("believes a positive failure assertion", () => {
    const v = classify("read", "ENOENT: no such file", undefined);
    assert.equal(v.outcome, "failed");
    assert.equal(v.error_class, "ToolError");
    assert.match(v.error, /ENOENT/);
  });

  it("does not treat an empty error string as a failure", () => {
    // The host omits `error` when falsy, so "" never signals a real failure.
    assert.equal(classify("read", "", "contents").outcome, "ok");
    assert.equal(classify("read", "   ", "contents").outcome, "ok");
  });

  it("scores a non-zero exit as failed even with no error field", () => {
    // The case this module exists for: `bash` exits 1, the tool call itself
    // "succeeded" in the host's terms and sets no error, and reading absence
    // as success would train on a failed command as a win.
    const v = classify("bash", undefined, { meta: { exitCode: 1 } });
    assert.equal(v.outcome, "failed");
    assert.equal(v.error_class, "ExitStatus:1");
  });

  it("scores a zero exit as ok", () => {
    assert.equal(classify("bash", undefined, { meta: { exitCode: 0 } }).outcome, "ok");
  });

  it("refuses to grade a shell whose exit it cannot read", () => {
    const v = classify("bash", undefined, { stdout: "hello" });
    assert.equal(v.outcome, "unknown");
    assert.equal(v.error_class, "ExitStatusUnreadable");
  });

  it("grades an unproven tool as unknown rather than ok", () => {
    // An MCP server returning {"error": ...} as ordinary content is invisible
    // from here. `unknown` is honest; `ok` would be a fabrication.
    assert.equal(
      classify("mcp__jira__transition", undefined, '{"ok":true}').outcome,
      "unknown",
    );
  });

  it("grades a proven tool as ok", () => {
    assert.equal(classify("read", undefined, "contents").outcome, "ok");
  });

  it("lets an explicit error beat an exit code that says otherwise", () => {
    const v = classify("bash", "killed by deadline", { meta: { exitCode: 0 } });
    assert.equal(v.outcome, "failed");
  });

  it("never emits a status_code, because there is no status to emit", () => {
    // An exit code in status_code would fail the contract's 100..599 check --
    // and would be a lie besides. 127 is not "Bad Request".
    for (const c of [
      classify("bash", undefined, { meta: { exitCode: 127 } }),
      classify("read", "x", undefined),
      classify("read", undefined, "x"),
      classify("zzz", undefined, "x"),
    ]) assert.equal(c.status_code, null);
  });
});

describe("reading an exit status", () => {
  it("accepts the shapes the host uses", () => {
    assert.equal(readExit({ meta: { exitCode: 2 } }), 2);
    assert.equal(readExit({ exitCode: 0 }), 0);
    assert.equal(readExit({ meta: { exit_code: "3" } }), 3);
    assert.equal(readExit({ details: { status: "error" } }), 1);
    assert.equal(readExit({ meta: { status: "ok" } }), 0);
  });

  it("returns undefined rather than a guess", () => {
    assert.equal(readExit({ stdout: "hi" }), undefined);
    assert.equal(readExit("plain string"), undefined);
    assert.equal(readExit(null), undefined);
    assert.equal(readExit({ meta: { exitCode: "not a number" } }), undefined);
  });
});
