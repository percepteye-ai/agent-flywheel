/**
 * Payloads captured from a real OpenClaw run, not hand-written.
 *
 * Both were dumped out of `after_tool_call` while OpenClaw 2026.7.1-2 executed
 * a shell command end to end. They are the contract with the host, so they are
 * checked in verbatim: if a future OpenClaw changes the shape, these fail and
 * someone re-reads the payload instead of discovering it from bad training
 * data months later.
 */
import assert from "node:assert/strict";
import { mkdtempSync, readFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { classify, readExit } from "../src/classify.js";
import { createRecorder } from "../src/record.js";
import { createWriter } from "../src/trajectory.js";

/** `echo hello-percepteye` — exit 0, no error field. */
const SUCCEEDED = {
  toolName: "exec",
  params: { command: "echo hello-percepteye" },
  runId: "983ab2ff-e307-4d00-a5e8-3ef1c35cafcd",
  toolCallId: "call_percepteye_1",
  result: {
    content: [{ type: "text", text: "hello-percepteye" }],
    details: {
      status: "completed", exitCode: 0, exitSignal: null, exitReason: "exit",
      durationMs: 16, aggregated: "hello-percepteye", noOutputTimedOut: false,
    },
  },
  durationMs: 93,
};

/** `exit 3` — exit 3, and note `status` still reads "completed". */
const FAILED = {
  toolName: "exec",
  params: { command: "exit 3" },
  runId: "bbed93b4-5a91-414b-95b6-c1776a2d27a4",
  toolCallId: "call_percepteye_1",
  result: {
    content: [{ type: "text", text: "\n\n(Command exited with code 3)" }],
    details: {
      status: "completed", exitCode: 3, exitSignal: null, exitReason: "exit",
      durationMs: 14, aggregated: "\n\n(Command exited with code 3)",
      noOutputTimedOut: false,
    },
  },
  error: "(Command exited with code 3)",
  durationMs: 91,
};

const CTX = {
  toolName: "exec", agentId: "main",
  sessionKey: "agent:main:cli:raw-ok2",
  sessionId: "c99ef80c-72d4-4b6b-b3a1-5256a13faee7",
  runId: "983ab2ff-e307-4d00-a5e8-3ef1c35cafcd",
  toolCallId: "call_percepteye_1",
};

function record(event, ctx = CTX) {
  const dir = mkdtempSync(join(tmpdir(), "pe-fx-"));
  createRecorder({ writer: createWriter({ dir }) }).handle(event, ctx);
  return JSON.parse(
    readFileSync(join(dir, "tool_calls.jsonl"), "utf8").trim(),
  );
}

describe("real host payloads", () => {
  it("grades the command that succeeded as ok", () => {
    const line = record(SUCCEEDED);
    assert.equal(line.outcome, "ok");
    assert.equal(line.name, "exec");
    assert.deepEqual(line.arguments, { command: "echo hello-percepteye" });
    assert.equal(line.tool_call_id, "call_percepteye_1");
    assert.equal(line.agent_name, "main");
    assert.equal(line.latency_ms, 93);
  });

  it("grades the command that exited 3 as failed", () => {
    const line = record(FAILED);
    assert.equal(line.outcome, "failed");
    assert.equal(line.error, "(Command exited with code 3)");
  });

  it("does NOT trust details.status, which says 'completed' either way", () => {
    // The trap, confirmed against a live run: OpenClaw reports
    // status:"completed" for a command that exited 3. `completed` means the
    // process finished, not that it worked. A classifier keying on it would
    // score every failed command as a success -- and it is the same word
    // OpenClaw's own itemStatus() falls back to for statuses it does not
    // recognise, so it is the single most dangerous field in this payload.
    assert.equal(FAILED.result.details.status, SUCCEEDED.result.details.status);
    assert.notEqual(record(FAILED).outcome, record(SUCCEEDED).outcome);
  });

  it("reads the exit code from the path the host actually uses", () => {
    assert.equal(readExit(SUCCEEDED.result), 0);
    assert.equal(readExit(FAILED.result), 3);
  });

  it("still grades the exit code when the host sets no error field", () => {
    // The host happened to set `error` on this failure, so `error` alone was
    // enough. It must not be the ONLY thing that works: strip it and the exit
    // code has to carry the verdict on its own.
    const { error, ...noError } = FAILED;
    assert.equal(classify("exec", undefined, noError.result).outcome, "failed");
    assert.equal(record(noError).error_class, "ExitStatus:3");
  });

  it("keeps the model's own call id as the join key", () => {
    // "call_percepteye_1" is the id the model minted, echoed back by the host.
    // It is what lets the control plane say which completion this call
    // followed without relying on clocks.
    assert.equal(record(SUCCEEDED).tool_call_id, "call_percepteye_1");
  });
});
