/**
 * The session store: the workspace it must remember, and the spool that keeps
 * a report from being lost on a truncated shutdown.
 */
import { strict as assert } from "node:assert";
import { appendFileSync, existsSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, it } from "node:test";
import { SessionStore, safeSegment, sessionDir } from "../src/session.js";

const home = () => mkdtempSync(join(tmpdir(), "pe-home-"));

describe("a session id is untrusted input", () => {
  it("cannot escape its directory", () => {
    assert.equal(safeSegment("../../etc/passwd"), "etcpasswd");
    assert.equal(safeSegment("/absolute/path"), "absolutepath");
    assert.ok(!safeSegment("a/../b").includes("/"));
  });
  it("never yields an empty segment", () => {
    // An empty segment would make the session dir the turns/ root itself,
    // and one session's spool would then be every session's spool.
    assert.equal(safeSegment(""), "unknown");
    assert.equal(safeSegment("!!!"), "unknown");
    assert.equal(safeSegment(null), "unknown");
  });
  it("is bounded, because a filename is bounded", () => {
    assert.ok(safeSegment("x".repeat(500)).length <= 120);
  });
});

describe("where capture lands", () => {
  it("honours an explicit trajectory dir above everything", () => {
    const env = { PERCEPTEYE_TRAJECTORY_DIR: "/tmp/explicit", HOME: "/tmp/h" };
    assert.equal(sessionDir("s1", env), "/tmp/explicit");
  });
  it("otherwise defaults under PERCEPTEYE_HOME, per session", () => {
    const env = { PERCEPTEYE_HOME: "/tmp/peh" };
    assert.equal(sessionDir("s1", env), "/tmp/peh/turns/s1");
    assert.notEqual(sessionDir("s1", env), sessionDir("s2", env));
  });
});

describe("the store remembers what session hooks cannot see", () => {
  it("an unknown session is a new one, never undefined", () => {
    const s = new SessionStore({ PERCEPTEYE_HOME: home() });
    const e = s.get("fresh");
    assert.equal(e.workspaceDir, null);
    assert.equal(e.lastAssistantText, "");
    assert.equal(e.reported, false);
  });

  it("returns the SAME entry for a repeated key, so a turn can enrich it", () => {
    // workspaceDir arrives on an AGENT-phase context; session_end does not
    // carry it. The store is the only place that bridge exists.
    const s = new SessionStore({ PERCEPTEYE_HOME: home() });
    s.get("k").workspaceDir = "/work/here";
    assert.equal(s.get("k").workspaceDir, "/work/here");
  });

  it("forgets on demand, so a long-lived gateway does not grow forever", () => {
    const s = new SessionStore({ PERCEPTEYE_HOME: home() });
    s.get("k").workspaceDir = "/work";
    s.forget("k");
    assert.equal(s.get("k").workspaceDir, null);
  });
});

describe("the spool survives what the network does not", () => {
  it("writes a report to disk and can read it back", () => {
    // The only awaited session_end emission is a shutdown drain with a ~2s
    // budget SHARED across every session. A report that exists only in flight
    // is lost on a busy shutdown.
    const h = home();
    const s = new SessionStore({ PERCEPTEYE_HOME: h });
    const e = s.get("s1");
    assert.equal(s.spool(e, { rollout_id: "r1", final_text: "done" }), true);
    const pending = s.drainable();
    assert.equal(pending.length, 1);
    assert.equal(pending[0].payload.rollout_id, "r1");
  });

  it("collects spools from every session, not just the current one", () => {
    const h = home();
    const s = new SessionStore({ PERCEPTEYE_HOME: h });
    s.spool(s.get("a"), { rollout_id: "r1", final_text: "x" });
    s.spool(s.get("b"), { rollout_id: "r2", final_text: "y" });
    assert.equal(s.drainable().length, 2);
  });

  it("clears a spool once it is delivered", () => {
    const h = home();
    const s = new SessionStore({ PERCEPTEYE_HOME: h });
    s.spool(s.get("a"), { rollout_id: "r1", final_text: "x" });
    const [pending] = s.drainable();
    s.clearSpool(pending.path);
    assert.equal(existsSync(pending.path), false);
    assert.equal(s.drainable().length, 0);
  });

  it("an empty spool root is not an error", () => {
    const s = new SessionStore({ PERCEPTEYE_HOME: home() });
    assert.deepEqual(s.drainable(), []);
  });

  it("a truncated final line costs that record and nothing else", () => {
    const h = home();
    const s = new SessionStore({ PERCEPTEYE_HOME: h });
    const e = s.get("a");
    s.spool(e, { rollout_id: "r1", final_text: "x" });
    // simulate a kill mid-write
    appendFileSync(join(e.dir, "pending-report.json"), '{"rollout_id":"r2"');
    assert.equal(s.drainable().length, 1);
  });

  it("never throws out of a lifecycle hook", () => {
    // A telemetry writer that can break someone's agent is worse than none.
    const s = new SessionStore({ PERCEPTEYE_HOME: "/proc/nonwritable-xyz" });
    assert.equal(s.spool(s.get("a"), { final_text: "x" }), false);
  });
});
