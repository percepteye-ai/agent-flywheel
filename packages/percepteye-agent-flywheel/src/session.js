/**
 * Per-session state, and a durable spool the network cannot lose.
 *
 * TWO OPENCLAW FACTS SHAPE THIS FILE.
 *
 * 1. SESSION HOOKS DO NOT CARRY THE WORKING DIRECTORY.
 *    `PluginHookSessionContext` is `{agentId?, sessionId, sessionKey?}` and
 *    nothing else, while `workspaceDir` lives on the AGENT-phase context. So
 *    the workspace has to be captured during a turn and remembered, or the
 *    session-end report has nowhere to look for the agent's deliverable.
 *
 * 2. THE END-OF-SESSION NETWORK CALL CAN BE TRUNCATED.
 *    In-line `session_end` emissions are fire-and-forget, and the only awaited
 *    emission is the shutdown drain -- bounded by a ~2s budget SHARED across
 *    every tracked session. A report that exists only in flight is a report
 *    that is lost on a busy shutdown.
 *
 *    So the report is written to a local spool SYNCHRONOUSLY first, and only
 *    then attempted over the network. A spooled report survives the process and
 *    is flushed on the next session's first turn. This is the same reasoning as
 *    the trajectory file itself: evidence goes to disk before it goes anywhere
 *    else, because disk is the part we control.
 */
import { appendFileSync, mkdirSync, readFileSync, readdirSync, rmSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

/** Where local capture and the report spool live when nothing else says. */
export function percepteyeHome(env = process.env) {
  return env.PERCEPTEYE_HOME || join(homedir(), ".percepteye");
}

/** A session id is untrusted input; it must never escape its directory. */
export function safeSegment(value) {
  const s = String(value ?? "");
  const cleaned = [...s].filter((c) => /[A-Za-z0-9_-]/.test(c)).join("");
  return cleaned.slice(0, 120) || "unknown";
}

export function sessionDir(sessionKey, env = process.env) {
  if (env.PERCEPTEYE_TRAJECTORY_DIR) return env.PERCEPTEYE_TRAJECTORY_DIR;
  return join(percepteyeHome(env), "turns", safeSegment(sessionKey));
}

const SPOOL = "pending-report.json";

export class SessionStore {
  constructor(env = process.env) {
    this._env = env;
    this._byKey = new Map();
  }

  /** Never returns undefined: an unknown session is a new one. */
  get(sessionKey) {
    const key = String(sessionKey ?? "unknown");
    let entry = this._byKey.get(key);
    if (!entry) {
      entry = {
        sessionKey: key,
        dir: sessionDir(key, this._env),
        workspaceDir: null,
        lastAssistantText: "",
        claimPromise: null,
        claim: null,
        reported: false,
      };
      this._byKey.set(key, entry);
    }
    return entry;
  }

  forget(sessionKey) {
    this._byKey.delete(String(sessionKey ?? "unknown"));
  }

  /**
   * Write a report to disk before attempting to send it.
   *
   * SYNCHRONOUS on purpose. The caller is on a shutdown path with a shared
   * ~2s budget; an async write there is a write that may never happen.
   */
  spool(entry, payload) {
    try {
      mkdirSync(entry.dir, { recursive: true });
      appendFileSync(
        join(entry.dir, SPOOL),
        JSON.stringify({ ...payload, spooled_at: null }) + "\n",
        "utf8",
      );
      return true;
    } catch {
      // Never throw out of a lifecycle hook. A telemetry writer that can break
      // someone's agent is worse than no telemetry.
      return false;
    }
  }

  /** Every spooled report still waiting to be sent, oldest session first. */
  drainable() {
    const root = join(percepteyeHome(this._env), "turns");
    let dirs;
    try {
      dirs = readdirSync(root);
    } catch {
      return [];
    }
    const out = [];
    for (const name of dirs) {
      const path = join(root, name, SPOOL);
      let text;
      try {
        text = readFileSync(path, "utf8");
      } catch {
        continue;
      }
      for (const line of text.split("\n")) {
        if (!line.trim()) continue;
        try {
          out.push({ path, payload: JSON.parse(line) });
        } catch {
          // A truncated final write costs that record and nothing else.
        }
      }
    }
    return out;
  }

  clearSpool(path) {
    try {
      rmSync(path, { force: true });
    } catch {
      /* nothing to clear */
    }
  }
}
