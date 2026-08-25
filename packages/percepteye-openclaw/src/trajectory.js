/**
 * The trajectory writer: append-only JSONL, one line per executed tool call.
 *
 * This file is the wire format, not an implementation detail. The Python SDK
 * reads exactly `$PERCEPTEYE_TRAJECTORY_DIR/tool_calls.jsonl`, so this module
 * and `percepteye_agent_flywheel.outcomes` are two ends of one ABI. Anything
 * that can append a line can feed the flywheel -- that is why the format is a
 * file and not a function call.
 *
 * Two rules the Python end depends on:
 *   - append-only and flushed per call, so a killed run keeps every call up to
 *     the kill rather than losing all of them;
 *   - one JSON object per line, because the reader recovers per line. A
 *     truncated final write costs that record and nothing else.
 *
 * Zero dependencies, `node:fs` only. This runs inside someone else's agent.
 */
import { appendFileSync, mkdirSync } from "node:fs";
import { dirname, join } from "node:path";

export const TRAJECTORY_DIR_ENV = "PERCEPTEYE_TRAJECTORY_DIR";
export const TOOL_CALLS_FILENAME = "tool_calls.jsonl";

/** Keys the contract accepts, in the order `to_wire()` emits them. */
const OPTIONAL_KEYS = [
  "output", "error", "error_class", "latency_ms", "agent_name", "tool_call_id",
];

/**
 * One record, shaped to match `ToolCallOutcome.to_wire()` exactly.
 *
 * `status_code` is always present and always null from here: OpenClaw's tool
 * stream carries no HTTP status, and inventing one is precisely the failure
 * the tri-state exists to prevent. A null status with a real outcome is the
 * honest shape; the contract requires the key, not a number.
 */
export function toWire({ name, args, outcome, status_code = null, entity_ids,
                         ...rest }) {
  const d = {
    name,
    arguments: args && typeof args === "object" ? args : {},
    outcome,
    status_code: status_code ?? null,
  };
  for (const key of OPTIONAL_KEYS) {
    const v = rest[key];
    if (v !== undefined && v !== null) d[key] = v;
  }
  // Emitted only when non-empty, and emitted for EVERY outcome including
  // `unknown`. An identity claims nothing about whether the call worked -- it
  // names an entity so something else can go and observe one. Withholding it
  // on `unknown` would suppress the only evidence able to settle that call.
  if (entity_ids && Object.keys(entity_ids).length > 0) {
    d.entity_ids = entity_ids;
  }
  return d;
}

/**
 * A writer bound to a directory, or an inert one when no directory is set.
 *
 * The env var is read per call rather than cached at construction: a host that
 * sets it late still gets capture, and a host that never sets it never gets a
 * stray file. Absence is the normal production case, not an error -- the same
 * plugin stays installed whether or not a rollout is running.
 */
export function createWriter({ dir = null, onError = null } = {}) {
  let warned = false;

  const resolveDir = () => dir ?? process.env[TRAJECTORY_DIR_ENV] ?? null;

  return {
    get active() {
      return Boolean(resolveDir());
    },
    /** @returns {boolean} whether the record was written. */
    write(record) {
      const base = resolveDir();
      if (!base) return false;
      const path = join(base, TOOL_CALLS_FILENAME);
      try {
        mkdirSync(dirname(path), { recursive: true });
        // 'a' is O_APPEND: concurrent writers interleave whole lines rather
        // than corrupting each other, which matters because a Gateway runs
        // tools concurrently.
        appendFileSync(path, JSON.stringify(record) + "\n", "utf8");
        return true;
      } catch (err) {
        // Never fail an agent over telemetry. Report once, then stay quiet:
        // a per-call warning on a full disk would bury the agent's own output.
        if (!warned) {
          warned = true;
          if (onError) onError(err);
        }
        return false;
      }
    },
  };
}
