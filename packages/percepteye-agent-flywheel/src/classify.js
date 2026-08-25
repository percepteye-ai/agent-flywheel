/**
 * Turning an OpenClaw `after_tool_call` event into a flywheel outcome.
 *
 * The tri-state is the whole point, and it is NOT "error field absent means
 * success". The host builds the event with
 *
 *     ...params.error ? { error: params.error } : {}
 *
 * so `error` is omitted whenever it is falsy -- which covers both "the call
 * succeeded" and "nobody set an error". Those are different facts and this
 * module refuses to merge them.
 *
 * OpenClaw's own durable-log path shows what merging them costs: `itemStatus()`
 * maps any status it does not recognise to "completed", so "cancelled" and
 * "unknown" -- which its own terminal-status audit keeps distinct -- arrive
 * downstream looking like success. We inherit the lesson rather than the bug:
 * a verdict nobody stated is `unknown`, never `ok`.
 */

/**
 * Tools whose silence really does mean the operation succeeded. These have no
 * in-band status channel, so a failure has nowhere to go but the error field.
 *
 * The names are the host's own: `bash, edit, exec, find, grep, ls, read, write`
 * are what OpenClaw types as tool names. `apply_patch` is deliberately NOT
 * here -- a patch can apply partially, and until a fixture shows how that is
 * reported, grading it `ok` on silence would be a guess in the dangerous
 * direction. Everything else is `unknown` until proven; promotion is earned,
 * never assumed.
 */
export const VERDICT_TOOLS = new Set([
  "read", "write", "edit", "grep", "ls", "find",
]);

/**
 * Tools that report the operation's real verdict INSIDE the result and set no
 * error field for an operation that plainly failed -- a non-zero exit. These
 * two names are the host's own definition of an exec tool
 * (`isExecToolName(n) => n === "exec" || n === "bash"`), so this set tracks
 * the host rather than guessing at aliases.
 */
export const IN_BAND_TOOLS = new Set(["bash", "exec"]);

const EXIT_PATHS = [
  ["meta", "exitCode"], ["meta", "exit_code"], ["meta", "status"],
  ["details", "exitCode"], ["details", "status"],
  ["exitCode"], ["exit_code"],
];

function dig(obj, path) {
  let cur = obj;
  for (const key of path) {
    if (cur === null || typeof cur !== "object") return undefined;
    cur = cur[key];
  }
  return cur;
}

/** The exit status if one is legible, else undefined. Never a guess. */
export function readExit(result) {
  for (const path of EXIT_PATHS) {
    const v = dig(result, path);
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && /^-?\d+$/.test(v)) return Number(v);
    if (v === "error" || v === "failed") return 1;
    if (v === "ok" || v === "success") return 0;
  }
  return undefined;
}

/**
 * @param {string} name       tool name, as the host reports it
 * @param {unknown} error     the host's error string; present only when truthy
 * @param {unknown} result    the tool result
 * @returns {{outcome:"ok"|"failed"|"unknown", status_code:null,
 *            error?:string, error_class?:string}}
 */
export function classify(name, error, result) {
  // A positive assertion by the host that the call failed. Believe it.
  if (typeof error === "string" && error.trim() !== "") {
    return {
      outcome: "failed",
      status_code: null,
      error: error.slice(0, 1000),
      error_class: "ToolError",
    };
  }

  if (IN_BAND_TOOLS.has(name)) {
    const exit = readExit(result);
    // Not found means we did not observe the verdict. Falling back to `ok`
    // here is the exact rounding-up this module exists to prevent.
    if (exit === undefined) {
      return {
        outcome: "unknown", status_code: null,
        error_class: "ExitStatusUnreadable",
      };
    }
    if (exit === 0) return { outcome: "ok", status_code: null };
    // An exit code is NOT an HTTP status. Putting 127 in `status_code` would
    // fail the contract's 100..599 range check on the Python side, and would
    // be a lie besides.
    return {
      outcome: "failed", status_code: null,
      error_class: `ExitStatus:${exit}`,
    };
  }

  if (VERDICT_TOOLS.has(name)) return { outcome: "ok", status_code: null };

  // MCP tools, third-party plugin tools, anything unproven. An MCP server that
  // returns {"error": ...} as ordinary content is invisible from here, so the
  // honest answer is that we did not observe the verdict.
  return { outcome: "unknown", status_code: null };
}
