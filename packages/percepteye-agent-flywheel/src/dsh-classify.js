/**
 * Turning a DeepSeek Harness `tools/result` outcome into a flywheel outcome.
 *
 * dsh types its result as a discriminated union on `isError`, which is a real
 * discriminant rather than an optional flag — so `isError: true` is a positive
 * assertion of failure and can simply be believed.
 *
 * `isError: false` is the interesting half, and it does NOT mean success. The
 * bash tool renders a failing exit as TEXT (`[exit code: 3]`) appended to the
 * model-facing content and still settles as a non-error result: the tool call
 * worked, the command did not. Reading the flag alone scores a failed command
 * as a win — the same trap OpenClaw sets with `status: "completed"`.
 *
 * What makes dsh better to classify is that its exec tools declare a real
 * output schema: `exitCode`, `signal`, `timedOut`, `aborted`, and a `kind`
 * discriminating foreground from background. That is enough to be honest about
 * three cases a cruder reading collapses into one.
 */
import { readExit } from "./classify.js";

/** dsh's exec tools, by their registered names. */
export const DSH_EXEC_TOOLS = new Set(["bash", "pwsh"]);

/**
 * Tools whose silence really is success: no in-band status channel, so a
 * failure has nowhere to go but `isError`. These are dsh's own registered
 * names. Anything absent from both sets grades `unknown` until a fixture
 * earns it a place.
 */
export const DSH_VERDICT_TOOLS = new Set([
  "read", "write", "edit", "read_image", "str_replace_editor", "todo_write",
]);

/**
 * @param {string} name    the registered tool name
 * @param {{isError?: boolean, value?: unknown, error?: {message?: string, info?: unknown}}} result
 * @returns {{outcome:"ok"|"failed"|"unknown", status_code:null,
 *            error?:string, error_class?:string}}
 */
export function classifyDsh(name, result) {
  const r = result && typeof result === "object" ? result : {};

  // A positive assertion by the harness that the call failed.
  if (r.isError === true) {
    const failure = r.error && typeof r.error === "object" ? r.error : {};
    const info = failure.info;
    return {
      outcome: "failed",
      status_code: null,
      error: typeof failure.message === "string"
        ? failure.message.slice(0, 1000)
        : undefined,
      // dsh carries an internal error class on `info`; it is far more useful
      // downstream than a generic label when it is a plain string.
      error_class: typeof info === "string" ? info
        : (info && typeof info === "object" && typeof info.code === "string")
          ? info.code
          : "ToolError",
    };
  }

  // `isError` absent is not `isError: false`. The harness did not say.
  if (r.isError !== false) return { outcome: "unknown", status_code: null };

  const value = r.value;
  const v = value && typeof value === "object" ? value : {};

  if (DSH_EXEC_TOOLS.has(name)) {
    // A backgrounded command has been LAUNCHED, not completed. There is no
    // exit code because nothing has exited. Grading it `ok` would claim a
    // result for work still running.
    if (v.kind === "background") {
      return { outcome: "unknown", status_code: null,
               error_class: "StillRunning" };
    }
    // Cancelled by the caller. The command did not get to say what it would
    // have done, and that is not the agent making a mistake.
    if (v.aborted === true) {
      return { outcome: "unknown", status_code: null, error_class: "Aborted" };
    }
    if (v.timedOut === true) {
      return { outcome: "failed", status_code: null, error_class: "TimedOut" };
    }
    // Killed by a signal: `exitCode` is null precisely here, so this must be
    // checked before treating a null exit as unreadable.
    if (typeof v.signal === "string" && v.signal) {
      return { outcome: "failed", status_code: null,
               error_class: `Signal:${v.signal}` };
    }
    const exit = readExit(value);
    if (exit === undefined) {
      // We did not observe the verdict. Falling back to `ok` here is the
      // rounding-up this module exists to prevent.
      return { outcome: "unknown", status_code: null,
               error_class: "ExitStatusUnreadable" };
    }
    if (exit === 0) return { outcome: "ok", status_code: null };
    // An exit code is not an HTTP status; putting 127 in `status_code` would
    // fail the contract's 100..599 check and would be a lie besides.
    return { outcome: "failed", status_code: null,
             error_class: `ExitStatus:${exit}` };
  }

  if (DSH_VERDICT_TOOLS.has(name)) return { outcome: "ok", status_code: null };

  // Everything unproven: web tools, terminal session tools, subagent tools,
  // and every tool a plugin registered. A tool that reports its own failure
  // inside a successful-looking value is invisible from here.
  return { outcome: "unknown", status_code: null };
}
