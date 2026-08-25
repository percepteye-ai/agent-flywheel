/**
 * percepteye-agent-flywheel — a DeepSeek Harness (Cordis) plugin.
 *
 * Records tool-call outcomes into the active rollout's trajectory. Observes
 * only, and here that is guaranteed by the harness rather than merely intended.
 *
 * WHY `tools/result`. dsh's tool pipeline exposes several seams, and most of
 * them are waterfalls: `tools/execute` wraps dispatch, `tools/post` can accept,
 * replace or BLOCK a result, and `tools/code-dispatch-log` rewrites what gets
 * logged. A listener on any of those can change what the model sees.
 *
 * `tools/result` is declared `@mode emit` and typed to return `undefined`. Its
 * documentation is explicit — "Observe the frozen, lossless-JSON final outcome.
 * Listener failures are contained." — and the registry DEEP-FREEZES both the
 * execution and the result before publishing them, with an invariant that
 * fails the harness if it ever publishes something unfrozen. So this plugin
 * cannot alter a tool result even by accident: there is no return channel, and
 * the object it receives is immutable.
 *
 * That is the same rule the OpenClaw plugin follows for the same reason:
 * capturing evidence must never be able to change the behaviour being
 * measured. dsh simply enforces it harder.
 *
 * Zero dependencies. `ctx` is supplied by the harness; nothing is imported
 * from it, so this module loads even when dsh is absent — which is what lets
 * one package serve two harnesses.
 */
import { classifyDsh } from "./dsh-classify.js";
import { extractEntityIds } from "./evidence.js";
import { createWriter, toWire, TRAJECTORY_DIR_ENV } from "./trajectory.js";

/** Display metadata; labels the plugin in Cordis diagnostics. */
export const name = "percepteye-agent-flywheel";

export const HOOK_NAME = "tools/result";

/** The agent that made the call, or undefined when nothing names one. */
function agentName(exec) {
  const agent = exec && typeof exec === "object" ? exec.agent : undefined;
  if (!agent || typeof agent !== "object") return undefined;
  for (const key of ["name", "id"]) {
    const v = agent[key];
    if (typeof v === "string" && v.trim()) return v;
  }
  return undefined;
}

/**
 * Build the trajectory record for one settled execution.
 *
 * Exported for tests: the harness supplies frozen objects, and a recorder that
 * can only be exercised through a live agent loop is a recorder nobody checks.
 */
export function recordFor(exec, result) {
  const toolName = typeof exec?.name === "string" ? exec.name : null;
  if (!toolName) return null;

  const verdict = classifyDsh(toolName, result);
  return toWire({
    name: toolName,
    // dsh types `arguments` as `unknown` but guarantees it is losslessly
    // JSON-serializable. A non-object stays out rather than being coerced
    // into a shape it never had.
    args: exec.arguments && typeof exec.arguments === "object"
      ? exec.arguments
      : {},
    outcome: verdict.outcome,
    status_code: null,
    error: verdict.error,
    error_class: verdict.error_class,
    agent_name: agentName(exec),
    // `callId` is the model's own id for a root call. A nested call under code
    // mode gets a registry-assigned one, which is still the only identifier
    // that execution has.
    tool_call_id: typeof exec.callId === "string" ? exec.callId : undefined,
    // Resolved from the canonical value, which dsh documents as
    // "execution-local ... deliberately omitted from durable events" -- so it
    // is available HERE and nowhere later. Identity is taken and the value
    // itself is dropped.
    entity_ids: extractEntityIds(result?.value),
  });
}

/**
 * Cordis plugin entry. Named `apply`, per the plugin contract.
 *
 * Deliberately no `default` export in this module: the Cordis loader
 * normalizes with `exports.default ?? exports`, so a default here would be
 * handed to the harness INSTEAD of `apply` and the plugin would fail to load.
 * The OpenClaw entry, which does export a default, is a separate module for
 * exactly that reason.
 */
export function apply(ctx) {
  if (!ctx || typeof ctx.on !== "function") return null;

  const logger = ctx.logger ?? null;
  const writer = createWriter({
    onError: (err) => logger?.warn?.(
      `[${name}] trajectory write failed; tool outcomes are not being ` +
      `recorded for this run: ${err?.message ?? err}`,
    ),
  });

  const dispose = ctx.on(HOOK_NAME, (exec, result) => {
    // dsh contains listener failures already, but it logs each one. Swallowing
    // here keeps a malformed execution from filling a customer's diagnostics
    // with our name: one lost record, never a failed run, never a noisy one.
    try {
      const record = recordFor(exec, result);
      if (record) writer.write(record);
    } catch {
      /* one lost record is not worth an interrupted agent */
    }
    // `@mode emit` listeners return undefined. Returning anything else here
    // would be meaningless at best.
    return undefined;
  });

  return { writer, dispose };
}

export { TRAJECTORY_DIR_ENV };
