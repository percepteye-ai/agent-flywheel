/**
 * percepteye-flywheel — an OpenClaw plugin that records tool-call outcomes.
 *
 * WHY THIS HOOK. OpenClaw exposes three places to see a tool result, and only
 * one of them is both complete and incapable of changing the run:
 *
 *   registerAgentToolResultMiddleware
 *     A waterfall. Its return value REPLACES the result the model reads --
 *     that is what the bundled output-compacting plugin uses it for. A bug
 *     here would silently rewrite a customer's agent output. It is also
 *     manifest-gated and refused unless the plugin is explicitly enabled.
 *     Rejected: a recorder must not sit on a mutation seam.
 *
 *   registerAgentEventSubscription (streams: ["tool"])
 *     A genuine observer -- the host `structuredClone`s the event and discards
 *     the return. But it is a UI/monitor stream: `data` is untyped, the
 *     arguments arrive on a separate `phase:"start"` event, and exec results
 *     are capped before dispatch. Correlating the two phases needs a pending
 *     map that either leaks or drops.
 *
 *   api.on("after_tool_call", ...)                            <-- what we use
 *     Also void-returning, so structurally incapable of altering, blocking or
 *     delaying anything; the host runs it best-effort inside its own
 *     try/catch. And it carries the whole call in one event: `params`,
 *     `result`, `error`, `durationMs`, `toolCallId`, plus agent and session
 *     identity on the context.
 *
 * The rule behind the choice: capturing evidence must never be able to change
 * the behaviour being measured. A collector that can corrupt the run it is
 * measuring is worse than no collector.
 *
 * Zero dependencies. Inert unless PERCEPTEYE_TRAJECTORY_DIR is set, so one
 * install is a no-op in production and a recorder during a rollout.
 */
import { createRecorder } from "./record.js";
import { createWriter, TRAJECTORY_DIR_ENV } from "./trajectory.js";

export const PLUGIN_ID = "percepteye-flywheel";
export const HOOK_NAME = "after_tool_call";

export function register(api) {
  if (typeof api?.on !== "function") return null;

  const logger = api.logger ?? api.runtime?.logger ?? null;
  const writer = createWriter({
    onError: (err) => logger?.warn?.(
      `[${PLUGIN_ID}] trajectory write failed; tool outcomes are not being ` +
      `recorded for this run: ${err?.message ?? err}`,
    ),
  });
  const recorder = createRecorder({ writer });

  api.on(HOOK_NAME, (event, ctx) => {
    // The host already isolates hook failures, but it logs a warning per
    // failure. Swallowing here keeps a malformed event from filling a
    // customer's logs with our name -- one lost record, never a failed run,
    // never a noisy one.
    try {
      recorder.handle(event, ctx ?? {});
    } catch {
      /* one lost record is not worth an interrupted agent */
    }
  });

  return { recorder, writer };
}

export default {
  id: PLUGIN_ID,
  name: "PerceptEye Flywheel",
  description:
    "Records tool-call outcomes (ok/failed/unknown) into the active " +
    `rollout's trajectory. Inert unless ${TRAJECTORY_DIR_ENV} is set.`,
  configSchema: { type: "object", additionalProperties: false, properties: {} },
  register,
};
