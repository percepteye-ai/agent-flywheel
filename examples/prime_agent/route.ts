/**
 * Point prime-agent's completions at the per-rollout Percepteye endpoint.
 *
 * This is the whole integration. prime-agent has no environment variable for a
 * model base URL — `/login` picks a built-in provider — so routing is done the
 * one way it supports: registering a provider from an extension. Extensions run
 * as Node modules with access to `process.env`, which is exactly where the SDK
 * puts the per-rollout endpoint and its single-use key.
 *
 * Both variables are set by `build_child_env` for every rollout, so this file
 * is static: the URL it reads is different on each one.
 *
 * Loaded with `-e`, so it is scoped to the run rather than installed into
 * `~/.prime/agent/extensions/` where it would also affect the developer's own
 * interactive sessions.
 */
function positiveInteger(name: string): number {
  const raw = process.env[name];
  const value = Number(raw);
  if (!raw || !Number.isSafeInteger(value) || value <= 0) {
    throw new Error(`${name} must be an explicit positive integer; got ${raw ?? "<unset>"}`);
  }
  return value;
}

export default function (pi: any) {
  const baseUrl = process.env.OPENAI_BASE_URL;
  if (!baseUrl) {
    // Refusing beats routing somewhere unrecorded. Without this the agent
    // silently falls back to its own provider, the rollout completes, and we
    // own none of the completion wire — which looks identical to a clean run.
    throw new Error(
      "OPENAI_BASE_URL is unset. This extension is meant to run under " +
        "agent-flywheel, which sets it per rollout.",
    );
  }

  // These values are selected in the parent and cross the rollout boundary on
  // a closed, non-secret allowlist. Requiring them here makes a future scrub
  // regression loud: the route cannot silently register "policy" while the
  // parent asks Prime to run a different --model.
  const modelId = process.env.PRIME_AGENT_FLYWHEEL_MODEL_ID;
  if (!modelId?.trim()) {
    throw new Error(
      "PRIME_AGENT_FLYWHEEL_MODEL_ID is unset. Parent/route model identity cannot be verified.",
    );
  }
  const contextWindow = positiveInteger("PRIME_AGENT_FLYWHEEL_CONTEXT_WINDOW");
  const maxTokens = positiveInteger("PRIME_AGENT_FLYWHEEL_MAX_TOKENS");
  if (maxTokens > contextWindow) {
    throw new Error(
      "PRIME_AGENT_FLYWHEEL_MAX_TOKENS cannot exceed " +
        "PRIME_AGENT_FLYWHEEL_CONTEXT_WINDOW",
    );
  }

  // Compaction makes a separate summarizer completion and replaces the
  // transcript whose token/logprob chain the rollout update needs.  Refuse it
  // here even if a user's ambient Prime settings enable it.  The Python runner
  // also excludes a trace if the event appears despite this hook.
  pi.on("session_before_compact", () => ({ cancel: true }));

  pi.registerProvider("percepteye", {
    baseUrl,
    // The NAME of the variable, not the value: prime-agent reads it itself.
    apiKey: "OPENAI_API_KEY",
    api: "openai-completions",
    models: [
      {
        id: modelId,
        name: "Policy under training",
        reasoning: false,
        input: ["text"],
        // 65536, not 32768: at 32k the agent compacted its own context
        // mid-rollout on ~40% of episodes in a measured run, and a compacted
        // rollout is UNTRAINABLE — compaction rewrites the history, breaking
        // the token/logprob chain the update is computed over. A third of
        // that run was discarded for exactly this reason.
        //
        // Size this to the TRAINER's context window, not the server's. A
        // serving endpoint often advertises far more than the trainer can
        // consume, and a captured sequence longer than the training window is
        // truncated at tokenize time — silently masking generated tokens out
        // of the objective. Raise this only alongside the trainer's window.
        contextWindow,
        maxTokens,
        cost: { input: 0, output: 0 },
      },
    ],
  });
}
