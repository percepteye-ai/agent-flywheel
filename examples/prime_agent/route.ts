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
export default function (pi: any) {
  const baseUrl = process.env.OPENAI_BASE_URL;
  if (!baseUrl) {
    // Refusing beats routing somewhere unrecorded. Without this the agent
    // silently falls back to its own provider, the rollout completes, and we
    // own none of the completion wire — which looks identical to a clean run.
    throw new Error(
      "OPENAI_BASE_URL is unset. This extension is meant to run under " +
        "percepteye-agent-flywheel, which sets it per rollout.",
    );
  }

  pi.registerProvider("percepteye", {
    baseUrl,
    // The NAME of the variable, not the value: prime-agent reads it itself.
    apiKey: "OPENAI_API_KEY",
    api: "openai-completions",
    models: [
      {
        id: process.env.PERCEPTEYE_MODEL_ID || "policy",
        name: "Policy under training",
        reasoning: false,
        input: ["text"],
        contextWindow: Number(process.env.PERCEPTEYE_CONTEXT_WINDOW || 32768),
        maxTokens: Number(process.env.PERCEPTEYE_MAX_TOKENS || 4096),
        cost: { input: 0, output: 0 },
      },
    ],
  });
}
