/**
 * Where the plugin's settings come from, and the ONE decision that gates the
 * network entirely.
 *
 * THE CONSENT BOUNDARY. If no API key resolves, `contributing()` returns null
 * and every control-plane path stays inert -- no claim, no report, no
 * registration, nothing leaves the machine. Local capture still runs, which is
 * the plugin's day-one value: your agent's real tool-failure rate, no account,
 * no network. A customer with no key is not misconfigured; they are the
 * ordinary first install.
 *
 * Precedence is OpenClaw config first, environment second. The config path is
 * preferred because it is hot-reloaded and puts the secret under OpenClaw's own
 * secret-ref machinery, while a gateway process reads its environment once at
 * launch and cannot be re-keyed without a restart.
 */
export const PLUGIN_ID = "percepteye-agent-flywheel";

const DEFAULT_CONTROL_PLANE = "https://launch.percepteye.ai/api/flywheel/v1";

const firstString = (...values) => {
  for (const v of values) {
    if (typeof v === "string" && v.trim()) return v.trim();
  }
  return null;
};

/**
 * @param {object} pluginConfig  `plugins.entries.<id>.config`, as OpenClaw hands it
 * @param {object} env           defaults to process.env
 */
export function resolveConfig(pluginConfig = {}, env = process.env) {
  const apiKey = firstString(pluginConfig.apiKey, env.PERCEPTEYE_API_KEY);
  return {
    apiKey,
    agentId: firstString(
      pluginConfig.agentId, env.PERCEPTEYE_AGENT_ID,
    ) ?? "openclaw",
    controlPlaneUrl: firstString(
      pluginConfig.controlPlaneUrl,
      env.PERCEPTEYE_CONTROL_PLANE_URL,
      env.CONTROL_PLANE_ENV_URL,
    ) ?? DEFAULT_CONTROL_PLANE,
    // Capture is on unless DELIBERATELY disabled. A typo must not silently
    // switch it off and leave the operator believing it is on.
    captureEnabled: !["0", "false", "no", "off"].includes(
      String(pluginConfig.capture ?? env.PERCEPTEYE_CAPTURE ?? "").trim().toLowerCase(),
    ),
    contributing: Boolean(apiKey),
  };
}
