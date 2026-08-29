/**
 * The agent describer: what this agent IS, observed from inside the host.
 *
 * WHY THIS EXISTS. The Python SDK discovers an agent by reflecting on a
 * constructed object. That works for an in-process Python agent and cannot
 * work for a CLI one: `fw.serve()` given a command entrypoint spawns an OS
 * subprocess, so `runner.resolved` is None, `introspect(None)` returns None,
 * and the control plane records `introspection_state = "unreadable"`. No tool
 * catalogue means no generated workflows, which means no evaluation suites,
 * no graded rollouts, and no dataset. The whole flywheel is downstream of this
 * one fact, and reflection cannot cross a process AND a language boundary.
 *
 * WHY THIS HOOK. `llm_input` is the only place the host offers what a
 * description needs, and it offers it already assembled:
 *
 *   systemPrompt  the FULLY ASSEMBLED prompt -- the host's several prompt
 *                 files, concatenated inside the Node process, from sources
 *                 the Python SDK never opens.
 *                 Reading those files ourselves would re-implement the host's
 *                 assembly order and drift from it silently.
 *   tools         the real tool list AS SENT TO THE MODEL. Not the config,
 *                 not the plugin registry -- the actual universe the agent
 *                 could act in on this turn. Nothing else in the host is
 *                 both complete and true at the same moment.
 *   provider,model the model config, resolved after all overrides.
 *
 * WHY THIS IS STILL "OBSERVES ONLY". The sibling recorder chose
 * `after_tool_call` over `registerAgentToolResultMiddleware` on one rule:
 * capturing evidence must never be able to change the behaviour being
 * measured. `llm_input` keeps that rule. It is typed
 * `(event, ctx) => Promise<void> | void` -- the same structurally
 * non-mutating class as `after_tool_call`. The host discards our return
 * value, so there is no seam here through which a description could alter a
 * prompt, a tool result, or a routing decision. Introspection is a READ.
 * The mutation seams OpenClaw does expose -- `before_prompt_build`'s
 * `{systemPrompt}`, `before_tool_call`'s `{block:true}` -- remain unused,
 * deliberately.
 *
 * THE ONE THING THIS HOOK COSTS. `llm_input` is a CONVERSATION hook
 * (`CONVERSATION_HOOK_NAMES`, hook-types:386). The host REFUSES to register
 * it for a non-bundled plugin unless the operator opts in with
 * `plugins.entries.percepteye-agent-flywheel.hooks.allowConversationAccess=true`
 * (registry:4226-4235). That refusal is a warn-level diagnostic, not a throw,
 * so a missing opt-in looks exactly like a working install that never fires.
 * `register()` below says so out loud at startup rather than letting the
 * operator discover it as an empty catalogue an hour later.
 *
 * Zero dependencies, `node:fs` only. This runs inside someone else's agent.
 */
import { renameSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

import { TRAJECTORY_DIR_ENV } from "./trajectory.js";

export const DISCOVERED_AGENT_FILENAME = "discovered_agent.json";
export const DESCRIBE_HOOK_NAME = "llm_input";

/**
 * `source_type` is a provenance claim, and it is NOT `sdk_runtime`.
 *
 * The Python probes mean something specific by `sdk_runtime`: the agent
 * described itself by reflection over a constructed object. This description
 * came from the HOST observing a real model call. Both are honest, they are
 * not the same act, and only one of them can be wrong in the ways reflection
 * is wrong. Naming it separately costs nothing -- the control plane stores
 * `source_type` and has no reader for it, and the discovery consumer stamps
 * its own source type when it builds its record rather than reading ours --
 * and it means a future reader can tell the two apart without archaeology.
 *
 * Kept short on purpose: `flywheel_agents.source_type` is a varchar(64)
 * written verbatim with no truncation, the exact column shape of a prior
 * overflow that refused 100% of inserts.
 */
export const SOURCE_TYPE = "openclaw_llm_input";

/**
 * `framework` is a LOOKUP KEY, not a label.
 *
 * The discovery consumer maps exactly `"openclaw"` to its OpenClaw
 * framework. Any other spelling -- `open_claw`,
 * `openclaw-cli`, the agent's own name -- silently degrades to
 * an UNKNOWN framework, which is logged and tolerated and loses the framework
 * identity in every generated workflow's provenance. Same varchar(64).
 */
export const FRAMEWORK = "openclaw";

/**
 * Postgres JSONB cannot store a NUL byte or a lone surrogate in any string:
 * it raises `unsupported Unicode escape sequence`, which surfaces as an
 * uncaught 500 on registration, 100% of the time, for that agent.
 *
 * This matters here specifically because the payload is not ours. The system
 * prompt is a concatenation of customer-authored markdown files, and
 * `event.tools` is typed `unknown[]` -- arbitrary provider-shaped JSON. A
 * description that cannot be stored is worth less than one missing a field,
 * so we scrub rather than let the registration fail.
 */
function scrub(value) {
  if (typeof value !== "string") return value;
  // eslint-disable-next-line no-control-regex
  return value.replace(/\u0000/g, "").replace(/[\uD800-\uDFFF]/g, (ch, i, s) => {
    const code = ch.charCodeAt(0);
    const isHigh = code <= 0xdbff;
    const partner = isHigh ? s.charCodeAt(i + 1) : s.charCodeAt(i - 1);
    const paired = isHigh
      ? partner >= 0xdc00 && partner <= 0xdfff
      : partner >= 0xd800 && partner <= 0xdbff;
    return paired ? ch : "";
  });
}

/** Recursively scrub every string in a JSON-ish value. */
function scrubDeep(value) {
  if (typeof value === "string") return scrub(value);
  if (Array.isArray(value)) return value.map(scrubDeep);
  if (value && typeof value === "object") {
    const out = {};
    for (const [k, v] of Object.entries(value)) out[scrub(k)] = scrubDeep(v);
    return out;
  }
  return value;
}

/**
 * Normalise ONE host tool object into the discovery protocol's tool shape.
 *
 * This is the load-bearing translation, and it is load-bearing because the
 * consumer is exact and silent. The consumer reads `input_schema` and NOTHING else: it does not accept the
 * host's native
 * `parameters` key, and it does not unwrap the OpenAI
 * `{type:"function", function:{...}}` envelope. Both failures are silent and
 * differently shaped -- `parameters` yields a tool with an EMPTY schema,
 * while the OpenAI envelope yields a tool with NO NAME, which the consumer
 * then drops entirely. Forward `event.tools` verbatim and the launch gate still
 * opens (the list is truthy) while the catalogue is empty or nameless.
 *
 * `event.tools` is `unknown[]` by contract, so all three shapes are possible
 * from one host depending on provider, and we handle all three rather than
 * betting on the one we saw first.
 *
 * @returns {{name:string,description:string,input_schema:object,original_format:string}|null}
 */
export function normalizeTool(raw) {
  if (!raw || typeof raw !== "object") return null;

  // The OpenAI function-calling envelope: the real tool is one level down.
  const isEnvelope =
    raw.type === "function" && raw.function && typeof raw.function === "object";
  const t = isEnvelope ? raw.function : raw;

  const name = typeof t.name === "string" ? t.name.trim() : "";
  // The consumer drops a nameless tool anyway; dropping it here means the count we
  // report is the count that survives, so "7 tools" never means "7 sent, 3
  // arrived".
  if (!name) return null;

  const schema =
    t.input_schema ?? t.inputSchema ?? t.parameters ?? t.schema ?? null;

  return {
    name: scrub(name),
    description: scrub(
      typeof t.description === "string" ? t.description
        : typeof t.label === "string" ? t.label
          : "",
    ),
    // `{type:"object",properties:{}}` is what the consumer substitutes for a falsy
    // schema. Emitting it here keeps the wire record self-describing rather
    // than relying on a default two repos away.
    input_schema:
      schema && typeof schema === "object"
        ? scrubDeep(schema)
        : { type: "object", properties: {} },
    original_format: isEnvelope ? "openai_function" : "openclaw_tool",
  };
}

/** Normalise the host's tool array. `undefined` in means `null` out. */
export function normalizeTools(rawTools) {
  // THE ABSENCE CONTRACT, preserved across the process boundary. `null` means
  // we could not look; `[]` means we looked and the agent has no tools. The
  // consumer stamps a `not observed` marker on the first and grades on the
  // difference, and the control plane's launch gate refuses a run whose agent
  // reports zero tools. Collapsing these two into `[]` would turn "we did not
  // observe" into a confident claim that the agent can do nothing.
  if (!Array.isArray(rawTools)) return null;
  const out = [];
  for (const raw of rawTools) {
    const tool = normalizeTool(raw);
    if (tool) out.push(tool);
  }
  return out;
}

/**
 * Build the wire record from one `llm_input` event.
 *
 * Field-for-field the shape of `DiscoveredAgent.to_wire()` in the Python SDK,
 * because that is what the control plane stores and reads. The two
 * are one ABI; this is the Node end of it, exactly as `trajectory.js` is the
 * Node end of the tool-outcome ABI.
 */
export function buildRecord(event, ctx = {}) {
  const tools = normalizeTools(event?.tools);
  const record = {
    agent_name: scrub(String(ctx.agentId || event?.sessionId || "openclaw-agent")),
    framework: FRAMEWORK,
    source_type: SOURCE_TYPE,
    role: "standalone",
    tool_definitions: tools,
  };

  if (typeof event?.systemPrompt === "string" && event.systemPrompt) {
    record.system_prompt = scrub(event.systemPrompt);
  }
  if (event?.provider || event?.model) {
    record.discovered_model_config = {
      provider: scrub(String(event.provider || "unknown")),
      model_name: scrub(String(event.model || "unknown")),
    };
  }

  // `incomplete` names the parts we could not read, so a consumer can tell a
  // gap from an absence. The host declares `systemPrompt` and `tools`
  // optional, so either can genuinely be missing from a real event.
  const incomplete = [];
  if (!record.system_prompt) incomplete.push("system_prompt");
  if (tools === null) incomplete.push("tool_definitions");
  if (incomplete.length) record.incomplete = incomplete;

  return record;
}

/**
 * A describer bound to a directory, or an inert one when none is set.
 *
 * WRITE POLICY. `llm_input` fires on every model call, and the tool universe
 * is not constant across them: a sub-agent turn or a post-compaction turn can
 * carry a SUBSET. So we keep the richest catalogue seen rather than the
 * newest -- last-write-wins would let a narrow sub-agent turn overwrite the
 * agent's real catalogue with a fraction of it, and the result would look
 * perfectly healthy at every layer downstream.
 *
 * The first event always writes, so an agent that genuinely has no tools
 * still produces a file and still reaches `described`.
 */
export function createDescriber({ dir = null, onError = null } = {}) {
  let best = -1;
  let warned = false;

  const resolveDir = () => dir ?? process.env[TRAJECTORY_DIR_ENV] ?? null;

  return {
    get active() {
      return Boolean(resolveDir());
    },
    /** @returns {boolean} whether this event was written. */
    handle(event, ctx = {}) {
      const base = resolveDir();
      if (!base) return false;

      const record = buildRecord(event, ctx);
      const count = record.tool_definitions?.length ?? 0;
      if (best >= 0 && count <= best) return false;

      const path = join(base, DISCOVERED_AGENT_FILENAME);
      const tmp = `${path}.${process.pid}.tmp`;
      try {
        mkdirSync(base, { recursive: true });
        // Write-then-rename. The Python SDK POLLS for this file while the
        // warm-up child runs, so a reader can arrive mid-write; rename is
        // atomic within a filesystem, so the reader sees either no file or a
        // complete one, never half a JSON object.
        writeFileSync(tmp, JSON.stringify(record), "utf8");
        renameSync(tmp, path);
        best = count;
        return true;
      } catch (err) {
        // Never fail an agent over telemetry. Report once, then stay quiet.
        if (!warned) {
          warned = true;
          if (onError) onError(err);
        }
        return false;
      }
    },
  };
}

/**
 * Register the describer on an OpenClaw plugin api.
 *
 * Returns the describer, or null when the host has no `on()`. The
 * conversation-access opt-in cannot be checked from here -- the host's
 * refusal is a diagnostic on ITS side and our handler simply never fires --
 * so the caller warns at startup instead.
 */
export function registerDescriber(api, { logger = null } = {}) {
  if (typeof api?.on !== "function") return null;

  const describer = createDescriber({
    onError: (err) => logger?.warn?.(
      `[percepteye-agent-flywheel] agent description write failed; this agent ` +
      `will register as introspection_state=unreadable and no workflows will ` +
      `be generated for it: ${err?.message ?? err}`,
    ),
  });

  api.on(DESCRIBE_HOOK_NAME, (event, ctx) => {
    try {
      describer.handle(event, ctx ?? {});
    } catch {
      /* one lost description is not worth an interrupted agent */
    }
  });

  return describer;
}
