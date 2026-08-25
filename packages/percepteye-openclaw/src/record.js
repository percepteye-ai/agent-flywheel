/**
 * One `after_tool_call` event becomes one trajectory record.
 *
 * There is no correlation state here, and that is the point of choosing this
 * hook: the event already carries the arguments, the result, the error, the
 * duration and the call id together. The alternative surface
 * (`registerAgentEventSubscription`) splits the same facts across a
 * `phase:"start"` event and a `phase:"result"` event, which forces a pending
 * map keyed by `toolCallId` -- an unbounded one leaks inside a long-lived
 * Gateway, and a bounded one silently drops calls under load. Neither failure
 * exists on this path.
 */
import { classify } from "./classify.js";
import { extractEntityIds } from "./evidence.js";
import { toWire } from "./trajectory.js";

export function createRecorder({ writer }) {
  let written = 0;

  return {
    get written() {
      return written;
    },
    /**
     * @param {{toolName?:string, params?:object, toolCallId?:string,
     *          result?:unknown, error?:string, durationMs?:number}} event
     * @param {{agentId?:string}} [ctx]
     * @returns {boolean} whether a record was written
     */
    handle(event, ctx = {}) {
      if (!event || typeof event !== "object") return false;
      const name = typeof event.toolName === "string" ? event.toolName : null;
      // A record with no tool name is unattributable, and the contract rejects
      // an empty name outright. Nothing useful can be said about it.
      if (!name) return false;

      const verdict = classify(name, event.error, event.result);
      // Resolve the identity out of the result and let the result itself go.
      // The record carries who was written, never what the response said.
      const entity_ids = extractEntityIds(event.result);
      const ok = writer.write(toWire({
        name,
        // The host passes the arguments the tool ACTUALLY ran with -- it
        // resolves any `before_tool_call` adjustment before dispatching here.
        // Recording the pre-adjustment arguments would describe a call that
        // never happened.
        args: event.params,
        outcome: verdict.outcome,
        status_code: null,
        error: verdict.error,
        error_class: verdict.error_class,
        // Host-measured, so it includes queueing and approval waits that a
        // timer started inside this plugin would miss.
        latency_ms: typeof event.durationMs === "number"
          ? event.durationMs
          : undefined,
        // Which agent made the call. Absent means unattributed, which is the
        // honest answer for a single agent -- never a claim that one agent
        // made everything.
        agent_name: typeof ctx.agentId === "string" ? ctx.agentId : undefined,
        // The id the model assigned. It is the join key that says which
        // completion this call followed, without clocks and without breaking
        // under concurrent tools.
        tool_call_id: typeof event.toolCallId === "string"
          ? event.toolCallId
          : undefined,
        entity_ids,
      }));
      if (ok) written += 1;
      return ok;
    },
  };
}
