/**
 * The nine control-plane calls, and the one that must never be replayed.
 *
 * A port of `percepteye_agent_flywheel/transport.py`. Everything here is
 * OUTBOUND-ONLY: there is no listening socket anywhere in this package, and a
 * customer's agent opens every connection.
 */
import { rolloutOutputToWire, rolloutRequestFromWire } from "./wire.js";

const clampInt = (v, lo, hi) => Math.max(lo, Math.min(hi, Number(v) || lo));

export class AttachTransport {
  /**
   * @param {import("./http.js").ControlPlaneClient} client
   * @param {{agentId: string, waitMs?: number, leaseMs?: number}} opts
   */
  constructor(client, { agentId, waitMs = 5000, leaseMs = 300_000 } = {}) {
    if (!agentId) throw new Error("agentId is required");
    this._c = client;
    this.agentId = agentId;
    this.waitMs = waitMs;
    this.leaseMs = leaseMs;
  }

  health() {
    return this._c.request("GET", "/health");
  }

  register(spec) {
    return this._c.request("POST", "/agents", {
      jsonBody: spec,
      idempotencyKey: `reg-${this.agentId}`,
    });
  }

  /**
   * What policy this agent should be running, per the control plane.
   *
   * FAILS OPEN, deliberately. A control plane that cannot answer must not stop
   * an agent from running, so an empty answer means "keep doing what you were
   * doing" -- exactly the behaviour before this endpoint existed.
   */
  async policyCurrent() {
    try {
      return (await this._c.request("GET", "/policy/current")) || {};
    } catch {
      return {};
    }
  }

  /**
   * The approved optimized system prompt, or `{}`.
   *
   * Same fail-open contract, and for a sharper reason: a control plane that
   * cannot answer must never BLANK a running agent's instructions.
   */
  async promptCurrent() {
    try {
      return (await this._c.request("GET", "/prompt/current")) || {};
    } catch {
      return {};
    }
  }

  /**
   * Claim up to `maxRollouts` rollouts. Returns `{rollouts, pollAfterMs}`.
   *
   * CLAIMING IS NOT REPLAYABLE, and this is the one call in the package that
   * turns ambiguous retries OFF. The control plane does not honour
   * Idempotency-Key on this route, so resending a claim that MAY already have
   * been processed leases a SECOND batch while the first is held by nobody --
   * those rollouts then sit untouched until their leases lapse. Only a failure
   * that provably never reached the server is retried.
   */
  async claim(maxRollouts) {
    const body = await this._c.request("POST", "/rollouts/claim", {
      jsonBody: {
        agent_id: this.agentId,
        max: clampInt(maxRollouts, 1, 16),
        wait_ms: this.waitMs,
        lease_ms: this.leaseMs,
      },
      idempotencyKey: crypto.randomUUID(),
      timeoutMs: this.waitMs + 10_000,
      retryIfAmbiguous: false,
    });
    const raw = body.rollouts || [];
    return {
      rollouts: raw.map(rolloutRequestFromWire),
      pollAfterMs: Number(body.poll_after_ms) || 1000,
    };
  }

  heartbeat(rollout, { progress } = {}) {
    const payload = {
      lease_token: rollout.leaseToken,
      extend_ms: Math.min(this.leaseMs, 600_000),
    };
    if (progress) payload.progress = progress;
    return this._c.request(
      "POST", `/rollouts/${rollout.rolloutId}/heartbeat`, { jsonBody: payload },
    );
  }

  /**
   * Report the result. REPLAYABLE: the server keys on (rollout_id, attempt),
   * so an identical replay is acknowledged as a duplicate rather than
   * double-counted -- which is why this one keeps ambiguous retries ON.
   */
  report(rollout, output) {
    return this._c.request("POST", `/rollouts/${rollout.rolloutId}/result`, {
      jsonBody: rolloutOutputToWire(output),
      idempotencyKey: `result-${rollout.rolloutId}`,
    });
  }

  /**
   * Give the work back, TYPED.
   *
   * This is not a low reward. It is "do not train on this" -- a substrate
   * fault, an expired credential, a crash. Reporting a crashed rollout as a
   * zero-scoring one teaches the policy that its own behaviour caused an
   * outcome it had nothing to do with.
   */
  abandon(rollout, { reason, detail = "" }) {
    if (!reason) throw new Error("abandon requires a typed reason");
    return this._c.request("POST", `/rollouts/${rollout.rolloutId}/abandon`, {
      jsonBody: { lease_token: rollout.leaseToken, reason, detail },
      idempotencyKey: `abandon-${rollout.rolloutId}`,
    });
  }

  artifacts(rollout, payload) {
    return this._c.request("POST", `/rollouts/${rollout.rolloutId}/artifacts`, {
      jsonBody: payload,
      idempotencyKey: `artifacts-${rollout.rolloutId}`,
    });
  }
}
