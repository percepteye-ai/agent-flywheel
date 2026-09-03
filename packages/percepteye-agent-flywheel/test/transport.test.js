/**
 * The transport, with emphasis on the ONE call that must never be replayed.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { AttachTransport } from "../src/transport.js";

/** Records every request the transport makes, and replays scripted bodies. */
function recorder(bodies = {}) {
  const calls = [];
  const client = {
    async request(method, path, opts = {}) {
      calls.push({ method, path, ...opts });
      if (bodies[path] instanceof Error) throw bodies[path];
      return bodies[path] ?? {};
    },
  };
  return { client, calls };
}

const t = (bodies) => {
  const r = recorder(bodies);
  return { tx: new AttachTransport(r.client, { agentId: "goose" }), calls: r.calls };
};

describe("claiming is not replayable", () => {
  it("turns ambiguous retries OFF for /rollouts/claim", async () => {
    // The control plane does not honour Idempotency-Key on this route, so a
    // resend that MAY have been processed leases a SECOND batch while the
    // first is held by nobody. transport.py:134-139 records the incident.
    const { tx, calls } = t({ "/rollouts/claim": { rollouts: [] } });
    await tx.claim(1);
    assert.equal(calls[0].retryIfAmbiguous, false);
  });

  it("leaves ambiguous retries ON for the replayable result call", async () => {
    const { tx, calls } = t();
    await tx.report({ rolloutId: "r1" }, { finalText: "done" });
    assert.notEqual(calls[0].retryIfAmbiguous, false);
    assert.equal(calls[0].idempotencyKey, "result-r1");
  });

  it("clamps the batch size to the server's bounds", async () => {
    const { tx, calls } = t({ "/rollouts/claim": { rollouts: [] } });
    await tx.claim(999);
    assert.equal(calls[0].jsonBody.max, 16);
    await tx.claim(0);
    assert.equal(calls[1].jsonBody.max, 1);
  });

  it("parses rollouts through the wire contract", async () => {
    const { tx } = t({
      "/rollouts/claim": {
        rollouts: [{ rollout_id: "r9", turn_input: "go", deadline_ts: 5 }],
        poll_after_ms: 2500,
      },
    });
    const { rollouts, pollAfterMs } = await tx.claim(1);
    assert.equal(rollouts[0].rolloutId, "r9");
    assert.equal(rollouts[0].turnInput, "go");
    assert.equal(pollAfterMs, 2500);
  });

  it("an empty queue is a normal answer, not an error", async () => {
    const { tx } = t({ "/rollouts/claim": {} });
    const { rollouts, pollAfterMs } = await tx.claim(1);
    assert.deepEqual(rollouts, []);
    assert.equal(pollAfterMs, 1000);
  });
});

describe("the reads fail OPEN", () => {
  it("policyCurrent returns {} when the control plane is unreachable", async () => {
    // A control plane that cannot answer must not stop an agent from running.
    const { tx } = t({ "/policy/current": new Error("down") });
    assert.deepEqual(await tx.policyCurrent(), {});
  });

  it("promptCurrent returns {} rather than blanking the prompt", async () => {
    const { tx } = t({ "/prompt/current": new Error("down") });
    assert.deepEqual(await tx.promptCurrent(), {});
  });
});

describe("abandon is typed", () => {
  it("refuses to abandon without a reason", async () => {
    // Abandon is "do not train on this", not a low reward. An untyped abandon
    // cannot be told apart from the agent simply doing badly.
    const { tx } = t();
    await assert.rejects(async () => tx.abandon({ rolloutId: "r" }, {}));
  });

  it("sends the reason and the lease token", async () => {
    const { tx, calls } = t();
    await tx.abandon({ rolloutId: "r1", leaseToken: "lt" },
                     { reason: "substrate_fault", detail: "oom" });
    assert.equal(calls[0].jsonBody.reason, "substrate_fault");
    assert.equal(calls[0].jsonBody.lease_token, "lt");
  });
});

describe("registration", () => {
  it("is idempotent per agent", async () => {
    const { tx, calls } = t();
    await tx.register({ agent_id: "goose" });
    assert.equal(calls[0].idempotencyKey, "reg-goose");
  });
  it("requires an agentId at construction", () => {
    assert.throws(() => new AttachTransport({}, {}));
  });
});
