/**
 * The retry semantics, pinned. Every case here names a defect the Python
 * client's docstring records, because this port exists to be the SAME client
 * in a second language, not a second client.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";
import { ControlPlaneClient } from "../src/http.js";
import { LeaseLost, TransportError } from "../src/errors.js";

/** A fetch stub that replays a scripted sequence and records what it saw. */
function stub(script) {
  const seen = [];
  const impl = async (url, init) => {
    seen.push({ url, init });
    const next = script[Math.min(seen.length - 1, script.length - 1)];
    if (typeof next === "function") return next();
    if (next instanceof Error) throw next;
    return new Response(next.body ?? null, {
      status: next.status ?? 200,
      headers: next.headers ?? {},
    });
  };
  return { impl, seen };
}

const client = (script, opts = {}) => {
  const s = stub(script);
  return {
    c: new ControlPlaneClient("https://cp.test", "pefw_k", {
      fetchImpl: s.impl, maxAttempts: 4, ...opts,
    }),
    seen: s.seen,
  };
};

function netError(code) {
  const e = new TypeError("fetch failed");
  e.cause = { code };
  return e;
}

describe("retrying reads the response, not just the exception", () => {
  it("retries a 503 and returns the eventual success", async () => {
    const { c, seen } = client([
      { status: 503, body: '{"detail":"warming"}' },
      { status: 200, body: '{"ok":true}' },
    ]);
    assert.deepEqual(await c.request("GET", "/health"), { ok: true });
    assert.equal(seen.length, 2);
  });

  it("does NOT retry a 400 — it will not become a 2xx", async () => {
    const { c, seen } = client([{ status: 400, body: '{"error":"bad"}' }]);
    await assert.rejects(() => c.request("POST", "/agents"), TransportError);
    assert.equal(seen.length, 1, "a terminal status was retried");
  });

  it("raises LeaseLost on 410 rather than a generic transport error", async () => {
    const { c } = client([{ status: 410, body: '{"detail":"lease lapsed"}' }]);
    await assert.rejects(() => c.request("POST", "/x"), LeaseLost);
  });

  it("honours Retry-After over its own backoff", async () => {
    const { c, seen } = client([
      { status: 429, headers: { "retry-after": "0" } },
      { status: 200, body: "{}" },
    ]);
    await c.request("GET", "/health");
    assert.equal(seen.length, 2);
  });
});

describe("ambiguity is the axis, not failure", () => {
  it("retryIfAmbiguous=false still retries a CONNECT failure (never sent)", async () => {
    const { c, seen } = client([netError("ECONNREFUSED"), { status: 200, body: "{}" }]);
    await c.request("POST", "/rollouts/claim", { retryIfAmbiguous: false });
    assert.equal(seen.length, 2, "a provably-unsent request was not retried");
  });

  it("retryIfAmbiguous=false does NOT retry a timeout", async () => {
    // THE DEFECT THIS PREVENTS: a read timeout may have been fully processed.
    // Resending /rollouts/claim double-leased batches that then sat until
    // their leases lapsed -- transport.py:134-139 records the incident.
    const abort = Object.assign(new Error("aborted"), { name: "AbortError" });
    const { c, seen } = client([abort, { status: 200, body: "{}" }]);
    await assert.rejects(
      () => c.request("POST", "/rollouts/claim", { retryIfAmbiguous: false }),
      TransportError,
    );
    assert.equal(seen.length, 1, "an ambiguous failure was retried");
  });

  it("a 429 retries EVEN when ambiguous retries are off", async () => {
    // The server explicitly said it did not do the work, so resending is safe
    // regardless. Python carries the same asymmetry.
    const { c, seen } = client([
      { status: 429, headers: { "retry-after": "0" } },
      { status: 200, body: "{}" },
    ]);
    await c.request("POST", "/rollouts/claim", { retryIfAmbiguous: false });
    assert.equal(seen.length, 2);
  });
});

describe("refusing to start wrong", () => {
  it("refuses an empty API key rather than defaulting one", async () => {
    // A previous SDK defaulted a missing key to a literal placeholder, so a
    // misconfigured deployment ran happily and reported nothing.
    assert.throws(() => new ControlPlaneClient("https://cp.test", ""), TransportError);
  });
  it("refuses an empty base URL", () => {
    assert.throws(() => new ControlPlaneClient("", "pefw_k"), TransportError);
  });
});

describe("bodies", () => {
  it("an empty body is {}, not a parse error", async () => {
    const { c } = client([{ status: 200, body: "" }]);
    assert.deepEqual(await c.request("POST", "/x"), {});
  });
  it("a non-object body is wrapped rather than returned bare", async () => {
    const { c } = client([{ status: 200, body: "[1,2]" }]);
    assert.deepEqual(await c.request("GET", "/x"), { data: [1, 2] });
  });
  it("an unparseable body raises instead of returning nothing", async () => {
    const { c } = client([{ status: 200, body: "not json" }]);
    await assert.rejects(() => c.request("GET", "/x"), TransportError);
  });
  it("sends the contract version on every request", async () => {
    const { c, seen } = client([{ status: 200, body: "{}" }]);
    await c.request("GET", "/health");
    assert.equal(seen[0].init.headers["Contract-Version"], "flywheel/1");
    assert.match(seen[0].init.headers.Authorization, /^Bearer /);
  });
  it("sets Idempotency-Key only when given one", async () => {
    const { c, seen } = client([{ status: 200, body: "{}" }, { status: 200, body: "{}" }]);
    await c.request("POST", "/a", { idempotencyKey: "k1" });
    await c.request("POST", "/b");
    assert.equal(seen[0].init.headers["Idempotency-Key"], "k1");
    assert.equal(seen[1].init.headers["Idempotency-Key"], undefined);
  });
});
