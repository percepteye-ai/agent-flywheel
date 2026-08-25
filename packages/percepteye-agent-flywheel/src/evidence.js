/**
 * Identity evidence: what a write produced, extracted and the body discarded.
 *
 * A tool outcome says whether a call worked. It cannot say whether the change
 * SURVIVED -- a create that returns 201 and then silently fails to persist
 * looks identical, from inside the process, to one that landed. Settling that
 * needs a read-back, and a read-back needs one thing: which entity to look for.
 *
 * That is all this produces. Not the response body -- the identity within it:
 *
 *   - the body is customer production data, and persisting it in a trajectory
 *     turns a capture plugin into a data processor;
 *   - the identity is a handful of opaque strings, and is nearly all a
 *     read-back needs.
 *
 * So identity is resolved here, at capture, and the body is dropped. This is
 * the mirror of `evidence.py` on the Python side and MUST agree with it --
 * same fields, same refusals, same ambiguity rule.
 */

/** High-entropy fields that uniquely identify the entity a write produced. */
export const IDENTITY_FIELDS = new Set(["id", "key", "uuid", "sid", "gid"]);

/**
 * Low-entropy fields a DIFFERENT entity can share by coincidence. Never
 * identity: matching on them confirms the wrong resource, which is a false
 * positive dressed as a verified success -- strictly worse than no evidence.
 */
export const WEAK_ID_FIELDS = new Set([
  "name", "number", "title", "label", "summary", "description",
]);

const MAX_DEPTH = 3;
const MAX_NODES = 2000;
const AMBIGUOUS = Symbol("ambiguous");

function scalar(value) {
  // `typeof true === "boolean"`, so booleans are excluded by omission rather
  // than by a check -- but say so, because `{"id": true}` is not an identity.
  if (typeof value === "number" && Number.isFinite(value)) return String(value);
  if (typeof value === "string") {
    const s = value.trim();
    return s === "" ? null : s;
  }
  return null;
}

/**
 * The identity a write produced, filed under the field it came from:
 * `{ id: "10001", key: "PROJ-42" }`.
 *
 * The per-field filing is not cosmetic. It is what lets a read-back compare a
 * candidate's `key` to the write's `key` and never to the write's `id` --
 * collapsing them into one flat set is how an entity whose literal `id` equals
 * another's `key` string gets falsely confirmed.
 *
 * Shallowest occurrence wins; a field seen twice at the same depth with
 * different values is dropped as ambiguous, because an ambiguous identity is
 * not an identity. Never throws: an unreadable body yields `{}`, which means
 * "no evidence" and therefore an abstention, never a failure.
 */
export function extractEntityIds(output) {
  try {
    let root = output;
    if (typeof root === "string") {
      try {
        root = JSON.parse(root);
      } catch {
        return {};              // prose; an id parsed out of it would be a guess
      }
    }
    const found = new Map();    // field -> { depth, value }
    let budget = MAX_NODES;

    const walk = (node, depth) => {
      if (depth > MAX_DEPTH || budget <= 0 || node === null) return;
      budget -= 1;
      if (Array.isArray(node)) {
        for (const item of node) walk(item, depth + 1);
        return;
      }
      if (typeof node !== "object") return;
      for (const [rawKey, value] of Object.entries(node)) {
        const field = String(rawKey).trim().toLowerCase();
        if (IDENTITY_FIELDS.has(field)) {
          const text = scalar(value);
          if (text !== null) {
            const prev = found.get(field);
            if (!prev || depth < prev.depth) {
              found.set(field, { depth, value: text });
            } else if (depth === prev.depth && prev.value !== text) {
              found.set(field, { depth, value: AMBIGUOUS });
            }
          }
        }
        walk(value, depth + 1);
      }
    };

    walk(root, 0);
    const out = {};
    for (const [field, { value }] of found) {
      if (value !== AMBIGUOUS) out[field] = value;
    }
    return out;
  } catch {
    return {};
  }
}
