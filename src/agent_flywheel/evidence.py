"""Identity evidence: what a write produced, extracted and the body discarded.

A tool outcome says whether a call worked. It cannot say whether the change
SURVIVED -- a create that returns 201 and then silently fails to persist looks
identical, from inside the process, to one that landed. The only thing that can
settle it is reading the resource back afterwards, and a read-back needs one
thing this SDK did not previously carry: **which entity to look for**.

That is all this module produces. Not the response body -- the identity within
it. The distinction is the entire point:

* The body is customer production data. Persisting it in a trajectory turns a
  capture library into a data processor, with everything that implies.
* The identity is a handful of opaque strings. It is what a read-back needs and
  it is nearly all of what it needs.

So identity is resolved HERE, at the moment of capture, and the body is free to
be dropped. Extract, then discard.

WHY THIS DOES NOT GUESS. The fields are high-entropy identifiers only. A
created resource routinely echoes a human ``name`` ("Story 1") or a small
positional ``number`` (7) that a completely different entity can share by
coincidence, and matching on those confirms foreign entities. Those fields are
refused by name. And when a field appears more than once with DIFFERENT values,
it is omitted rather than picked between -- an ambiguous identity is not an
identity, and the honest result of ambiguity is no evidence rather than a coin
flip. Empty is a supported, expected answer: it means a later read-back must
abstain, which is the correct behaviour and the same rule ``unknown`` follows.
"""
from __future__ import annotations

import json
from typing import Any

__all__ = ["IDENTITY_FIELDS", "WEAK_ID_FIELDS", "extract_entity_ids"]

#: High-entropy identity fields that uniquely identify the entity a write
#: produced. Ordered by how commonly they carry the canonical id.
IDENTITY_FIELDS: tuple[str, ...] = ("id", "key", "uuid", "sid", "gid")

#: Low-entropy fields a DIFFERENT entity can share by coincidence. Never
#: identity, however tempting: matching on them confirms the wrong resource,
#: which is a false positive dressed as a verified success -- strictly worse
#: than having no evidence at all.
WEAK_ID_FIELDS: tuple[str, ...] = (
    "name", "number", "title", "label", "summary", "description",
)

#: How deep to look. A create's identity sits at the top of its response or
#: just inside an envelope (``{"data": {...}}``, ``{"result": {...}}``). Deeper
#: than that and a match is more likely to be some nested unrelated object's id
#: than the entity's own.
_MAX_DEPTH = 3

#: Guards a pathological body from costing real time at capture, which happens
#: on the customer's own request path.
_MAX_NODES = 2_000

_AMBIGUOUS = object()


def _scalar(value: Any) -> str | None:
    """A usable identity string, or None.

    ``bool`` is checked before ``int`` because it IS an ``int`` in Python, and
    ``{"id": true}`` is not an identity.
    """
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        s = value.strip()
        return s or None
    return None


def extract_entity_ids(output: Any) -> dict[str, str]:
    """The identity a write produced, filed under the field it came from.

    Returns e.g. ``{"id": "10001", "key": "PROJ-42"}``. The per-field filing
    matters and is not cosmetic: it is what lets a later read-back compare a
    candidate's ``key`` to the write's ``key`` and never to the write's ``id``.
    Collapsing both into one flat set is how an entity whose literal ``id``
    happens to equal another's ``key`` string gets falsely confirmed.

    Shallowest occurrence wins. A field seen twice at the same depth with
    different values is dropped as ambiguous.

    Never raises. A body this cannot read yields ``{}``, which downstream means
    "no identity evidence" -- an abstention, never a failure.
    """
    try:
        return _extract(output)
    except Exception:
        # Capture runs inside the customer's process, on their request path.
        # Nothing here is worth an exception reaching them.
        return {}


def _extract(output: Any) -> dict[str, str]:
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except (ValueError, TypeError):
            # Prose. An id parsed out of free text would be a guess.
            return {}

    #: field -> (depth, value | _AMBIGUOUS)
    found: dict[str, tuple[int, Any]] = {}
    budget = [_MAX_NODES]

    def walk(node: Any, depth: int) -> None:
        if depth > _MAX_DEPTH or budget[0] <= 0:
            return
        budget[0] -= 1
        if isinstance(node, dict):
            for raw_key, value in node.items():
                field = str(raw_key).strip().lower()
                if field in IDENTITY_FIELDS:
                    text = _scalar(value)
                    if text is not None:
                        prev = found.get(field)
                        if prev is None or depth < prev[0]:
                            found[field] = (depth, text)
                        elif depth == prev[0] and prev[1] != text:
                            # Two equally-shallow answers that disagree. We do
                            # not know which entity is THE entity.
                            found[field] = (depth, _AMBIGUOUS)
                walk(value, depth + 1)
        elif isinstance(node, list):
            for item in node:
                walk(item, depth + 1)

    walk(output, 0)
    return {
        field: value
        for field, (_, value) in found.items()
        if value is not _AMBIGUOUS
    }
