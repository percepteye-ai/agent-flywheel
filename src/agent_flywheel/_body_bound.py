"""How a captured tool body is shortened. ONE definition for this SDK.

Every producer of a tool call's ``output`` or ``error`` that the SDK itself
observes shortens it here: the event-stream parser, the LangChain and ADK
adapters, and the host recorder (``hosts/scripts/percepteye_record.py``). The
Node SDK's ``src/body-bound.js`` is the same rule in the other language, and
``tests/test_body_bound_parity.py`` proves the two return the same string for
the same input. A producer that cut its own way would give one wire field a
shape per producer -- the 1000-char head cut this replaced was exactly that.

STANDARD LIBRARY ONLY, AND IT IMPORTS NOTHING. The host recorder runs once per
tool call inside someone else's agent and must never pay for ``import
agent_flywheel``, so it loads this file by path; an import here would bring
that cost back on the path it exists to keep cheap.
"""
from __future__ import annotations

#: One budget for an output body and an error body alike. An error used to get
#: a quarter of the output budget, though a raw traceback is the load-bearing
#: signal for learned self-correction and the one payload whose informative
#: part is reliably at the END: the exception class and its message, after
#: however many frames. A 1000-char head cut deleted exactly that.
MAX_BODY_CHARS = 4000

#: Stamped into a body that had to be shortened. A fixed, greppable string
#: with the removed count in it: it lets a downstream census tell "the tool
#: printed 300 chars" from "the tool printed 300,000 and we kept a window"
#: without a wire field every producer would have to learn to set.
ELISION = "\n…[{dropped} chars elided by the SDK capture bound]…\n"


def bounded(text: str, limit: int = MAX_BODY_CHARS) -> str:
    """Shorten ``text`` to ``limit`` characters while keeping BOTH ends.

    A traceback opens with a banner and closes with the exception type and
    message; a REPL transcript opens with a command and closes with what it
    produced. Head-truncation keeps the less informative half of both, and
    does it silently. So: half the room from the front, the rest from the
    back, with a marker between them stating how many chars of ``text`` it
    replaced. Characters are code points, which is what the Node twin counts
    too.
    """
    if len(text) <= limit:
        return text
    # The marker is part of the budget: a bound that its own annotation can
    # push past is not a bound. So what goes is ``len(text) - room``, and room
    # depends on the marker's length, which depends on the count's digits.
    # The count only grows, so this settles in a step or two.
    dropped = len(text) - limit
    while True:
        marker = ELISION.format(dropped=dropped)
        room = limit - len(marker)
        if room <= 0:
            # Too small to hold the marker: keep the tail, the half worth
            # keeping. (Not ``text[-limit:]``, which is all of it at 0.)
            return text[len(text) - limit:]
        if len(text) - room == dropped:
            break
        dropped = len(text) - room
    head = room // 2
    return text[:head] + marker + text[len(text) - (room - head):]
