"""Resolving the customer's agent to something callable.

Resolution happens ONCE, at registration, and raises there. It never happens
mid-rollout. That is not tidiness: a rollout is one member of a comparison
group, and a rollout that fails to emit because a method name was wrong does
not merely lose itself -- it biases the score of every sibling it was supposed
to be compared against. A typo must stop the agent at startup, in front of
the person who made it.

The input shape is DECLARED by the caller, never inferred. An earlier SDK
guessed the call convention from which method name it happened to find, so
`run(str)` and `run(dict)` were indistinguishable until something failed at
depth.
"""
from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Literal

from .errors import EntrypointError

__all__ = ["ENTRYPOINT_ORDER", "InputShape", "ResolvedEntrypoint", "resolve_entrypoint"]

InputShape = Literal["text", "messages", "dict"]

#: Probed in this order, first match wins.
ENTRYPOINT_ORDER = ("invoke", "run", "execute", "chat", "__call__")


class _TaskSentinel:
    """The position in an argv where this rollout's task text belongs.

    A SENTINEL rather than a ``"{task}"`` string, and that is a correctness
    choice rather than a stylistic one: a literal argument that happens to
    contain the placeholder cannot be mistaken for the marker, and the marker
    cannot survive into a real argv by accident -- substitution is the only
    thing that removes it, so an unsubstituted one is a bug that shows up as a
    type error rather than as an agent receiving the four characters
    ``{task}`` and doing something confident with them.
    """

    __slots__ = ()

    def __repr__(self) -> str:      # what a registration label shows
        return "<task>"


#: Marks where the task goes in a command argv::
#:
#:     fw.serve(["dsh", "--profile", "headless", fw.TASK], ...)
#:
#: Absent, the argv is untouched and the task arrives on stdin and in
#: PERCEPTEYE_TURN_INPUT exactly as before.
TASK = _TaskSentinel()


class ResolvedEntrypoint:
    """A callable plus everything we needed to learn about it at registration."""

    __slots__ = ("fn", "input_shape", "is_async", "name")

    def __init__(
        self,
        fn: Callable[..., Any],
        name: str,
        is_async: bool,
        input_shape: InputShape,
    ) -> None:
        self.fn = fn
        self.name = name
        self.is_async = is_async
        self.input_shape = input_shape

    def build_arg(self, turn_input: str) -> Any:
        if self.input_shape == "text":
            return turn_input
        if self.input_shape == "messages":
            return [{"role": "user", "content": turn_input}]
        return {"input": turn_input}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        kind = "async" if self.is_async else "sync"
        return f"<ResolvedEntrypoint {self.name}() {kind} shape={self.input_shape}>"


def resolve_entrypoint(
    obj: Any,
    *,
    input_shape: InputShape,
    method: str | None = None,
) -> ResolvedEntrypoint:
    """Find the callable that drives the customer's agent.

    Args:
        obj: the agent object, or a plain callable.
        input_shape: how ``turn_input`` should be passed. Declared, not guessed.
        method: force a specific attribute instead of probing.

    Raises:
        EntrypointError: at registration, listing what was actually available.
    """
    if input_shape not in ("text", "messages", "dict"):
        raise EntrypointError(
            f"input_shape must be 'text', 'messages' or 'dict', got {input_shape!r}"
        )

    if method is not None:
        fn = getattr(obj, method, None)
        if fn is None or not callable(fn):
            raise EntrypointError(
                f"entrypoint method {method!r} is not callable on "
                f"{type(obj).__name__}. Available callables: {_callables(obj)}"
            )
        return ResolvedEntrypoint(fn, method, _is_async(fn), input_shape)

    # A bare function or lambda is already the entrypoint.
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return ResolvedEntrypoint(obj, getattr(obj, "__name__", "<callable>"),
                                  _is_async(obj), input_shape)

    for name in ENTRYPOINT_ORDER:
        fn = getattr(obj, name, None)
        if callable(fn):
            return ResolvedEntrypoint(fn, name, _is_async(fn), input_shape)

    raise EntrypointError(
        f"no entrypoint found on {type(obj).__name__}. Tried, in order: "
        f"{', '.join(ENTRYPOINT_ORDER)}. Available callables: {_callables(obj)}.\n"
        f"Pass method='...' to name one explicitly, or pass a plain callable."
    )


def _is_async(fn: Callable[..., Any]) -> bool:
    if inspect.iscoroutinefunction(fn):
        return True
    inner = getattr(fn, "__call__", None)  # noqa: B004 - deliberate
    return bool(inner and inspect.iscoroutinefunction(inner))


def _callables(obj: Any) -> str:
    names = [
        n for n in dir(obj)
        if not n.startswith("_") and callable(getattr(obj, n, None))
    ]
    return ", ".join(sorted(names)[:12]) or "(none)"
