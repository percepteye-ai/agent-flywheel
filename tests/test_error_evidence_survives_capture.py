"""Raw error payloads are the load-bearing signal, not a diagnostic nicety.

ReTool §3.3 attributes emergent self-correction to feeding verbatim tracebacks
back to the model. Everything downstream that learns from failure — recovery-
window preference mining, judge distillation, hint extraction — reads this
text. If it does not survive capture, those channels are starved silently.

The failure mode this pins is specific and was live: the error body got a
QUARTER of the ordinary output budget and was head-truncated, so on any real
traceback the first thing deleted was the exception class and message at the
end — the only part a model can learn to correct from.
"""
from __future__ import annotations

from percepteye_agent_flywheel.event_stream import (
    _ELISION,
    _MAX_ERROR_CHARS,
    _MAX_OUTPUT_CHARS,
    _bounded,
)

_TRACEBACK_TAIL = 'ValueError: city "Paris " has a trailing space'


def _long_traceback(n_frames: int = 400) -> str:
    frames = "".join(
        f'  File "/app/mod{i}.py", line {i}, in handler\n    return step{i}()\n'
        for i in range(n_frames)
    )
    return "Traceback (most recent call last):\n" + frames + _TRACEBACK_TAIL


def test_a_body_within_the_bound_is_returned_byte_identical():
    """The common case must not acquire a marker it did not earn."""
    body = "stdout: fine\n"
    assert _bounded(body) == body
    assert _ELISION.format(dropped=1) not in _bounded(body)


def test_the_exception_line_survives_a_long_traceback():
    """Head-truncation deleted exactly this."""
    out = _bounded(_long_traceback(), _MAX_ERROR_CHARS)
    assert out.endswith(_TRACEBACK_TAIL)
    assert out.startswith("Traceback (most recent call last):")


def test_the_bound_is_honoured_including_its_own_marker():
    """A bound that its own annotation can push past is not a bound."""
    out = _bounded(_long_traceback(), _MAX_ERROR_CHARS)
    assert len(out) <= _MAX_ERROR_CHARS


def test_the_elision_is_visible_and_counts_what_went():
    """A reader must be able to tell a window from a whole, and a census must
    be able to count it without a new wire field."""
    text = _long_traceback()
    out = _bounded(text, _MAX_ERROR_CHARS)
    assert "elided by the SDK capture bound" in out
    dropped = len(text) - _MAX_ERROR_CHARS
    assert str(dropped) in out


def test_errors_are_not_budgeted_below_ordinary_output():
    """They used to get a quarter. The payload most worth keeping was the one
    kept least."""
    assert _MAX_ERROR_CHARS >= _MAX_OUTPUT_CHARS


def test_a_limit_too_small_for_the_marker_keeps_the_tail():
    """Degenerate, but it must degrade toward the half worth keeping."""
    out = _bounded(_long_traceback(), 20)
    assert len(out) == 20
    assert out.endswith(_TRACEBACK_TAIL[-20:])
