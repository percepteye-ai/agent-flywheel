"""What a user did with a PRODUCTION turn, on that turn's trace.

A production turn has no episode for ``POST /turn-feedback`` to join, so
``report_turn_feedback(signal, turn_id=)`` puts the feedback where the turn
went: the customer's TracerProvider, which an OTLP export delivers to the
control plane's intake with the turn. The turn's span has ended by then, so
the event rides a span of its own, linked to the turn's.

``tests/fixtures/turn_feedback_span.json`` is that export, byte for byte the
same in the Node SDK (``test/fixtures``), and what the control plane's reader
is driven with: the two SDKs and the reader are held to one artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_flywheel import InvalidFeedback
from agent_flywheel.contract import rfc3339_unix_nanos, turn_feedback_on_a_span

from .node_sdk import require_node_sdk, run_node
from .test_production_turn_upload import _attached, _Transport, _turn

pytest.importorskip("opentelemetry.sdk")

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "turn_feedback_span.json"
TURN = "turn-7"
AT = "2026-11-10T09:00:00.25Z"


@pytest.fixture
def spans(monkeypatch):
    """A real provider with sequential ids and an in-memory exporter, found by
    the SDK as a customer's would be."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )
    from opentelemetry.sdk.trace.id_generator import IdGenerator

    class _Sequential(IdGenerator):
        def __init__(self):
            self.trace = self.span = 0

        def generate_trace_id(self):
            self.trace += 1
            return self.trace

        def generate_span_id(self):
            self.span += 1
            return self.span

    import agent_flywheel.tracing as mod

    exporter = InMemorySpanExporter()
    provider = TracerProvider(id_generator=_Sequential())
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(mod, "_TRACER", provider.get_tracer("agent-flywheel"))
    monkeypatch.setattr(mod, "_RESOLVED", True)
    monkeypatch.setattr(mod, "_TURN_SPANS", type(mod._TURN_SPANS)())
    return exporter


def _value(v):
    if isinstance(v, bool):
        return {"boolValue": v}
    if isinstance(v, int):
        return {"intValue": str(v)}
    if isinstance(v, float):
        return {"doubleValue": v}
    return {"stringValue": str(v)}


def _attributes(attrs) -> list[dict]:
    return [{"key": k, "value": _value(v)} for k, v in sorted(dict(attrs).items())]


def otlp_json(finished) -> dict:
    """OTLP/JSON (hex ids, string nanos, sorted attributes) of the spans an
    exporter finished, with each span's own start and end replaced by its
    position -- the only values a run cannot repeat."""
    out = []
    for position, span in enumerate(finished):
        row = {
            "traceId": f"{span.context.trace_id:032x}",
            "spanId": f"{span.context.span_id:016x}",
            "name": span.name,
            "kind": 1,
            "startTimeUnixNano": str(1794301100000000000 + position * 2),
            "endTimeUnixNano": str(1794301100000000001 + position * 2),
            "attributes": _attributes(span.attributes),
        }
        if span.parent is not None:
            row["parentSpanId"] = f"{span.parent.span_id:016x}"
        if span.events:
            row["events"] = [{
                "name": e.name, "timeUnixNano": str(e.timestamp),
                "attributes": _attributes(e.attributes)} for e in span.events]
        if span.links:
            row["links"] = [{
                "traceId": f"{link.context.trace_id:032x}",
                "spanId": f"{link.context.span_id:016x}"} for link in span.links]
        out.append(row)
    return {"resourceSpans": [{"resource": {"attributes": []}, "scopeSpans": [
        {"scope": {"name": "agent-flywheel"}, "spans": out}]}]}


def _a_turn_and_its_feedback(monkeypatch, tmp_path, **feedback) -> str | None:
    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        _turn(px, TURN, ask="where is my order?", answer="It ships tomorrow.")
        return px.report_turn_feedback(
            "corrected", turn_id=TURN, occurred_at=AT, **feedback)


def test_reported_inside_another_span_it_is_still_a_trace_of_its_own(
    monkeypatch, tmp_path, spans,
):
    """The caller's request span is not the turn the user reacted to."""
    import agent_flywheel.tracing as mod

    with mod._TRACER.start_as_current_span("request"):
        _a_turn_and_its_feedback(monkeypatch, tmp_path)
    (feedback,) = [s for s in spans.get_finished_spans()
                   if s.name == "percepteye.turn.feedback"]
    assert feedback.parent is None


def test_feedback_on_a_production_turn_is_an_event_linked_to_its_turn(
    monkeypatch, tmp_path, spans,
):
    got = _a_turn_and_its_feedback(monkeypatch, tmp_path, feedback_id="fb-1")
    assert got == "fb-1"
    *turn, feedback = spans.get_finished_spans()
    assert turn and all(s.attributes["percepteye.turn.id"] == TURN for s in turn)
    assert feedback.name == "percepteye.turn.feedback"
    assert feedback.parent is None, "a trace of its own"
    assert [link.context.span_id for link in feedback.links] == [
        s.context.span_id for s in turn]
    assert feedback.attributes["percepteye.turn.id"] == TURN
    (event,) = feedback.events
    assert event.name == "percepteye.turn.feedback"
    assert event.timestamp == rfc3339_unix_nanos(AT) == 1794301200250000000
    assert dict(event.attributes) == {
        "percepteye.turn.feedback.signal": "corrected",
        "percepteye.turn.feedback.id": "fb-1"}


def test_the_export_is_the_shared_fixture(monkeypatch, tmp_path, spans):
    """Byte for byte what the Node SDK's copy holds and the control plane's
    reader is driven with."""
    _a_turn_and_its_feedback(monkeypatch, tmp_path, feedback_id="fb-1")
    emitted = json.dumps(otlp_json(spans.get_finished_spans()), indent=2) + "\n"
    assert emitted == FIXTURE.read_text(encoding="utf-8")


def test_a_turn_this_process_never_recorded_is_still_named(spans):
    import agent_flywheel.tracing as mod

    assert mod.record_turn_feedback_span(
        turn_id="elsewhere", signal="accepted",
        occurred_at_unix_nano=rfc3339_unix_nanos(AT))
    (span,) = spans.get_finished_spans()
    assert not span.links
    assert span.attributes["percepteye.turn.id"] == "elsewhere"
    assert "percepteye.turn.feedback.id" not in span.events[0].attributes


def test_a_rollouts_turn_is_not_linked(spans):
    """An episode's feedback goes to POST /turn-feedback; its turn's span is
    not remembered for a production report."""
    import agent_flywheel.tracing as mod

    mod.record_turn_span(output_text="x", turn_id=TURN, rollout_id="rl_1")
    assert mod.record_turn_feedback_span(
        turn_id=TURN, signal="accepted",
        occurred_at_unix_nano=rfc3339_unix_nanos(AT))
    assert not spans.get_finished_spans()[-1].links


def test_nothing_records_it_without_a_tracer(monkeypatch, tmp_path):
    import agent_flywheel.tracing as mod

    monkeypatch.setattr(mod, "_TRACER", None)
    monkeypatch.setattr(mod, "_RESOLVED", True)
    assert _a_turn_and_its_feedback(monkeypatch, tmp_path) is None


def test_no_episode_route_is_asked_for_a_production_turn(
    monkeypatch, tmp_path, spans,
):
    _a_turn_and_its_feedback(monkeypatch, tmp_path)
    assert not list(tmp_path.glob(".feedback.*")), "nothing queued for the route"


@pytest.mark.parametrize("kwargs, words", [
    ({"signal": "liked", "turn_id": TURN}, "signal must be one of"),
    ({"signal": "accepted", "turn_id": "../x"}, "turn_id is not a valid"),
    ({"signal": "accepted", "turn_id": TURN, "episode_id": "rl_1",
      "turn_index": 0}, "not both"),
    ({"signal": "accepted", "turn_id": TURN, "occurred_at": "2026-11-10"},
     "occurred_at must be an RFC 3339 time"),
])
def test_what_the_wire_cannot_carry_is_refused_at_the_call(
    monkeypatch, tmp_path, spans, kwargs, words,
):
    with _attached(monkeypatch, tmp_path, _Transport()) as px:
        with pytest.raises(InvalidFeedback, match=words):
            px.report_turn_feedback(**kwargs)
    assert spans.get_finished_spans() == ()


# ── one rule in both SDKs ──────────────────────────────────────────────────

#: name -> what a caller passes; validated and timed by both SDKs.
SPAN_FEEDBACK = {
    "every field": {"feedback_id": "fb-1", "turn_id": TURN,
                    "signal": "accepted", "occurred_at": AT},
    "an offset and nanoseconds": {"feedback_id": "fb-2", "turn_id": TURN,
                                  "signal": "abandoned",
                                  "occurred_at": "2026-11-10T09:00:00.123456789+05:30"},
    "more than nine fraction digits": {
        "feedback_id": "fb-3", "turn_id": TURN, "signal": "repeated",
        "occurred_at": "2026-11-10T09:00:00.1234567891Z"},
    "a signal outside the vocabulary": {
        "feedback_id": "fb-4", "turn_id": TURN, "signal": "liked",
        "occurred_at": AT},
    "a turn id that is a path": {"feedback_id": "fb-5", "turn_id": "a/b",
                                 "signal": "accepted", "occurred_at": AT},
    "an episode field": {"feedback_id": "fb-6", "turn_id": TURN,
                         "signal": "accepted", "occurred_at": AT,
                         "episode_id": "rl_1"},
    "no time": {"feedback_id": "fb-7", "turn_id": TURN, "signal": "accepted"},
    "a year below 100": {"feedback_id": "fb-8", "turn_id": TURN,
                         "signal": "escalated",
                         "occurred_at": "0050-01-01T00:00:00Z"},
}

_NODE = r"""
import { turnFeedbackOnASpan, rfc3339UnixNanos } from './src/wire.js';
let raw = '';
for await (const chunk of process.stdin) raw += chunk;
const out = {};
for (const [name, wire] of Object.entries(JSON.parse(raw))) {
  try {
    const ok = turnFeedbackOnASpan(wire);
    out[name] = { ok, nanos: String(rfc3339UnixNanos(ok.occurred_at)) };
  } catch (err) {
    out[name] = { refused: err.message };
  }
}
process.stdout.write(JSON.stringify(out));
"""


def _python(wire: dict) -> dict:
    try:
        ok = turn_feedback_on_a_span(wire)
    except InvalidFeedback as exc:
        return {"refused": str(exc)}
    return {"ok": ok, "nanos": str(rfc3339_unix_nanos(ok["occurred_at"]))}


def test_both_sdks_take_the_same_feedback_on_a_span():
    sdk, binary = require_node_sdk()
    theirs = json.loads(run_node(sdk, binary, _NODE, stdin=json.dumps(SPAN_FEEDBACK)))
    for name, wire in SPAN_FEEDBACK.items():
        assert theirs[name] == _python(wire), (
            f"{name}: Node {theirs[name]!r}, Python {_python(wire)!r}")
    assert theirs["an offset and nanoseconds"]["nanos"] == "1794281400123456789"
    assert "refused" in theirs["a signal outside the vocabulary"]
    # ...and the export both are held to is one file.
    assert (sdk / "test" / "fixtures" / FIXTURE.name).read_bytes() == (
        FIXTURE.read_bytes())
