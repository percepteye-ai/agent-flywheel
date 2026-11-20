"""The turn span: a Flywheel key in its text is redacted, its content stays the
form Mission Control's reader rebuilds a conversation from, and the tracer is
published before the flag that says it was resolved."""
from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

import pytest

import agent_flywheel.tracing as mod
from agent_flywheel._http import REDACTED_KEY

pytest.importorskip("opentelemetry.sdk")

KEY = "pefw_" + "A" * 43


@pytest.fixture
def spans(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(mod, "_TRACER", provider.get_tracer("test"))
    monkeypatch.setattr(mod, "_RESOLVED", True)
    return exporter


def test_a_flywheel_key_never_reaches_the_turn_span(spans):
    mod.record_turn_span(input_text=f"my key is {KEY}, help",
                         output_text=f"use {KEY}", turn_id="t1")
    mod.record_turn_span(output_text=" " * (mod._MAX_TURN_CHARS - 10) + KEY,
                         turn_id="t2")
    texts = [v for s in spans.get_finished_spans()
             for k, v in s.attributes.items() if k.startswith("gen_ai.")
             and k.endswith(".messages")]
    assert len(texts) == 3
    assert all(KEY[:20] not in t for t in texts), texts
    assert REDACTED_KEY in texts[0] and REDACTED_KEY in texts[1]


def test_the_tracer_is_published_before_the_flag(monkeypatch):
    """A second thread asking while the first resolved read the flag, then
    the ``None`` beside it, and dropped its span."""
    from opentelemetry import trace

    monkeypatch.setattr(mod, "_TRACER", None)
    monkeypatch.setattr(mod, "_RESOLVED", False)
    resolving, release = threading.Event(), threading.Event()
    real = trace.get_tracer

    def slow(name, *a, **kw):
        resolving.set()
        release.wait(5)
        return real(name, *a, **kw)

    monkeypatch.setattr(trace, "get_tracer", slow)
    got: dict[str, object] = {}
    first = threading.Thread(target=lambda: got.update(a=mod._tracer()))
    first.start()
    assert resolving.wait(5)
    second = threading.Thread(target=lambda: got.update(b=mod._tracer()))
    second.start()
    second.join(0.3)
    release.set()
    first.join(5)
    second.join(5)
    assert got["a"] is not None and got["b"] is not None


MC_READER = Path("/Users/srini/Projects/MissionControl/agent_protocol")


def test_mission_controls_reader_rebuilds_the_conversation(spans, tmp_path):
    """The format is the reader's, not ours to change alone: the half-spans
    this SDK emits are rebuilt into one conversation. Spec message arrays on
    the same half-spans are not -- the answer half is refused for recording no
    prompt and every turn becomes its own segment."""
    if not (MC_READER / "agent_protocol" / "genai_traces.py").exists():
        pytest.skip("Mission Control's reader is not checked out here")
    sys.path.insert(0, str(MC_READER))
    try:
        from agent_protocol.genai_traces import decode_genai_conversations
    finally:
        sys.path.remove(str(MC_READER))

    for tid, q, a in (("T1", "hi", "hello"), ("T2", "and you?", "fine")):
        mod.record_turn_span(input_text=q, turn_id=tid, conversation_id="c1")
        mod.record_turn_span(output_text=a, turn_id=tid, conversation_id="c1")

    def otlp(span):
        def value(v):
            return {"stringValue": v} if isinstance(v, str) else {
                "intValue": str(v)}
        return {
            "traceId": format(span.context.trace_id, "032x"),
            "spanId": format(span.context.span_id, "016x"),
            "name": span.name,
            "startTimeUnixNano": str(span.start_time),
            "endTimeUnixNano": str(span.end_time),
            "attributes": [{"key": k, "value": value(v)}
                           for k, v in span.attributes.items()],
        }

    result = decode_genai_conversations(
        [otlp(s) for s in spans.get_finished_spans()])
    (conversation,) = result.conversations
    assert conversation.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "and you?"},
        {"role": "assistant", "content": "fine"},
    ]
    assert json.dumps(conversation.messages)
