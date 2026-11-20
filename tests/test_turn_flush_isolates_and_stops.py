"""The turn upload goes through the outcomes' sender: a turn the intake or a
proxy refuses as a record is isolated, an outage ends the flush, every unsent
turn is counted once, and a turn whose scope is still open is never sent.

Before, a 413 or a batch-level 4xx on one chunk was a blip: that one oversized
or invalid turn held back the 63 healthy turns beside it on every flush, for
good. A transport failure moved on to the next chunk, which waited out the
same outage. A flush in which every turn failed to render said ``failed: 0``.
"""
from __future__ import annotations

import warnings
from contextlib import contextmanager

import pytest

import agent_flywheel as fw
from agent_flywheel.errors import TransportError

from .test_production_turn_upload import _attached, _Transport


class _Intake(_Transport):
    """``refuse(ids)`` answers a batch: ``None`` to accept it, else the error
    to raise. Every batch tried is recorded."""

    def __init__(self, refuse, **kw):
        super().__init__(capture=True, **kw)
        self.refuse = refuse
        self.tried: list[list[str]] = []

    def register(self, spec):
        return {"capture": {"turns": True, "model_content": True}}

    def report_turns(self, agent_id, turns):
        ids = [t["turn_id"] for t in turns]
        self.tried.append(ids)
        err = self.refuse(ids)
        if err is not None:
            raise err
        return super().report_turns(agent_id, turns)


@contextmanager
def _with_turns(monkeypatch, tmp_path, tr, n):
    monkeypatch.setattr(fw.Attachment, "_flush_soon", lambda self: None)
    with _attached(monkeypatch, tmp_path, tr) as px:
        for i in range(n):
            with px.turn(turn_id=f"t{i:03d}", input_text="q"):
                pass
            px.record_answer(f"t{i:03d}", "a")
        yield px


def _too_large(ids):
    if "t007" in ids:
        return TransportError("POST /turns -> HTTP 413: Request Entity Too "
                              "Large", status_code=413)
    return None


def test_one_turn_too_large_never_holds_back_the_rest(monkeypatch, tmp_path):
    tr = _Intake(_too_large)
    with _with_turns(monkeypatch, tmp_path, tr, 20) as px:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            counts = px.flush(batch_size=64)
        assert sorted(t["turn_id"] for t in tr.sent) == [
            f"t{i:03d}" for i in range(20) if i != 7]
        assert counts["sent"] == 19 and counts["dropped"] == 1
        assert counts["failed"] == 0
        assert any("413" in str(w.message) and "too large" in str(w.message)
                   for w in caught)
        # Never sent again, and never deleted: the directory is the customer's.
        tr.tried.clear()
        assert px.flush()["sent"] == 0 and tr.tried == []
        assert (tmp_path / "t007").is_dir()


def test_an_outage_ends_the_flush_and_counts_each_turn_once(
        monkeypatch, tmp_path):
    tr = _Intake(lambda ids: TransportError("POST /turns failed after 4 "
                                            "attempts: HTTP 503"))
    with _with_turns(monkeypatch, tmp_path, tr, 10) as px:
        counts = px.flush(batch_size=4)
        assert len(tr.tried) == 1, "the next chunk waited out the same outage"
        assert counts["sent"] == 0 and counts["failed"] == 10
        assert px.may_upload is True, "a blip is not a refusal"


def test_a_403_counts_each_unsent_turn_once(monkeypatch, tmp_path):
    calls = []

    def blip_then_refuse(ids):
        calls.append(ids)
        if len(calls) == 1:
            return None
        return TransportError("POST /turns -> HTTP 403: capture is off",
                              status_code=403)

    tr = _Intake(blip_then_refuse)
    with _with_turns(monkeypatch, tmp_path, tr, 10) as px:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            counts = px.flush(batch_size=4)
        assert (counts["sent"], counts["failed"]) == (4, 6)
        assert counts["not_permitted"] == 1


def test_turns_that_cannot_be_rendered_are_counted(monkeypatch, tmp_path):
    tr = _Intake(lambda ids: None)
    with _with_turns(monkeypatch, tmp_path, tr, 3) as px:
        def broken(d):
            raise ValueError("unreadable")
        monkeypatch.setattr(px, "_turn_wire", broken)
        assert px.flush()["failed"] == 3


def test_a_turn_still_running_is_not_sent_even_at_close(monkeypatch, tmp_path):
    tr = _Intake(lambda ids: None)
    monkeypatch.setattr(fw.Attachment, "_flush_soon", lambda self: None)
    with _attached(monkeypatch, tmp_path, tr) as px:
        with px.turn(turn_id="live", input_text="q"):
            px.record_answer("live", "partial")
            counts = px.flush(include_unanswered=True)
            assert tr.sent == [] and counts["skipped"] == 1
        assert px.flush(include_unanswered=True)["sent"] == 1


def test_the_answer_span_names_its_conversation(monkeypatch, tmp_path):
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import agent_flywheel.tracing as mod

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(mod, "_TRACER", provider.get_tracer("test"))
    monkeypatch.setattr(mod, "_RESOLVED", True)
    px = fw.attach(agent_id="a", trajectory_root=str(tmp_path), register=False)
    with px.turn(turn_id="t1", conversation_id="c-9", input_text="q"):
        pass
    px.record_answer("t1", "a")
    px.close()
    chats = [s for s in exporter.get_finished_spans() if s.name == "chat"]
    assert [s.attributes.get("percepteye.conversation.id")
            for s in chats] == ["c-9", "c-9"]
