"""The model-assigned call id, threaded end to end.

``tool_call_id`` is the join key that lets the control plane say WHICH
completion a tool call followed: the same id reappears on the ``role="tool"``
message of the agent's next request, so attribution needs no clocks and
survives concurrent tools. Without it, every reported call can only be placed
"somewhere in this rollout" -- which is why, downstream, an entire episode's
tool signal lands on its final step today.

The field crosses four serialisation boundaries (ledger write, ledger read,
report wire, wire reconstruct). Each one is a silent-drop point: a boundary
that forgets the key loses attribution with no error anywhere. These tests pin
every boundary individually, because a single end-to-end test would pass while
two boundaries cancel each other out.
"""
from __future__ import annotations

import asyncio
import json
import pathlib

import pytest

from percepteye_agent_flywheel.contract import ToolCallOutcome
from percepteye_agent_flywheel.outcomes import (
    LEDGER_DIR_ENV,
    ROLLOUT_ID_ENV,
    OutcomeLedger,
    _RolloutScope,
    record_tool_call,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


# ── the contract ────────────────────────────────────────────────────────────


def test_the_id_is_optional_and_defaults_to_none():
    """None means the framework exposed no id. The call still counts."""
    rec = ToolCallOutcome(name="t", arguments={}, outcome="ok", status_code=200)
    assert rec.tool_call_id is None
    assert "tool_call_id" not in rec.to_wire(), (
        "an absent id must be ABSENT on the wire, not null -- absent and "
        "unattributable are the same claim, a null key is a second way to "
        "spell it"
    )


def test_the_id_rides_the_wire_when_present():
    rec = ToolCallOutcome(name="t", arguments={}, outcome="ok",
                          status_code=200, tool_call_id="call_abc123")
    assert rec.to_wire()["tool_call_id"] == "call_abc123"


def test_an_unknown_outcome_keeps_its_id():
    """The unknown-path scrubbing removes OUTCOME evidence (status, status-like
    body keys) because an unknown outcome must not smuggle a verdict. The id is
    ATTRIBUTION, not a verdict -- and the unknown calls are exactly the ones
    where knowing which completion they followed matters most."""
    rec = ToolCallOutcome(name="t", arguments={}, outcome="unknown",
                          status_code=200, tool_call_id="call_x")
    assert rec.status_code is None, "sanity: the scrub still fires"
    assert rec.tool_call_id == "call_x", (
        "the scrub took the id with it -- attribution was conflated with "
        "outcome evidence"
    )


# ── every serialisation boundary, individually ──────────────────────────────


def test_ledger_roundtrip_preserves_the_id(tmp_path):
    with _RolloutScope("rl_id", str(tmp_path)):
        record_tool_call("deploy", {"svc": "api"}, outcome="ok",
                         status_code=200, tool_call_id="call_dep_1")
        record_tool_call("lookup", {}, outcome="ok", status_code=200)

    calls = OutcomeLedger(str(tmp_path), "rl_id").read()
    by = {c.name: c for c in calls}
    assert by["deploy"].tool_call_id == "call_dep_1"
    assert by["lookup"].tool_call_id is None


def test_the_report_wire_reconstruct_preserves_the_id():
    """runner._output_from_wire is the LAST boundary before the control plane;
    a drop here survives every other test in this file."""
    from percepteye_agent_flywheel.runner import _output_from_wire

    out = _output_from_wire({
        "final_text": "done",
        "tool_calls": [
            {"name": "t", "arguments": {}, "outcome": "ok",
             "status_code": 200, "tool_call_id": "call_w1"},
            {"name": "u", "arguments": {}, "outcome": "ok", "status_code": 200},
        ],
    })
    assert out.tool_calls[0].tool_call_id == "call_w1"
    assert out.tool_calls[1].tool_call_id is None


def test_the_wire_shape_is_legal_under_the_schema():
    """ToolCallRecord is additionalProperties:false -- a wire key the schema
    does not name makes EVERY record carrying it invalid at the server. This
    is the test that fails when someone adds a field to to_wire and not to the
    schema."""
    schema = json.loads((ROOT / "schema" / "flywheel-1.json").read_text())
    allowed = set(schema["$defs"]["ToolCallRecord"]["properties"])
    rec = ToolCallOutcome(name="t", arguments={"a": 1}, outcome="failed",
                          status_code=409, output={"e": "conflict"},
                          error="HTTP 409", error_class="Conflict",
                          latency_ms=12.0, agent_name="planner",
                          tool_call_id="call_s1")
    wire = set(rec.to_wire())
    assert wire <= allowed, (
        f"to_wire emits keys the schema does not allow: {wire - allowed}"
    )
    assert "tool_call_id" in allowed


# ── the adapters actually populate it ───────────────────────────────────────


def test_langchain_adapter_reads_the_id_off_the_tool_message(monkeypatch, tmp_path):
    lc = pytest.importorskip("langchain_core")  # noqa: F841
    from langchain_core.messages.tool import ToolMessage

    from percepteye_agent_flywheel.adapters.langchain import (
        LangChainAdapter,
        _build_handler_cls,
    )

    monkeypatch.setenv(ROLLOUT_ID_ENV, "rl_lcid")
    monkeypatch.setenv(LEDGER_DIR_ENV, str(tmp_path))
    adapter = LangChainAdapter()
    unbind = adapter.bind(llm_url="https://gw/x", api_key="k")
    try:
        h = _build_handler_cls()()
        h.on_tool_start({"name": "search"}, "q", run_id="r1")
        h.on_tool_end(
            ToolMessage(content="ok", tool_call_id="call_lc_7", status="success"),
            run_id="r1",
        )
        # A tool returning a plain value has no id: None, never invented.
        h.on_tool_start({"name": "plain"}, "q", run_id="r2")
        h.on_tool_end("just a string", run_id="r2")
    finally:
        unbind()

    by = {c.name: c for c in OutcomeLedger(str(tmp_path), "rl_lcid").read()}
    assert by["search"].tool_call_id == "call_lc_7"
    assert by["plain"].tool_call_id is None


def test_adk_adapter_uses_the_function_call_id(tmp_path):
    pytest.importorskip("google.adk")
    from google.adk.plugins.plugin_manager import PluginManager

    from percepteye_agent_flywheel.adapters.adk import ADKAdapter

    class _Tool:
        name = "lookup"

    class _Ctx:
        function_call_id = "fc_42"
        agent_name = "planner"

    adapter = ADKAdapter()
    unbind = adapter.bind(llm_url="https://gw.invalid/v1", api_key="perk_x")
    try:
        pm = PluginManager()

        async def drive():
            await pm.run_before_tool_callback(
                tool=_Tool(), tool_args={"q": "x"}, tool_context=_Ctx()
            )
            await pm.run_after_tool_callback(
                tool=_Tool(), tool_args={"q": "x"}, tool_context=_Ctx(),
                result={"ok": True},
            )

        with _RolloutScope("rl_adkid", str(tmp_path)):
            asyncio.run(drive())
    finally:
        unbind()

    calls = OutcomeLedger(str(tmp_path), "rl_adkid").read()
    assert calls and calls[0].tool_call_id == "fc_42"


# ── the OTel projection carries the convention's own attribute ──────────────


def test_the_span_carries_gen_ai_tool_call_id(monkeypatch):
    """gen_ai.tool.call.id is one attribute the convention DOES define. A trace
    consumer joins the tool span to the completion that requested it by this
    value -- the same join this SDK's control plane performs on the ledger."""
    pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    import percepteye_agent_flywheel.tracing as mod

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(mod, "_TRACER", provider.get_tracer("test"))
    monkeypatch.setattr(mod, "_RESOLVED", True)

    mod.record_tool_call_span("lookup", {}, outcome="ok", status_code=200,
                              tool_call_id="call_span_1")
    mod.record_tool_call_span("bare", {}, outcome="ok", status_code=200)

    spans = exporter.get_finished_spans()
    assert spans[0].attributes["gen_ai.tool.call.id"] == "call_span_1"
    assert "gen_ai.tool.call.id" not in spans[1].attributes
