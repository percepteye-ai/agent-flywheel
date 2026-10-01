"""Framework-neutral projections from a command's structured event stream.

Core owns the generic projection, pairing, and absence rules. A host-specific
adapter supplies its event vocabulary and any lifecycle semantics. This keeps
customer agent, harness, control-event, and transcript heuristics out of the
SDK core.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from ._body_bound import bounded
from .contract import Outcome, ToolCallOutcome
from .evidence import extract_entity_ids

__all__ = [
    "FORMATS",
    "OTEL_GENAI",
    "CommandOutputAdapter",
    "CommandOutputProjection",
    "StreamFormat",
    "ToolCallParseResult",
    "iter_json_events",
    "parse_tool_calls",
    "parse_tool_calls_with_omissions",
]


@dataclass(frozen=True)
class CommandOutputProjection:
    """Evidence one explicit adapter projected from command stdout.

    None preserves absence. Core never guesses a model-call count from
    transcript shape and never treats lifecycle JSON as the command's answer.
    """

    recognized: bool = False
    final_text: str | None = None
    llm_call_count: int | None = None
    tool_calls: list[ToolCallOutcome] | None = None
    tool_calls_omitted_count: int = 0
    exclusion_reason: str | None = None
    exclusion_detail: str | None = None


class CommandOutputAdapter(Protocol):
    """Optional adapter that interprets one command's stdout evidence."""

    @property
    def adapter_name(self) -> str: ...

    def project(self, stdout: str) -> CommandOutputProjection: ...


@dataclass(frozen=True)
class StreamFormat:
    """Declarative vocabulary for tool-call lifecycle events."""

    name: str
    start_type: str
    end_type: str
    call_id: tuple[str, ...] = ("toolCallId", "tool_call_id", "id")
    tool_name: tuple[str, ...] = ("toolName", "tool_name", "name")
    args: tuple[str, ...] = ("args", "arguments", "input")
    result: tuple[str, ...] = ("result", "output", "response")
    error_flag: tuple[str, ...] = ("isError", "is_error", "error")
    # An adapter may declare structured failure evidence that is stronger than
    # a generic success flag. Core applies the rule without knowing the host.
    result_status_path: tuple[str, ...] | None = None
    failed_status_values: tuple[str, ...] = ()


@dataclass(frozen=True)
class ToolCallParseResult:
    """Representable calls plus executions known to be incomplete."""

    calls: list[ToolCallOutcome] | None
    omitted_count: int = 0


# A standards vocabulary is safe as the inspection helper's default. The
# rollout runner does not apply even this automatically: a command must opt in
# with a CommandOutputAdapter before stdout is interpreted as telemetry.
OTEL_GENAI = StreamFormat(
    name="otel-genai",
    start_type="gen_ai.tool.start",
    end_type="gen_ai.tool.end",
    call_id=("gen_ai.tool.call.id", "toolCallId", "id"),
    tool_name=("gen_ai.tool.name", "toolName", "name"),
    args=("gen_ai.tool.input", "args", "arguments"),
    result=("gen_ai.tool.output", "result", "output"),
    error_flag=("error.type", "isError", "is_error"),
)
FORMATS: tuple[StreamFormat, ...] = (OTEL_GENAI,)


def iter_json_events(stdout: str):
    """Yield JSON objects from a noisy JSONL stream."""
    for line in stdout.splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if isinstance(value, dict):
            yield value


def _first(value: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if key in value:
            return value[key]
    return None


def _at_path(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for part in path:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _outcome_of(end_event: dict[str, Any], fmt: StreamFormat) -> Outcome:
    """Read an adapter-declared verdict while preserving the tri-state."""
    result = _first(end_event, fmt.result)
    if fmt.result_status_path and fmt.failed_status_values:
        status = _at_path(result, fmt.result_status_path)
        if (
            isinstance(status, str)
            and status.strip().lower() in fmt.failed_status_values
        ):
            return "failed"

    saw_success = False
    for holder in (end_event, result):
        if not isinstance(holder, dict):
            continue
        flag = _first(holder, fmt.error_flag)
        if isinstance(flag, bool):
            if flag:
                return "failed"
            saw_success = True
        elif isinstance(flag, str) and flag.strip():
            return "failed"
    return "ok" if saw_success else "unknown"


def _text_of(result: Any) -> Any:
    """The readable body, bounded, or the raw structured value.

    Every readable body is bounded HERE, once, and a failed call's error is
    that same body: an output and an error get one budget, so a second bound
    at the error would cut nothing. A result that is itself a string (a
    ``gen_ai.tool.output`` can be one) is a readable body too; its output
    used to pass through unbounded.
    """
    if isinstance(result, str):
        return bounded(result)
    if not isinstance(result, dict):
        return result
    content = result.get("content")
    if isinstance(content, list):
        parts = [
            item["text"]
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        if parts:
            return bounded("".join(parts))
    details = result.get("details")
    if isinstance(details, dict) and isinstance(details.get("stdout"), str):
        return bounded(details["stdout"])
    return result


def parse_tool_calls_with_omissions(
    stdout: str,
    *,
    formats: tuple[StreamFormat, ...] = FORMATS,
) -> ToolCallParseResult:
    """Project tool outcomes under an explicit event vocabulary.

    None means no event from any supplied vocabulary was observed. Starts
    left open, duplicate starts, and unnamed ends increment omitted_count so a
    surviving partial list cannot receive a training grade.
    """
    if not stdout or not formats:
        return ToolCallParseResult(calls=None)

    matched: StreamFormat | None = None
    started: dict[str, dict[str, Any]] = {}
    anonymous_started: list[dict[str, Any]] = []
    calls: list[ToolCallOutcome] = []
    saw_any = False
    omitted = 0

    for event in iter_json_events(stdout):
        kind = event.get("type")
        fmt = matched
        if fmt is None:
            for candidate in formats:
                if kind in (candidate.start_type, candidate.end_type):
                    fmt = matched = candidate
                    break
        if fmt is None or kind not in (fmt.start_type, fmt.end_type):
            continue

        if kind == fmt.start_type:
            call_id = str(_first(event, fmt.call_id) or "")
            if call_id:
                if call_id in started:
                    omitted += 1
                started[call_id] = event
            else:
                anonymous_started.append(event)
            saw_any = True
            continue

        saw_any = True
        call_id = str(_first(event, fmt.call_id) or "")
        if call_id:
            start = started.pop(call_id, {})
        else:
            start = anonymous_started.pop(0) if anonymous_started else {}
        name = str(
            _first(event, fmt.tool_name) or _first(start, fmt.tool_name) or ""
        )
        if not name:
            omitted += 1
            continue

        arguments = _first(event, fmt.args)
        if arguments is None:
            arguments = _first(start, fmt.args)
        if not isinstance(arguments, dict):
            arguments = {} if arguments is None else {"_raw": arguments}
        result = _first(event, fmt.result)
        outcome = _outcome_of(event, fmt)

        output = _text_of(result)
        error: str | None = None
        if outcome == "failed" and isinstance(output, str) and output.strip():
            error = output

        latency = None
        details = result.get("details") if isinstance(result, dict) else None
        if isinstance(details, dict):
            milliseconds = details.get("durationMs")
            if (
                isinstance(milliseconds, (int, float))
                and not isinstance(milliseconds, bool)
            ):
                latency = float(milliseconds)

        calls.append(
            ToolCallOutcome(
                name=name,
                arguments=arguments,
                outcome=outcome,
                status_code=None,
                output=output,
                error=error,
                latency_ms=latency,
                tool_call_id=call_id or None,
                entity_ids=extract_entity_ids(result),
            )
        )

    omitted += len(started) + len(anonymous_started)
    if not saw_any:
        return ToolCallParseResult(calls=None)
    return ToolCallParseResult(calls=calls, omitted_count=omitted)


def parse_tool_calls(
    stdout: str,
    *,
    formats: tuple[StreamFormat, ...] = FORMATS,
) -> list[ToolCallOutcome] | None:
    """Inspection projection without omission metadata."""
    return parse_tool_calls_with_omissions(stdout, formats=formats).calls
