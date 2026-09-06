#!/usr/bin/env python3
"""Your agent's real tool failure rate, from a trajectory this plugin recorded.

    python3 percepteye_report.py ./traj

Reads ``tool_calls.jsonl`` -- the file the hook recorder appends to -- and prints
what actually happened. No account, no network, no SDK: standard library only,
reading a local file. Someone who installed nothing but this plugin can run it.

WHY THIS EXISTS SEPARATELY FROM THE FLYWHEEL. The recorder's output is training
evidence, and training pays off on a training timescale. But the same file
answers a question people have on day one and cannot otherwise answer: of the
tool calls my agent made, which ones actually worked? A transcript will not tell
you -- a tool that swallows its exception and returns a string reads exactly
like one that succeeded. This does tell you, because the outcome came from the
host's own success/failure branch rather than from prose.

THE `unknown` COLUMN IS THE POINT, not a rounding error. Calls we could not
observe are held out of the failure rate entirely rather than being counted as
successes. A rate computed over calls nobody observed is the specific lie this
whole design exists to avoid, so it is reported as its own number.
"""
from __future__ import annotations

import json
import os
import sys

TOOL_CALLS_FILENAME = "tool_calls.jsonl"
_BAR_WIDTH = 28
_ORDER = ("ok", "failed", "unknown")


def _tty() -> bool:
    return sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _tty() else text


def _dim(t: str) -> str: return _c(t, "2")
def _bold(t: str) -> str: return _c(t, "1")


_COLOR = {"ok": "32", "failed": "31", "unknown": "33"}


def load(path: str) -> tuple[list[dict], int]:
    """Records and a malformed-line count.

    One unreadable line costs that line and nothing else: a rollout killed on
    its deadline truncates the final write, and discarding every outcome around
    it would turn a normal ending into an empty report.
    """
    records, malformed = [], 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if isinstance(rec, dict) and rec.get("name"):
                records.append(rec)
            else:
                malformed += 1
    return records, malformed


def _bar(n: int, total: int) -> str:
    # max(1, ...) because a non-zero count that rounds to zero blocks renders
    # as "no failures" beside a non-zero percentage -- 1 in 100 rounds to 0.
    filled = 0 if not n or not total else max(1, round(_BAR_WIDTH * n / total))
    return "█" * filled + _dim("░" * (_BAR_WIDTH - filled))


def _pct(n: int, total: int) -> str:
    return "  0%" if not total else f"{round(100 * n / total):3d}%"


def render(records: list[dict], malformed: int, source: str) -> str:
    total = len(records)
    counts = {k: 0 for k in _ORDER}
    for rec in records:
        outcome = rec.get("outcome")
        counts[outcome if outcome in counts else "unknown"] += 1

    out = ["", f"  {_bold('percepteye')} {_dim('·')} {source}", ""]
    out.append(f"  {_bold(str(total))} tool call{'' if total == 1 else 's'}")
    out.append(f"  {_dim('─' * 46)}")
    for key in _ORDER:
        n = counts[key]
        label = _c(f"{key:<8}", _COLOR[key]) if n else f"{key:<8}"
        out.append(f"  {label} {n:>4}  {_bar(n, total)} {_pct(n, total)}")
    out.append("")

    # The rate is computed over OBSERVED calls only. Including `unknown` in the
    # denominator would silently dilute the failure rate with calls nobody
    # watched, which reads as "things are fine" for exactly the wrong reason.
    observed = counts["ok"] + counts["failed"]
    if observed:
        failed = counts["failed"]
        rate = round(100 * failed / observed)
        headline = _bold(f"Real failure rate: {rate}%")
        detail = _dim(f"({failed} of {observed} observed calls)")
        out.append(f"  {headline}  {detail}")
    else:
        out.append(f"  {_dim('No observed outcomes yet.')}")
    if counts["unknown"]:
        out.append(_dim(f"  {counts['unknown']} call(s) where nothing observed the outcome — "
                        f"held out of that rate, never counted as success."))
    out.append("")

    by_tool: dict[str, list[int]] = {}
    for rec in records:
        slot = by_tool.setdefault(rec["name"], [0, 0])
        slot[1] += 1
        if rec.get("outcome") == "failed":
            slot[0] += 1
    failing = sorted(((n, f, t) for n, (f, t) in by_tool.items() if f),
                     key=lambda r: (-r[1], r[0]))
    if failing:
        out.append(f"  {_bold('Where the failures are')}")
        for name, failed, tool_total in failing[:8]:
            out.append(f"    {name:<24} {failed:>3} failed / {tool_total:>3} calls"
                       f"   {_pct(failed, tool_total)}")
        out.append("")

    recent = [r for r in records if r.get("outcome") == "failed" and r.get("error")]
    if recent:
        out.append(f"  {_bold('Recent failures')}")
        for rec in recent[-3:]:
            err = " ".join(str(rec["error"]).split())[:64]
            out.append(f"    {rec['name']:<24} {_dim(err)}")
        out.append("")

    if malformed:
        out.append(_dim(f"  {malformed} unreadable line(s) skipped."))
        out.append("")
    return "\n".join(out)


def _empty_help(target: str) -> str:
    return "\n".join([
        "",
        f"  No trajectory at {_bold(target)}.",
        "",
        "  Run your agent once with the plugin installed and it records a",
        f"  directory per session; {_bold('percepteye report')} then reads the most",
        "  recent one. Check the plugin is registered with:",
        "",
        f"    {_bold('percepteye status')}",
        "",
        "  To capture one run into a directory you name, set",
        f"  {_bold('PERCEPTEYE_TRAJECTORY_DIR')} and point this at the same path.",
        f"  {_bold('PERCEPTEYE_CAPTURE=0')} turns recording off entirely.",
        "",
    ])


def main(argv: list[str]) -> int:
    target = argv[1] if len(argv) > 1 else os.environ.get("PERCEPTEYE_TRAJECTORY_DIR") or "."
    path = target if os.path.isfile(target) else os.path.join(target, TOOL_CALLS_FILENAME)
    if not os.path.isfile(path):
        print(_empty_help(target))
        return 1

    # Show the shorter of absolute and cwd-relative: an absolute trajectory
    # path is usually long enough to wrap the header onto a second line.
    try:
        shown = os.path.relpath(path)
    except ValueError:
        # No relative path exists (different drives on Windows, or a deleted
        # cwd). A display nicety must never take down the report.
        shown = os.path.abspath(path)
    if len(os.path.abspath(path)) < len(shown):
        shown = os.path.abspath(path)

    records, malformed = load(path)
    if not records and not malformed:
        print(_empty_help(target))
        return 1
    print(render(records, malformed, shown))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
