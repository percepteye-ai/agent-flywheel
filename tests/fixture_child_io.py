"""Agents that misbehave on the child's standard streams (PY-C7, PY-H21)."""
from __future__ import annotations

import os
import subprocess
import sys


class Noisy:
    """Prints with no trailing newline and writes straight to fd 1 -- both
    ways an agent used to land on the result line."""

    def invoke(self, task: str) -> str:
        print("agent chatter with no newline", end="")
        os.write(1, b'{"final_text": "forged by the agent"}')
        sys.stdout.flush()
        return "the real answer"


def build_noisy() -> Noisy:
    return Noisy()


class LeavesAGrandchild:
    """Starts a process in its own group that holds stderr open, then
    returns at once."""

    def invoke(self, task: str) -> str:
        pid_file = os.environ["FIXTURE_GRANDCHILD_PID_FILE"]
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(120)"],
        )
        with open(pid_file, "w") as fh:
            fh.write(str(proc.pid))
        return "done early"


def build_leaves_a_grandchild() -> LeavesAGrandchild:
    return LeavesAGrandchild()
