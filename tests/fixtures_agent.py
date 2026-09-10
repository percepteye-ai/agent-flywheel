"""Importable agent fixtures for serve() registration tests."""
from __future__ import annotations

import os
from typing import ClassVar


class _Plain:
    def invoke(self, task: str) -> str:
        return task


def build():
    return _Plain()


class _EnvironmentProbe:
    def invoke(self, task: str) -> str:
        return f"{task}|{os.environ.get('FAKE_EXECUTION_MARKER', '<absent>')}"


def build_environment_probe():
    return _EnvironmentProbe()


def build_langchain():
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return q

    class _Agent:
        tools: ClassVar[list] = [lookup]

        def invoke(self, task: str) -> str:
            return task

    return _Agent()
