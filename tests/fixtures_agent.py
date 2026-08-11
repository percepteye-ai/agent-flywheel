"""Importable agent fixtures for serve() registration tests."""
from __future__ import annotations


class _Plain:
    def invoke(self, task: str) -> str:
        return task


def build():
    return _Plain()


def build_langchain():
    from langchain_core.tools import tool

    @tool
    def lookup(q: str) -> str:
        """Look something up."""
        return q

    class _Agent:
        tools = [lookup]

        def invoke(self, task: str) -> str:
            return task

    return _Agent()
