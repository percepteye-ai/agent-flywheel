"""``read_only`` on a tool: three-valued, framework-neutral, joined by segment.

A tool that declares nothing is NOT a tool that declares it writes. Downstream,
"undeclared" keeps a task out of the held-out pool (conservative-write), while
"declares read" makes it eligible. Collapsing the two either grades a write as a
read or refuses every honest agent, so the distinction is carried end to end:
absent from the wire when unset, never serialised as ``null``.

The trait itself never reaches the host's conversation hook — by the time tools
are handed to a model they are in the provider's function-call shape, which has
no field for it. A transport-level observer writes what each server declared
into ``tool_traits.<pid>.json`` beside the description, and it is joined here.
"""
from __future__ import annotations

import json

from percepteye_agent_flywheel.host_description import (
    _suffix_is_a_segment,
    _tool_from_wire,
    read_discovered_agent,
)
from percepteye_agent_flywheel.introspection import ToolDefinition


def _write_agent(d, names):
    (d / "discovered_agent.json").write_text(json.dumps({
        "agent_name": "a", "framework": "openclaw", "source_type": "sdk_runtime",
        "role": "r", "system_prompt": "p", "discovered_model_config": {},
        "tool_definitions": [{"name": n} for n in names],
    }), encoding="utf-8")


class TestWireContract:
    def test_undeclared_is_absent_not_null(self):
        assert "read_only" not in ToolDefinition(name="t").to_wire()

    def test_declared_survives_both_ways(self):
        for value in (True, False):
            wire = ToolDefinition(name="t", read_only=value).to_wire()
            assert wire["read_only"] is value
            assert _tool_from_wire(wire).read_only is value

    def test_only_a_real_bool_counts(self):
        """null / a string / a number are all 'the host did not say'."""
        for raw in (None, "yes", 1, [], {}):
            assert _tool_from_wire({"name": "t", "read_only": raw}).read_only is None


class TestSegmentJoin:
    def test_namespaced_tool_is_claimed_by_its_bare_name(self):
        assert _suffix_is_a_segment("multivendor-cli-mcp__server_info", "server_info")

    def test_a_word_ending_is_not_a_segment(self):
        """The one that bites: a trait for `facts` must not claim `get_facts`.

        A single `_` cannot be the test — it is both the namespace separator's
        last character and an ordinary word separator.
        """
        assert not _suffix_is_a_segment("multivendor-cli-mcp__get_facts", "facts")

    def test_other_host_separators_work_without_being_hardcoded(self):
        assert _suffix_is_a_segment("memory-mcp.list_items", "list_items")
        assert _suffix_is_a_segment("prefix/list", "list")


class TestMerge:
    def test_declared_traits_reach_the_catalogue(self, tmp_path):
        _write_agent(tmp_path, ["mcp__server_info", "mcp__run_fleet"])
        (tmp_path / "tool_traits.1.json").write_text(
            json.dumps({"server_info": True, "run_fleet": False}), encoding="utf-8")
        got = {t.name: t.read_only for t in read_discovered_agent(tmp_path).tool_definitions}
        assert got == {"mcp__server_info": True, "mcp__run_fleet": False}

    def test_two_servers_disagreeing_leaves_BOTH_undeclared(self, tmp_path):
        """Ambiguity resolves to absence, never to a winner.

        Two servers each declare `list_items` and disagree. Picking one would
        invent a fact about somebody's tool out of a name collision.
        """
        _write_agent(tmp_path, ["a-mcp__list_items", "b-mcp__list_items"])
        (tmp_path / "tool_traits.1.json").write_text(
            json.dumps({"list_items": True}), encoding="utf-8")
        (tmp_path / "tool_traits.2.json").write_text(
            json.dumps({"list_items": False}), encoding="utf-8")
        for t in read_discovered_agent(tmp_path).tool_definitions:
            assert t.read_only is None, t.name

    def test_a_tool_nobody_described_stays_undeclared(self, tmp_path):
        _write_agent(tmp_path, ["exec"])
        (tmp_path / "tool_traits.1.json").write_text(
            json.dumps({"server_info": True}), encoding="utf-8")
        assert read_discovered_agent(tmp_path).tool_definitions[0].read_only is None

    def test_no_trait_files_is_not_an_error(self, tmp_path):
        _write_agent(tmp_path, ["exec"])
        agent = read_discovered_agent(tmp_path)
        assert agent is not None and agent.tool_definitions[0].read_only is None
