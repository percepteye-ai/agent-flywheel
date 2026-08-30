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


class TestNoMergeMachinery:
    """The trait arrives already namespaced, so there is nothing to join.

    An earlier version of this had the SDK glue bare per-server names onto
    namespaced catalogue entries by suffix. That was solving a problem the
    describe path does not have: it asks each server for `tools/list` itself and
    prefixes the name as the host bridges it, so `read_only` is set on the entry
    that already carries its final name. The suffix-join had no producer left
    once that was true, so it is gone rather than kept warm.
    """

    def test_no_join_helper_survives(self):
        import percepteye_agent_flywheel.host_description as hd
        assert not hasattr(hd, "_suffix_is_a_segment")
        assert not hasattr(hd, "_merge_tool_traits")


class TestReadThrough:
    def test_a_declared_trait_survives_the_description_file(self, tmp_path):
        (tmp_path / "discovered_agent.json").write_text(json.dumps({
            "agent_name": "a", "framework": "openclaw", "source_type": "sdk_runtime",
            "role": "r", "system_prompt": "p", "discovered_model_config": {},
            "tool_definitions": [
                {"name": "srv__read_one", "read_only": True},
                {"name": "srv__write_one", "read_only": False},
                {"name": "srv__undeclared"},
            ],
        }), encoding="utf-8")
        got = {t.name: t.read_only
               for t in read_discovered_agent(tmp_path).tool_definitions}
        assert got == {"srv__read_one": True, "srv__write_one": False,
                       "srv__undeclared": None}
