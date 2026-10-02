"""One spelling for the control-plane address, everywhere in this package.

The goose recorder resolved it as ``CONTROL_PLANE_ENV_URL or
PERCEPTEYE_CONTROL_PLANE_URL``. Nothing else here reads the un-prefixed name --
not ``policy.py``, not ``production.py``, not ``__init__`` -- and the Node SDK
ranks it LAST where this ranked it FIRST. So the same two variables, exported
together, sent the recorder and the library in one process to two different
control planes with nothing logged either way: a customer's rollouts land in
staging while the SDK reports to production, and the only symptom is rows that
never appear in the dashboard.

The recorder cannot import the SDK on the path that runs per tool call (46 ms
against a 16 ms interpreter floor), so it cannot delegate to
``missing_credentials_cause``. What it CAN do is read the same one name, and
that is what this asserts -- against the shipped file, which is the artifact a
customer's host actually executes.
"""
from __future__ import annotations

import ast

from agent_flywheel import ENV_CONTROL_PLANE_URL
from agent_flywheel.hosts import RECORDER

from .repo_files import repo_files

#: Spellings that must not appear. Written out rather than derived, because the
#: point is that no OTHER name is consulted at all.
_FOREIGN = ("CONTROL_PLANE_ENV_URL", "MC_CONTROL_PLANE_URL", "CONTROL_PLANE_URL\"")


def _env_names_read(source: str) -> set[str]:
    """Every literal passed to ``os.environ.get`` / ``os.environ[...]``."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get"
                and isinstance(node.func.value, ast.Attribute)
                and node.func.value.attr == "environ"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)):
            names.add(node.args[0].value)
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and node.value.attr == "environ"
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            names.add(node.slice.value)
    return names


def test_the_recorder_reads_the_control_plane_under_one_name():
    source = RECORDER.read_text(encoding="utf-8")
    read = _env_names_read(source)
    assert ENV_CONTROL_PLANE_URL in read, (
        "the recorder no longer reads the documented control-plane variable")
    for foreign in _FOREIGN:
        assert foreign.rstrip('"') not in read, (
            f"the recorder reads {foreign!r} as well as "
            f"{ENV_CONTROL_PLANE_URL!r}; two names for one address is how the "
            f"plugin and the SDK end up reporting to different control planes")


def test_no_library_module_reads_a_second_spelling():
    """The other half. A name only the recorder honours is still two answers."""
    import pathlib

    src_dir = pathlib.Path(RECORDER).resolve().parents[2]
    offenders = []
    for path in repo_files(src_dir, suffix=".py"):
        read = _env_names_read(path.read_text(encoding="utf-8"))
        for foreign in _FOREIGN:
            if foreign.rstrip('"') in read:
                offenders.append(f"{path.name}: {foreign}")
    assert not offenders, offenders
