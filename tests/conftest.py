import pathlib
import sys

import pytest

# The subprocess test spawns a child that must import ``fixture_agent`` by name,
# so the tests directory has to be importable from both processes.
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(autouse=True)
def _no_ambient_agent_mode(monkeypatch):
    """Keep the developer's own AGENT_MODE out of the suite.

    `serve()` reads it before everything else, so a developer who exported
    AGENT_MODE=production -- which the README now tells them to do -- got
    18 failures across test_serve_registration and test_host_description, all
    surfacing as `KeyError: 'tr'` because serve() returned 0 before
    registering. Tests that want a mode set it themselves.

    The name is unnamespaced, so this is likelier than it was: AGENT_MODE may
    already be exported for something that has nothing to do with us.
    """
    monkeypatch.delenv("AGENT_MODE", raising=False)
