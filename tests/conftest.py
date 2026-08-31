import pathlib
import sys

import pytest

# The subprocess test spawns a child that must import ``fixture_agent`` by name,
# so the tests directory has to be importable from both processes.
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(autouse=True)
def _no_ambient_percepteye_mode(monkeypatch):
    """Keep the developer's own PERCEPTEYE_MODE out of the suite.

    `serve()` reads it before everything else, so a developer who exported
    PERCEPTEYE_MODE=production -- which the README now tells them to do -- got
    18 failures across test_serve_registration and test_host_description, all
    surfacing as `KeyError: 'tr'` because serve() returned 0 before
    registering. Tests that want a mode set it themselves.
    """
    monkeypatch.delenv("PERCEPTEYE_MODE", raising=False)
