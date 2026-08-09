import pathlib
import sys

# The subprocess test spawns a child that must import ``fixture_agent`` by name,
# so the tests directory has to be importable from both processes.
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
