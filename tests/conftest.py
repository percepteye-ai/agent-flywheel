import os
import pathlib
import shutil
import subprocess
import sys

import pytest

# The subprocess test spawns a child that must import ``fixture_agent`` by name,
# so the tests directory has to be importable from both processes.
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@pytest.fixture(autouse=True)
def _no_ambient_agent_mode(monkeypatch):
    """Keep the developer's own PERCEPTEYE_AGENT_MODE out of the suite.

    `serve()` reads it before everything else, so a developer who exported
    PERCEPTEYE_AGENT_MODE=production -- which the README now tells them to do -- got
    18 failures across test_serve_registration and test_host_description, all
    surfacing as `KeyError: 'tr'` because serve() returned 0 before
    registering. Tests that want a mode set it themselves.

    The name is unnamespaced, so this is likelier than it was: PERCEPTEYE_AGENT_MODE may
    already be exported for something that has nothing to do with us.
    """
    monkeypatch.delenv("PERCEPTEYE_AGENT_MODE", raising=False)


def _isolate_home(patch: pytest.MonkeyPatch, home: pathlib.Path) -> None:
    """Point HOME (and USERPROFILE, its Windows spelling) at ``home``.

    A PERCEPTEYE_HOME the developer exported names their real install, so it
    is removed rather than left to stand in.
    """
    patch.setenv("HOME", str(home))
    patch.setenv("USERPROFILE", str(home))
    patch.delenv("PERCEPTEYE_HOME", raising=False)


def _assert_untouched(home: pathlib.Path, who: str) -> None:
    reached = home / ".percepteye"
    assert not reached.exists(), (
        f"{who} reached the default PERCEPTEYE_HOME (~/.percepteye, the "
        f"developer's real home) and wrote {sorted(p.name for p in reached.iterdir())}; "
        "give it a directory under tmp_path instead"
    )


def _uv_directories() -> dict[str, str]:
    """uv's cache and Python-install directories, as uv itself reports them.

    Kept where they are when HOME moves: they are the developer's tool caches,
    not this package's home, and Prime's kernel bootstrap (the Prime binary
    tests' ``provisioned_kernel``) would otherwise download a Python and every
    package again on every run, about 330 MB.
    """
    uv = shutil.which("uv")
    found: dict[str, str] = {}
    for name, command in (("UV_CACHE_DIR", ("cache", "dir")),
                          ("UV_PYTHON_INSTALL_DIR", ("python", "dir"))):
        if uv is None or os.environ.get(name):
            continue
        try:
            out = subprocess.run([uv, *command], capture_output=True, text=True,
                                 timeout=30, check=True).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
        if out:
            found[name] = out
    return found


@pytest.fixture(scope="session", autouse=True)
def _no_real_home_in_the_session(tmp_path_factory):
    """The guard below, for what runs before any test does.

    A module- or session-scoped fixture is set up before every
    function-scoped one, so the per-test guard cannot see it -- and one did
    reach the real home: ``provisioned_kernel`` imported the Prime example,
    which created ``~/.percepteye/prime-agent-workspace``. This keeps the
    whole session's HOME off the real one, and fails the session if a
    ``.percepteye`` appears in it, naming what was written.
    """
    home = tmp_path_factory.mktemp("session-home")
    with pytest.MonkeyPatch.context() as patch:
        for name, value in _uv_directories().items():
            patch.setenv(name, value)
        _isolate_home(patch, home)
        yield
    _assert_untouched(home, "a module- or session-scoped fixture")


@pytest.fixture(autouse=True)
def _no_real_home(monkeypatch, tmp_path_factory):
    """No test reaches the developer's real home, and one that tries fails.

    ``~/.percepteye`` is where a real install keeps what it captured
    (``turns_root``) and where the Prime example puts its workspace. A test
    that resolved it wrote into the developer's own home: the Prime binary
    tests' loader created ``~/.percepteye/prime-agent-workspace`` on every run
    that had the binary. The Node SDK's suite had the same shape worse --
    concurrent test files drained each other's held results there -- and
    guards it the same way.

    Each test gets an empty HOME of its own, so a test that reaches the
    default -- in this process or in a child that inherits the environment --
    lands there, never in the real one, and the failure names the test.
    """
    home = tmp_path_factory.mktemp("home")
    _isolate_home(monkeypatch, home)
    yield
    _assert_untouched(home, "this test")


@pytest.fixture(autouse=True)
def _never_the_production_control_plane(monkeypatch):
    """No test may reach the production control plane.

    ``PERCEPTEYE_CONTROL_PLANE_URL`` defaults to it (``agent_flywheel.
    control_plane``, the Node SDK's rule), so a test that leaves the variable
    unset and sets a key would otherwise register, upload and read policy
    against the live service. A request to that host is refused as if the host
    were unreachable -- the SDK fails open on that, so raising alone would be
    swallowed -- and the test then FAILS at teardown, naming the request. A
    test sets its own URL, or a mock transport.
    """
    import httpx

    from agent_flywheel.control_plane import DEFAULT_CONTROL_PLANE_URL

    host = httpx.URL(DEFAULT_CONTROL_PLANE_URL).host
    real = httpx.HTTPTransport.handle_request
    attempted: list[str] = []

    def guarded(self, request):
        if request.url.host == host:
            attempted.append(f"{request.method} {request.url}")
            raise httpx.ConnectError("the test suite never reaches production",
                                     request=request)
        return real(self, request)

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", guarded)
    yield
    if attempted:
        pytest.fail(
            "this test sent requests to the production control plane "
            f"({', '.join(sorted(set(attempted)))}); set "
            "PERCEPTEYE_CONTROL_PLANE_URL or a mock transport instead")
