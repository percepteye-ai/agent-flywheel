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
