"""The package version, written down once.

``agent_flywheel.__version__`` re-exports it, and ``_http`` builds the
User-Agent from it -- which it cannot import from the package root without an
import cycle. That cycle is why the User-Agent used to carry its own copy of
the number, so a release that bumped the version everywhere else would still
have announced itself to the control plane as the previous one.
``pyproject.toml`` states it too, for the build and for the release workflow's
tag check, as do the goose manifest and ``uv.lock``, and ``CHANGELOG.md`` opens
with its entry; ``tests/test_one_version.py`` holds them all together.
"""
__version__ = "0.1.3"
