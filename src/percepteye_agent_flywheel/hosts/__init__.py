"""Host payloads — the files a CLI agent's own plugin system loads.

These are DATA, not code this package imports. They ship in the wheel so that
``pip install percepteye-agent-flywheel`` is the only install a customer needs:
``percepteye install-hooks`` points the host at them.

The split that used to exist — a separate marketplace repository for the goose
plugin — was a packaging accident, not an architecture. Its own commit says so:
"Both marketplaces install from a repository ROOT and neither can install a
subdirectory." Nothing about the design required two artifacts, and two copies of
a recorder that must agree byte-for-byte with ``outcomes.Trajectory`` is exactly
the drift this package refuses everywhere else.

``scripts/percepteye_record.py`` is STANDARD LIBRARY ONLY and must stay that way.
It runs once per tool call, inside someone else's agent, and it BLOCKS that agent
until it exits — goose awaits each hook command. Measured on CPython 3.14:
importing this package costs 46 ms against a 16 ms interpreter floor, so an
``import percepteye_agent_flywheel`` in the recorder would nearly triple the
per-tool-call latency of every rollout. The once-per-session entry points can
afford it; this one cannot.
"""
from __future__ import annotations

from pathlib import Path

#: The directory this package's host payloads live in, wherever pip put it.
HOSTS_DIR = Path(__file__).resolve().parent

#: The shared recorder. One file, two host vocabularies (goose's
#: PostToolUse/PostToolUseFailure split, Gemini's inline ``tool_response.error``).
RECORDER = HOSTS_DIR / "scripts" / "percepteye_record.py"

#: The human-facing report — "your agent's real tool failure rate", no account
#: and no network. The day-one payoff, and the reason the recorder must work with
#: nothing else installed.
REPORTER = HOSTS_DIR / "scripts" / "percepteye_report.py"

GOOSE_MANIFEST = HOSTS_DIR / "goose" / "plugin.json"
GEMINI_MANIFEST = HOSTS_DIR / "gemini" / "gemini-extension.json"
