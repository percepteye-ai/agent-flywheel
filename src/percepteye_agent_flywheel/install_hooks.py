"""Point a CLI agent's plugin system at the payload this wheel already ships.

WHY THIS IS A COMMAND AND NOT A POST-INSTALL STEP. A wheel has no post-install
hook — pip executes no code from one — so ``pip install`` alone cannot register
anything. It also cannot be deferred to :func:`serve`, because ``serve()`` is a
worker loop and the customer this is FOR has no Python process at all: they run
the goose CLI, and the plugin's day-one value ("your agent's real tool failure
rate", no account, no network) never involves us. So: an explicit command, run
once.

WHY THE HOST GETS A RENDERED DIRECTORY AND NOT A SYMLINK TO THIS PACKAGE.
Both work — goose's ``list_dir_children`` filters on ``path.is_dir()``, which
follows symlinks, and it keeps the UNRESOLVED path so ``${PLUGIN_ROOT}`` expands
to the link (measured against goose 1.48.0: a plugin symlinked out of a
site-packages directory was discovered and fired). The reason not to is the
INTERPRETER. A hook runs as ``sh -c`` with goose's PATH — the login shell's, for
the desktop app — so a bare ``python3`` there is whatever the user's shell
resolves, not the virtualenv that has this package importable. The recorder is
standard-library-only so it survives that today, but the once-per-session entry
points are not, and a plugin that works for tool outcomes and silently fails to
claim a rollout is the worst of both.

So the rendered ``hooks.json`` names ``sys.executable`` and the recorder's
ABSOLUTE path inside site-packages. Nothing depends on PATH, and nothing depends
on ``${PLUGIN_ROOT}``.

The consequence, said out loud because it is the trade: the plugin directory is a
SNAPSHOT of two paths, not of the code. A ``pip install --upgrade`` updates the
recorder immediately, because the rendered command points straight at it. It does
NOT update the event list — if a release subscribes to a new hook event, this
command has to be re-run, and :func:`installed_state` is how a caller finds out.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .contract import CONTRACT_VERSION
from .hosts import GOOSE_MANIFEST, RECORDER


#: Where goose looks. ``Paths::plugins_dir`` is ``$HOME/.agents/plugins``, or
#: ``$GOOSE_PATH_ROOT/.agents/plugins`` when that variable holds an ABSOLUTE
#: path (goose ``config/paths.rs``). The env override is honoured because it is
#: what an isolated test — and a customer with a relocated goose — relies on.
def goose_plugins_dir() -> Path:
    root = os.environ.get("GOOSE_PATH_ROOT")
    if root and Path(root).is_absolute():
        return Path(root) / ".agents" / "plugins"
    return Path.home() / ".agents" / "plugins"


#: The directory name IS the plugin name to goose (it reads
#: ``path.file_name()``), so this string is user-visible and stable.
PLUGIN_NAME = "percepteye-agent-flywheel"

#: Written beside the manifest so a later run can tell what it is looking at.
#: A plugin directory with no stamp was put there by something else, and this
#: command refuses to overwrite it rather than guessing.
STAMP_FILENAME = ".percepteye-install.json"


def _hooks_document(python: str, recorder: str) -> dict[str, Any]:
    """The ``hooks.json`` goose loads.

    ONE COMMAND SHAPE, absolute on both halves. Quoted because a virtualenv
    path routinely contains spaces on macOS ("Application Support"), and this
    string is handed to ``sh -c``.

    WHAT KEEPS THIS OBSERVE-ONLY CHANGED, and the honest version is narrower.
    It used to be a property of WHICH events we registered for: the post-tool
    events are emitted via ``emit``, which cannot carry a decision. ``Stop`` is
    not -- it goes through ``emit_blocking`` and its chain CAN deny a turn. So
    the property is now a property of the SCRIPT: ``classify_output`` produces a
    denial only on exit code 2 or a JSON verdict on stdout, and the recorder's
    two standing invariants are that every path exits 0 and it writes nothing to
    stdout. Those two lines are now load-bearing for something they were not
    written for, and the recorder says so.

    ``PreToolUse`` remains deliberately absent. It is the event whose whole
    purpose is to allow or deny a call before it runs, and nothing here has any
    business on that path.
    """
    cmd = f'"{python}" "{recorder}"'
    def _hook(name: str, description: str) -> dict[str, Any]:
        return {"hooks": [{"name": name, "type": "command",
                           "command": cmd, "description": description}]}
    return {
        "hooks": {
            # goose: the split IS goose's own verdict on the call
            # (``Ok(r) if r.is_error != Some(true)`` in agents/agent.rs), which
            # is why `ok`/`failed` here are observed rather than reconstructed.
            "PostToolUse": [_hook(
                "percepteye-record-ok",
                "Record a succeeded tool call into the flywheel trajectory")],
            "PostToolUseFailure": [_hook(
                "percepteye-record-failed",
                "Record a failed tool call into the flywheel trajectory")],
            # Gemini CLI reports the failure inline on one event instead.
            "AfterTool": [_hook(
                "percepteye-record",
                "Record a tool call outcome into the flywheel trajectory")],
            # THE TURN, without which the trajectory is tool calls with no task
            # and no answer -- nothing that can be rendered as a trainable turn,
            # and nothing a human report can title. goose populates both fields
            # already; the recorder simply was not subscribed to them.
            #
            # These land in turns.jsonl, NEVER in tool_calls.jsonl: that
            # reader builds rows with required subscripts and counts anything
            # else as malformed, which forbids the training grade.
            "UserPromptSubmit": [_hook(
                "percepteye-record-task",
                "Record the task this turn was given")],
            "Stop": [_hook(
                "percepteye-record-answer",
                "Record the agent's answer when the turn ends")],
        }
    }


def installed_state(plugins_dir: Path | None = None) -> dict[str, Any]:
    """What is installed at the host, without changing anything.

    Returns ``{"installed": bool, ...}``. ``foreign`` means a directory of that
    name exists with no stamp of ours — a hand-installed marketplace copy, most
    likely — and is reported rather than silently replaced.
    """
    root = (plugins_dir or goose_plugins_dir()) / PLUGIN_NAME
    if not root.exists():
        return {"installed": False, "root": str(root), "foreign": False}
    stamp = root / STAMP_FILENAME
    if not stamp.is_file():
        return {"installed": False, "root": str(root), "foreign": True}
    try:
        data = json.loads(stamp.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"installed": False, "root": str(root), "foreign": True}
    data.update({"installed": True, "root": str(root), "foreign": False})
    # The recorder is pointed at by absolute path, so it tracks the package.
    # Report whether that path still resolves — a rebuilt or moved virtualenv
    # leaves a plugin whose command silently does nothing.
    data["recorder_present"] = Path(data.get("recorder", "")).is_file()
    data["stale"] = (
        data.get("contract_version") != CONTRACT_VERSION
        or not data["recorder_present"]
    )
    return data


def install_goose(
    plugins_dir: Path | None = None,
    *,
    force: bool = False,
    python: str | None = None,
) -> dict[str, Any]:
    """Render the goose plugin directory. Idempotent; returns what it did.

    ``force`` is required to overwrite a directory this command did not write —
    a marketplace install of the standalone plugin is the expected case, and
    clobbering it without being asked would replace a thing the customer chose
    with a thing we chose.
    """
    root = (plugins_dir or goose_plugins_dir()) / PLUGIN_NAME
    state = installed_state(plugins_dir)
    if state.get("foreign") and not force:
        raise FileExistsError(
            f"{root} exists and was not installed by this command — most likely "
            f"a marketplace install of the standalone plugin. Re-run with "
            f"force=True (CLI: --force) to replace it with the one this wheel "
            f"ships, which is the same recorder pointed at this virtualenv."
        )

    interpreter = python or sys.executable
    recorder = str(RECORDER)
    if not RECORDER.is_file():                       # pragma: no cover - packaging
        raise FileNotFoundError(
            f"the wheel is missing its recorder at {recorder}; this is a "
            f"packaging fault, not a configuration one"
        )

    if root.is_symlink() or root.is_file():
        root.unlink()
    elif root.is_dir() and force:
        shutil.rmtree(root)
    (root / "hooks").mkdir(parents=True, exist_ok=True)

    (root / "hooks" / "hooks.json").write_text(
        json.dumps(_hooks_document(interpreter, recorder), indent=2) + "\n",
        encoding="utf-8")
    # goose needs only hooks/hooks.json (``HookManager::from_plugins`` checks
    # exactly that path); the manifest is copied because the marketplace and
    # ``goose plugin`` listings read it, and a plugin with no name reads as
    # broken to a human even when it works.
    if GOOSE_MANIFEST.is_file():
        shutil.copyfile(GOOSE_MANIFEST, root / "plugin.json")
    (root / STAMP_FILENAME).write_text(json.dumps({
        "contract_version": CONTRACT_VERSION,
        "python": interpreter,
        "recorder": recorder,
    }, indent=2) + "\n", encoding="utf-8")

    return {"root": str(root), "python": interpreter, "recorder": recorder,
            "replaced_foreign": bool(state.get("foreign"))}
