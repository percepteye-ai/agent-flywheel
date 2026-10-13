"""``agent-flywheel`` — the one command a CLI-agent customer runs.

Deliberately tiny, and deliberately NOT importing the rest of the package at
module scope beyond what it needs: this is the entry point for someone whose
agent is a binary and who may never call :func:`serve` at all.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence

from .errors import ConfigurationError


def _contribution_line() -> str:
    """Whether this install will claim rollouts, where, and if not why.

    The key is the one thing that can be missing: the control-plane URL
    resolves through the one rule the library uses (a default when unset), and
    the line names the URL it resolved to, because a default a customer cannot
    see is a default they cannot tell apart from a control plane that is down.
    Asked through the same predicate the library asks, so the CLI and the SDK
    cannot disagree about what "configured" means.
    """
    from .control_plane import resolve_control_plane_url
    from .policy import missing_credentials_cause

    url = resolve_control_plane_url()
    cause = missing_credentials_cause(os.environ.get("PERCEPTEYE_API_KEY"))
    if cause is None:
        return f"rollouts  claimed — from {url}"
    return f"rollouts  NOT claimed — {cause}"


def _cmd_install_hooks(args: argparse.Namespace) -> int:
    from .install_hooks import PLUGIN_NAME, goose_plugins_dir, install_goose

    try:
        out = install_goose(force=args.force)
    except FileExistsError as exc:
        print(f"agent-flywheel: {exc}", file=sys.stderr)
        return 2
    except (OSError, FileNotFoundError) as exc:
        print(f"agent-flywheel: could not install hooks: {exc}", file=sys.stderr)
        return 1

    print(f"agent-flywheel: goose plugin {PLUGIN_NAME!r} installed at {out['root']}")
    print(f"                recorder  {out['recorder']}")
    print(f"                python    {out['python']}")
    if out["replaced_foreign"]:
        print("                (replaced a plugin this command did not install)")
    _print_endpoint(out["endpoint"])
    print(f"                {_contribution_line()}")
    print()
    print("goose will load it on its next run. Nothing else is required for tool")
    print("outcomes; `agent-flywheel report` prints what it recorded.")
    _ = goose_plugins_dir  # imported for the error path's message symmetry
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    """Print the tool-failure rate the plugin recorded. The day-one payoff.

    THE CONSUMER THIS SCRIPT NEVER HAD. ``percepteye_report.py`` shipped in
    every wheel and was named by exactly one line of code -- the constant
    binding it -- so the number the README leads with was reachable only by
    finding the file inside site-packages and running it by hand.

    Loaded from the same path the wheel installs rather than reimplemented: the
    script is standard-library-only on purpose and is the one place that knows
    how to read the file the recorder writes.
    """
    import importlib.util

    from .hosts import RECORDER, REPORTER

    def _load(path, name):
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:      # pragma: no cover
            raise ImportError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    try:
        reporter = _load(REPORTER, "_percepteye_report")
        recorder = _load(RECORDER, "_percepteye_record")
    except (OSError, ImportError) as exc:
        print(f"agent-flywheel: this install is missing its reporter: {exc}",
              file=sys.stderr)
        return 1

    target = args.directory
    if target is None:
        # THE MOST RECENT SESSION, because that is the one a customer who just
        # ran their agent means. The recorder writes one directory per session
        # under a root only it knew; asking it for that root rather than
        # rebuilding the join here is what keeps the reader pointed at the
        # same place the writer writes.
        root = recorder.turns_root()
        try:
            sessions = [
                os.path.join(root, d) for d in os.listdir(root)
                if os.path.isfile(
                    os.path.join(root, d, reporter.TOOL_CALLS_FILENAME))
            ]
        except OSError:
            sessions = []
        if sessions:
            target = max(sessions, key=os.path.getmtime)
            print(f"agent-flywheel: the most recent session under {root}\n")
        else:
            # The reporter's own empty-state help, which names the directory
            # and how to record one, is better than anything said here.
            target = root
    return int(reporter.main(["agent-flywheel report", target]))


def _print_endpoint(ep: dict) -> None:
    """Say where the agent's traffic goes, and who decided.

    Printed on BOTH commands because routing is the one thing this install
    changes about the customer's own agent, and a customer who cannot see it
    has to read our source to find out. The ``shadowed_by`` line is the one
    that saves an hour: an env var outranks the config file, so a perfectly
    installed route can be inert for a reason nothing else would reveal.
    """
    from .endpoint import GOOSE_ENDPOINT_KEY

    print(f"                mode      {ep.get('mode')} — {ep.get('reason')}")
    if ep.get("value"):
        print(f"                endpoint  {ep['value']}")
    elif ep.get("mode") == "production":
        print("                endpoint  (unset — goose uses its own default)")
    if ep.get("shadowed_by"):
        print(f"                NOTE: ${ep['shadowed_by']} is set in your environment "
              f"and outranks\n                      {GOOSE_ENDPOINT_KEY} in "
              f"{ep.get('path')}. That variable wins.")


def _cmd_status(args: argparse.Namespace) -> int:
    from .install_hooks import installed_state

    st = installed_state()
    if not st["installed"]:
        if st["foreign"]:
            print(f"agent-flywheel: a plugin exists at {st['root']} that this command "
                  f"did not install; run `agent-flywheel install-hooks --force` to "
                  f"replace it.")
        else:
            print("agent-flywheel: goose plugin NOT installed — run "
                  "`agent-flywheel install-hooks`.")
        return 1
    print(f"agent-flywheel: installed at {st['root']}")
    print(f"                contract  {st.get('contract_version')}")
    print(f"                recorder  {st.get('recorder')}"
          f"{'' if st.get('recorder_present') else '   [MISSING]'}")
    # Reported from a RECONCILE, not from the stamp: it re-reads the config and
    # the mode, so this prints where traffic will actually go on the next run
    # rather than where the install once pointed it.
    from .install_hooks import reconcile_endpoint

    try:
        _print_endpoint(reconcile_endpoint())
    except OSError as exc:
        print(f"                endpoint  UNREADABLE — {exc}")
    # Printed on STATUS as well as install, because "where does my traffic go"
    # and "am I contributing" are the two questions this command exists to
    # answer and only the first one was answered.
    print(f"                {_contribution_line()}")
    if st.get("stale"):
        # Said loudly: the rendered command is a snapshot of two paths, so a
        # moved virtualenv or a contract bump leaves a plugin that loads and
        # records nothing. That is exactly the silent-capture-loss this package
        # front-loads its checks to prevent.
        print("                STALE — re-run `agent-flywheel install-hooks`.")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="agent-flywheel",
        description="Agent Flywheel — CLI-agent setup.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_install = sub.add_parser(
        "install-hooks",
        help="point goose at the tool-outcome recorder this wheel ships",
    )
    p_install.add_argument(
        "--force", action="store_true",
        help="replace a plugin directory this command did not install",
    )
    p_install.set_defaults(func=_cmd_install_hooks)

    p_status = sub.add_parser(
        "status", help="report whether the host plugin is installed and current")
    p_status.set_defaults(func=_cmd_status)

    p_report = sub.add_parser(
        "report",
        help="print your agent's real tool-failure rate from what was recorded")
    p_report.add_argument(
        "directory", nargs="?", default=None,
        help="a trajectory directory; defaults to the most recent session")
    p_report.set_defaults(func=_cmd_report)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except ConfigurationError as exc:
        # A bad PERCEPTEYE_AGENT_MODE reaches here from either subcommand, both
        # of which resolve the mode to decide the endpoint. The MESSAGE is
        # already the right one -- it names the variable, the value, and every
        # spelling that works -- so the only thing wrong with a traceback here
        # is that it buries that message under a stack the customer cannot act
        # on. Print it and exit; do not guess a mode.
        print(f"agent-flywheel: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
