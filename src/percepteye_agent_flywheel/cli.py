"""``percepteye`` — the one command a CLI-agent customer runs.

Deliberately tiny, and deliberately NOT importing the rest of the package at
module scope beyond what it needs: this is the entry point for someone whose
agent is a binary and who may never call :func:`serve` at all.
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def _cmd_install_hooks(args: argparse.Namespace) -> int:
    from .install_hooks import PLUGIN_NAME, goose_plugins_dir, install_goose

    try:
        out = install_goose(force=args.force)
    except FileExistsError as exc:
        print(f"percepteye: {exc}", file=sys.stderr)
        return 2
    except (OSError, FileNotFoundError) as exc:
        print(f"percepteye: could not install hooks: {exc}", file=sys.stderr)
        return 1

    print(f"percepteye: goose plugin {PLUGIN_NAME!r} installed at {out['root']}")
    print(f"            recorder  {out['recorder']}")
    print(f"            python    {out['python']}")
    if out["replaced_foreign"]:
        print("            (replaced a plugin this command did not install)")
    print()
    print("goose will load it on its next run. Nothing else is required for tool")
    print("outcomes; set PERCEPTEYE_API_KEY to contribute rollouts.")
    _ = goose_plugins_dir  # imported for the error path's message symmetry
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from .install_hooks import installed_state

    st = installed_state()
    if not st["installed"]:
        if st["foreign"]:
            print(f"percepteye: a plugin exists at {st['root']} that this command "
                  f"did not install; run `percepteye install-hooks --force` to "
                  f"replace it.")
        else:
            print("percepteye: goose plugin NOT installed — run "
                  "`percepteye install-hooks`.")
        return 1
    print(f"percepteye: installed at {st['root']}")
    print(f"            contract  {st.get('contract_version')}")
    print(f"            recorder  {st.get('recorder')}"
          f"{'' if st.get('recorder_present') else '   [MISSING]'}")
    if st.get("stale"):
        # Said loudly: the rendered command is a snapshot of two paths, so a
        # moved virtualenv or a contract bump leaves a plugin that loads and
        # records nothing. That is exactly the silent-capture-loss this package
        # front-loads its checks to prevent.
        print("            STALE — re-run `percepteye install-hooks`.")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="percepteye",
        description="Percepteye Agent Flywheel — CLI-agent setup.",
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

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":                                # pragma: no cover
    raise SystemExit(main())
