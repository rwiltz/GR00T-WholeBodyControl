# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Run Isaac Lab's stock teleoperation and data-collection scripts against this repo's tasks.

Usage::

    python -m gear_sonic.lab_teleop.scripts.run teleop        --task <id> --viz kit
    python -m gear_sonic.lab_teleop.scripts.run record_demos  --task <id> --dataset_file demos.hdf5
    python -m gear_sonic.lab_teleop.scripts.run replay        --task <id> --replay_file <mcap>

Everything after the script name is forwarded to the stock script untouched, so its ``--help``,
flags, dataset format and future fixes all apply verbatim.

Why this exists
---------------
Isaac Lab resolves a task by name through ``gym.make``, and ids are registered as an import side
effect. ``isaaclab_tasks`` only auto-imports its *own* subpackages
(``isaaclab_tasks/__init__.py`` -> ``import_packages(__name__, ...)``), and there is no plugin hook
for task packages outside it, so a stock script cannot see our ids on its own.

This mirrors Isaac Lab's own guidance for external projects. Its template generator emits a
project-local runner that does exactly this (``tools/template/templates/external/train``)::

    import {{ name }}.tasks  # noqa: F401
    from isaaclab_rl.entrypoints import run_train_cli

That shape only works for workflows Isaac Lab has factored into library entry points —
``isaaclab_rl.entrypoints.dispatch`` covers train / play / zero-agent / random-agent. Teleoperation
and demo recording have no such entry point; they are only available as scripts. So instead of
calling a library function we execute the stock script itself, which keeps us on the upstream code
path rather than vendoring a copy that would drift.

Importing the task registry first is safe: :mod:`gear_sonic.lab_teleop.tasks` touches only
``gymnasium`` and registers *string* entry points, so nothing from ``isaaclab`` or Isaac Sim is
imported before the stock script builds its ``AppLauncher``. Locating the checkout likewise uses
``importlib.util.find_spec``, which resolves ``isaaclab`` without executing it.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pathlib
import runpy
import sys

__all__ = ["STOCK_SCRIPTS", "main", "resolve_isaaclab_root", "run_stock_script"]

#: Stock scripts we expose, mapped to their path within the Isaac Lab checkout.
STOCK_SCRIPTS = {
    "teleop": "scripts/environments/teleoperation/teleop_se3_agent.py",
    "record_demos": "scripts/tools/record_demos.py",
    "replay": "scripts/environments/teleoperation/teleop_replay_agent.py",
}


def _looks_like_checkout(path: pathlib.Path) -> bool:
    return (path / "scripts").is_dir() and (path / "source").is_dir()


def resolve_isaaclab_root(explicit: str | None = None) -> pathlib.Path:
    """Locate the Isaac Lab checkout.

    Resolution order: ``explicit`` argument, then ``$ISAACLAB_PATH``, then the directory containing
    the installed ``isaaclab`` package.

    Args:
        explicit: Path supplied on the command line, if any.

    Returns:
        Absolute path to the checkout root.

    Raises:
        FileNotFoundError: If no checkout can be found, or the candidate lacks ``scripts/``.
    """
    for candidate in (explicit, os.environ.get("ISAACLAB_PATH")):
        if candidate:
            root = pathlib.Path(candidate).expanduser().resolve()
            if not _looks_like_checkout(root):
                raise FileNotFoundError(
                    f"{root} does not look like an Isaac Lab checkout (needs 'scripts/' and"
                    " 'source/')."
                )
            return root

    # find_spec resolves the package without executing it, so Isaac Sim is not pulled in here.
    spec = importlib.util.find_spec("isaaclab")
    if spec is not None and spec.origin:
        for parent in pathlib.Path(spec.origin).resolve().parents:
            if _looks_like_checkout(parent):
                return parent

    raise FileNotFoundError(
        "Could not locate the Isaac Lab checkout. Point at it explicitly, e.g.\n"
        "    export ISAACLAB_PATH=~/repo/IsaacLab\n"
        "or pass --isaaclab-path. Note a wheel-only Isaac Lab install has no 'scripts/'"
        " directory; the stock teleoperation scripts require a source checkout."
    )


def run_stock_script(
    script_key: str, argv: list[str], isaaclab_path: str | None = None
) -> None:
    """Register this repo's tasks, then execute a stock Isaac Lab script in this interpreter.

    Args:
        script_key: Key into :data:`STOCK_SCRIPTS`.
        argv: Arguments forwarded to the script, excluding the program name.
        isaaclab_path: Override for the Isaac Lab checkout root.

    Raises:
        FileNotFoundError: If Isaac Lab or the requested script cannot be found.
    """
    root = resolve_isaaclab_root(isaaclab_path)
    script = root / STOCK_SCRIPTS[script_key]
    if not script.is_file():
        raise FileNotFoundError(
            f"Stock script not found: {script}\n"
            "Check that the Isaac Lab checkout matches the version this repo targets."
        )

    import gear_sonic.lab_teleop.tasks  # noqa: F401  (registers the gym ids)

    # The stock scripts parse sys.argv at module scope, so present the argv they expect.
    sys.argv = [str(script), *argv]
    runpy.run_path(str(script), run_name="__main__")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Unrecognised arguments are forwarded to the stock script."""
    parser = argparse.ArgumentParser(
        prog="python -m gear_sonic.lab_teleop.scripts.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "script", choices=sorted(STOCK_SCRIPTS), help="Stock Isaac Lab script to run"
    )
    parser.add_argument(
        "--isaaclab-path",
        default=None,
        help="Isaac Lab checkout root (default: $ISAACLAB_PATH, else auto-detected)",
    )
    known, forwarded = parser.parse_known_args(argv)
    run_stock_script(known.script, forwarded, known.isaaclab_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
