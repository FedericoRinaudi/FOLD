"""Compute RF TreeSHAP, grouped FOLD SHAP or redraw saved attributions.

    python -m fold.explainability explain --help
    python -m fold.explainability plot --help

Explanation requires explicit train/test trace IDs in bundle metadata
``split_ids`` or a JSON ``--split-file``. Both use the schema in
``fold.explainability.core.data``. The sampling seed does not reconstruct a split.
"""
import argparse

from .rf.cli import add_commands as add_rf_commands, run as run_rf


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    add_rf_commands(commands)
    from .end_to_end.cli import add_commands

    add_commands(commands)
    from .open_world.cli import add_commands as add_open_world_commands

    add_open_world_commands(commands)
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    try:
        if args.command in ("explain-open-world", "plot-open-world"):
            from .open_world.cli import run

            run(args)
            return
        if args.command in ("explain-fold", "plot-fold"):
            from .end_to_end.cli import run

            run(args)
            return
        run_rf(args)
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
