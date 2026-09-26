"""Compute RF TreeSHAP or redraw figures from saved attributions.

    python -m fold.explainability explain --help
    python -m fold.explainability plot --help

Explanation requires explicit train/test trace IDs in bundle metadata
``split_ids`` or a JSON ``--split-file``. Both use the schema in
``fold.explainability.data``. The sampling seed does not reconstruct a split.
"""
import argparse
from pathlib import Path

from .artifacts import _digest, load_forest_bundle
from .data import select_traces
from .explainers import ForestTreeExplainer
from .results import ForestExplanation


def _plot_args(parser):
    parser.add_argument("--sample-index", type=int, default=0,
                        help="saved sample row for the local waterfall")
    parser.add_argument("--feature", default=None,
                        help="dependence feature name; default: highest mean absolute SHAP")
    parser.add_argument("--max-display", type=int, default=15)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    explain = commands.add_parser("explain", help="explain the final RF, not FOLD multi_output")
    explain.add_argument("--bundle", type=Path, required=True)
    explain.add_argument("--trusted", action="store_true",
                         help="acknowledge that the bundle's joblib file comes from a trusted source")
    explain.add_argument("--allow-version-mismatch", action="store_true",
                         help="warn instead of rejecting RF dependency version mismatches")
    explain.add_argument("--dataset", type=Path, required=True)
    explain.add_argument("--split-file", type=Path,
                         help="JSON dataset_sha256 and train/test trace_id lists; overrides missing split_ids")
    explain.add_argument("--target-class", type=int, required=True)
    explain.add_argument("--seed", type=int, default=0, help="sampling seed within the supplied splits")
    explain.add_argument("--background-size", type=int, default=32)
    explain.add_argument("--max-samples", type=int, default=10)
    explain.add_argument("--allow-multiclass", action="store_true",
                         help="accept TreeSHAP's internal all-class computation; save only the target")
    explain.add_argument("--out-dir", type=Path, required=True)
    explain.add_argument("--skip-plots", action="store_true",
                         help="save attributions without requiring matplotlib")
    _plot_args(explain)
    plot = commands.add_parser("plot", help="redraw figures without a model or SHAP computation")
    plot.add_argument("--attributions", type=Path, required=True)
    plot.add_argument("--out-dir", type=Path, required=True)
    _plot_args(plot)
    return parser, parser.parse_args()


def main():
    parser, args = parse_args()
    try:
        if args.command == "plot":
            result = ForestExplanation.load(args.attributions)
        else:
            path = args.out_dir / f"rf_class_{args.target_class:03d}_shap.npz"
            if path.exists():
                raise ValueError(f"Attributions already exist: {path}; choose another output directory")
            bundle = load_forest_bundle(args.bundle, trusted=args.trusted,
                                        strict_versions=not args.allow_version_mismatch)
            background, traces, background_ids, sample_ids, selection = select_traces(
                args.dataset, bundle.metadata, split_file=args.split_file, seed=args.seed,
                background_size=args.background_size, max_samples=args.max_samples)
            metadata = {**bundle.metadata, "selection": selection,
                        "bundle_manifest_sha256": _digest(args.bundle / "manifest.json")}
            print(f"RF TreeSHAP: {len(traces)} test traces, {len(background)} training references, "
                  f"target {args.target_class}; {len(bundle.forest.clf.classes_)} internal class outputs")
            explainer = ForestTreeExplainer(
                bundle.encoder, bundle.forest, background, background_ids=background_ids,
                target_class=args.target_class, class_names=bundle.class_names,
                metadata=metadata, allow_multiclass=args.allow_multiclass)
            result = explainer.explain(traces, sample_ids=sample_ids)
            result.save(path)
            print(f"RF attributions saved to {path}")
            if args.skip_plots:
                return
        from .plots import plot_forest

        paths = plot_forest(result, args.out_dir, sample_index=args.sample_index,
                            feature=args.feature, max_display=args.max_display)
        print(f"Saved {len(paths)} RF figures to {args.out_dir}")
    except (ValueError, TypeError, KeyError, OSError, ImportError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
