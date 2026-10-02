"""CLI entry points for exact grouped FOLD explanations and saved figures."""
from pathlib import Path
import platform

from ..core.artifacts import _digest, _versions, load_bundle
from ..core.data import select_traces
from .results import FoldExplanation


def add_commands(commands):
    explain = commands.add_parser("explain-fold", help="exact group SHAP of FOLD multi_output")
    explain.add_argument("--bundle", type=Path, required=True)
    explain.add_argument("--trusted", action="store_true", help="load only trusted joblib bundles")
    explain.add_argument("--allow-version-mismatch", action="store_true")
    explain.add_argument("--dataset", type=Path, required=True)
    explain.add_argument("--target-class", type=int, required=True)
    explain.add_argument("--sample-ids-file", type=Path)
    explain.add_argument("--max-samples", type=int,
                         help="random count (default 10), or ordered explicit-ID prefix limit")
    explain.add_argument("--num-groups", type=int, choices=range(1, 9), default=8)
    explain.add_argument("--reference-size", type=int, default=16,
                         help="SHAP references from recorded training IDs; not open-world traffic")
    explain.add_argument("--seed", type=int, default=0)
    explain.add_argument("--batch-size", type=int, default=16, help="inference batch size")
    explain.add_argument("--out-dir", type=Path, required=True)
    explain.add_argument("--skip-plots", action="store_true")
    explain.add_argument("--sample-index", type=int, default=0, help="local figure sample index")
    plot = commands.add_parser("plot-fold", help="redraw grouped FOLD figures from NPZ only")
    plot.add_argument("--attributions", type=Path, required=True)
    plot.add_argument("--out-dir", type=Path, required=True)
    plot.add_argument("--sample-index", type=int, default=0)


def run(args):
    if args.command == "plot-fold":
        result = FoldExplanation.load(args.attributions)
    else:
        from .explainer import FoldGroupExplainer
        from .predictor import FoldPredictor

        path = args.out_dir / f"fold_class_{args.target_class:03d}_shap.npz"
        if path.exists():
            raise ValueError(f"Attributions already exist: {path}; choose another output directory")
        if args.reference_size < 1 or args.batch_size < 1 or args.seed < 0:
            raise ValueError("Reference and batch sizes must be positive; seed must be nonnegative")
        bundle = load_bundle(args.bundle, trusted=args.trusted,
                             strict_versions=not args.allow_version_mismatch)
        references, traces, reference_ids, sample_ids, selection = select_traces(
            args.dataset, bundle.metadata, sample_ids_file=args.sample_ids_file,
            target_class=args.target_class, seed=args.seed,
            background_size=args.reference_size, max_samples=args.max_samples,
            reference_num_groups=args.num_groups)
        if not 0 <= args.sample_index < len(sample_ids):
            raise ValueError("sample_index is outside the selected samples")
        predictor = FoldPredictor(bundle.encoder, bundle.forest, bundle.model,
                                  batch_size=args.batch_size)
        source_dir = Path(__file__).resolve().parent
        # Keep the saved provenance keys stable across module moves.
        source_paths = {"grouped.py": source_dir / "explainer.py",
                        "wrappers.py": source_dir / "predictor.py",
                        "fold_results.py": source_dir / "results.py",
                        "data.py": source_dir.parent / "core" / "data.py"}
        metadata = {**bundle.metadata, "selection": selection,
                    "bundle_manifest_sha256": _digest(args.bundle / "manifest.json"),
                    "xai_source_sha256": {name: _digest(path) for name, path in source_paths.items()},
                    "inference_batch_size": args.batch_size,
                    "runtime_versions": _versions(),
                    "python": platform.python_version()}
        explainer = FoldGroupExplainer(predictor, references, reference_ids=reference_ids,
                                      target_class=args.target_class, class_names=bundle.class_names,
                                      metadata=metadata, num_groups=args.num_groups, seed=args.seed,
                                      eligible_reference_count=selection["eligible_reference_count"])
        upper = len(references) + len(traces) * (
            1 + len(references) * (2**selection["effective_num_groups"] - 2))
        print(f"Exact FOLD SHAP: {len(traces)} samples, {len(references)} training references; "
              f"at most {upper} trace evaluations, batched by {args.batch_size}", flush=True)
        result = explainer.explain(traces, sample_ids=sample_ids)
        result.save(path)
        print(f"FOLD attributions saved to {path}; effective groups: {result.group_counts.tolist()}")
        if args.skip_plots:
            return
    from .plots import plot_fold

    paths = plot_fold(result, args.out_dir, sample_index=args.sample_index)
    print(f"Saved {len(paths)} FOLD figures to {args.out_dir}")
