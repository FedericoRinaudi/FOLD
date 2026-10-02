"""CLI for binary explanations of already trained open-world FOLD bundles."""
from pathlib import Path

from ..core.artifacts import _digest, _versions, load_bundle
from .data import select_open_world
from .results import BinaryExplanation


def add_commands(commands):
    explain = commands.add_parser("explain-open-world", help="exact group Shapley of P(monitored)")
    explain.add_argument("--bundle", type=Path, required=True)
    explain.add_argument("--trusted", action="store_true", help="load only trusted joblib bundles")
    explain.add_argument("--allow-version-mismatch", action="store_true")
    explain.add_argument("--dataset", type=Path, required=True)
    explain.add_argument("--background-pool", type=Path, required=True,
                         help="the original packet CSV pool, verified against its recorded hash")
    explain.add_argument("--sample-ids-file", type=Path, required=True)
    explain.add_argument("--reference-ids-file", type=Path,
                         help="optional explicit balanced training IDs; must match reference-size")
    explain.add_argument("--num-groups", type=int, choices=range(1, 9), default=8)
    explain.add_argument("--reference-size", type=int, default=16,
                         help="total references, at least 2; odd extra goes to monitored")
    explain.add_argument("--seed", type=int, default=0)
    explain.add_argument("--batch-size", type=int, default=16)
    explain.add_argument("--out-dir", type=Path, required=True)
    explain.add_argument("--skip-plots", action="store_true")
    explain.add_argument("--sample-index", type=int, default=0)
    plot = commands.add_parser("plot-open-world", help="redraw binary figures from NPZ only")
    plot.add_argument("--attributions", type=Path, required=True)
    plot.add_argument("--out-dir", type=Path, required=True)
    plot.add_argument("--sample-index", type=int, default=0)


def run(args):
    if args.command == "plot-open-world":
        result = BinaryExplanation.load(args.attributions)
    else:
        from .explainer import BinaryGroupExplainer
        from .predictor import BinaryFoldPredictor

        path = args.out_dir / "binary_monitored_shap.npz"
        if path.exists():
            raise ValueError(f"Attributions already exist: {path}; choose another output directory")
        if args.reference_size < 2 or args.batch_size < 1 or args.seed < 0:
            raise ValueError("Require reference-size >= 2, positive batch-size and nonnegative seed")
        bundle = load_bundle(args.bundle, trusted=args.trusted,
                             strict_versions=not args.allow_version_mismatch)
        selection = select_open_world(
            args.dataset, args.background_pool, bundle.metadata,
            sample_ids_file=args.sample_ids_file, reference_ids_file=args.reference_ids_file,
            num_groups=args.num_groups, reference_size=args.reference_size, seed=args.seed)
        if not 0 <= args.sample_index < len(selection.samples):
            raise ValueError("sample_index is outside the selected samples")
        source_dir = Path(__file__).resolve().parent
        source_paths = [*source_dir.glob("*.py"),
                        source_dir.parent / "end_to_end" / "explainer.py",
                        source_dir.parent / "end_to_end" / "predictor.py",
                        source_dir.parent / "end_to_end" / "results.py"]
        selection.metadata.update(
            bundle_manifest_sha256=_digest(args.bundle / "manifest.json"),
            inference_batch_size=args.batch_size, runtime_versions=_versions(),
            xai_source_sha256={p.relative_to(source_dir.parent).as_posix(): _digest(p)
                               for p in sorted(source_paths)})
        info = selection.metadata["selection"]
        cost = len(selection.references) + len(selection.samples) * (
            1 + len(selection.references) * (2**info["effective_num_groups"] - 2))
        print(f"Exact binary FOLD Shapley: {len(selection.samples)} samples, "
              f"{len(selection.references)} balanced references; {cost} trace evaluations", flush=True)
        predictor = BinaryFoldPredictor(bundle, batch_size=args.batch_size)
        result = BinaryGroupExplainer(predictor, selection).explain()
        result.save(path)
        print(f"Binary attributions saved to {path}")
        if args.skip_plots:
            return
    from .plots import plot_binary

    paths = plot_binary(result, args.out_dir, sample_index=args.sample_index)
    print(f"Saved {len(paths)} binary figures to {args.out_dir}")
