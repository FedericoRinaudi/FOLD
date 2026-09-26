"""Run one original RF + FOLD experiment and save its inference bundle.

Uses run_experiment.py arguments, with --model fold. --out-dir is the root
under which the scenario cell and exp_NNN directories are created.
"""
import csv
from pathlib import Path
import platform
import sys

import numpy as np

from .artifacts import _class_names, _digest, _metadata, fold_config, save_bundle
from .configuration import experiment_metadata, scenario_fields, scenario_tag, validate_split


def options_from_args(args, *, argv=()):
    """Adapt the existing CLI to Config without importing TensorFlow."""
    from run_experiment import defense_config

    if args.model != "fold" or args.n1_head or args.holdout_location != "random":
        raise ValueError("This runner supports only --model fold with the original two heads")
    if args.epochs < 1 or args.batch_size < 1 or not 0 <= args.experiment_id <= (2**32 - 43) // 17:
        raise ValueError("Invalid experiment id, epoch ceiling or batch size")
    defense = defense_config(args)
    scenario = "open-world" if args.open_world else "defense" if defense else "closed-world"
    fields = scenario_fields(scenario, args.background_ratio if args.open_world else None, defense)
    def specified(option):
        return any(a.startswith("--") and option.startswith(a.split("=", 1)[0]) for a in argv)

    explicit_ratio = specified("--background-ratio")
    if not args.open_world and (explicit_ratio or args.background_ratio != 0.5):
        raise ValueError("--background-ratio requires --open-world")
    if specified("--sweep-probs") or specified("--holdout-location"):
        raise ValueError("Sweep and location options do not apply to single-configuration training")
    return {"experiment_id": args.experiment_id,
            "out_dir": args.out_dir / scenario_tag(fields), "dataset": args.dataset,
            "scenario": scenario, "defense": defense,
            "background_ratio": args.background_ratio, "background_pool": args.background_pool,
            "epochs": args.epochs, "batch_size": args.batch_size, "n1_head": False}


def class_names(dataset):
    """Read the CSV's existing label-to-site mapping."""
    names = {}
    with Path(dataset).open(newline="") as stream:
        for row in csv.DictReader(stream):
            label = int(float(row["categorical_label"]))
            site = row["site"]
            if label in names and names[label] != site:
                raise ValueError("One categorical label names multiple sites")
            names[label] = site
    if set(names) != set(range(len(names))):
        raise ValueError("Dataset labels must be contiguous from zero")
    return _class_names([names[i] for i in range(len(names))], len(names))


def split_record(split, digest):
    return validate_split({"dataset_sha256": digest,
                           **{name: getattr(split, name).ids for name in ("train", "val", "test")}})


def source_hashes():
    root = Path(__file__).resolve().parents[2]
    paths = sorted((root / "fold").rglob("*.py"))
    paths += [root / "run_experiment.py", root / "run_campaign.sh"]
    return {path.relative_to(root).as_posix(): _digest(path) for path in paths}


def run(cfg):
    """Keep the final RF alive across the two unchanged training runners."""
    from fold.experiments import run_fold, run_forest, seed_everything

    metadata = experiment_metadata(cfg)
    digest = _digest(cfg.dataset)
    names = class_names(cfg.dataset)
    sources = source_hashes()
    pool_digest = _digest(cfg.background_pool) if cfg.is_open_world else None
    # Reserve a new run before either original runner can overwrite predictions.
    cfg.exp_dir.mkdir(parents=True, exist_ok=False)
    seed_everything(cfg.seed)
    rf_split, rf_background, rf_encoder, forest = run_forest(cfg, return_artifacts=True)
    split_ids = split_record(rf_split, digest)
    metadata.update(dataset_sha256=digest, split_ids=split_ids, source_sha256=sources)
    if cfg.is_open_world:
        metadata["background_split_ids"] = validate_split(
            {"dataset_sha256": pool_digest, **rf_background.ids})
        metadata["background_order"] = "monitored_then_background"
        metadata["realized_background_ratio"] = {
            name: len(getattr(rf_background, name)) /
                  (len(getattr(rf_split, name)) + len(getattr(rf_background, name)))
            for name in ("train", "val", "test")}
    if cfg.is_defended:
        metadata["defense_replay"] = {
            "python_random_state": rf_split.defense_state,
            "python_version": platform.python_version(),
            "order": ["train", "val", "test"],
            "source_sha256": sources["fold/data.py"]}
    metadata = _metadata(metadata)
    # Each original CLI invocation starts from this seed, including defense RNG.
    seed_everything(cfg.seed)
    split, background, encoder, model = run_fold(cfg, return_artifacts=True)
    if split_record(split, digest) != split_ids or split.defense_state != rf_split.defense_state:
        raise ValueError("FOLD and RF used different splits or defense RNG states")
    for name in ("train", "val", "test"):
        left, right = getattr(rf_split, name), getattr(split, name)
        if left.labels != right.labels or left.traces != right.traces:
            raise ValueError("FOLD and RF used different monitored traces")
        if getattr(rf_background, name) != getattr(background, name):
            raise ValueError("FOLD and RF used different background traces")
    if (background.ids != rf_background.ids or encoder.num_classes != rf_encoder.num_classes
            or encoder.target_length != rf_encoder.target_length
            or not np.array_equal(encoder.size_edges, rf_encoder.size_edges)):
        raise ValueError("FOLD and RF inference contexts differ")
    if (_digest(cfg.dataset) != digest or source_hashes() != sources
            or (cfg.is_open_world and _digest(cfg.background_pool) != pool_digest)):
        raise ValueError("Dataset, pool or source changed during training")
    save_bundle(cfg.exp_dir / "bundle", encoder=encoder, forest=forest, model=model,
                model_config=fold_config(encoder), class_names=names, metadata=metadata)
    print(f"Inference bundle: {cfg.exp_dir / 'bundle'}")


def main():
    from run_experiment import parse_args

    args = parse_args()
    try:
        options = options_from_args(args, argv=sys.argv[1:])
        from fold.experiments import Config

        run(Config(**options))
    except (ValueError, TypeError, OSError, ImportError) as error:
        raise SystemExit(f"error: {error}") from error


if __name__ == "__main__":
    main()
