"""One bundle load per cell/run: RF, FOLD multiclass and open-world binary.

Writes immutable per-class NPZ plus long CSV. Completed outputs are skipped
only after checking all content hashes, bundle binding and selection binding.
Incomplete output directories are preserved and reported, never overwritten.
"""
import argparse
import csv
import json
from pathlib import Path
import uuid

import numpy as np

from ..core.artifacts import _digest, load_bundle
from ..core.configuration import scenario_tag
from ..core.data import select_traces
from ..end_to_end.explainer import FoldGroupExplainer
from ..end_to_end.predictor import FoldPredictor
from ..rf.explainer import ForestTreeExplainer
from .selection import for_run, validate_selection
from .runner import persist_selection


LONG_FIELDS = ("configuration", "scenario", "intensity", "background_ratio", "experiment_id", "model",
               "sample_id", "class", "site_id", "site", "is_reused_trace_id", "cohort", "output", "target_class", "dimension", "phi",
               "probability", "base_value", "reference_ids", "selection_seed", "metadata")


def emit(writer, result, metadata, *, model, site, label, cohort="monitored", output="multi_output"):
    fields = {"configuration": scenario_tag(metadata), "scenario": metadata["scenario"],
              "intensity": metadata.get("defense_intensity"), "background_ratio": metadata.get("background_ratio"),
              "experiment_id": metadata["experiment_id"], "model": model, "class": label, "site": site,
              "site_id": label,
              "cohort": cohort, "output": output, "target_class": label if output == "multi_output" else 0,
              "selection_seed": 0, "reference_ids": json.dumps(list(getattr(result, "reference_ids", getattr(result, "background_ids", ())))),
              "metadata": json.dumps({"algorithm": result.metadata.get("algorithm", result.metadata.get("explainer")),
                                      "additivity_atol": result.metadata.get("additivity_atol"),
                                      "metadata_storage": "corresponding NPZ header; complete.json binds all files"}, separators=(",", ":"))}
    dimensions = result.feature_names if model == "rf" else tuple(f"G{i+1}" for i in range(result.values.shape[1]))
    bases = np.full(len(result.sample_ids), result.base_value) if model == "rf" else result.base_values
    for row, tid in enumerate(result.sample_ids):
        reused = tid in metadata.get("selection", {}).get("reused_trace_ids_this_run", [])
        for col, name in enumerate(dimensions):
            writer.writerow({**fields, "sample_id": tid, "is_reused_trace_id": reused, "dimension": name, "phi": result.values[row, col],
                             "probability": result.probabilities[row], "base_value": bases[row]})


def background_splits(pool_ids, monitored_split, run):
    """Original draw_background slicing, needed before bundles exist."""
    from .preflight import original_seed

    order = np.random.default_rng(original_seed(run)).permutation(len(pool_ids))
    result = {}
    for ratio in (.30, .40, .50, .60, .70):
        sizes = [int(round(len(monitored_split[part]) * ratio / (1-ratio))) for part in ("train", "val", "test")]
        if sum(sizes) > len(pool_ids):
            raise ValueError(f"Background pool too small for ratio {ratio}")
        chosen = [pool_ids[i] for i in order[:sum(sizes)]]
        a, b, _ = sizes
        result[ratio] = dict(train=chosen[:a], val=chosen[a:a+b], test=chosen[a+b:])
    return result


def binary_files(dataset, pool, metadata, monitored_selection, out):
    """Common background test/reference IDs across ratios when feasible.

    train/test boundaries move with ratio; intersections are explicitly used.
    Reference semantics stay 8 monitored + 8 background training traces.
    """
    from fold.data import load_traces
    from ..open_world.data import load_background_pool
    from ..end_to_end.explainer import resolution_edges

    background = load_background_pool(pool)
    splits = background_splits(list(background), metadata["split_ids"], metadata["experiment_id"])
    rng = np.random.default_rng(0)
    common_test = set.intersection(*(set(s["test"]) for s in splits.values()))
    actual = metadata["background_split_ids"]
    expected = splits[metadata["background_ratio"]]
    if any(expected[k] != actual[k] for k in ("train", "val", "test")):
        raise ValueError("Background split reconstruction differs from bundle")
    # Match monitored cohort size when possible, otherwise use all eligible test
    # backgrounds (binary background has no 15-per-site requirement).
    source = sorted(common_test) if common_test else actual["test"]
    source = [
        tid for tid in source
        if len(resolution_edges(background[tid])) - 1 >= 4
    ]
    if not source:
        raise ValueError("No G1-G4 eligible binary background TEST traces")
    chosen = rng.choice(
        source,
        min(len(source), len(monitored_selection["sample_ids"])),
        replace=False,
    ).tolist()
    samples = {"output": "binary_output", "monitored_sample_ids": monitored_selection["sample_ids"],
               "background_sample_ids": chosen}
    persist_selection(out / "binary_samples.json", samples)
    traces, _, _, ids = load_traces(dataset, return_ids=True)
    monitored = dict(zip(ids, traces))
    common_train = set.intersection(*(set(s["train"]) for s in splits.values()))
    mtrain = [tid for tid in metadata["split_ids"]["train"] if len(resolution_edges(monitored[tid]))-1 >= 4]
    btrain = sorted(tid for tid in common_train if len(resolution_edges(background[tid]))-1 >= 4)
    if len(mtrain) < 8 or len(btrain) < 8:
        raise ValueError("Insufficient eligible common binary training references (8 per population)")
    # Match the existing independent-population samplers with the same seed;
    # reference draws must not depend on a ratio's background sample count.
    refs = {"output": "binary_output", "monitored_reference_ids": np.random.default_rng(0).choice(mtrain, 8, replace=False).tolist(),
            "background_reference_ids": np.random.default_rng(0).choice(btrain, 8, replace=False).tolist()}
    persist_selection(out / "binary_references.json", refs)
    return {"background_samples_common_across_ratios": bool(common_test),
            "background_common_test_count": len(common_test), "references_common_across_ratios": True}


def verify_output(out, binding):
    completion = json.loads((out / "complete.json").read_text())
    if completion["binding"] != binding:
        raise ValueError("Existing XAI output has different bundle/selection binding")
    if "raw_long.csv" not in completion["sha256"]:
        raise ValueError("Missing raw CSV in completed XAI manifest")
    for name, digest in completion["sha256"].items():
        if _digest(out / name) != digest:
            raise ValueError(f"Existing XAI output integrity failure: {name}")
    return completion


def run(args):
    binding = {"bundle_manifest": _digest(args.bundle / "manifest.json"), "selection": _digest(args.selection),
               "dataset": _digest(args.dataset), "background_pool": _digest(args.background_pool)
               if args.background_pool.exists() else None}
    binding["assigned_experiment_id"] = args.experiment_id
    binding["xai_source"] = {path.relative_to(Path(__file__).parent.parent).as_posix(): _digest(path) for path in
                             (Path(__file__), Path(__file__).with_name("preflight.py"),
                              Path(__file__).parent.parent / "end_to_end/explainer.py",
                              Path(__file__).parent.parent / "rf/explainer.py", Path(__file__).with_name("selection.py"))}
    if args.out_dir.exists():
        verify_output(args.out_dir, binding)
        print(f"Verified existing explanations: {args.out_dir}")
        return
    full_selection = json.loads(args.selection.read_text(encoding="utf-8"))
    bundle = load_bundle(args.bundle, trusted=args.trusted)
    metadata = bundle.metadata
    selection = for_run(full_selection, args.experiment_id)
    if selection["split_ids"] != metadata["split_ids"] or selection["experiment_id"] != metadata["experiment_id"]:
        raise ValueError("Selection and bundle use different repetition/split")
    from fold.data import load_traces
    _, labels, _, ids = load_traces(args.dataset, return_ids=True)
    label_by_id = dict(zip(ids, labels))
    splits = {int(run): (split, label_by_id, len(bundle.class_names)) for run, split in full_selection["splits"].items()}
    validate_selection(full_selection, splits, bundle.class_names)

    fold_policy = full_selection.get("fold_grouped_cohort")
    fold_records_all = full_selection.get("fold_records")
    if not isinstance(fold_policy, dict) or not isinstance(fold_records_all, list):
        raise ValueError("Selection lacks the fixed G1-G4 eligible FOLD cohort")

    fold_records = [
        r for r in fold_records_all
        if r["assigned_experiment_id"] == args.experiment_id
    ]
    if not fold_records:
        raise ValueError(f"No G1-G4 eligible FOLD observations for exp_{args.experiment_id:03d}")

    fold_selection = {
        "sample_ids": [r["sample_id"] for r in fold_records],
        "target_classes": [r["site_id"] for r in fold_records],
        "experiment_id": args.experiment_id,
        "split_ids": full_selection["splits"][str(args.experiment_id)],
        "sampling_seed": full_selection["sampling_seed"],
        "protocol": fold_policy["protocol"],
        "records": fold_records,
        "reuse_exceptions": [],
        "cohort_description": fold_policy["primary_scope"],
    }

    if metadata["dataset_sha256"] != binding["dataset"]:
        raise ValueError("Selected cohort dataset differs from bundle")
    pending = args.out_dir.parent / (args.out_dir.name + ".pending-" + uuid.uuid4().hex)
    pending.mkdir(parents=True)
    assigned_file = pending / "assigned_selection.json"
    persist_selection(assigned_file, selection)

    fold_assigned_file = pending / "fold_g4_assigned_selection.json"
    persist_selection(fold_assigned_file, fold_selection)

    rf_cohort_info = {
        "protocol": selection["protocol"],
        "cohort_description": selection["cohort_description"],
        "global_selection_sha256": binding["selection"],
        "reuse_exceptions": selection["reuse_exceptions"],
        "reused_trace_ids_this_run": [
            r["sample_id"]
            for r in selection["records"]
            if r["is_reused_trace_id"]
        ],
        "cohort_role": "rf_and_performance_primary_100_site",
    }

    fold_cohort_info = {
        "protocol": fold_policy["protocol"],
        "cohort_description": fold_policy["primary_scope"],
        "global_selection_sha256": binding["selection"],
        "parent_selection_sha256": fold_policy["parent_selection_sha256"],
        "eligibility": fold_policy["eligibility"],
        "group_definition": fold_policy["group_definition"],
        "n_sites": fold_policy["n_sites"],
        "n_observations": fold_policy["n_observations"],
        "n_full15_sites": fold_policy["n_full15_sites"],
        "reused_trace_ids_this_run": [],
        "cohort_role": "fold_grouped_g1_g4_eligible",
    }

    predictor = FoldPredictor(
        bundle.encoder,
        bundle.forest,
        bundle.model,
    )

    # FOLD uses only the fixed G1-G4 eligible cohort.
    refs, fold_samples, ref_ids, fold_sample_ids, info = select_traces(
        args.dataset,
        metadata,
        background_size=16,
        sample_ids_file=fold_assigned_file,
        reference_num_groups=4,
        seed=0,
    )

    if info["effective_num_groups"] != 4:
        raise ValueError(
            "Fixed FOLD cohort unexpectedly failed the exact G1-G4 gate"
        )

    fold_metadata = {
        **metadata,
        "selection": {
            **info,
            **fold_cohort_info,
        },
    }

    engine = FoldGroupExplainer(
        predictor,
        refs,
        reference_ids=ref_ids,
        target_class=0,
        class_names=bundle.class_names,
        metadata=fold_metadata,
        eligible_reference_count=info["eligible_reference_count"],
        num_groups=4,
    )

    # RF retains the original immutable 1500-observation cohort.
    rf_refs, rf_samples, rf_ids, rf_sample_ids, rf_info = select_traces(
        args.dataset,
        metadata,
        background_size=32,
        sample_ids_file=assigned_file,
        seed=0,
    )

    rf_metadata = {
        **metadata,
        "selection": {
            **rf_info,
            **rf_cohort_info,
        },
    }

    rf = ForestTreeExplainer(
        bundle.encoder,
        bundle.forest,
        rf_refs,
        background_ids=rf_ids,
        target_class=0,
        class_names=bundle.class_names,
        metadata=rf_metadata,
        allow_multiclass=True,
    )

    if len(rf.feature_names) != 97:
        raise ValueError(
            "All-sites RF requires the original 97 features"
        )
    with (pending / "raw_long.csv").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=LONG_FIELDS)
        writer.writeheader()
        fold_targets = fold_selection["target_classes"]
        rf_targets = selection["target_classes"]

        # Explain FOLD and RF on their explicitly different, fixed cohorts.
        # Both cohorts are shared across every configuration, preserving
        # exact pairing against baseline.
        for c in range(100):

            fold_indices = [
                i for i, target in enumerate(fold_targets)
                if target == c
            ]

            if fold_indices:
                batch = [
                    fold_samples[i]
                    for i in fold_indices
                ]
                bids = [
                    fold_sample_ids[i]
                    for i in fold_indices
                ]

                result = engine.explain_targets(
                    batch,
                    sample_ids=bids,
                    target_classes=[c] * len(batch),
                )[c]

                result.save(
                    pending / f"fold_class_{c:03d}.npz"
                )

                emit(
                    writer,
                    result,
                    fold_metadata,
                    model="fold",
                    site=bundle.class_names[c],
                    label=c,
                )

            rf_indices = [
                i for i, target in enumerate(rf_targets)
                if target == c
            ]

            if rf_indices:
                batch = [
                    rf_samples[i]
                    for i in rf_indices
                ]
                bids = [
                    rf_sample_ids[i]
                    for i in rf_indices
                ]

                forest = rf.explain_targets(
                    batch,
                    sample_ids=bids,
                    target_classes=[c] * len(batch),
                )[c]

                forest.save(
                    pending / f"rf_class_{c:03d}.npz"
                )

                emit(
                    writer,
                    forest,
                    rf_metadata,
                    model="rf",
                    site=bundle.class_names[c],
                    label=c,
                )

        if metadata["scenario"] == "open-world":
            from ..open_world.data import select_open_world
            from ..open_world.predictor import BinaryFoldPredictor
            from ..open_world.explainer import BinaryGroupExplainer

            policy = binary_files(args.dataset, args.background_pool, fold_metadata, fold_selection, pending)
            binary_selection = select_open_world(args.dataset, args.background_pool, fold_metadata,
                                                 sample_ids_file=pending / "binary_samples.json", num_groups=4,
                                                 reference_ids_file=pending / "binary_references.json")
            if binary_selection.metadata["selection"]["effective_num_groups"] != 4:
                raise ValueError("Binary all-sites requires exactly four nonempty resolution groups")
            binary = BinaryGroupExplainer(BinaryFoldPredictor(bundle), binary_selection).explain()
            binary.metadata.update(policy)
            binary.metadata["cross_run_cohort"] = fold_cohort_info
            binary.save(pending / "binary.npz")
            # Emit by population/site with a lightweight row view.
            from types import SimpleNamespace
            for i, (tid, population) in enumerate(zip(binary.sample_ids, binary.sample_types)):
                label = label_by_id[tid] if population == "monitored" else -1
                site = bundle.class_names[label] if label >= 0 else "background"
                view = SimpleNamespace(values=binary.values[i:i+1], base_values=binary.base_values[i:i+1],
                                       probabilities=binary.probabilities[i:i+1], sample_ids=(tid,),
                                       reference_ids=binary.reference_ids, metadata=binary.metadata)
                emit(writer, view, fold_metadata, model="fold", site=site, label=label,
                     cohort=population, output="binary_output")
    files = {f.name: _digest(f) for f in pending.iterdir() if f.is_file()}
    (pending / "complete.json").write_text(json.dumps({"binding": binding, "sha256": files}, indent=2), encoding="utf-8")
    args.out_dir.parent.mkdir(parents=True, exist_ok=True)
    if args.out_dir.exists():
        raise ValueError("Output appeared concurrently; pending results preserved")
    pending.rename(args.out_dir)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--selection", type=Path, required=True)
    p.add_argument("--experiment-id", type=int, choices=range(1, 6), required=True)
    p.add_argument("--dataset", type=Path, default=Path("data/odoh_traces.csv"))
    p.add_argument("--background-pool", type=Path, default=Path("data/background_pool.csv"))
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--trusted", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
