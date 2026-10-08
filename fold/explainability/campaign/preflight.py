"""Reusable, TensorFlow-free audit of the ORIGINAL monitored splits.

Uses load_balanced and split_data, and reads Config.seed's expression from its
AST (rather than importing experiments, which eagerly imports TensorFlow).
Fails closed if the original seed implementation changes.
"""
import argparse
import ast
import csv
import json
from pathlib import Path

import numpy as np

from fold.data import load_balanced, split_data
from ..core.artifacts import _digest
from ..train import class_names


def original_seed(experiment_id):
    tree = ast.parse((Path(__file__).parents[2] / "experiments.py").read_text(encoding="utf-8"))
    config = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Config")
    method = next(n for n in config.body if isinstance(n, ast.FunctionDef) and n.name == "seed")
    expression = next(n.value for n in method.body if isinstance(n, ast.Return))
    expected = ast.parse("self.experiment_id * 17 + 42", mode="eval").body
    if ast.dump(expression) != ast.dump(expected):
        raise ValueError("Config.seed changed: update the pre-flight adapter before running")
    return experiment_id * 17 + 42


def reconstruct(dataset, experiment_id, loaded=None):
    traces, labels, classes, ids = loaded or load_balanced(dataset, return_ids=True)
    order = np.random.default_rng(original_seed(experiment_id)).permutation(len(ids))
    ordered_ids = [ids[i] for i in order]
    ordered_labels = [labels[i] for i in order]
    parts = split_data(ordered_ids, ordered_labels)
    split = {"dataset_sha256": _digest(dataset), **dict(zip(("train", "val", "test"), parts[0]))}
    label_by_id = dict(zip(ids, labels))
    return split, label_by_id, classes


def audit(dataset, repetitions=range(1, 6), size=15):
    loaded = load_balanced(dataset, return_ids=True)
    names = class_names(dataset)
    rows, splits, insufficient = [], {}, []
    for run in repetitions:
        split, labels, classes = reconstruct(dataset, run, loaded)
        if classes != 100:
            raise ValueError(f"Expected 100 sites, found {classes}")
        splits[run] = (split, labels, classes)
        for c in range(classes):
            row = {"experiment_id": run, "seed": original_seed(run), "class": c, "site": names[c],
                   "total": sum(v == c for v in labels.values()),
                   **{part: sum(labels[tid] == c for tid in split[part]) for part in ("train", "val", "test")}}
            row["required_test"] = size
            rows.append(row)
            if row["test"] < size:
                insufficient.append(row)
    return rows, splits, insufficient


def require_available(insufficient):
    if insufficient:
        details = "; ".join(f"exp_{r['experiment_id']:03d} class={r['class']} site={r['site']} test={r['test']}"
                            for r in insufficient)
        raise ValueError("15 TEST samples/site unavailable; campaign blocked before training: " + details)


def write_audit(out, rows, insufficient):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for filename, records in (("availability.csv", rows), ("insufficient.csv", insufficient)):
        with (out / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(records)
    (out / "availability.json").write_text(json.dumps({"rows": rows, "insufficient": insufficient}, indent=2), encoding="utf-8")


def cross_run_availability(splits, names, *, size=15):
    """Count DISTINCT test IDs in the union, without selecting or moving IDs.

    An ID present in multiple test repetitions counts once. Only recorded test
    membership is consulted; train/validation never contribute to the union.
    """
    if set(splits) != set(range(1, 6)):
        raise ValueError("Cross-run feasibility requires exactly experiment_id 1..5")
    labels = splits[1][1]
    if any(classes != len(names) or run_labels != labels
           for _, run_labels, classes in splits.values()):
        raise ValueError("Class/ID mapping differs across repetitions")
    rows = []
    for c, name in enumerate(names):
        row = {"site_id": c, "site_name": name}
        union = set()
        by_run = {}
        for run in range(1, 6):
            split = splits[run][0]
            ids = [tid for tid in split["test"] if labels[tid] == c]
            if len(ids) != len(set(ids)):
                raise ValueError(f"Duplicate test IDs in exp_{run:03d}")
            by_run[run] = ids
            row[f"n_test_exp{run:03d}"] = len(ids)
            union.update(ids)
        occurrences = sum(len(ids) for ids in by_run.values())
        row.update(n_unique_test_union=len(union), n_test_occurrences_cross_run=occurrences,
                   requires_cross_run_duplicate=len(union) < size and occurrences >= size,
                   feasible_15=occurrences >= size)
        # Also retain the actual IDs for reproducible inspection of every union.
        row.update({f"test_ids_exp{run:03d}": by_run[run] for run in range(1, 6)})
        row["test_ids_union"] = sorted(union)
        rows.append(row)
    return rows, [row for row in rows if not row["feasible_15"]]


def write_cross_run_audit(out, rows):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cross_run_availability.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, list) else value
                         for key, value in row.items()} for row in rows)
    (out / "cross_run_availability.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def require_cross_run_available(insufficient):
    if insufficient:
        details = "; ".join(f"site_id={r['site_id']} site={r['site_name']} unique_test_union={r['n_unique_test_union']}"
                            for r in insufficient)
        raise ValueError("Cross-run gate BLOCKED: requires 15 TEST sample-run observations/site across exp_001..exp_005; " + details)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("data/odoh_traces.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis/cross_run/preflight"))
    parser.add_argument("--cross-run", action="store_true", help="Audit 15 TEST sample-run observations/site (default)")
    parser.add_argument("--legacy-per-run", action="store_true", help="Historical 15 TEST traces/site/run diagnostic only")
    parser.add_argument("--prepare-selection", action="store_true", help="Save the deterministic 1500-observation selection")
    args = parser.parse_args()
    rows, splits, insufficient = audit(args.dataset)
    write_audit(args.out_dir, rows, insufficient)
    print(f"500 site/repetition rows; insufficient={len(insufficient)}")
    if not args.legacy_per_run:
        cross_rows, cross_insufficient = cross_run_availability(splits, class_names(args.dataset))
        write_cross_run_audit(args.out_dir, cross_rows)
        counts = [r["n_unique_test_union"] for r in cross_rows]
        print(f"Cross-run: {len(cross_rows)} sites; min/mean/max unique TEST union="
              f"{min(counts)}/{np.mean(counts):.2f}/{max(counts)}; insufficient={len(cross_insufficient)}")
        require_cross_run_available(cross_insufficient)
        if args.prepare_selection:
            from .selection import prepare_selection, save_selection
            selection = prepare_selection(splits, class_names(args.dataset))
            save_selection(selection, args.out_dir)
    else:
        require_available(insufficient)


if __name__ == "__main__":
    main()
