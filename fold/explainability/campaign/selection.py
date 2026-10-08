"""Immutable TEST sample-run cohorts: maximize diversity, then balance runs.

Per site, a small binary optimization selects exactly 15 valid occurrences,
with exactly min(15, union size) distinct IDs, and minimizes sum(run_count**2).
Seeded tie costs (<1 in total) cannot override the integer balance objective.
No replacement or repeated (sample_id, experiment_id) pairs are possible.
"""
import csv
import json
from pathlib import Path

import numpy as np

from .preflight import cross_run_availability, require_cross_run_available


PROTOCOL = "test_sample_run_observations_v1"
COHORT_DESCRIPTION = (
    "15 held-out sample-run observations per site. For 99/100 sites these correspond "
    "to 15 distinct traces; Spotify has 14 distinct traces, with one trace evaluated "
    "under two independent repetitions in which it belongs to the test split.")


def choose_site(availability, *, size=15, seed=0):
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy.sparse import lil_matrix

    pairs = [(tid, run) for run in range(1, 6) for tid in sorted(availability[f"test_ids_exp{run:03d}"])]
    ids = sorted({tid for tid, _ in pairs})
    if len(pairs) < size:
        raise ValueError("Insufficient distinct TEST sample-run observations")
    nx, ny, nz = len(pairs), len(ids), 5*size
    # x: occurrence chosen; y: distinct ID chosen; z: run occupancy slots.
    cost = np.zeros(nx+ny+nz)
    rng = np.random.default_rng(np.random.SeedSequence([seed, availability["site_id"]]))
    cost[:nx] = rng.permutation(nx) / (nx*size*10)
    cost[nx+ny:] = np.tile(np.arange(1, 2*size, 2), 5)
    matrix = lil_matrix((2+2*ny+5, len(cost)), dtype=float)
    lower, upper = [], []
    matrix[0, :nx] = 1
    lower.append(size); upper.append(size)
    matrix[1, nx:nx+ny] = 1
    lower.append(min(size, ny)); upper.append(min(size, ny))
    row = 2
    for j, tid in enumerate(ids):
        occurrences = [i for i, pair in enumerate(pairs) if pair[0] == tid]
        matrix[row, occurrences] = 1
        matrix[row, nx+j] = -1
        lower.append(0); upper.append(np.inf)
        row += 1
        matrix[row, occurrences] = 1
        matrix[row, nx+j] = -len(occurrences)
        lower.append(-np.inf); upper.append(0)
        row += 1
    for run in range(1, 6):
        matrix[row, [i for i, pair in enumerate(pairs) if pair[1] == run]] = 1
        start = nx+ny+(run-1)*size
        matrix[row, start:start+size] = -1
        lower.append(0); upper.append(0)
        row += 1
    result = milp(cost, integrality=np.ones(len(cost)), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"mip_rel_gap": 0.0})
    if not result.success:
        raise ValueError(f"Cross-run selection optimization failed: {result.message}")
    selected = sorted(pairs[i] for i in range(nx) if result.x[i] > .5)
    seen, records = set(), []
    for tid, run in selected:
        records.append(dict(sample_id=tid, site_id=availability["site_id"], site_name=availability["site_name"],
                            assigned_experiment_id=run, is_reused_trace_id=tid in seen))
        seen.add(tid)
    return records


def prepare_selection(splits, names, *, seed=0, size=15):
    availability, insufficient = cross_run_availability(splits, names, size=size)
    require_cross_run_available(insufficient)
    records = [record for site in availability for record in choose_site(site, seed=seed, size=size)]
    selection = {"protocol": PROTOCOL, "sampling_seed": seed, "observations_per_site": size,
                 "n_sites": len(names), "n_observations": len(records), "records": records,
                 "splits": {str(run): split for run, (split, _, _) in splits.items()},
                 "selection_objective": "maximum_distinct_IDs_then_minimum_sum_squared_run_counts_then_seeded_ties",
                 "cohort_description": COHORT_DESCRIPTION if len(names) == 100 else "held-out sample-run observations",
                 "reuse_exceptions": [dict(site_id=site["site_id"], site_name=site["site_name"],
                                            n_unique_traces=site["n_unique_test_union"], n_observations=size)
                                      for site in availability if site["requires_cross_run_duplicate"]]}
    validate_selection(selection, splits, names, size=size)
    return selection


def validate_selection(selection, splits, names, *, size=15):
    if selection.get("protocol") != PROTOCOL or selection.get("observations_per_site") != size:
        raise ValueError("Unsupported cross-run selection protocol/quota")
    records = selection["records"]
    if selection.get("n_sites") != len(names) or selection.get("n_observations") != len(records) or len(records) != len(names)*size:
        raise ValueError("Expected exactly 15 sample-run observations for every site")
    if selection.get("splits") != {str(run): split for run, (split, _, _) in splits.items()}:
        raise ValueError("Selection split provenance differs")
    availability, insufficient = cross_run_availability(splits, names, size=size)
    require_cross_run_available(insufficient)
    observed = set()
    for record in records:
        tid, run, site = record["sample_id"], record["assigned_experiment_id"], record["site_id"]
        if type(run) is not int or run not in splits or type(site) is not int or not 0 <= site < len(names):
            raise ValueError("Invalid assigned run/site")
        if (tid, run) in observed:
            raise ValueError("Duplicate sample_id/experiment_id pair")
        observed.add((tid, run))
        split, labels, _ = splits[run]
        if tid not in split["test"] or tid in split.get("train", []) or tid in split.get("val", []):
            raise ValueError("Observation must belong exclusively to assigned TEST split")
        if labels.get(tid) != site or record["site_name"] != names[site] or type(record["is_reused_trace_id"]) is not bool:
            raise ValueError("Observation class/site/reuse metadata mismatch")
    for site in availability:
        chosen = [r for r in records if r["site_id"] == site["site_id"]]
        unique = {r["sample_id"] for r in chosen}
        if len(chosen) != size or len(unique) != min(size, site["n_unique_test_union"]):
            raise ValueError("Site quota or maximum distinct-trace objective violated")
        if sum(r["is_reused_trace_id"] for r in chosen) != size-len(unique):
            raise ValueError("Reuse flag count differs from actual duplicate trace count")
        for tid in unique:
            occurrences = [r for r in chosen if r["sample_id"] == tid]
            if sum(not r["is_reused_trace_id"] for r in occurrences) != 1:
                raise ValueError("Exactly one occurrence per trace must be marked original")
    expected_exceptions = [dict(site_id=site["site_id"], site_name=site["site_name"],
                                n_unique_traces=site["n_unique_test_union"], n_observations=size)
                           for site in availability if site["requires_cross_run_duplicate"]]
    if selection.get("reuse_exceptions") != expected_exceptions:
        raise ValueError("Missing or incorrect explicit reuse exception metadata")
    return records


def for_run(selection, run):
    records = [r for r in selection["records"] if r["assigned_experiment_id"] == run]
    if not records:
        raise ValueError(f"No assigned observations for exp_{run:03d}")
    return {"sample_ids": [r["sample_id"] for r in records], "target_classes": [r["site_id"] for r in records],
            "experiment_id": run, "split_ids": selection["splits"][str(run)],
            "sampling_seed": selection["sampling_seed"], "protocol": PROTOCOL,
            "records": records, "reuse_exceptions": selection["reuse_exceptions"],
            "cohort_description": selection["cohort_description"]}


def save_selection(selection, out_dir=Path("analysis/cross_run/preflight"), path=Path("analysis/cross_run/selections/all_sites_cross_run.json")):
    from .runner import persist_selection

    persist_selection(path, selection)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "all_sites_cross_run_selection.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("sample_id", "site_id", "site_name", "assigned_experiment_id", "is_reused_trace_id"))
        writer.writeheader()
        writer.writerows(selection["records"])
