"""Aggregate all-sites long CSV and performance JSON; plotting never reruns XAI."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .statistics import hierarchical_ci, paired_delta, repetition_ci


KEYS = ["configuration", "model", "output", "cohort"]


def cells(frame, value="phi"):
    dimensions = sorted(frame.dimension.unique())
    result = {}
    for (run, site), group in frame.groupby(["experiment_id", "site"]):
        table = group.pivot(index="sample_id", columns="dimension", values=value).reindex(columns=dimensions)
        if table.isna().any().any():
            raise ValueError("Missing attribution dimensions")
        result.setdefault(int(run), {})[site] = table.to_numpy()
    return result, dimensions


def summary(frame, *, bootstrap, seed, statistic="abs", scope="all_sites"):
    data = frame.copy()
    if scope == "instagram":
        data = data[data.site == "instagram.com"]
    if data.empty:
        return []
    if statistic == "abs":
        data["phi"] = data.phi.abs()
    hierarchy, dimensions = cells(data)
    ci = hierarchical_ci(hierarchy, bootstrap=bootstrap, seed=seed, resample_sites=(scope == "all_sites"))
    n_unique = len(data[["site", "sample_id"]].drop_duplicates())
    return [{"dimension": dimension, "statistic": statistic, "scope": scope,
             "n_observations": ci["n_samples"], "n_unique_traces": n_unique,
             **{k: float(ci[k][i]) for k in ("mean", "ci_low", "ci_high")},
             **{k: ci[k] for k in ("n_runs", "n_sites", "n_samples", "bootstrap", "bootstrap_seed", "ci_method")}}
            for i, dimension in enumerate(dimensions)]


def validate_raw_cohort(group, selection):
    """Require exactly the assigned sample-run pairs, including explicit reuse."""
    records = (
        selection.get("fold_records", selection["records"])
        if group.model.iloc[0] == "fold"
        else selection["records"]
    )
    expected = {
        (r["assigned_experiment_id"], r["site_id"], r["sample_id"]): r
        for r in records
    }
    observations = group.drop_duplicates(["experiment_id", "site_id", "sample_id"])
    actual = {(int(r.experiment_id), int(r.site_id), r.sample_id): r for r in observations.itertuples()}
    if set(expected) != set(actual):
        raise ValueError("CSV cohort differs from immutable assigned sample-run observations")
    for key, record in expected.items():
        row = actual[key]
        if row.site != record["site_name"] or row.is_reused_trace_id != record["is_reused_trace_id"]:
            raise ValueError("CSV site/reuse metadata differs from selection")
    dimensions = 97 if group.model.iloc[0] == "rf" else 4
    counts = group.groupby(["experiment_id", "site_id", "sample_id"]).dimension.nunique()
    if not (counts == dimensions).all():
        raise ValueError("Missing RF feature or grouped attribution dimensions")
    if group.output.iloc[0] == "multi_output" and not (group.target_class == group.site_id).all():
        raise ValueError("Multiclass target must equal sample's true site class")
    if group.output.iloc[0] == "binary_output" and not (group.target_class == 0).all():
        raise ValueError("Binary output must target P(monitored)")


def aggregate(raw, *, bootstrap=10000, seed=20261008, require_complete=True, selection=None):
    unique = KEYS + ["experiment_id", "site_id" if "site_id" in raw else "site", "sample_id", "dimension"]
    if raw.duplicated(unique).any():
        raise ValueError("Unexpected duplicate long CSV records")
    if not np.isfinite(raw[["phi", "probability", "base_value"]].to_numpy()).all():
        raise ValueError("Nonfinite raw CSV values")
    if require_complete:
        if selection is None:
            raise ValueError("Complete cross-run aggregation requires the immutable selection")
        for key, group in raw.groupby(KEYS):
            if set(group.experiment_id) != set(range(1, 6)):
                raise ValueError(f"Requires all five repetitions: {key}")
            if key[-1] == "monitored":
                validate_raw_cohort(group, selection)
    site = raw.assign(abs_phi=raw.phi.abs()).groupby(KEYS + ["experiment_id", "site", "dimension"], as_index=False).agg(
        mean=("abs_phi", "mean"), mean_signed=("phi", "mean"), n_samples=("sample_id", "nunique"))
    site["n_observations"] = site.n_samples  # no repeated pair is permitted in a site/run
    site["n_unique_traces"] = site.n_samples
    fold_counts = {}
    if selection is not None:
        for record in selection.get("fold_records", []):
            site_name = record["site_name"]
            fold_counts[site_name] = fold_counts.get(site_name, 0) + 1

    full15_sites = {
        site_name
        for site_name, count in fold_counts.items()
        if count == 15
    }

    global_rows, delta_rows, pairing = [], [], []
    for key, group in raw.groupby(KEYS):
        fields = dict(zip(KEYS, key))
        statistics = ("abs", "signed") if fields["output"] == "binary_output" else ("abs",)
        for statistic in statistics:
            for scope in ("instagram", "all_sites") if fields["cohort"] == "monitored" else ("all_sites",):
                global_rows.extend({**fields, **row} for row in summary(group, bootstrap=bootstrap, seed=seed, statistic=statistic, scope=scope))
        # Probability summary is sample-level, avoiding feature/group replication.
        probability = group.drop_duplicates(["experiment_id", "site", "sample_id"]).copy()
        probability["dimension"], probability["phi"] = "target_probability", probability.probability
        for scope in ("instagram", "all_sites") if fields["cohort"] == "monitored" else ("all_sites",):
            global_rows.extend({**fields, **row} for row in summary(probability, bootstrap=bootstrap, seed=seed, statistic="signed", scope=scope))
        if fields["configuration"] != "baseline" and fields["output"] == "multi_output":
            baseline = raw[(raw.configuration == "baseline") & (raw.model == fields["model"]) & (raw.output == "multi_output") & (raw.cohort == fields["cohort"])]
            try:
                delta = paired_delta(group, baseline)
            except ValueError as error:
                pairing.append({**fields, "paired": False, "reason": str(error)})
                continue
            pairing.append({**fields, "paired": True, "reason": "exact ID match"})
            delta_rows.extend(delta.to_dict("records"))
            for scope in ("instagram", "all_sites"):
                global_rows.extend({**fields, **row, "pairing": "exact_run_site_sample"} for row in summary(
                    delta, bootstrap=bootstrap, seed=seed, statistic="delta_abs", scope=scope))
    # --------------------------------------------------------
    # Prespecified sensitivity analysis:
    # 76 sites with the full quota of 15 DISTINCT eligible
    # G1-G4 traces. This is separate from the primary 92-site
    # result and does not recompute SHAP.
    # --------------------------------------------------------
    full15_rows = []

    fold_monitored = raw[
        (raw.model == "fold")
        & (raw.output == "multi_output")
        & (raw.cohort == "monitored")
        & (raw.site.isin(full15_sites))
    ].copy()

    if not fold_monitored.empty:
        for key, group in fold_monitored.groupby(KEYS):
            fields = dict(zip(KEYS, key))

            for statistic in ("abs",):
                rows = summary(
                    group,
                    bootstrap=bootstrap,
                    seed=seed,
                    statistic=statistic,
                    scope="all_sites",
                )

                for row in rows:
                    row = dict(row)
                    row["scope"] = "all_sites_full15"
                    row["sensitivity_definition"] = (
                        "sites_with_15_distinct_g1_g4_eligible_test_traces"
                    )
                    full15_rows.append({**fields, **row})

            probability = group.drop_duplicates(
                ["experiment_id", "site", "sample_id"]
            ).copy()

            probability["dimension"] = "target_probability"
            probability["phi"] = probability.probability

            rows = summary(
                probability,
                bootstrap=bootstrap,
                seed=seed,
                statistic="signed",
                scope="all_sites",
            )

            for row in rows:
                row = dict(row)
                row["scope"] = "all_sites_full15"
                row["sensitivity_definition"] = (
                    "sites_with_15_distinct_g1_g4_eligible_test_traces"
                )
                full15_rows.append({**fields, **row})

    full15_delta_rows = []

    delta_frame = pd.DataFrame(delta_rows)

    if not delta_frame.empty and full15_sites:
        subset = delta_frame[
            delta_frame.site.isin(full15_sites)
        ].copy()

        if not subset.empty:
            for key, group in subset.groupby(KEYS):
                fields = dict(zip(KEYS, key))

                rows = summary(
                    group,
                    bootstrap=bootstrap,
                    seed=seed,
                    statistic="delta_abs",
                    scope="all_sites",
                )

                for row in rows:
                    row = dict(row)
                    row["scope"] = "all_sites_full15"
                    row["pairing"] = "exact_run_site_sample"
                    row["sensitivity_definition"] = (
                        "sites_with_15_distinct_g1_g4_eligible_test_traces"
                    )
                    full15_delta_rows.append({**fields, **row})

    global_frame = pd.DataFrame(global_rows)
    if not ((global_frame.ci_low <= global_frame["mean"]) & (global_frame["mean"] <= global_frame.ci_high)).all():
        raise ValueError("Bootstrap percentile interval does not enclose point estimate; inspect data/draw count")
    # Retain historical per-run winners, but the primary winner is based on
    # each site's equal-run aggregate BEFORE argmax, not a vote over runs.
    groups = site[(site.model == "fold") & (site.output == "multi_output")].copy()
    pivot = groups.pivot(index=KEYS + ["experiment_id", "site"], columns="dimension", values="mean").sort_index()
    winners = pivot.idxmax(axis=1).rename("dominant_group").reset_index()
    distributions = []
    for key, frame in winners.groupby(KEYS):
        for run, run_frame in frame.groupby("experiment_id"):
            counts = run_frame.dominant_group.value_counts()
            for g in ("G1", "G2", "G3", "G4"):
                distributions.append({**dict(zip(KEYS, key)), "experiment_id": run, "group": g,
                                      "n_sites": len(run_frame), "count": int(counts.get(g, 0)),
                                      "fraction": counts.get(g, 0)/len(run_frame)})
    distribution = pd.DataFrame(distributions)
    heatmap = groups.groupby(KEYS + ["site", "dimension"], as_index=False).agg(mean=("mean", "mean"), n_runs=("experiment_id", "nunique"))
    cross_pivot = heatmap.pivot(index=KEYS + ["site"], columns="dimension", values="mean").sort_index()
    cross_winners = cross_pivot.idxmax(axis=1).rename("dominant_group").reset_index()
    dominant_summary = []
    for key, frame in cross_winners.groupby(KEYS):
        counts = frame.dominant_group.value_counts()
        for g in ("G1", "G2", "G3", "G4"):
            dominant_summary.append({**dict(zip(KEYS, key)), "group": g, "count": int(counts.get(g, 0)),
                                     "mean": counts.get(g, 0)/len(frame), "n_sites": len(frame),
                                     "meaning": "argmax of each site's equal-run mean absolute attribution; each site counted once"})
    observations = raw.drop_duplicates(
        [
            "configuration",
            "model",
            "output",
            "cohort",
            "experiment_id",
            "site",
            "sample_id",
        ]
    )

    coverage = observations.groupby(
        KEYS + ["site"],
        as_index=False,
    ).agg(
        n_observations=("sample_id", "size"),
        n_unique_traces=("sample_id", "nunique"),
        n_runs=("experiment_id", "nunique"),
    )

    coverage["full15_fold_sensitivity_site"] = (
        coverage.site.isin(full15_sites)
        & (coverage.model == "fold")
        & (coverage.output == "multi_output")
        & (coverage.cohort == "monitored")
    )

    sensitivity = sensitivity_check(raw, selection)
    return {"site_summary": site, "global_summary": global_frame, "delta_raw_long": pd.DataFrame(delta_rows),
            "pairing": pd.DataFrame(pairing), "dominant_sites": cross_winners,
            "dominant_sites_per_run": winners, "dominant_per_run": distribution,
            "dominant_summary": pd.DataFrame(dominant_summary), "heatmap": heatmap,
            "coverage": coverage,
            "full15_global_summary": pd.DataFrame(full15_rows),
            "full15_delta_summary": pd.DataFrame(full15_delta_rows),
            "reuse_sensitivity": sensitivity}


def sensitivity_check(raw, selection):
    """Remove ONLY the marked reused occurrence; retain its other run's trace.

    Includes all 97 RF dimensions, G1-G4, signed/absolute means, probabilities
    and base values. Uses equal-run means per site before equal-site averaging.
    No XAI is recomputed and no unmeasured real-data result is invented.
    """
    if selection is None:
        return pd.DataFrame(columns=["status", "model", "statistic", "dimension", "mean_A", "mean_B", "difference", "abs_difference"])
    reused = {(r["assigned_experiment_id"], r["site_name"], r["sample_id"]) for r in selection["records"] if r["is_reused_trace_id"]}
    baseline = raw[(raw.configuration == "baseline") & (raw.output == "multi_output") & (raw.cohort == "monitored")]
    results = []
    for model, frame in baseline.groupby("model"):
        present = set(zip(frame.experiment_id, frame.site, frame.sample_id))
        if not reused <= present:
            results.append(dict(status="unavailable_missing_reused_observation", model=model))
            continue
        for statistic in ("abs", "signed", "probability", "base_value"):
            data = frame.copy()
            if statistic in ("probability", "base_value"):
                data = data.drop_duplicates(["experiment_id", "site", "sample_id"])
                data["dimension"] = "target_probability" if statistic == "probability" else "base_value"
                data["phi"] = data[statistic]
            elif statistic == "abs":
                data["phi"] = data.phi.abs()
            exclude = np.array([(r.experiment_id, r.site, r.sample_id) in reused for r in data.itertuples()])
            def estimate(source):
                run = source.groupby(["site", "experiment_id", "dimension"]).phi.mean()
                return run.groupby(["site", "dimension"]).mean().groupby("dimension").mean()
            a, b = estimate(data), estimate(data.loc[~exclude])
            for dimension in a.index:
                difference = float(b[dimension]-a[dimension])
                results.append(dict(status="measured_from_supplied_baseline_attributions", model=model, statistic=statistic,
                                    dimension=dimension, mean_A=float(a[dimension]), mean_B=float(b[dimension]),
                                    difference=difference, abs_difference=abs(difference)))
    return pd.DataFrame(results)


def performance(root, *, bootstrap, seed, complete=True):
    rows = []
    from .runner import configurations
    from ..core.configuration import scenario_tag

    for fields, _ in configurations():
        configuration = scenario_tag(fields)
        for run in range(1, 6):
            path = root / configuration / f"exp_{run:03d}" / "result.json"
            if not path.exists():
                if complete:
                    raise ValueError(f"Missing metrics: {path}")
                continue
            report = json.loads(path.read_text())
            if report["experiment_id"] != run or report["scenario"] != fields["scenario"]:
                raise ValueError(f"Metrics provenance mismatch: {path}")
            for model in ("forest", "fold"):
                for metric, value in report[model].items():
                    values = value if isinstance(value, dict) else {metric: value}
                    for name, number in values.items():
                        if isinstance(number, (int, float)):
                            rows.append({"configuration": configuration, "experiment_id": run, "model": model,
                                         "metric": f"{metric}.{name}" if isinstance(value, dict) else name, "value": number})
    raw = pd.DataFrame(rows)
    summaries = []
    if not raw.empty:
        for key, frame in raw.groupby(["configuration", "model", "metric"]):
            ci = repetition_ci(frame.value.to_numpy(), bootstrap=bootstrap, seed=seed)
            summaries.append({**dict(zip(("configuration", "model", "metric"), key)),
                              **{k: float(ci[k][0]) for k in ("mean", "ci_low", "ci_high")},
                              "n_runs": len(frame), "ci_method": "repetition_percentile_95", "bootstrap": bootstrap,
                              "bootstrap_seed": seed})
    return raw, pd.DataFrame(summaries)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("explanations/all_sites"))
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--out-dir", type=Path, default=Path("analysis/cross_run/summary"))
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--allow-partial", action="store_true", help="Diagnostic only; records actual n_runs/n_sites")
    p.add_argument("--selection", type=Path, default=Path("analysis/cross_run/selections/all_sites_cross_run.json"))
    args = p.parse_args()
    files = sorted(args.input.glob("*/exp_*/raw_long.csv"))
    if not files:
        raise ValueError("No all-sites CSV; run explanations after passing pre-flight")
    frames = []
    from .worker import verify_output
    for file in files:
        completion = json.loads((file.parent / "complete.json").read_text())
        verify_output(file.parent, completion["binding"])
        frames.append(pd.read_csv(file))
    raw = pd.concat(frames, ignore_index=True)
    if not args.allow_partial:
        from .runner import configurations
        from ..core.configuration import scenario_tag
        expected = {scenario_tag(fields) for fields, _ in configurations()}
        if set(raw.configuration) != expected:
            raise ValueError("Requires all 26 configurations")
        for configuration, frame in raw.groupby("configuration"):
            actual = set(zip(frame.model, frame.output, frame.cohort))
            required = {("fold", "multi_output", "monitored"), ("rf", "multi_output", "monitored")}
            if configuration.startswith("openworld"):
                required.update({("fold", "binary_output", "monitored"), ("fold", "binary_output", "background")})
            if actual != required:
                raise ValueError(f"Missing or unexpected RF/FOLD/binary output population: {configuration}")
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    tables = aggregate(raw, bootstrap=args.bootstrap, seed=args.seed, require_complete=not args.allow_partial, selection=selection)
    perf, perf_summary = performance(args.artifacts, bootstrap=args.bootstrap, seed=args.seed, complete=not args.allow_partial)
    tables.update(performance_raw=perf, performance_summary=perf_summary)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(args.out_dir / f"{name}.csv", index=False)
    print(f"Wrote {len(tables)} CSV tables; no SHAP recomputation")


if __name__ == "__main__":
    main()
