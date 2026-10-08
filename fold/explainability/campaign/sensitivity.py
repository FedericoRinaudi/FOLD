"""Economical Spotify reuse check from saved cross-run BASELINE CSV only."""
import argparse
import json
from pathlib import Path

import pandas as pd

from .worker import verify_output
from .summarize import sensitivity_check, validate_raw_cohort


def run(input_dir, selection, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted((input_dir / "baseline").glob("exp_*/raw_long.csv"))
    report = {"method": "equal-run site means then equal-site mean; B removes only marked reuse",
              "n_observations_A": selection["n_observations"],
              "n_observations_B": selection["n_observations"]-sum(r["is_reused_trace_id"] for r in selection["records"])}
    if not paths:
        report.update(status="NOT_MEASURED", reason="No baseline cross-run attributions available; historical Instagram NPZ is insufficient")
    else:
        frames = []
        for path in paths:
            completion = json.loads((path.parent / "complete.json").read_text())
            verify_output(path.parent, completion["binding"])
            frames.append(pd.read_csv(path))
        raw = pd.concat(frames, ignore_index=True)
        for model in ("fold", "rf"):
            subset = raw[(raw.model == model) & (raw.output == "multi_output") & (raw.cohort == "monitored")]
            if subset.empty:
                raise ValueError(f"Missing baseline {model} observations")
            validate_raw_cohort(subset, selection)
        table = sensitivity_check(raw, selection)
        table.to_csv(out_dir / "reuse_sensitivity.csv", index=False)
        report.update(status="MEASURED", max_abs_difference=table.groupby(["model", "statistic"]).abs_difference.max().to_dict())
        report["max_abs_difference"] = {f"{model}.{statistic}": value for (model, statistic), value in report["max_abs_difference"].items()}
    # Conservative bounds, not empirical results: |phi| and probabilities have
    # range [0,1] (each exact/interventional Shapley value lies in [-1,1]).
    bounds = []
    for record in selection["records"]:
        if record["is_reused_trace_id"]:
            run, site = record["assigned_experiment_id"], record["site_id"]
            count = sum(r["site_id"]==site and r["assigned_experiment_id"]==run for r in selection["records"])
            nruns = len({r["assigned_experiment_id"] for r in selection["records"] if r["site_id"]==site})
            if count > 1:
                bounds.append(dict(site_id=site, removed_run=run, n_in_removed_cell=count,
                                   absolute_phi_or_probability_max_change=1/(count*nruns*selection["n_sites"]),
                                   signed_phi_max_change=2/(count*nruns*selection["n_sites"]),
                                   meaning="theoretical bound, not a measured sensitivity result"))
    report["bounds"] = bounds
    (out_dir / "reuse_sensitivity_status.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("explanations/all_sites"))
    p.add_argument("--selection", type=Path, default=Path("analysis/cross_run/selections/all_sites_cross_run.json"))
    p.add_argument("--out-dir", type=Path, default=Path("analysis/cross_run/preflight"))
    args = p.parse_args()
    print(json.dumps(run(args.input, json.loads(args.selection.read_text()), args.out_dir), indent=2))


if __name__ == "__main__":
    main()
