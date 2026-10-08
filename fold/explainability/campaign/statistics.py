"""Equal-site hierarchical means and reproducible percentile 95% bootstrap CI.

XAI: draw R repetitions with replacement; in each drawn repetition draw S
sites with replacement; for each drawn site/run draw its assigned observations
with replacement. Average observations, then available runs within each site,
then sites (uniform weights at both levels). Instagram omits
the site-resampling level. Background binary has one population-level site.
Performance resamples repetitions only. Paired deltas resample DIFFERENCES,
not two independent marginal intervals. No normalized attribution shares.
"""
import numpy as np


def hierarchical_ci(cells, *, bootstrap=10000, seed=20261008, resample_sites=True):
    """cells[run][site]: finite (samples, dimensions) arrays; equal site weight."""
    if bootstrap < 100 or not cells:
        raise ValueError("Use at least 100 bootstrap draws and nonempty data")
    runs = sorted(cells)
    if any(not cells[run] for run in runs):
        raise ValueError("Empty repetition")
    arrays = [[np.asarray(cells[run][site], dtype=float) for site in sorted(cells[run])] for run in runs]
    width = arrays[0][0].shape[1]
    if any(a.ndim != 2 or not len(a) or a.shape[1] != width or not np.isfinite(a).all() for sites in arrays for a in sites):
        raise ValueError("Finite nonempty sample-by-dimension arrays required")
    # A run with four assigned observations must not outweigh a run with two.
    # Average available run means WITHIN each site, then average the sites.
    sites = sorted(set().union(*(set(cells[r]) for r in runs)))
    mean = np.mean([np.mean([np.asarray(cells[r][s]).mean(0) for r in runs if s in cells[r]], axis=0)
                    for s in sites], axis=0)
    rng = np.random.default_rng(seed)
    # Shared repetition resampling, then equal-site resampling, then independent
    # observations in every available site/run. Multinomial weights represent
    # exact with-replacement draws without allocating giant RF row tensors.
    coverage = np.array([[s in cells[r] for s in sites] for r in runs], dtype=int)
    estimates = np.empty((bootstrap, width))
    for start in range(0, bootstrap, 100):
        count = min(100, bootstrap-start)
        draws = np.zeros((count, width))
        run_weights = rng.multinomial(len(runs), np.full(len(runs), 1/len(runs)), size=count)
        # Sparse cohorts are allowed. Never synthesize unavailable cells: reject
        # a repetition draw that omits every observed run of any site. The
        # current cohort covers all 100 sites in all five runs (no rejection).
        for attempt in range(10000):
            denominator = run_weights @ coverage
            invalid = np.any(denominator == 0, axis=1)
            if not invalid.any():
                break
            run_weights[invalid] = rng.multinomial(len(runs), np.full(len(runs), 1/len(runs)), size=int(invalid.sum()))
        else:
            raise ValueError("Sparse site/run coverage prevents repetition bootstrap")
        site_weights = (rng.multinomial(len(sites), np.full(len(sites), 1/len(sites)), size=count)
                        if resample_sites else np.ones((count, len(sites)), dtype=int))
        for sindex, site in enumerate(sites):
            numerator = np.zeros((count, width))
            for rindex, run in enumerate(runs):
                if site not in cells[run]:
                    continue
                values = np.asarray(cells[run][site], dtype=float)
                repeats = run_weights[:, rindex] * site_weights[:, sindex]
                weights = rng.multinomial(len(values)*repeats, np.full(len(values), 1/len(values)))
                numerator += weights @ values / len(values)
            draws += numerator / denominator[:, sindex, None] / len(sites)
        estimates[start:start+count] = draws
    low, high = np.quantile(estimates, [.025, .975], axis=0)
    # Degenerate constant/delta intervals can differ from the point estimate
    # by a few float64 ulps after differently ordered sums. Correct rounding
    # only; genuinely non-enclosing percentile intervals remain detectable.
    tolerance = 64*np.finfo(float).eps*np.maximum(1, np.abs(mean))
    low = np.where((low > mean) & (low-mean <= tolerance), mean, low)
    high = np.where((high < mean) & (mean-high <= tolerance), mean, high)
    return {"mean": mean, "ci_low": low, "ci_high": high, "n_runs": len(runs),
            "n_sites": len(sites), "n_samples": sum(len(a) for group in arrays for a in group),
            "bootstrap": bootstrap, "bootstrap_seed": seed, "ci_method": "cross_run_hierarchical_percentile_95",
            "sparse_run_policy": "condition_repetition_draws_on_site_coverage; average_available_run_means_per_site"}


def repetition_ci(values, *, bootstrap=10000, seed=20261008):
    return hierarchical_ci({i: {"performance": np.asarray([[v]])} for i, v in enumerate(values)},
                           bootstrap=bootstrap, seed=seed, resample_sites=False)


def paired_delta(current, baseline):
    """Strict ID pairing. Never drop unmatched records or claim partial pairing."""
    site_key = "site_id" if "site_id" in current and "site_id" in baseline else "site"
    keys = ["experiment_id", site_key, "sample_id", "model", "cohort", "output", "dimension"]
    if current.duplicated(keys).any() or baseline.duplicated(keys).any():
        raise ValueError("Duplicate pairing keys")
    left, right = current.set_index(keys), baseline.set_index(keys)
    if set(left.index) != set(right.index):
        raise ValueError("Comparison is not fully pairable: sample/site/run/dimension IDs differ")
    out = current.copy().set_index(keys)
    out["phi"] = left["phi"].abs() - right["phi"].abs().reindex(left.index)
    out["pairing"] = "exact_run_site_sample_output_dimension"
    return out.reset_index()
