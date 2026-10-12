"""Equal-site means and reproducible percentile 95% bootstrap confidence intervals.

XAI intervals are conditional on the observed trained repetitions.  Sites are
resampled for all-sites summaries and observations are resampled independently
inside every available site/repetition cell.  Each site's available repetition
means are averaged with equal weight, then sites are averaged with equal weight.

Instagram omits the site-resampling level.  Background binary has one
population-level site.

Performance intervals are different: they resample the five trained
repetitions themselves.

Paired deltas bootstrap paired differences, not two independent marginal
intervals.  No normalized attribution shares are used.
"""
import numpy as np


def hierarchical_ci(cells, *, bootstrap=10000, seed=20261008, resample_sites=True):
    """cells[run][site]: finite (samples, dimensions) arrays; equal site weight."""
    if bootstrap < 100 or not cells:
        raise ValueError("Use at least 100 bootstrap draws and nonempty data")

    runs = sorted(cells)
    if any(not cells[run] for run in runs):
        raise ValueError("Empty repetition")

    arrays = [
        [
            np.asarray(cells[run][site], dtype=float)
            for site in sorted(cells[run])
        ]
        for run in runs
    ]
    width = arrays[0][0].shape[1]

    if any(
        a.ndim != 2
        or not len(a)
        or a.shape[1] != width
        or not np.isfinite(a).all()
        for sites in arrays
        for a in sites
    ):
        raise ValueError("Finite nonempty sample-by-dimension arrays required")

    sites = sorted(set().union(*(set(cells[r]) for r in runs)))

    # Point estimate:
    # observations -> equal-weight available repetitions within site
    #              -> equal-weight sites.
    mean = np.mean(
        [
            np.mean(
                [
                    np.asarray(cells[r][site], dtype=float).mean(0)
                    for r in runs
                    if site in cells[r]
                ],
                axis=0,
            )
            for site in sites
        ],
        axis=0,
    )

    rng = np.random.default_rng(seed)
    estimates = np.empty((bootstrap, width))

    # The trained repetitions are intentionally NOT resampled here.
    # They are the fixed five fitted models on which the XAI analysis is
    # conditioned.  Sparse site/run cells remain absent rather than being
    # synthesized or forcing repetition draws.
    for start in range(0, bootstrap, 100):
        count = min(100, bootstrap - start)
        draws = np.zeros((count, width))

        if resample_sites:
            site_weights = rng.multinomial(
                len(sites),
                np.full(len(sites), 1 / len(sites)),
                size=count,
            )
        else:
            site_weights = np.ones((count, len(sites)), dtype=int)

        for sindex, site in enumerate(sites):
            available_runs = [run for run in runs if site in cells[run]]
            if not available_runs:
                raise ValueError(f"Site without observed repetition: {site}")

            # site_weights[:, sindex] is the number of bootstrap copies of
            # this site in every draw.  Drawing n*k observations is
            # distributionally equivalent to summing k independent
            # bootstrap cell means, while avoiding explicit duplicate-site
            # tensors.
            repeats = site_weights[:, sindex]
            site_sum = np.zeros((count, width))

            for run in available_runs:
                values = np.asarray(cells[run][site], dtype=float)
                weights = rng.multinomial(
                    len(values) * repeats,
                    np.full(len(values), 1 / len(values)),
                )
                site_sum += weights @ values / len(values)

            # Every represented bootstrap copy of a site averages all of that
            # site's OBSERVED repetitions with equal weight.
            draws += site_sum / len(available_runs) / len(sites)

        estimates[start:start + count] = draws

    low, high = np.quantile(estimates, [0.025, 0.975], axis=0)

    # Only repair floating-point ulp inversions for degenerate intervals.
    tolerance = 64 * np.finfo(float).eps * np.maximum(1, np.abs(mean))
    low = np.where((low > mean) & (low - mean <= tolerance), mean, low)
    high = np.where((high < mean) & (mean - high <= tolerance), mean, high)

    return {
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "n_runs": len(runs),
        "n_sites": len(sites),
        "n_samples": sum(len(a) for group in arrays for a in group),
        "bootstrap": bootstrap,
        "bootstrap_seed": seed,
        "ci_method": "site_observation_stratified_percentile_95_fixed_repetitions",
        "sparse_run_policy": (
            "condition_on_observed_repetitions;"
            "average_available_run_means_per_site"
        ),
    }


def repetition_ci(values, *, bootstrap=10000, seed=20261008):
    """Percentile CI obtained by resampling trained repetitions."""
    values = np.asarray(values, dtype=float).reshape(-1)

    if bootstrap < 100 or not len(values) or not np.isfinite(values).all():
        raise ValueError(
            "Use at least 100 bootstrap draws and finite repetition values"
        )

    rng = np.random.default_rng(seed)
    n = len(values)

    weights = rng.multinomial(
        n,
        np.full(n, 1 / n),
        size=bootstrap,
    )
    estimates = weights @ values / n

    mean = np.asarray([values.mean()])
    low, high = np.quantile(estimates, [0.025, 0.975])
    low = np.asarray([low])
    high = np.asarray([high])

    tolerance = 64 * np.finfo(float).eps * np.maximum(1, np.abs(mean))
    low = np.where((low > mean) & (low - mean <= tolerance), mean, low)
    high = np.where((high < mean) & (mean - high <= tolerance), mean, high)

    return {
        "mean": mean,
        "ci_low": low,
        "ci_high": high,
        "n_runs": n,
        "n_sites": 1,
        "n_samples": n,
        "bootstrap": bootstrap,
        "bootstrap_seed": seed,
        "ci_method": "repetition_percentile_95",
    }


def paired_delta(current, baseline):
    """Strict ID pairing. Never drop unmatched records or claim partial pairing."""
    site_key = (
        "site_id"
        if "site_id" in current and "site_id" in baseline
        else "site"
    )
    keys = [
        "experiment_id",
        site_key,
        "sample_id",
        "model",
        "cohort",
        "output",
        "dimension",
    ]

    if current.duplicated(keys).any() or baseline.duplicated(keys).any():
        raise ValueError("Duplicate pairing keys")

    left, right = current.set_index(keys), baseline.set_index(keys)

    if set(left.index) != set(right.index):
        raise ValueError(
            "Comparison is not fully pairable: "
            "sample/site/run/dimension IDs differ"
        )

    out = current.copy().set_index(keys)
    out["phi"] = (
        left["phi"].abs()
        - right["phi"].abs().reindex(left.index)
    )
    out["pairing"] = "exact_run_site_sample_output_dimension"
    return out.reset_index()
