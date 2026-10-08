"""Figures from aggregated CSV only; deterministic site order, no SHAP calls."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bars(ax, data, title, *, top=None):
    if top:
        data = data.nlargest(top, "mean").sort_values("mean")
    else:
        data = data.sort_values("dimension")
    y = np.arange(len(data))
    errors = np.vstack([data["mean"]-data.ci_low, data.ci_high-data["mean"]])
    ax.barh(y, data["mean"], xerr=errors, capsize=3)
    ax.set_yticks(y, data.dimension)
    ax.set_title(title)
    ax.set_xlabel("Mean |attribution| (probability units), 95% CI")


def plot(input_dir, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    global_frame = pd.read_csv(input_dir / "global_summary.csv")
    for model in ("rf", "fold"):
        data = global_frame[(global_frame.configuration == "baseline") & (global_frame.model == model) &
                            (global_frame.output == "multi_output") & (global_frame.statistic == "abs")]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
        for ax, scope in zip(axes, ("instagram", "all_sites")):
            bars(ax, data[data.scope == scope], f"{model.upper()}: {scope.replace('_', ' ')}", top=15 if model == "rf" else None)
        # Each panel starts at zero; RF ranks features independently per scope.
        for ax in axes:
            ax.set_xlim(left=0)
        fig.savefig(out_dir / f"baseline_{model}.png", dpi=180)
        plt.close(fig)
    for model in ("fold",):
        families = ("padding", "dummy", "fragmentation", "combined", "openworld")
        for family in families:
            for statistic in ("abs", "delta_abs", "probability"):
                dimension_filter = "target_probability" if statistic == "probability" else None
                selected_statistic = "signed" if statistic == "probability" else statistic
                mask = global_frame.configuration.str.startswith("openworld" if family == "openworld" else f"defense_{family}_")
                data = global_frame[mask & (global_frame.model == model) & (global_frame.output == "multi_output") &
                                    (global_frame.statistic == selected_statistic)]
                if dimension_filter:
                    data = data[data.dimension == dimension_filter]
                if data.empty:
                    continue
                fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
                for ax, scope in zip(axes, ("instagram", "all_sites")):
                    frame = data[data.scope == scope].copy()
                    frame["x"] = frame.configuration.str.extract(r"(?:p|r)(\d+\.\d+)$")[0].astype(float)
                    for group, values in frame.groupby("dimension"):
                        values = values.sort_values("x")
                        ax.plot(values.x, values["mean"], marker="o", label=group)
                        ax.fill_between(values.x, values.ci_low, values.ci_high, alpha=.2)
                    if statistic == "delta_abs":
                        ax.axhline(0, color="black", linewidth=.7)
                    ax.set_title(f"{family}: {scope.replace('_', ' ')}")
                    ax.set_xlabel("Background ratio" if family == "openworld" else "Defense intensity")
                    ax.set_ylabel("Delta mean |phi| vs closed-world" if statistic == "delta_abs" else
                                  "Target probability" if statistic == "probability" else "Mean |phi|")
                    ax.legend()
                fig.savefig(out_dir / f"{family}_{statistic}.png", dpi=180)
                plt.close(fig)
    binary = global_frame[(global_frame.output == "binary_output") & (global_frame.dimension != "target_probability") &
                          (global_frame.scope == "all_sites")]
    for statistic in ("abs", "signed"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
        for ax, population in zip(axes, ("monitored", "background")):
            frame = binary[(binary.statistic == statistic) & (binary.cohort == population)].copy()
            frame["x"] = frame.configuration.str.extract(r"r(\d+\.\d+)$")[0].astype(float)
            for group, values in frame.groupby("dimension"):
                values = values.sort_values("x")
                ax.plot(values.x, values["mean"], marker="o", label=group)
                ax.fill_between(values.x, values.ci_low, values.ci_high, alpha=.2)
            ax.set_title(f"P(monitored): {population}")
            ax.set_xlabel("Background ratio")
            ax.set_ylabel("Mean signed phi" if statistic == "signed" else "Mean |phi|")
            ax.axhline(0, color="black", linewidth=.7)
            if not frame.empty:
                ax.legend()
        fig.savefig(out_dir / f"openworld_binary_{statistic}.png", dpi=180)
        plt.close(fig)
    heatmap = pd.read_csv(input_dir / "heatmap.csv")
    # Common scale across configurations makes the requested maps comparable.
    vmax = heatmap["mean"].max()
    for configuration, frame in heatmap.groupby("configuration"):
        matrix = frame.pivot(index="site", columns="dimension", values="mean").sort_index()
        fig, ax = plt.subplots(figsize=(7, 22), constrained_layout=True)
        picture = ax.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(4), matrix.columns)
        ax.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
        ax.set_title(f"{configuration}: mean over repetition/site means")
        fig.colorbar(picture, ax=ax, label="Mean |phi| (probability units)")
        fig.savefig(out_dir / f"heatmap_{configuration}.png", dpi=160)
        plt.close(fig)
    dominant = pd.read_csv(input_dir / "dominant_summary.csv")
    for configuration, frame in dominant.groupby("configuration"):
        frame = frame.sort_values("group")
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.bar(frame.group, frame["mean"]*100)
        ax.set_ylabel("% of sites (argmax after equal-run aggregation)")
        ax.set_title(configuration + ": dominant group")
        fig.savefig(out_dir / f"dominant_{configuration}.png", dpi=180)
        plt.close(fig)
    performance_path = input_dir / "performance_summary.csv"
    if performance_path.stat().st_size > 1:
        performance = pd.read_csv(performance_path)
        for (model, metric), frame in performance.groupby(["model", "metric"]):
            frame = frame.sort_values("configuration")
            fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
            ax.errorbar(range(len(frame)), frame["mean"], yerr=np.vstack([frame["mean"]-frame.ci_low, frame.ci_high-frame["mean"]]), fmt="o", capsize=3)
            ax.set_xticks(range(len(frame)), frame.configuration, rotation=90, fontsize=7)
            ax.set_title(f"{model}: {metric}, mean and 95% CI between repetitions")
            fig.savefig(out_dir / f"performance_{model}_{metric}.png", dpi=180)
            plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=Path("analysis/cross_run/summary"))
    p.add_argument("--out-dir", type=Path, default=Path("analysis/cross_run/summary/figures"))
    args = p.parse_args()
    plot(args.input, args.out_dir)


if __name__ == "__main__":
    main()
