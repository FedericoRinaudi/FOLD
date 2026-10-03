"""Build open-world summaries and plot XAI and performance comparisons.

With complete runtime inputs, rebuild the three summary CSVs as before.
Otherwise, or with --from-csv, plot the existing CSVs without rewriting them.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RATIOS = ["0.30", "0.40", "0.50", "0.60", "0.70"]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"


def build_metrics():
    rows = []

    for r in RATIOS:
        path = ROOT / f"artifacts/openworld_r{r}/exp_001/result.json"

        with path.open() as f:
            result = json.load(f)

        for model in ("forest", "fold"):
            x = result[model]

            rows.append({
                "ratio": float(r),
                "model": model,
                "accuracy": x["accuracy"],
                "macro_f1": x["f1"],
                "fnr": x["fnr"],
                "fpr_multiclass": x["fpr"],
                "detection_tpr": x["detection"]["tpr"],
                "detection_fpr": x["detection"]["fpr"],
                "detection_auc": x["detection"]["auc"],
            })

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "openworld_metrics.csv", index=False)
    return frame


def build_multi():
    rows = []

    for r in RATIOS:
        path = ROOT / (
            f"explanations/openworld_r{r}/exp_001/"
            "fold_instagram/fold_class_008_shap.npz"
        )

        with np.load(path, allow_pickle=False) as z:
            values = np.asarray(z["values"], dtype=float)
            probs = np.asarray(z["probabilities"], dtype=float)
            base = np.asarray(z["base_values"], dtype=float)

        row = {
            "ratio": float(r),
            "n_samples": len(values),
            "mean_probability_class8": probs.mean(),
            "std_probability_class8": probs.std(),
            "mean_base_value": base.mean(),
        }

        for g in range(values.shape[1]):
            row[f"G{g+1}_mean_abs_phi"] = np.mean(
                np.abs(values[:, g])
            )
            row[f"G{g+1}_mean_signed_phi"] = np.mean(
                values[:, g]
            )

        rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "openworld_multi_shap.csv", index=False)
    return frame


def build_binary():
    rows = []

    for r in RATIOS:
        path = ROOT / (
            f"explanations/openworld_r{r}/exp_001/"
            "binary/binary_monitored_shap.npz"
        )

        with np.load(path, allow_pickle=False) as z:
            values = np.asarray(z["values"], dtype=float)
            probs = np.asarray(z["probabilities"], dtype=float)
            base = np.asarray(z["base_values"], dtype=float)
            types = np.asarray(z["sample_types"]).astype(str)

        for population in ("monitored", "background"):
            mask = types == population
            values_pop = values[mask]

            row = {
                "ratio": float(r),
                "population": population,
                "n_samples": int(mask.sum()),
                "mean_P_monitored": probs[mask].mean(),
                "std_P_monitored": probs[mask].std(),
                "mean_base_value": base[mask].mean(),
            }

            for g in range(values_pop.shape[1]):
                row[f"G{g+1}_mean_abs_phi"] = np.mean(
                    np.abs(values_pop[:, g])
                )
                row[f"G{g+1}_mean_signed_phi"] = np.mean(
                    values_pop[:, g]
                )

            rows.append(row)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "openworld_binary_shap.csv", index=False)
    return frame


def plot_groups(frame, title, filename):
    fig, ax = plt.subplots(figsize=(7.5, 4.8))

    for g in range(1, 5):
        ax.plot(
            frame["ratio"],
            frame[f"G{g}_mean_abs_phi"],
            marker="o",
            label=f"G{g}",
        )

    ax.set_xlabel("Background ratio")
    ax.set_ylabel("Mean |Shapley value|")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)


def validate_metrics(frame):
    """Require exactly the five observed configurations for each model."""
    columns = ["accuracy", "macro_f1", "detection_tpr", "detection_fpr",
               "detection_auc"]
    required = {"ratio", "model", *columns}
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing performance columns: {sorted(required - set(frame.columns))}")
    expected = {(float(r), model) for r in RATIOS for model in ("forest", "fold")}
    actual = set(frame[["ratio", "model"]].itertuples(index=False, name=None))
    if frame.duplicated(["ratio", "model"]).any() or actual != expected:
        raise ValueError("Expected one row per ratio (0.30 to 0.70) and model (forest, fold)")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Performance metrics must be finite values in [0, 1]")


def plot_performance(frame):
    """Plot CSV observations, without smoothing or cross-run error bars.

    run_evaluate uses include_background=False for macro website metrics,
    but accuracy includes all monitored/background test decisions. Detection
    takes monitored as positive (score > 0.5); AUC spans all ROC thresholds.
    See fold.metrics.open_world/detection and fold.predictions.Predictions.views.
    """
    validate_metrics(frame)
    plots = (
        ("accuracy", "Multiclass accuracy (all test traces)",
         "Open-world classification: website or background"),
        ("macro_f1", "Macro-F1 (monitored website classes)",
         "Open-world monitored website classification"),
        ("detection_tpr", "TPR (monitored correctly accepted)",
         "Monitored/background detection (threshold 0.5)"),
        ("detection_fpr", "FPR (background incorrectly accepted)",
         "Monitored/background detection (threshold 0.5)"),
        ("detection_auc", "ROC AUC (positive: monitored)",
         "Monitored/background detection (all thresholds)"),
    )
    for column, ylabel, title in plots:
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for model, label in (("forest", "Random Forest"), ("fold", "FOLD")):
            selected = frame.loc[frame["model"] == model].sort_values("ratio")
            ax.plot(selected["ratio"], selected[column], marker="o", label=label)
        ax.set_xticks([float(r) for r in RATIOS], RATIOS)
        ax.set_xlabel("Background ratio")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(OUT / f"openworld_{column}.png", dpi=180)
        plt.close(fig)


def load_summaries(from_csv=False):
    """Use one source mode for all summaries; never mix partial runtime inputs."""
    inputs = [
        path
        for r in RATIOS
        for path in (
            ROOT / f"artifacts/openworld_r{r}/exp_001/result.json",
            ROOT / f"explanations/openworld_r{r}/exp_001/fold_instagram/fold_class_008_shap.npz",
            ROOT / f"explanations/openworld_r{r}/exp_001/binary/binary_monitored_shap.npz",
        )
    ]
    if from_csv or not all(path.is_file() for path in inputs):
        reason = "--from-csv" if from_csv else "Runtime inputs are incomplete"
        print(f"{reason}: reading existing summary CSVs without rewriting them.")
        return tuple(pd.read_csv(OUT / filename) for filename in (
            "openworld_metrics.csv", "openworld_multi_shap.csv", "openworld_binary_shap.csv"
        ))
    return build_metrics(), build_multi(), build_binary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-csv", action="store_true",
                        help="plot saved summary CSVs without reading runtime inputs or rewriting CSVs")
    args = parser.parse_args()
    metrics, multi, binary = load_summaries(from_csv=args.from_csv)

    plot_performance(metrics)

    plot_groups(
        multi,
        "FOLD multi_output - Instagram",
        "openworld_multi_mean_abs_phi.png",
    )

    plot_groups(
        binary[binary["population"] == "monitored"],
        "FOLD binary_output - monitored traces",
        "openworld_binary_monitored_mean_abs_phi.png",
    )

    plot_groups(
        binary[binary["population"] == "background"],
        "FOLD binary_output - background traces",
        "openworld_binary_background_mean_abs_phi.png",
    )

    print("Available outputs (CSV summaries and generated figures):")
    for path in sorted(OUT.glob("openworld_*")):
        print(" ", path.relative_to(ROOT))

    print("\nFOLD metrics:")
    print(
        metrics[metrics["model"] == "fold"][
            [
                "ratio",
                "accuracy",
                "macro_f1",
                "detection_tpr",
                "detection_fpr",
                "detection_auc",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
