"""Build compact open-world metrics, SHAP summaries and comparison plots."""

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


def main():
    metrics = build_metrics()
    multi = build_multi()
    binary = build_binary()

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

    print("Generated:")
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
