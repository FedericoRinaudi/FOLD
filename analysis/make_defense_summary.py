from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis"

FAMILIES = ["padding", "dummy", "fragmentation", "combined"]
LEVELS = [
    ("0.10", 0.10),
    ("0.30", 0.30),
    ("0.50", 0.50),
    ("0.70", 0.70),
    ("1.00", 1.00),
]


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_shap(path):
    with np.load(path, allow_pickle=False) as z:
        values = z["values"]
        base_values = z["base_values"]
        probabilities = z["probabilities"]

        reconstructed = base_values + values.sum(axis=1)
        additivity_error = float(
            np.max(np.abs(reconstructed - probabilities))
        )

        row = {
            "n_samples": values.shape[0],
            "n_groups": values.shape[1],
            "mean_probability": float(np.mean(probabilities)),
            "mean_base_value": float(np.mean(base_values)),
            "max_additivity_error": additivity_error,
        }

        for i in range(values.shape[1]):
            row[f"mean_phi_g{i+1}"] = float(
                np.mean(values[:, i])
            )
            row[f"mean_abs_phi_g{i+1}"] = float(
                np.mean(np.abs(values[:, i]))
            )

        return row


# ============================================================
# Performance metrics
# ============================================================

metric_rows = []

baseline = load_json(
    ROOT / "artifacts/baseline/exp_001/result.json"
)

metric_rows.append({
    "scenario": "baseline",
    "family": "baseline",
    "intensity": 0.0,
    "forest_accuracy": baseline["forest"]["accuracy"],
    "forest_macro_f1": baseline["forest"]["f1"],
    "forest_fnr": baseline["forest"]["fnr"],
    "forest_fpr": baseline["forest"]["fpr"],
    "fold_accuracy": baseline["fold"]["accuracy"],
    "fold_macro_f1": baseline["fold"]["f1"],
    "fold_fnr": baseline["fold"]["fnr"],
    "fold_fpr": baseline["fold"]["fpr"],
})

for family in FAMILIES:
    for token, intensity in LEVELS:
        scenario = f"defense_{family}_p{token}"

        result = load_json(
            ROOT
            / "artifacts"
            / scenario
            / "exp_001"
            / "result.json"
        )

        metric_rows.append({
            "scenario": scenario,
            "family": family,
            "intensity": intensity,
            "forest_accuracy": result["forest"]["accuracy"],
            "forest_macro_f1": result["forest"]["f1"],
            "forest_fnr": result["forest"]["fnr"],
            "forest_fpr": result["forest"]["fpr"],
            "fold_accuracy": result["fold"]["accuracy"],
            "fold_macro_f1": result["fold"]["f1"],
            "fold_fnr": result["fold"]["fnr"],
            "fold_fpr": result["fold"]["fpr"],
        })

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(OUT / "defense_metrics.csv", index=False)


# ============================================================
# SHAP summaries
# ============================================================

shap_rows = []

baseline_shap_path = (
    ROOT
    / "explanations"
    / "baseline"
    / "fold_instagram"
    / "fold_class_008_shap.npz"
)

shap_rows.append({
    "scenario": "baseline",
    "family": "baseline",
    "intensity": 0.0,
    **load_shap(baseline_shap_path),
})

for family in FAMILIES:
    for token, intensity in LEVELS:
        scenario = f"defense_{family}_p{token}"

        shap_path = (
            ROOT
            / "explanations"
            / scenario
            / "exp_001"
            / "fold_instagram"
            / "fold_class_008_shap.npz"
        )

        shap_rows.append({
            "scenario": scenario,
            "family": family,
            "intensity": intensity,
            **load_shap(shap_path),
        })

shap = pd.DataFrame(shap_rows)
shap.to_csv(OUT / "defense_shap.csv", index=False)


# ============================================================
# Helpers
# ============================================================

def get_series(df, family, column):
    baseline_row = df[df["family"] == "baseline"].iloc[0]
    subset = (
        df[df["family"] == family]
        .sort_values("intensity")
    )

    x = np.concatenate((
        [0.0],
        subset["intensity"].to_numpy()
    ))

    y = np.concatenate((
        [baseline_row[column]],
        subset[column].to_numpy()
    ))

    return x, y


def plot_metric(column, ylabel, filename):
    plt.figure(figsize=(8, 5))

    for family in FAMILIES:
        x, y = get_series(metrics, family, column)
        plt.plot(x, y, marker="o", label=family)

    plt.xlabel("Defense intensity")
    plt.ylabel(ylabel)
    plt.xticks([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / filename, dpi=180)
    plt.close()


# ============================================================
# Performance figures
# ============================================================

plot_metric(
    "fold_accuracy",
    "FOLD accuracy",
    "defense_fold_accuracy.png",
)

plot_metric(
    "fold_macro_f1",
    "FOLD macro-F1",
    "defense_fold_macro_f1.png",
)

plot_metric(
    "forest_accuracy",
    "Random Forest accuracy",
    "defense_forest_accuracy.png",
)

plot_metric(
    "forest_macro_f1",
    "Random Forest macro-F1",
    "defense_forest_macro_f1.png",
)


# ============================================================
# Instagram probability
# ============================================================

plt.figure(figsize=(8, 5))

for family in FAMILIES:
    x, y = get_series(
        shap,
        family,
        "mean_probability",
    )
    plt.plot(x, y, marker="o", label=family)

plt.xlabel("Defense intensity")
plt.ylabel("Mean P(Instagram) on common cohort")
plt.xticks([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(
    OUT / "defense_instagram_mean_probability.png",
    dpi=180,
)
plt.close()


# ============================================================
# Grouped Shapley figures
# ============================================================

for family in FAMILIES:
    plt.figure(figsize=(8, 5))

    for group in range(1, 5):
        column = f"mean_abs_phi_g{group}"

        x, y = get_series(
            shap,
            family,
            column,
        )

        plt.plot(
            x,
            y,
            marker="o",
            label=f"G{group}",
        )

    plt.xlabel("Defense intensity")
    plt.ylabel("Mean |Shapley value|")
    plt.xticks([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
    plt.title(f"{family}: Instagram grouped Shapley")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUT / f"defense_{family}_mean_abs_phi.png",
        dpi=180,
    )
    plt.close()


print("Generated files:")
for path in sorted(OUT.glob("defense_*")):
    print(" ", path.relative_to(ROOT))

print()
print("===== FOLD PERFORMANCE =====")
print(
    metrics[
        [
            "family",
            "intensity",
            "fold_accuracy",
            "fold_macro_f1",
        ]
    ].to_string(index=False)
)

print()
print("===== INSTAGRAM SHAP =====")
print(
    shap[
        [
            "family",
            "intensity",
            "mean_probability",
            "mean_abs_phi_g1",
            "mean_abs_phi_g2",
            "mean_abs_phi_g3",
            "mean_abs_phi_g4",
        ]
    ].to_string(index=False)
)
