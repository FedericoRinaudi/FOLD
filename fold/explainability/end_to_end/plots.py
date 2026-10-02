"""Reconstruct grouped FOLD figures from saved numeric attributions only."""
from pathlib import Path

import numpy as np


def plot_fold(result, out_dir, *, sample_index=0):
    result.__post_init__()
    if not 0 <= sample_index < len(result.sample_ids):
        raise ValueError("sample_index is outside the saved samples")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"fold_class_{result.target_class:03d}"
    title = f"FOLD P({result.class_name})"
    row = sample_index
    count = int(result.group_counts[row])
    values = result.values[row, :count]
    colors = np.where(values >= 0, "#D55E00", "#0072B2")
    paths = []

    def save(figure, name):
        try:
            figure.suptitle(title)
            figure.tight_layout()
            path = out_dir / f"{prefix}_{name}.png"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            paths.append(path)
        finally:
            plt.close(figure)

    figure, ax = plt.subplots()
    cumulative = result.base_values[row] + np.r_[0, np.cumsum(values)]
    ax.bar(0, result.base_values[row], color="gray")
    ax.bar(np.arange(count) + 1, values, bottom=cumulative[:-1], color=colors)
    ax.bar(count + 1, result.probabilities[row], color="gray")
    ax.set_xticks(np.arange(count + 2), ["Base", *[f"G{i + 1}" for i in range(count)], "FOLD"])
    ax.set_ylabel("Target probability / signed contribution")
    ax.set_title(result.sample_ids[row], fontsize=8)
    save(figure, f"waterfall_sample_{row:04d}")

    active = np.arange(result.values.shape[1])[None, :] < result.group_counts[:, None]
    support = active.sum(axis=0)
    columns = np.flatnonzero(support)
    means = np.abs(result.values).sum(axis=0)[columns] / support[columns]
    figure, ax = plt.subplots()
    ax.bar(columns + 1, means)
    ax.set_xticks(columns + 1, [f"G{i + 1}\n(n={support[i]})" for i in columns])
    ax.set_ylabel("Mean |SHAP| over samples with this group")
    ax.set_xlabel("Ordinal group position")
    save(figure, "bar")

    figure, ax = plt.subplots()
    limit = max(float(np.max(np.abs(result.values))), 1e-12)
    heatmap = ax.imshow(np.ma.array(result.values, mask=~active), aspect="auto",
                        cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks(np.arange(result.values.shape[1]),
                  [f"G{i + 1}" for i in range(result.values.shape[1])])
    ax.set_ylabel("Sample index in saved order")
    figure.colorbar(heatmap, ax=ax, label="Signed probability contribution")
    save(figure, "heatmap")

    figure, ax = plt.subplots()
    spans = result.unit_bounds[row, :count]
    ax.bar(spans[:, 0], values, width=spans[:, 1] - spans[:, 0], align="edge",
           color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlim(0, spans[-1, 1])
    ax.set_xlabel("Resolution index (zero-based, half-open groups)")
    ax.set_ylabel("Group contribution to target probability")
    ax.set_title(result.sample_ids[row], fontsize=8)
    save(figure, f"trace_sample_{row:04d}")
    return paths
