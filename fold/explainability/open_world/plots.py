"""Binary labels and population-aware figures from saved attributions only.

The multiclass plotter hardcodes class filenames and lacks population labels;
render these four views locally without changing its API or stored results.
"""
from pathlib import Path

import numpy as np


def plot_binary(result, out_dir, *, sample_index=0):
    result.__post_init__()
    if not 0 <= sample_index < len(result.sample_ids):
        raise ValueError("sample_index is outside the saved samples")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    def save(figure, name):
        try:
            figure.suptitle("FOLD binary_output: P(monitored)")
            figure.tight_layout()
            path = out_dir / f"binary_monitored_{name}.png"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            paths.append(path)
        finally:
            plt.close(figure)

    row = sample_index
    count = int(result.group_counts[row])
    values = result.values[row, :count]
    colors = np.where(values >= 0, "#D55E00", "#0072B2")
    label = f"{result.sample_types[row]}: {result.sample_ids[row]}"
    figure, ax = plt.subplots()
    cumulative = result.base_values[row] + np.r_[0, np.cumsum(values)]
    ax.bar(0, result.base_values[row], color="gray")
    ax.bar(np.arange(count) + 1, values, bottom=cumulative[:-1], color=colors)
    ax.bar(count + 1, result.probabilities[row], color="gray")
    ax.set_xticks(np.arange(count + 2), ["Base", *[f"G{i+1}" for i in range(count)], "P(monitored)"])
    ax.set_ylabel("Probability / signed group contribution")
    ax.set_title(label, fontsize=8)
    save(figure, f"waterfall_sample_{row:04d}")

    active = np.arange(result.values.shape[1])[None, :] < result.group_counts[:, None]
    columns = np.flatnonzero(active.sum(axis=0))
    figure, ax = plt.subplots()
    ax.bar(columns + 1, np.abs(result.values).sum(axis=0)[columns] / active.sum(axis=0)[columns])
    ax.set_xticks(columns + 1, [f"G{i+1}" for i in columns])
    ax.set_ylabel("Mean |Shapley| over samples with this group")
    ax.set_title(
        f"{result.sample_types.count('monitored')} monitored / "
        f"{result.sample_types.count('background')} background samples", fontsize=9)
    save(figure, "bar")

    figure, ax = plt.subplots(figsize=(7, max(3, 0.25 * len(result.sample_ids))))
    limit = max(float(np.max(np.abs(result.values))), 1e-12)
    heatmap = ax.imshow(np.ma.array(result.values, mask=~active), aspect="auto",
                        cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks(np.arange(result.values.shape[1]), [f"G{i+1}" for i in range(result.values.shape[1])])
    ax.set_yticks(np.arange(len(result.sample_ids)),
                  [f"{i}: {p}" for i, p in enumerate(result.sample_types)])
    ax.set_ylabel("Saved sample index and population")
    figure.colorbar(heatmap, ax=ax, label="Signed contribution to P(monitored)")
    save(figure, "heatmap")

    figure, ax = plt.subplots()
    spans = result.unit_bounds[row, :count]
    ax.bar(spans[:, 0], values, width=spans[:, 1] - spans[:, 0], align="edge",
           color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlim(0, spans[-1, 1])
    ax.set_xlabel("Resolution index (zero-based, half-open groups)")
    ax.set_ylabel("Group contribution to P(monitored)")
    ax.set_title(label, fontsize=8)
    save(figure, f"trace_sample_{row:04d}")
    return paths
