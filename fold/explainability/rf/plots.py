"""Figures from saved forest attributions for a fixed class probability."""
from pathlib import Path

import numpy as np

from .results import ForestExplanation


def plot_forest(result: ForestExplanation, out_dir, *, sample_index=0,
                 feature=None, max_display=15) -> list[Path]:
    """Save waterfall, bar, beeswarm, dependence, heatmap and group bar as PNG.

    The global bar uses mean absolute SHAP. The group bar sums signed feature
    values per sample, then averages positive and negative parts separately.
    The other plots retain signed probability contributions. ``feature`` is a
    column name or index; by default it is the largest mean-absolute attribution.
    """
    result.check_additivity()
    if not 0 <= sample_index < len(result.sample_ids) or max_display < 1:
        raise ValueError("Invalid sample_index or max_display")
    if feature is None:
        column = int(np.argmax(np.mean(np.abs(result.values), axis=0)))
    elif isinstance(feature, str):
        if feature not in result.feature_names:
            raise ValueError(f"Unknown feature: {feature}")
        column = result.feature_names.index(feature)
    elif isinstance(feature, (int, np.integer)) and 0 <= feature < len(result.feature_names):
        column = int(feature)
    else:
        raise ValueError("feature must be a column name or a valid numeric index")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap
    # Reuse the paper's taxonomy only during plotting: its module sets mpl style.
    from make_figures import feature_groups

    explanation = shap.Explanation(
        values=result.values,
        base_values=np.full(len(result.sample_ids), result.base_value),
        data=result.features, feature_names=list(result.feature_names))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"rf_class_{result.target_class:03d}"
    title = f"RF P(class {result.target_class})"
    paths = []

    def save(name, draw, subtitle):
        figure = plt.figure()
        try:
            draw()
            plotted = plt.gcf()
            plotted.suptitle(f"{title} — {subtitle}")
            path = out_dir / f"{prefix}_{name}.png"
            plotted.savefig(path, dpi=200, bbox_inches="tight")
            paths.append(path)
        finally:
            plt.close(plt.gcf())
            plt.close(figure)

    save(f"waterfall_sample_{sample_index:04d}",
         lambda: shap.plots.waterfall(explanation[sample_index],
                                      max_display=max_display, show=False),
         f"sample {sample_index}: signed contributions")
    save("bar", lambda: shap.plots.bar(explanation.abs.mean(0),
                                       max_display=max_display, show=False),
         "mean absolute SHAP")
    save("beeswarm", lambda: shap.plots.beeswarm(explanation,
                                                 max_display=max_display, show=False),
         "signed contributions")
    save(f"dependence_feature_{column:03d}",
         lambda: shap.plots.scatter(explanation[:, column], show=False),
         result.feature_names[column])
    save("heatmap", lambda: shap.plots.heatmap(
        explanation, instance_order=np.arange(len(result.sample_ids)),
        max_display=max_display, show=False), "signed contributions in saved sample order")

    def group_bar():
        groups = feature_groups(len(result.feature_names))
        indices = np.concatenate(list(groups.values()))
        if not np.array_equal(np.sort(indices), np.arange(len(result.feature_names))):
            raise ValueError("Semantic groups must partition the saved feature columns")
        values = np.column_stack([result.values[:, idx].sum(axis=1)
                                  for idx in groups.values()])
        positions = np.arange(len(groups))
        ax = plt.gca()
        ax.barh(positions, np.maximum(values, 0).mean(axis=0), color="#D55E00",
                label="Positive part")
        ax.barh(positions, np.minimum(values, 0).mean(axis=0), color="#0072B2",
                label="Negative part")
        ax.set_yticks(positions, ["Time values" if g == "Timings" else g for g in groups])
        ax.axvline(0, color="black", linewidth=0.7)
        ax.set_xlabel("Mean signed group contribution to RF probability")
        ax.legend()

    save("groups", group_bar, "positive and negative parts of group sums")
    return paths
