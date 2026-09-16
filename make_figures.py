#!/usr/bin/env python
"""The figures and tables of the paper, from a finished campaign.

    python make_figures.py                 # everything
    python make_figures.py fig4 table4     # only these

Each figure is written as one file per panel, plus the legend as a file of its
own, at the exact width of its slot in the LaTeX source: nothing is rescaled on
inclusion, so the font sizes on the page are the ones declared here.

    fig4   open world: FOLD against its two branches
    fig5   open world: FOLD against the convolutional baseline
    fig6   ROC of the monitored/background decision
    fig7   where the fingerprinting signal is, and how many queries a site makes
    fig8   two heads against a single softmax over N+1 classes
    fig9   under each countermeasure: FOLD against its two branches
    fig10  under each countermeasure: FOLD against the convolutional baseline
    table4 closed-world performance on undefended traffic
    table5 location generalization

Results are read from ``results/`` by default; set WF_RESULTS to point
elsewhere.  Figures are written to ``figures/``, or to WF_FIGURES.
"""
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fold.metrics import closed_world  # noqa: E402
from fold.predictions import Predictions  # noqa: E402

RES = Path(os.environ.get("WF_RESULTS", "results"))
FIG = Path(os.environ.get("WF_FIGURES", "figures"))
DATASET = Path(os.environ.get("WF_DATASET", Path(__file__).resolve().parent / "data" / "odoh_traces.csv"))

#: Background ratios of the open-world experiments.
RATIOS = (0.30, 0.40, 0.50, 0.60, 0.70)

#: Probabilities each countermeasure is applied with.
PROBS = (0.10, 0.30, 0.50, 0.70, 1.00)

#: The countermeasures, as (file name, directory name).
MECHANISMS = (("padding", "padding"), ("dummy", "dummy"),
              ("fragmentation", "fragmentation"), ("combined", "combined"))

#: Display name -> the file its predictions live in.
FILES = {"FOLD": "fold", "TR": "transformer", "RF": "forest", "CNN": "cnn"}

# ---- palette (Okabe-Ito, colourblind-safe; identity is also carried by the
# marker and the legend, never by hue alone) --------------------------------
C = {"FOLD": "#0072B2", "RF": "#009E73", "TR": "#CC79A7", "CNN": "#D55E00"}
MK = {"FOLD": "D", "RF": "s", "TR": "^", "CNN": "o"}
# The two arms of the head ablation: "two heads" is FOLD, so it keeps its
# colour and marker; the N+1 softmax takes the one hue no model uses.
C["two"], C["n1"] = C["FOLD"], "#E69F00"
MK["two"], MK["n1"] = MK["FOLD"], "o"

# Point sizes are real points on the page, since every figure is generated at
# the width of its slot. They stay under the body text (8.9pt): a figure should
# not shout louder than the prose.
mpl.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "lines.linewidth": 1.4, "lines.markersize": 3.4,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#DDDDDD", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "figure.dpi": 200, "savefig.pad_inches": 0.02,
})

# ---- geometry, taken from the LaTeX source --------------------------------
TEXT_WIDTH, COLUMN_WIDTH = 7.16, 3.5     # inches, measured on the built PDF

PANEL_OW = 0.32 * TEXT_WIDTH             # a panel of a three-across block
PANEL_HALF = 0.48 * COLUMN_WIDTH         # half a column, for two-panel blocks
PANEL_FIRST = 0.268 * TEXT_WIDTH         # four across: the first carries the y label
PANEL_NEXT = 0.241 * TEXT_WIDTH
PANEL_ROW = 0.31 * TEXT_WIDTH            # the row of independent figures

H_OW = 1.62
H_HALF = 1.35                            # at half a column, this keeps the box
                                         # from turning square
H_DEFENSE = 1.45
H_ROW = 1.45
H_LEGEND = 0.24

LEG_OW = 0.18 * TEXT_WIDTH
LEG_DEFENSE_BRANCHES = 0.33 * TEXT_WIDTH
LEG_DEFENSE_CNN = 0.53 * TEXT_WIDTH

# Inner margins in inches. Shared by every open-world block, whose frames only
# line up on the page if they match.
MARGINS_OW = dict(left=0.55, bottom=0.389, right=0.034, top=0.049)
MARGINS_HALF = dict(left=0.44, bottom=0.34, right=0.06, top=0.05)
MARGINS_ROW = dict(left=0.46, bottom=0.34, right=0.06, top=0.05)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def mean_ci(x):
    """Mean and 95% Student-t interval over the repetitions.

    Student rather than a bootstrap percentile: with ten values the percentile
    interval is anticonservative, and Student's is symmetric by construction,
    so the "mean +/- half-width" of the tables is exact rather than an
    approximation.  No outlier removal: dropping repetitions to tighten error
    bars is harder to defend than any choice of interval.
    """
    from scipy import stats
    x = np.asarray(x, float)
    if len(x) < 2:
        return (float(x.mean()) if len(x) else float("nan"),) * 3
    h = float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return float(x.mean()), float(x.mean()) - h, float(x.mean()) + h


def margins(fig, width, height, left, bottom, right=0.04, top=0.05):
    """Margins in inches rather than fractions, so the drawing box has the same
    area in panels of different width (the first is wider because it carries
    the y label)."""
    fig.subplots_adjust(left=left / width, right=1 - right / width,
                        bottom=bottom / height, top=1 - top / height)


def boxed(ax):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(True)


def save(fig, name, tight=True):
    """Write the figure as PDF and PNG; ``name`` may contain subdirectories.

    ``bbox_inches="tight"`` trims the margins, so the final width is no longer
    the one asked for.  Panels destined for the LaTeX source pass tight=False
    and control their margins explicitly, which keeps the scale at 1.
    """
    (FIG / name).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", **({"bbox_inches": "tight"} if tight else {}))
    plt.close(fig)
    print("wrote", FIG / f"{name}.pdf")


def save_legend(handles, name, width, ncol=None):
    """The legend as a file of its own, as the multi-panel figures expect."""
    fig = plt.figure(figsize=(width, H_LEGEND))
    fig.legend(handles=handles, loc="center", ncol=ncol or len(handles),
               frameon=False, fontsize=7, columnspacing=1.3,
               handlelength=1.8, handletextpad=0.5, borderpad=0)
    save(fig, name)      # trimmed: include it without a width, or it rescales


def legend_order(models):
    """FOLD first, everywhere.  Drawing order is untouched."""
    return sorted(models, key=lambda m: m != "FOLD")


# ---------------------------------------------------------------------------
# Reading a finished campaign
# ---------------------------------------------------------------------------

def clean_metrics(model, key="accuracy"):
    """One value per repetition on undefended traffic, from the static sweep.

    This is the ceiling the open-world and defended curves are read against.
    """
    out = []
    for d in sorted(glob.glob(str(RES / "static_sweep" / "preds" / "exp_*"))):
        y = np.load(f"{d}/y_test.npy")
        p = np.load(f"{d}/clean.npz")[FILES[model]]
        out.append(closed_world(p, y)[key])
    return 100 * np.array(out, float)


def defended_accuracy(model, mechanism, p):
    """One accuracy per repetition under a countermeasure at probability p."""
    values = []
    for f in sorted(glob.glob(str(RES / f"defense_{mechanism}_p{p:.2f}" / "exp_*" / "result.json"))):
        r = json.load(open(f)).get(FILES[model])
        if r is not None:
            values.append(r["accuracy"])
    return 100 * np.array(values, float)


def openworld_cells(model, r):
    """Every repetition of one open-world cell: its predictions and its labels."""
    out = []
    for d in sorted(glob.glob(str(RES / f"openworld_r{r:.2f}" / "exp_*"))):
        f = Path(d) / f"{FILES[model]}.npz"
        if not f.exists():
            continue
        out.append((Predictions.load(f), np.load(f"{d}/y_test.npy")))
    return out


def openworld_end_to_end(model, r):
    """Share of monitored traces the whole chain gets right: accepted by the
    monitored decision *and* attributed to the right website.

    Skipping the gate would let a model with a high miss rate look good, since
    a rejected monitored trace would not count as an error.  The denominator
    stays the number of monitored traces, so the metric does not move with the
    composition of the test set.
    """
    out = []
    for pred, y in openworld_cells(model, r):
        mon, _, score, _ = pred.views()
        out.append(((score > 0.5) & (np.argmax(mon, 1) == y)).mean())
    return 100 * np.array(out, float)


def openworld_macro(model, r, key):
    """The paper's macro One-vs-Rest metrics, one value per repetition."""
    from fold.metrics import open_world
    out = []
    for pred, y in openworld_cells(model, r):
        mon, back, mon_score, back_score = pred.views()
        out.append(open_world(mon, back, y, mon_score, back_score,
                              include_background=False)[key])
    return 100 * np.array(out, float)


def detection_scores(directory, model):
    """(monitored scores, background scores) for every repetition."""
    out = []
    for d in sorted(glob.glob(str(RES / directory / "exp_*"))):
        f = Path(d) / f"{FILES[model]}.npz"
        if not f.exists():
            continue
        _, _, mon_score, back_score = Predictions.load(f).views()
        out.append((mon_score, back_score))
    return out


def pooled_roc(per_seed):
    """ROC over the scores of every repetition pooled together.

    Pooling costs about 1e-4 of AUC against averaging the per-repetition AUCs,
    and costs slightly more to the better models, so the curve is conservative.
    """
    from sklearn.metrics import roc_curve
    mon = np.concatenate([a for a, _ in per_seed])
    back = np.concatenate([b for _, b in per_seed])
    y = np.r_[np.ones(len(mon)), np.zeros(len(back))]
    fpr, tpr, _ = roc_curve(y, np.r_[mon, back])
    return 100 * fpr, 100 * tpr


# ---------------------------------------------------------------------------
# Figs. 4 and 5 — the open world
# ---------------------------------------------------------------------------

def _openworld_figure(models, directory, ylim, per_model_reference, legend_style,
                      legend_width=LEG_OW):
    """Two panels against the background ratio: accuracy, then macro-F1.

    The dashed line is the undefended closed-world level, the ceiling from
    which the cost of the open world is read.  Both panels share one fixed axis
    rather than autoscaling: with two different autoscales one could not say
    whether the F1 sits above or below the accuracy without reading the ticks.
    """
    x = [int(r * 100) for r in RATIOS]
    panels = [("accuracy", "accuracy (%)", None), ("f1", "macro-F1 (%)", "f1")]

    for name, ylabel, key in panels:
        fig, ax = plt.subplots(figsize=(PANEL_HALF, H_HALF))
        for m in (models if per_model_reference else [models[0]]):
            ax.axhline(mean_ci(clean_metrics(m, key or "accuracy"))[0],
                       ls="--", color=C[m], lw=1.1, zorder=1)
        for m in models:
            ys, lo, hi = [], [], []
            for r in RATIOS:
                v = (openworld_end_to_end(m, r) if key is None
                     else openworld_macro(m, r, key))
                a, l, h = mean_ci(v)
                ys.append(a); lo.append(l); hi.append(h)
            ax.fill_between(x, lo, hi, color=C[m], alpha=0.15, lw=0)
            ax.plot(x, ys, color=C[m], marker=MK[m], label=m)
        ax.set_xticks(x)
        ax.set_xlabel("background ratio (%)")
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        step = 5 if 100 - ylim[0] >= 10 else 2
        ax.set_yticks(range(int(ylim[0]), 101, step))
        boxed(ax)
        margins(fig, PANEL_HALF, H_HALF, **MARGINS_HALF)
        save(fig, f"{directory}/{name}", tight=False)

    handles = [Line2D([0], [0], color=C[m], marker=MK[m], ms=4, lw=1.6, label=m)
               for m in legend_order(models)]
    if legend_style == "per_model":
        handles += [Line2D([0], [0], color=C[m], ls="--", lw=1.1,
                           label=f"{m}, no background") for m in models]
        ncol = len(models)
    else:
        handles.append(Line2D([0], [0], color=C[models[0]], ls="--", lw=1.1,
                              label=f"{models[0]}, no background"))
        ncol = None
    save_legend(handles, f"{directory}/legend", legend_width, ncol=ncol)


def fig4():
    """Fig. 4: FOLD and its two branches as the background grows.

    A common 94-100 axis on both panels. Not 90: the drawing area is 0.912 inch
    tall, and at 90 the one significant gap here -- about half a point of F1
    between FOLD and TR -- would shrink to 1.2 mm, less than the gap on the
    accuracy panel that is *not* significant and would invite the comparison.
    """
    _openworld_figure(["FOLD", "TR", "RF"], "fig4_open_world", (94, 100),
                      per_model_reference=False, legend_style="single")


def fig5():
    """Fig. 5: FOLD against the convolutional baseline, as the background grows.

    One dashed line per model here, unlike Fig. 4: the two undefended levels
    are 8.9 points apart (98.67 against 89.75), and without the baseline's own
    reference the reader would measure its drop against FOLD's, which confuses
    "worse to begin with" with "degrades more".
    """
    _openworld_figure(["FOLD", "CNN"], "fig5_open_world_cnn", (80, 100),
                      per_model_reference=True, legend_style="per_model")


# ---------------------------------------------------------------------------
# Fig. 6 — the monitored/background decision
# ---------------------------------------------------------------------------
# This figure and Fig. 8 measure the binary decision, not the One-vs-Rest macro
# of the tables: the latter averages over 101 classes of which 100 are
# monitored, so it describes the ranking of websites and not membership. The
# two false-positive rates differ by two orders of magnitude.

def _roc_axes(ax):
    ax.set_xscale("log")
    ax.set_xlim(0.05, 100)
    ax.set_ylim(88, 100.4)
    # Explicit ticks: the automatic y ones would read "97.5"/"100.0", five
    # characters that do not fit the margin.
    ax.set_xticks([0.1, 1, 10, 100])
    ax.set_yticks(range(90, 101, 2))
    ax.set_xlabel("FPR (%)")
    ax.set_ylabel("TPR (%)")
    ax.grid(True, which="both")
    # The last x label sits on the frame; right-aligning extends it inward,
    # where there is room.
    ax.get_figure().canvas.draw()
    ax.get_xticklabels()[-1].set_horizontalalignment("right")
    boxed(ax)


def fig6():
    """Fig. 6: one ROC panel per background ratio, three models each.

    Three ratios out of five for space: the two ends plus the middle, a regular
    subsample of the sweep.  All five stay in the tables.
    """
    models = ["RF", "TR", "FOLD"]
    for r in (RATIOS[0], RATIOS[len(RATIOS) // 2], RATIOS[-1]):
        fig, ax = plt.subplots(figsize=(PANEL_OW, H_OW))
        for m in models:
            per_seed = detection_scores(f"openworld_r{r:.2f}", m)
            if not per_seed:
                continue
            x, y = pooled_roc(per_seed)
            ax.plot(x, y, color=C[m], lw=1.4, zorder=3)
        _roc_axes(ax)
        margins(fig, PANEL_OW, H_OW, **MARGINS_OW)
        save(fig, f"fig6_detection_roc/panel_bg{int(r * 100)}", tight=False)

    handles = [Line2D([0], [0], color=C[m], lw=1.6, label=m) for m in legend_order(models)]
    save_legend(handles, "fig6_detection_roc/legend", LEG_DEFENSE_BRANCHES)


# ---------------------------------------------------------------------------
# Fig. 7 — where the signal is, and what the traces look like
# ---------------------------------------------------------------------------

def feature_groups(n_features):
    """The forest's inputs, grouped by the quantity they measure.

    One level of taxonomy: *what* a feature measures, not how it is
    represented.  The size histogram is one way of representing sizes, so it
    joins "Sizes" rather than standing beside them.

    Layout of the 33 aggregate statistics (fold.data.aggregate_representation):
        0-4    global   : n_packets, total_size, time_span, mean_size, std_size
        5-28   per role : count, mean_size, std_size, sum_size, mean_ts, std_ts
        29-32  deltas   : mean_dt, std_dt, mean|dsize|, std|dsize|
    followed by the 64 bins of the packet-size histogram.
    """
    sizes = [1, 3, 4]                      # total_size, mean_size, std_size
    counts = [0]                           # n_packets
    times = [2]                            # time_span
    for role in range(4):
        base = 5 + 6 * role
        counts.append(base)                # count
        sizes += [base + 1, base + 2, base + 3]
        times += [base + 4, base + 5]
    times += [29, 30]
    sizes += [31, 32]
    sizes += list(range(33, n_features))   # the size histogram
    return {"Sizes": np.array(sizes), "Counts": np.array(counts),
            "Timings": np.array(times)}


def forest_importances(n_seeds=10):
    """Gini importance per group, one row per seed.

    This measures the order-independent branch, not FOLD as a whole: the
    aggregate statistics reach the fusion only through the forest's
    probabilities.  Gini splits the credit between correlated features
    according to the split a tree chose, which is how three separate bars can
    arise from inputs none of which is individually indispensable.
    """
    from sklearn.ensemble import RandomForestClassifier
    from fold.data import Encoder, load_traces

    traces, labels, _ = load_traces(DATASET)
    # The same features the attack's forest reads: the aggregate statistics and
    # the packet-size histogram.
    encoder = Encoder.fit(traces, max(labels) + 1, max(len(t) for t in traces))
    X, y = encoder.forest(traces), np.array(labels)
    groups = feature_groups(X.shape[1])

    rows = []
    for seed in range(n_seeds):
        rf = RandomForestClassifier(n_estimators=500, criterion="gini",
                                    max_features="sqrt", n_jobs=-1, random_state=seed)
        rf.fit(X, y)
        imp = rf.feature_importances_ / rf.feature_importances_.sum()
        rows.append([100 * imp[idx].sum() for idx in groups.values()])
    return np.array(rows), list(groups)


def queries_per_site():
    """Mean number of ODoH queries a website's visit produces."""
    import pandas as pd
    df = pd.read_csv(DATASET, usecols=["trace_id", "site"])
    per_trace = df.groupby(["site", "trace_id"]).size().rename("q").reset_index()
    return per_trace.groupby("site")["q"].mean().values


def fig7():
    """Fig. 7: the contribution of each group of features, and the distribution
    of the number of queries per website."""
    F = 6.5      # one body size for the whole row: labels and ticks alike

    values, labels = forest_importances()
    fig, ax = plt.subplots(figsize=(PANEL_ROW, H_ROW))
    m = values.mean(0)
    lo = np.array([m[i] - mean_ci(values[:, i])[1] for i in range(values.shape[1])])
    hi = np.array([mean_ci(values[:, i])[2] - m[i] for i in range(values.shape[1])])
    ax.bar(np.arange(len(m)), m, 0.6, yerr=[lo, hi], capsize=2, edgecolor="black",
           linewidth=0.6, facecolor="#7EA6E0", error_kw=dict(lw=0.8, ecolor="#333"),
           zorder=3)
    ax.set_xticks(np.arange(len(m)))
    ax.set_xticklabels(labels, fontsize=F)
    ax.set_ylabel("importance (%)", fontsize=F)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 25))
    ax.tick_params(labelsize=F)
    boxed(ax)
    margins(fig, PANEL_ROW, H_ROW, **MARGINS_ROW)
    save(fig, "fig7_signal/importance", tight=False)

    q = np.sort(queries_per_site())
    fig, ax = plt.subplots(figsize=(PANEL_ROW, H_ROW))
    ax.plot(q, 100 * np.arange(1, len(q) + 1) / len(q), lw=1.4, color="#0072B2")
    # "CDF" belongs on the axis: the y value is the share of websites with at
    # most that many queries, and without saying so it reads as a density.
    ax.set_xlabel("ODoH queries", fontsize=F)
    ax.set_ylabel("Website CDF (%)", fontsize=F)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_xlim(0, 80)
    ax.set_xticks(range(0, 81, 20))
    ax.tick_params(labelsize=F)
    boxed(ax)
    margins(fig, PANEL_ROW, H_ROW, **MARGINS_ROW)
    save(fig, "fig7_signal/query_distribution", tight=False)


# ---------------------------------------------------------------------------
# Fig. 8 — two heads against one softmax over N+1 classes
# ---------------------------------------------------------------------------

ARMS = (("two", "two heads"), ("n1", "N+1 softmax"))


def _heads_false_positives(arm, r):
    """Detection false-positive rate at the working threshold, per repetition."""
    out = []
    for d in sorted(glob.glob(str(RES / f"heads_{arm}_r{r:.2f}" / "exp_*"))):
        f = Path(d) / "fold.npz"
        if not f.exists():
            continue
        score = Predictions.load(f).test_background_score
        out.append(100 * (np.asarray(score).ravel() > 0.5).mean())
    return np.array(out, float)


def fig8():
    """Fig. 8: a dedicated monitored head against a single softmax over N+1.

    Two panels, both on the membership decision: the false-positive rate at the
    working threshold used throughout the paper, and the same decision at every
    threshold -- the two formulations calibrate differently, so a comparison at
    one fixed threshold answers only half the question.

    No accuracy panel.  It would need a definition different from Fig. 4's:
    restricted to monitored traces it would favour the more permissive gate,
    and over all traces it would not be comparable with Fig. 4.  The comparison
    stands on the membership decision, which is the only thing the two
    formulations differ on.
    """
    x = [int(r * 100) for r in RATIOS]

    fig, ax = plt.subplots(figsize=(PANEL_HALF, H_HALF))
    for arm, _ in ARMS:
        ys, lo, hi = [], [], []
        for r in RATIOS:
            m, l, h = mean_ci(_heads_false_positives(arm, r))
            ys.append(m); lo.append(l); hi.append(h)
        ax.fill_between(x, lo, hi, color=C[arm], alpha=0.15, lw=0)
        ax.plot(x, ys, color=C[arm], marker=MK[arm], zorder=3)
    ax.set_xticks(x)
    ax.set_xlabel("background ratio (%)")
    ax.set_ylabel("FPR (%)")
    boxed(ax)
    margins(fig, PANEL_HALF, H_HALF, left=0.44, bottom=0.34, right=0.10, top=0.05)
    save(fig, "fig8_heads/fpr", tight=False)

    r_roc = RATIOS[len(RATIOS) // 2]
    fig, ax = plt.subplots(figsize=(PANEL_HALF, H_HALF))
    for arm, _ in ARMS:
        per_seed = detection_scores(f"heads_{arm}_r{r_roc:.2f}", "FOLD")
        if not per_seed:
            continue
        x_roc, y_roc = pooled_roc(per_seed)
        ax.plot(x_roc, y_roc, color=C[arm], lw=1.4, zorder=3)
    _roc_axes(ax)
    margins(fig, PANEL_HALF, H_HALF, left=0.44, bottom=0.34, right=0.13, top=0.05)
    save(fig, "fig8_heads/roc", tight=False)

    handles = [Line2D([0], [0], color=C[a], marker=MK[a], ms=4, lw=1.6, label=label)
               for a, label in ARMS]
    save_legend(handles, "fig8_heads/legend", LEG_DEFENSE_CNN)


# ---------------------------------------------------------------------------
# Figs. 9 and 10 — under the countermeasures
# ---------------------------------------------------------------------------

def _defense_panel(mechanism, models, baselines, with_ylabel, width, ylim, dashed):
    x = [int(p * 100) for p in PROBS]
    fig, ax = plt.subplots(figsize=(width, H_DEFENSE))
    for m in models:
        ys, lo, hi = [], [], []
        for p in PROBS:
            a, l, h = mean_ci(defended_accuracy(m, mechanism, p))
            ys.append(a); lo.append(l); hi.append(h)
        ax.fill_between(x, lo, hi, color=C[m], alpha=0.18, lw=0)
        ax.plot(x, ys, color=C[m], marker=MK[m], zorder=3)
    for m in dashed:
        ax.axhline(baselines[m][0], ls="--", color=C[m], lw=1.1, zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in x])
    ax.set_xlabel("$p$ (%)")
    ax.set_ylim(*ylim)
    ax.set_yticks(range(ylim[0], 101, 20 if ylim[0] == 0 else 10))
    if with_ylabel:
        ax.set_ylabel("accuracy (%)")
    boxed(ax)
    margins(fig, width, H_DEFENSE, left=0.49 if with_ylabel else 0.30, bottom=0.36)
    return fig


def _defense_figure(models, directory, legend_width, handles, ylim, dashed):
    baselines = {m: mean_ci(clean_metrics(m)) for m in models}
    for k, (name, mechanism) in enumerate(MECHANISMS):
        fig = _defense_panel(mechanism, models, baselines, with_ylabel=(k == 0),
                             width=PANEL_FIRST if k == 0 else PANEL_NEXT,
                             ylim=ylim, dashed=dashed)
        save(fig, f"{directory}/panel_{name}", tight=False)
    save_legend(handles, f"{directory}/legend", legend_width)


def fig9():
    """Fig. 9: FOLD against its two branches, under each countermeasure.

    One dashed reference only: the three undefended levels lie within 0.4
    points of each other and three lines would merge into a band.  The y axis
    starts at 40, below the lowest point drawn, so the 3 to 13 points that
    separate the fusion from its branches where the defense bites are read at
    1.67x.
    """
    models = ["RF", "TR", "FOLD"]
    handles = [Line2D([0], [0], color=C[m], marker=MK[m], ms=4, lw=1.6, label=m)
               for m in legend_order(models)]
    handles.append(Line2D([0], [0], color=C["FOLD"], ls="--", lw=1.1,
                          label="no defense FOLD"))
    _defense_figure(models, "fig9_defense_branches", LEG_DEFENSE_BRANCHES,
                    handles, ylim=(40, 100), dashed=("FOLD",))


def fig10():
    """Fig. 10: FOLD against the convolutional baseline, under each countermeasure.

    A dashed line per model: the two undefended levels are 8.9 points apart,
    and the baseline is already at 56.45 at p=10%, so its starting point cannot
    be read off its own curve.  The axis runs to 0 because it falls to 8%.
    """
    models = ["FOLD", "CNN"]
    handles = [Line2D([0], [0], color=C[m], marker=MK[m], ms=4, lw=1.6, label=m)
               for m in models]
    handles += [Line2D([0], [0], color=C[m], ls="--", lw=1.1, label=f"no defense {m}")
                for m in models]
    _defense_figure(models, "fig10_defense_cnn", LEG_DEFENSE_CNN, handles,
                    ylim=(0, 100), dashed=tuple(models))


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _escape(t) -> str:
    return str(t).replace("%", r"\%").replace("&", r"\&")


def _write_table(rows, headers, caption, label, name, alignment):
    """Write the table as LaTeX and as Markdown, side by side."""
    tex = [r"\begin{table}[t]\centering",
           r"\caption{%s}" % caption,
           r"\label{%s}" % label,
           r"{\footnotesize",
           r"\begin{tabular}{%s}" % alignment,
           r"\hline \hline",
           " " + " & ".join(r"\textbf{%s}" % _escape(h) for h in headers) + r"\\",
           r"\hline"]
    for row in rows:
        tex.append(" & ".join(_escape(c) for c in row) + r"\\")
    tex += [r"\hline \hline", r"\end{tabular}", "}", r"\end{table}"]

    md = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    md += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]

    FIG.mkdir(parents=True, exist_ok=True)
    (FIG / f"{name}.tex").write_text("\n".join(tex) + "\n")
    (FIG / f"{name}.md").write_text(f"**{caption}**\n\n" + "\n".join(md) + "\n")
    print("wrote", FIG / f"{name}.tex", "and .md")


def table4():
    """Table IV: closed-world performance on undefended traffic.

    Mean and 95% interval over the repetitions, from the undefended cell of the
    static sweep.
    """
    rows = []
    for m in ("FOLD", "TR", "RF", "CNN"):
        acc = mean_ci(clean_metrics(m, "accuracy"))
        f1 = mean_ci(clean_metrics(m, "f1"))
        rows.append((m, f"{acc[0]:.2f} $\\pm$ {acc[0] - acc[1]:.2f}",
                     f"{f1[0]:.2f} $\\pm$ {f1[0] - f1[1]:.2f}"))
    _write_table(rows, ["Model", "Accuracy (%)", "Macro-F1 (%)"],
                 r"Closed-world attack performance on undefended traffic over the "
                 r"$N = 100$ monitored websites.",
                 "tab:closed-world", "table4_closed_world", "l cc")


def table5():
    """Table V: leave one location out against a random split.

    The metrics come from the per-trace predictions, with the same macro
    One-vs-Rest definitions as everywhere else, rather than from the aggregate
    CSV, which carries only accuracy and macro-F1.
    """
    files = sorted(glob.glob(str(RES / "location" / "preds" / "exp_*" / "*.npz")))
    rows = []
    for m in ("FOLD", "TR", "RF", "CNN"):
        cells = []
        for held_out in (True, False):
            values = []
            for f in files:
                if (Path(f).stem != "random") != held_out:
                    continue
                z = np.load(f)
                values.append(closed_world(z[FILES[m]], z["y_test"]))
            cells += [f"{np.mean([v[k] for v in values]):.4f}"
                      for k in ("accuracy", "fnr", "fpr")]
        rows.append((m, *cells))
    _write_table(rows, ["Algorithm", "LOO Accuracy", "LOO FNR", "LOO FPR",
                        "Random Accuracy", "Random FNR", "Random FPR"],
                 r"Location generalization. \emph{Leave-one-out}: the model is trained "
                 r"on all client locations but one and tested on the held-out location, "
                 r"averaged over the held-out locations. \emph{Random split}: a random "
                 r"train/test split over the same pool, so that training and test "
                 r"contain traces from every location. FNR/FPR are macro One-vs-Rest "
                 r"over the $N$ classes.",
                 "tab:location", "table5_location", "l ccc ccc")


FIGURES = {"fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8,
           "fig9": fig9, "fig10": fig10, "table4": table4, "table5": table5}


if __name__ == "__main__":
    wanted = sys.argv[1:] or list(FIGURES)
    for k in wanted:
        if k not in FIGURES:
            raise SystemExit(f"unknown figure {k!r}; choose from {list(FIGURES)}")
        FIGURES[k]()
