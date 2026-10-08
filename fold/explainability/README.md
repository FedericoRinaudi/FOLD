# FOLD explainability

Train RF and FOLD inference bundles, explain their predictions, and reproduce
the 26-configuration TSR campaign. Run all commands from the repository root;
the shell examples use Bash (including the loops).

## What is here

| Path | Purpose |
| --- | --- |
| `fold/explainability/` | Bundle export, RF TreeSHAP, grouped FOLD Shapley and plotting |
| `analysis/` | Campaign summaries and aggregate CSV/PNG outputs |
| `selections/` | Recorded explanation samples and common binary references |
| `data/odoh_traces.csv` | Included monitored traffic dataset |
| `make_background.py` | Generate synthetic background traffic |
| `run_experiment.py` | Original experiment runner, including evaluation |

## Getting started

Use Python 3.12. `requirements.txt` pins TensorFlow, scikit-learn, NumPy,
pandas, SciPy, matplotlib and SDV. SHAP and joblib are additional explainability
dependencies; their versions are not pinned in that file.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install shap joblib
```

CPU execution is supported; a GPU is optional and useful for training.

The included `data/target_websites.csv` lists the 100 monitored websites.
Training obtains the label-to-site mapping from `data/odoh_traces.csv` itself.
Its temporal fields are **intervals between packets**, not absolute timestamps.

## The TSR campaign

| Scenario | Configurations |
| --- | --- |
| Baseline | 1: no background, no defense |
| Open-world | Background 30/40/50/60/70% |
| Padding | 10/30/50/70/100%, no background |
| Dummy | 10/30/50/70/100%, no background |
| Fragmentation | 10/30/50/70/100%, no background |
| Combined | All three defenses at 10/30/50/70/100%, no background |

There are **26 configurations**, each with experiment ID 1 and one repetition.
Intensity sets padding probability per processed record/fragment, dummy insertion probability after an original record, and recursive fragmentation probability within implementation constraints; combined enables all three mechanisms at the same intensity.
Background and defense are never combined. The original `run_campaign.sh`
runs a larger campaign and does not replace the procedure below.

The experiment ID and configuration determine the splits and seeds used by the code.

## Preparing the background pool

Generate the pool once, before open-world training, and reuse that exact file:

```bash
python make_background.py \
  --dataset data/odoh_traces.csv --out data/background_pool.csv \
  --size 40000 --n-fit 1800 --seed 10 \
  --vocab-ratio 0.95 --proxy-vocab-ratio 0.35 --seq-repeat 0.55
```

The command uses the code defaults and requires SDV from `requirements.txt`.
The pool is generated locally and is not versioned. Reuse the same pool
throughout a campaign, with bundles and selections that correspond to it.

## Running one experiment

The export runner trains the final RF, its five OOF forests, and FOLD, then
saves the inference bundle. Evaluation is a separate command.
**Export appends the scenario to `--out-dir`; evaluation does not.** Both
append `exp_001`. Keep the dataset, ID and scenario flags identical.

```bash
# Baseline
python -m fold.explainability.train --model fold --experiment-id 1 --dataset data/odoh_traces.csv --out-dir artifacts
python run_experiment.py --model evaluate --experiment-id 1 --dataset data/odoh_traces.csv --out-dir artifacts/baseline

# Open-world, 30% background
python -m fold.explainability.train --model fold --experiment-id 1 --dataset data/odoh_traces.csv --out-dir artifacts --open-world --background-ratio 0.30 --background-pool data/background_pool.csv
python run_experiment.py --model evaluate --experiment-id 1 --dataset data/odoh_traces.csv --out-dir artifacts/openworld_r0.30 --open-world --background-ratio 0.30 --background-pool data/background_pool.csv

# Padding, 30%
python -m fold.explainability.train --model fold --experiment-id 1 --dataset data/odoh_traces.csv --out-dir artifacts --defense-padding 0.30
python run_experiment.py --model evaluate --experiment-id 1 --dataset data/odoh_traces.csv --out-dir artifacts/defense_padding_p0.30 --defense-padding 0.30

# Combined, 50% for each mechanism
python -m fold.explainability.train --model fold --experiment-id 1 --dataset data/odoh_traces.csv --out-dir artifacts --defense-padding 0.50 --defense-dummy 0.50 --defense-fragmentation 0.50
python run_experiment.py --model evaluate --experiment-id 1 --dataset data/odoh_traces.csv --out-dir artifacts/defense_combined_p0.50 --defense-padding 0.50 --defense-dummy 0.50 --defense-fragmentation 0.50
```

Choose these individual examples or the full campaign below. Training creates
a new directory for each configuration.

## Running the full TSR campaign

After preparing the pool, this Bash block trains and evaluates all 26 cells,
once each, and stops on the first error. It requires no preceding single-run
example.

```bash
(
  set -euo pipefail
  run_cell() {
    local scenario="$1"
    shift
    python -m fold.explainability.train --model fold --experiment-id 1 \
      --dataset data/odoh_traces.csv --out-dir artifacts "$@"
    python run_experiment.py --model evaluate --experiment-id 1 \
      --dataset data/odoh_traces.csv --out-dir "artifacts/$scenario" "$@"
  }

  run_cell baseline
  for r in 0.30 0.40 0.50 0.60 0.70; do
    run_cell "openworld_r$r" --open-world --background-ratio "$r" \
      --background-pool data/background_pool.csv
  done
  for family in padding dummy fragmentation combined; do
    for p in 0.10 0.30 0.50 0.70 1.00; do
      if [ "$family" = combined ]; then
        run_cell "defense_${family}_p$p" \
          --defense-padding "$p" --defense-dummy "$p" --defense-fragmentation "$p"
      else
        run_cell "defense_${family}_p$p" "--defense-$family" "$p"
      fi
    done
  done
)
```

## What a run leaves behind

After training and evaluation:

```text
artifacts/<scenario>/exp_001/
    forest.npz          RF predictions
    fold.npz            FOLD predictions
    y_test.npy          evaluation labels
    result.json         evaluation metrics
    background.pkl      sampled background and IDs (open-world only)
    bundle/
        encoder.npz
        forest.joblib
        fold.weights.h5
        manifest.json
```

Training generates predictions and the inference bundle; evaluation writes
`result.json` and `y_test.npy`. The bundle serves inference and XAI. These
model artifacts are not versioned; baseline explanation NPZ files and
aggregate CSV results are included.

## Explainability workflows

Run these after exporting the corresponding bundles. Explanation commands
write NPZ files and figures; `--skip-plots` saves only the NPZ. Use a new
output directory when regenerating explanations.

### Random Forest

TreeSHAP explains the final RF's aggregate features for one monitored class.

```bash
python -m fold.explainability explain \
  --bundle artifacts/baseline/exp_001/bundle --trusted \
  --dataset data/odoh_traces.csv --target-class 0 \
  --background-size 32 --max-samples 10 --seed 0 --allow-multiclass \
  --out-dir explanations/reproduction/baseline/rf
python -m fold.explainability plot \
  --attributions explanations/reproduction/baseline/rf/rf_class_000_shap.npz \
  --out-dir explanations/reproduction/baseline/rf/figures
```

### FOLD multiclass

Grouped exact Shapley explains `multi_output` by replacing trace groups and
recomputing the full encoder/RF/FOLD pipeline. The baseline uses target 8
(`instagram.com`), 15 selected traces, 4 groups, 16 references and seed 0.

View the baseline result already included in the clone, without a bundle:

```bash
python -m fold.explainability plot-fold \
  --attributions explanations/baseline/fold_instagram/fold_class_008_shap.npz \
  --out-dir explanations/reproduction/included_baseline_figures --sample-index 0
```

Generate a new explanation in a different directory:

```bash
python -m fold.explainability explain-fold \
  --bundle artifacts/baseline/exp_001/bundle --trusted \
  --dataset data/odoh_traces.csv --target-class 8 \
  --sample-ids-file selections/baseline_fold_instagram.json \
  --num-groups 4 --reference-size 16 --seed 0 --batch-size 16 \
  --out-dir explanations/reproduction/baseline/fold_instagram
python -m fold.explainability plot-fold \
  --attributions explanations/reproduction/baseline/fold_instagram/fold_class_008_shap.npz \
  --out-dir explanations/reproduction/baseline/fold_instagram/figures --sample-index 0
```

### Defense FOLD

Use the same baseline cohort and parameters for all 20 defended bundles.

```bash
(
  set -euo pipefail
  for family in padding dummy fragmentation combined; do
    for p in 0.10 0.30 0.50 0.70 1.00; do
      scenario="defense_${family}_p$p"
      python -m fold.explainability explain-fold \
        --bundle "artifacts/$scenario/exp_001/bundle" --trusted \
        --dataset data/odoh_traces.csv --target-class 8 \
        --sample-ids-file selections/baseline_fold_instagram.json \
        --num-groups 4 --reference-size 16 --seed 0 --batch-size 16 \
        --out-dir "explanations/$scenario/exp_001/fold_instagram"
    done
  done
)
```

Each output contains `fold_class_008_shap.npz` and four figures. Use
`plot-fold` with the corresponding NPZ to redraw them.

### Open-world multiclass and binary

The multiclass workflow explains the Instagram probability on the monitored
cohort; the binary workflow explains `binary_output = P(monitored)` on both
populations, with 8 monitored and 8 background references shared across ratios.

```bash
(
  set -euo pipefail
  for r in 0.30 0.40 0.50 0.60 0.70; do
    python -m fold.explainability explain-fold \
      --bundle "artifacts/openworld_r$r/exp_001/bundle" --trusted \
      --dataset data/odoh_traces.csv --target-class 8 \
      --sample-ids-file selections/baseline_fold_instagram.json \
      --num-groups 4 --reference-size 16 --seed 0 --batch-size 16 \
      --out-dir "explanations/openworld_r$r/exp_001/fold_instagram"
    python -m fold.explainability explain-open-world \
      --bundle "artifacts/openworld_r$r/exp_001/bundle" --trusted \
      --dataset data/odoh_traces.csv --background-pool data/background_pool.csv \
      --sample-ids-file "selections/openworld_r${r}_binary.json" \
      --reference-ids-file selections/openworld_common_binary_references.json \
      --num-groups 4 --reference-size 16 --seed 0 --batch-size 16 \
      --out-dir "explanations/openworld_r$r/exp_001/binary"
  done
)
```

The multiclass NPZ is `fold_instagram/fold_class_008_shap.npz`; the binary NPZ
is `binary/binary_monitored_shap.npz` and contains both populations. Each
workflow writes four figures. Redraw binary figures, for example, with:

```bash
python -m fold.explainability plot-open-world \
  --attributions explanations/openworld_r0.30/exp_001/binary/binary_monitored_shap.npz \
  --out-dir explanations/openworld_r0.30/exp_001/binary --sample-index 0
```

Use `plot-fold` for the multiclass NPZ. In binary results, monitored rows precede
background rows; `--sample-index` refers to this combined order.

## Selections

| File under `selections/` | Purpose |
| --- | --- |
| `baseline_fold_instagram.json` | 15 monitored evaluation IDs, target class 8; reused for multiclass comparisons |
| `openworld_r0.30_binary.json` | 15 monitored + 15 background evaluation IDs at 30% |
| `openworld_r0.40_binary.json` | Same population counts at 40% |
| `openworld_r0.50_binary.json` | Same population counts at 50% |
| `openworld_r0.60_binary.json` | Same population counts at 60% |
| `openworld_r0.70_binary.json` | Same population counts at 70% |
| `openworld_common_binary_references.json` | 8 monitored + 8 background training IDs, common across ratios |

Selections are tied to the corresponding split/background pool and may need
to be regenerated for a new campaign. There is no selection-generation CLI.

## Aggregating results

Both scripts use fixed repository-relative paths and write to `analysis/`,
replacing aggregate CSV/PNG outputs when regenerated.

### Open-world

With all five `result.json` files and all ten multiclass/binary NPZ files at
the paths above, rebuild summaries and figures:

```bash
python analysis/make_openworld_summary.py
```

With incomplete runtime inputs, the script redraws the existing CSV summaries.
To explicitly plot those summaries without training or rewriting the CSVs:

```bash
python analysis/make_openworld_summary.py --from-csv
```

| Output | Files under `analysis/` |
| --- | --- |
| CSV | `openworld_metrics.csv`, `openworld_multi_shap.csv`, `openworld_binary_shap.csv` |
| XAI figures | `openworld_multi_mean_abs_phi.png`, `openworld_binary_monitored_mean_abs_phi.png`, `openworld_binary_background_mean_abs_phi.png` |
| Performance figures | `openworld_accuracy.png`, `openworld_macro_f1.png`, `openworld_detection_tpr.png`, `openworld_detection_fpr.png`, `openworld_detection_auc.png` |

### Defense

`analysis/make_defense_summary.py` reads baseline plus all 20 defense metrics
and explanation NPZ files at fixed repository-relative paths. For a new
campaign, place its inputs at the paths expected by the script, including
the new baseline explanation at
`explanations/baseline/fold_instagram/fold_class_008_shap.npz`.

```bash
python analysis/make_defense_summary.py
```

| Output | Files under `analysis/` |
| --- | --- |
| CSV | `defense_metrics.csv`, `defense_shap.csv` |
| Performance figures | `defense_fold_accuracy.png`, `defense_fold_macro_f1.png`, `defense_forest_accuracy.png`, `defense_forest_macro_f1.png` |
| Cohort probability | `defense_instagram_mean_probability.png` |
| XAI figures | `defense_padding_mean_abs_phi.png`, `defense_dummy_mean_abs_phi.png`, `defense_fragmentation_mean_abs_phi.png`, `defense_combined_mean_abs_phi.png` |

### Metric definitions

| Metric | Population and decision |
| --- | --- |
| Accuracy | Correct website predictions; in open world, includes correct background rejections over the full evaluation set |
| Macro-F1 | Mean F1 over monitored website classes; background contributes negatives but is excluded from the class average |
| Detection TPR | Fraction of monitored evaluation traces with monitored score > 0.5 |
| Detection FPR | Fraction of background evaluation traces incorrectly accepted with score > 0.5 |
| Detection ROC AUC | Monitored positives versus background negatives over all score thresholds |

Detection measures monitored/background membership, separately from website
classification accuracy and macro-F1.

## Reproducibility and trust

The bundle manifest records model configuration, runtime versions, provenance
and artifact hashes. Loaders check artifact integrity and version compatibility;
use the same environment for training and inference. `--allow-version-mismatch`
accepts version warnings without guaranteeing equivalent predictions.

Use `--trusted` only for bundles you trust: joblib and pickle loading can execute
code. This also applies to the generated background pickle.

## Five-repetition all-sites campaign

The campaign runs 26 configurations in five repetitions (experiment IDs 1–5):
closed world, five open-world ratios, and four defense families at five intensities.
Original seeds, splits, preprocessing and training settings are preserved.

The immutable selection is
`analysis/cross_run/selections/all_sites_cross_run.json`: 1,500 held-out
sample-run observations, 15 per site, and 1,499 distinct traces. For 99 sites
all 15 traces are distinct; Spotify has 14 distinct traces and one trace assigned
to two repetitions in which it is TEST. Each observation is identified by
`(sample_id, experiment_id)`; train/validation observations are excluded.

Multiclass explanations target each observation's true site dynamically.
FOLD uses exact G1–G4 Shapley values; RF uses probability TreeSHAP over 97 features.
Open-world binary explanations target P(monitored) with separate monitored and
background cohorts. Raw attributions retain run IDs and explicit trace reuse.

Run these commands from the repository root:

```bash
python -m fold.explainability.campaign.preflight --prepare-selection
python -m fold.explainability.campaign --dry-run --stage all
# On the training server, with TensorFlow and the generated background pool:
python -m fold.explainability.campaign --stage all --trusted
python -m fold.explainability.campaign.summarize --bootstrap 10000 --seed 20261008
python -m fold.explainability.campaign.figures
python -m fold.explainability.campaign.sensitivity
```

The preflight validates TEST membership and cross-run availability; the dry-run
prints 130 training/evaluation cells without training or inference. Selection
content is immutable. Verified bundles and explanations can be resumed;
inconsistent provenance or incomplete outputs are rejected.

Outputs are `artifacts/` for bundles and evaluation,
`explanations/all_sites/` for raw XAI, `analysis/cross_run/preflight/` for
regenerable diagnostics, and `analysis/cross_run/summary/` for aggregate CSVs
and figures. Generated outputs are ignored; the selection JSON is committed.

Attributions average observations within site/run, then run means within site,
then site means with equal weights. The 95% hierarchical percentile bootstrap
resamples repetitions, sites and observations (10,000 draws, seed 20261008).
Paired comparisons match run/site/sample keys. Dominant groups and heatmaps use
the same site means; sensitivity removes only the marked Spotify reuse.
Final attribution results require server inference.
