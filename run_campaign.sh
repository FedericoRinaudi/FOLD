#!/usr/bin/env bash
# The campaign behind the paper: every experiment, ten times.
#
#   bash run_campaign.sh                 # run all of it here
#   bash run_campaign.sh --list          # print the jobs, run nothing
#   SHARD=2/5 bash run_campaign.sh       # run this machine's fifth of the jobs
#   REPS=1 bash run_campaign.sh          # a single repetition, to try it out
#
# A job is one cell of the campaign for one repetition: the models it needs,
# trained one process at a time so each training releases its GPU memory, then
# the evaluation that turns their predictions into result.json.  Each job
# writes to its own directory, so any subset can run on any machine and the
# results merge by copying directories together.
#
# Ten repetitions come to 470 jobs.  Shards are assigned round-robin, which
# spreads the slow cells (the sweep, the open world) evenly over the machines.
#
# Environment:
#   REPS     repetitions (default 10)
#   SHARD    i/N, run every N-th job starting at the i-th (default 1/1)
#   RESULTS  where results go (default results/)
#   PYTHON   interpreter (default: the active virtualenv, else python3)
set -euo pipefail
cd "$(dirname "$0")"

REPS="${REPS:-10}"
RESULTS="${RESULTS:-results}"
SHARD="${SHARD:-1/1}"
SHARD_ID="${SHARD%%/*}"
SHARD_N="${SHARD##*/}"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then PY="${VIRTUAL_ENV}/bin/python"
  else PY="$(command -v python3 || command -v python)"; fi
fi

# The probabilities every defense is swept over, and the background ratios of
# the open-world experiments.
PROBS=(0.10 0.30 0.50 0.70 1.00)
RATIOS=(0.30 0.40 0.50 0.60 0.70)

# The client locations of the dataset, spelled as the dataset spells them
# ("viriginia" included: the published data carries that typo, and correcting
# it here would simply fail to match). Texas and Colorado hold 172 and 98
# traces against about 290 for the others, so as holdouts they give a smaller
# test set and a noisier estimate.
LOCATIONS=(georgia missouri newyork south-carolina viriginia washington wisconsin texas colorado random)

JOBS=()
job() { JOBS+=("$1|$2|$3|$4"); }   # models | directory | repetition | flags

for e in $(seq 1 "$REPS"); do

  # -- Defenses, against an adversary that knows about them (Figs. 9 and 10).
  # Training, validation and test are all defended: the attack is retrained on
  # defended traffic, which is what measures the residual leakage rather than
  # the surprise.
  for mech in padding dummy fragmentation combined; do
    for p in "${PROBS[@]}"; do
      case "$mech" in
        padding)       flags="--defense-padding $p" ;;
        dummy)         flags="--defense-dummy $p" ;;
        fragmentation) flags="--defense-fragmentation $p" ;;
        combined)      flags="--defense-padding $p --defense-dummy $p --defense-fragmentation $p" ;;
      esac
      job "forest,transformer,fold,cnn,evaluate" "$RESULTS/defense_${mech}_p${p}" "$e" "$flags"
    done
  done

  # -- Open world (Figs. 4, 5 and 6).
  for r in "${RATIOS[@]}"; do
    job "forest,transformer,fold,cnn,evaluate" "$RESULTS/openworld_r${r}" "$e" \
        "--open-world --background-ratio $r"
  done

  # -- Two heads against one softmax over N+1 classes (Fig. 8).
  # The same five ratios as above, so the ablation lives on the same axis as
  # the main open-world results. The two-head arm repeats that configuration
  # on purpose: the comparison needs both arms on the same seed and the same
  # split, and here TR and the convolutional baseline are not needed.
  for r in "${RATIOS[@]}"; do
    job "forest,fold,evaluate" "$RESULTS/heads_two_r${r}" "$e" \
        "--open-world --background-ratio $r"
    job "forest,fold,evaluate" "$RESULTS/heads_n1_r${r}" "$e" \
        "--open-world --background-ratio $r --n1-head"
  done

  # -- Trained once on clean traffic, evaluated under every defense.
  # This is where the undefended reference of Table IV comes from.
  job "static-sweep" "$RESULTS/static_sweep" "$e" ""

  # -- Deep Fingerprinting, closed world only (the ~20% of Section VI-A).
  job "df" "$RESULTS/deep_fingerprinting" "$e" ""

  # -- Leave one client location out (Table V).
  for loc in "${LOCATIONS[@]}"; do
    job "location" "$RESULTS/location" "$e" "--holdout-location $loc"
  done
done

total=${#JOBS[@]}

if [ "${1:-}" = "--list" ]; then
  for j in "${JOBS[@]}"; do
    IFS='|' read -r models outdir exp flags <<< "$j"
    echo "exp $exp  $outdir  [$models]  $flags"
  done
  echo "# $total jobs"
  exit 0
fi

echo ">> shard $SHARD_ID/$SHARD_N | $total jobs total | interpreter: $PY"
done_count=0
failed=0

for ((i = 0; i < total; i++)); do
  (( i % SHARD_N == SHARD_ID - 1 )) || continue
  IFS='|' read -r models outdir exp flags <<< "${JOBS[i]}"
  echo ""
  echo ">>> job $((i + 1))/$total : exp $exp  $outdir  $flags"
  t0=$(date +%s)
  ok=1
  for model in ${models//,/ }; do
    # shellcheck disable=SC2086
    "$PY" run_experiment.py --model "$model" --experiment-id "$exp" \
        --out-dir "$outdir" $flags || { ok=0; echo "!! $model failed"; break; }
  done
  [ "$ok" = 1 ] || failed=$((failed + 1))
  done_count=$((done_count + 1))
  echo ">>> done in $(( $(date +%s) - t0 ))s  (this shard: $done_count run, $failed failed)"
done

echo ""
echo ">> shard $SHARD_ID/$SHARD_N finished: $done_count jobs, $failed failed."
