#!/usr/bin/env python
"""Run one model of one experiment.

One invocation trains (or evaluates) exactly one model, so that each training
gets a fresh process and releases its GPU memory when it ends.  The campaign
script chains the invocations; see ``run_campaign.sh``.

    # closed world, undefended
    python run_experiment.py --model forest --experiment-id 1 --out-dir results/cw
    python run_experiment.py --model fold   --experiment-id 1 --out-dir results/cw
    python run_experiment.py --model evaluate --experiment-id 1 --out-dir results/cw

    # against a defense the adversary knows about
    python run_experiment.py --model fold --experiment-id 1 \\
        --defense-padding 0.7 --out-dir results/padding

    # open world, half the traffic outside the monitored set
    python run_experiment.py --model fold --experiment-id 1 \\
        --open-world --background-ratio 0.5 --out-dir results/ow
"""
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATASET = HERE / "data" / "odoh_traces.csv"
DEFAULT_BACKGROUND = HERE / "data" / "background_pool.csv"

MODELS = ("forest", "transformer", "fold", "cnn", "df", "evaluate",
          "static-sweep", "location")


def parse_args():
    p = argparse.ArgumentParser(
        description="Run one model of one experiment of the FOLD campaign.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--model", required=True, choices=MODELS,
                   help="forest, transformer and fold are the attack and its two "
                        "branches; cnn and df are the baselines; evaluate reads "
                        "back the predictions and writes result.json; "
                        "static-sweep trains once on clean traffic and evaluates "
                        "on many defended test sets; location holds one client "
                        "location out")
    p.add_argument("--experiment-id", type=int, required=True,
                   help="the repetition; it seeds the split and the initialisation")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="directory of this campaign cell")
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)

    p.add_argument("--open-world", action="store_true",
                   help="mix in traffic from outside the monitored set")
    p.add_argument("--background-ratio", type=float, default=0.5,
                   help="fraction of the traces that is background, in open world")
    p.add_argument("--background-pool", type=Path, default=DEFAULT_BACKGROUND,
                   help="the synthetic background pool, from make_background.py")

    p.add_argument("--defense-padding", type=float, default=0.0,
                   help="per-packet padding probability")
    p.add_argument("--defense-dummy", type=float, default=0.0,
                   help="per-packet dummy-injection probability")
    p.add_argument("--defense-fragmentation", type=float, default=0.0,
                   help="per-packet fragmentation probability")

    p.add_argument("--n1-head", action="store_true",
                   help="replace FOLD's two heads with one softmax over N+1 "
                        "classes, the ablation of the paper's Fig. 8")
    p.add_argument("--holdout-location", default="random",
                   help="for --model location: the client location to hold out, "
                        "or 'random' for the reference split")
    p.add_argument("--sweep-probs", type=float, nargs="+",
                   default=[0.10, 0.30, 0.50, 0.70, 1.00],
                   help="for --model static-sweep: the defense probabilities")

    p.add_argument("--epochs", type=int, default=500,
                   help="ceiling; training ends on early stopping")
    p.add_argument("--batch-size", type=int, default=16)
    return p.parse_args()


def defense_config(args) -> dict:
    """The mechanisms that are switched on, and with what probability."""
    return {name: value
            for value, name in ((args.defense_padding, "padding"),
                                (args.defense_dummy, "dummy"),
                                (args.defense_fragmentation, "fragmentation"))
            if value}


def main():
    args = parse_args()
    from fold.experiments import (Config, run_cnn, run_df, run_evaluate, run_fold,
                                  run_forest, run_location, run_static_sweep,
                                  run_transformer, seed_everything)

    RUNNERS = {"forest": run_forest, "transformer": run_transformer, "fold": run_fold,
               "cnn": run_cnn, "df": run_df, "evaluate": run_evaluate}

    defense = defense_config(args)
    scenario = ("open-world" if args.open_world
                else "defense" if defense
                else "closed-world")

    if args.n1_head and not args.open_world:
        # The N+1 head puts background in class N, which only exists in the
        # open world; silently ignoring the flag would look like it worked.
        raise SystemExit("--n1-head only applies with --open-world")

    cfg = Config(experiment_id=args.experiment_id, out_dir=args.out_dir,
                 dataset=args.dataset, scenario=scenario, defense=defense,
                 background_ratio=args.background_ratio,
                 background_pool=args.background_pool,
                 batch_size=args.batch_size, epochs=args.epochs,
                 n1_head=args.n1_head)
    seed_everything(cfg.seed)

    # The two experiments that vary the test set take what they vary it by.
    if args.model == "static-sweep":
        run_static_sweep(cfg, args.sweep_probs)
    elif args.model == "location":
        run_location(cfg, args.holdout_location)
    else:
        RUNNERS[args.model](cfg)


if __name__ == "__main__":
    main()
