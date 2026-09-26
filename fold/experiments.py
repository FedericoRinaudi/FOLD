"""The experiments of the paper, one function per model and per scenario.

Every model of an experiment runs in its own process (see ``run_campaign.sh``)
and writes its predictions to the experiment's directory; ``evaluate`` reads
them back and writes ``result.json``.  Splitting it this way keeps the GPU
memory of one training out of the next, and lets a rerun pick up where it
stopped.  :mod:`fold.predictions` describes what ends up on disk.
"""
from __future__ import annotations

import csv
import json
import pickle
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from fold import metrics as M
from fold.data import (Encoder, apply_defense, load_balanced, load_traces,
                       one_hot, split_data)
from fold.models.forest import RandomForest
from fold.predictions import Predictions
from fold.training import (configure_gpu, fit, fit_closed_world, fit_open_world,
                           predict_sequence, sequence_model, single_head_model)

if TYPE_CHECKING:                      # only for the annotations below
    from tensorflow import keras

#: Weight of the monitored head's loss, by scenario.
#: In the closed world every trace is monitored, so the head is degenerate:
#: TR keeps it at the same weight it has elsewhere, while FOLD, whose own
#: detection stem is fed by the forest's confidence, zeroes its gradient.
#: In the open world detection is the task being measured and the weight is
#: twice the closed-world one.
BINARY_WEIGHT = {"transformer": 0.3, "fold": 0.0}
BINARY_WEIGHT_OPEN_WORLD = 0.6

#: Deep Fingerprinting trains with its own batch size and epoch ceiling, from
#: the original implementation.
DF_EPOCHS = 300
DF_BATCH_SIZE = 32

#: The models of a full campaign cell, in the order they must run: FOLD reads
#: the forest's output, so the forest goes first.
FAMILY = ("forest", "transformer", "fold", "cnn")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and TensorFlow for this repetition.

    Two things depend on it.  The defense is stochastic and is applied
    separately in each model's process, so without a seed the models of one
    experiment would each see a differently perturbed test set; with it they
    are compared on exactly the same defended traces.  And model
    initialisation becomes a function of the repetition, which is what the
    error bars across repetitions are meant to capture.
    """
    import tensorflow as tf
    tf.keras.utils.set_random_seed(seed)    # Python, NumPy and TensorFlow at once


# ---------------------------------------------------------------------------
# What a run is made of
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Everything one run of one model needs to know."""
    experiment_id: int
    out_dir: Path
    dataset: Path
    scenario: str = "closed-world"      # closed-world | defense | open-world
    defense: dict = field(default_factory=dict)
    background_ratio: float = 0.5
    background_pool: Optional[Path] = None
    batch_size: int = 16
    epochs: int = 500
    n1_head: bool = False

    @property
    def is_open_world(self) -> bool:
        return self.scenario == "open-world"

    @property
    def is_defended(self) -> bool:
        return bool(self.defense)

    @property
    def exp_dir(self) -> Path:
        return self.out_dir / f"exp_{self.experiment_id:03d}"

    @property
    def seed(self) -> int:
        """The seed of this repetition; it drives both splits and initialisation."""
        return self.experiment_id * 17 + 42


@dataclass
class TraceSet:
    """Traces and the websites they belong to."""
    traces: List
    labels: List[int]
    ids: Optional[List[str]] = None

    def __len__(self) -> int:
        return len(self.traces)


@dataclass
class Split:
    """The monitored traces of one repetition, divided in three."""
    train: TraceSet
    val: TraceSet
    test: TraceSet
    num_classes: int
    target_length: int
    defense_state: Optional[tuple] = None

    def encoder(self) -> Encoder:
        """The encoder of this run: fit on the training traces, reused for the
        other two."""
        return Encoder.fit(self.train.traces, self.num_classes, self.target_length)


@dataclass
class Background:
    """The traffic from outside the monitored set drawn for one repetition."""
    train: List = field(default_factory=list)
    val: List = field(default_factory=list)
    test: List = field(default_factory=list)
    ids: Optional[dict] = None


# ---------------------------------------------------------------------------
# Data, splits and the background pool
# ---------------------------------------------------------------------------

def load_and_split(cfg: Config, *, capture_ids: bool = False) -> Split:
    """Load the dataset, reshuffle it with this repetition's seed, split it."""
    loaded = load_balanced(cfg.dataset, return_ids=capture_ids)
    traces, labels, num_classes = loaded[:3]
    target_length = max(len(t) for t in traces)

    idx = np.random.default_rng(seed=cfg.seed).permutation(len(traces))
    traces = [traces[i] for i in idx]
    labels = [labels[i] for i in idx]

    arrays = [traces, labels]
    if capture_ids:
        arrays.append([loaded[3][i] for i in idx])
    parts = split_data(*arrays)
    return Split(*(TraceSet(*items) for items in zip(*parts)),
                 num_classes, target_length)


def defend(cfg: Config, split: Split) -> Split:
    """Apply the defense to every split.

    The adversary is adaptive: it knows the defense is deployed and collects
    defended traffic to train on, so training, validation and test are all
    defended.  Evaluating a clean-trained attack on defended traces would
    measure the surprise, not the residual leakage.
    """
    if not cfg.is_defended:
        return split
    print(f"[exp {cfg.experiment_id}] defense applied to every split: {cfg.defense}")
    state = random.getstate() if split.train.ids is not None else None
    return Split(
        *(TraceSet(apply_defense(s.traces, cfg.defense), s.labels, s.ids)
          for s in (split.train, split.val, split.test)),
        split.num_classes, split.target_length, state)


def draw_background(cfg: Config, split: Split, *, capture_ids: bool = False) -> Background:
    """Draw disjoint background slices from the synthetic pool.

    The pool is generated once, offline (``make_background.py``), because
    background traffic depends neither on the monitored split nor on any
    defense.  Each repetition reshuffles it with its own seed and carves its
    own disjoint train/validation/test slices, so that the error bars across
    repetitions also reflect background sampling, and no single background test
    set is baked into the results.
    """
    import pandas as pd

    path = cfg.background_pool
    if path is None or not Path(path).exists():
        raise SystemExit(
            f"background pool not found at {path}.\n"
            f"Generate it once with:  python make_background.py --out {path}")

    scale = cfg.background_ratio / (1 - cfg.background_ratio)
    n_train, n_val, n_test = (int(round(len(s) * scale))
                              for s in (split.train, split.val, split.test))

    # comment="#": the pool carries a provenance header recording the seed and
    # every parameter used to build it.
    df = pd.read_csv(path, comment="#")
    pool = [[[float(t), float(s), float(r)] for t, s, r in zip(g["t"], g["size"], g["role"])]
            for _, g in df.sort_values(["trace_id", "pkt_idx"]).groupby("trace_id", sort=True)]

    need = n_train + n_val + n_test
    if len(pool) < need:
        raise SystemExit(
            f"the pool holds {len(pool)} background traces but {need} are needed "
            f"({n_train}+{n_val}+{n_test}) at this background ratio. "
            f"Generate a larger one with make_background.py.")

    idx = np.random.default_rng(seed=cfg.seed).permutation(len(pool))[:need]
    drawn = [pool[i] for i in idx]
    ids = None
    if capture_ids:
        pool_ids = list(df.groupby("trace_id", sort=True).groups)
        selected = [str(pool_ids[i]) for i in idx]
        ids = {"train": selected[:n_train], "val": selected[n_train:n_train + n_val],
               "test": selected[n_train + n_val:need]}
    print(f"[exp {cfg.experiment_id}] background pool: {len(pool)} traces, "
          f"drew {n_train}/{n_val}/{n_test}")
    return Background(drawn[:n_train], drawn[n_train:n_train + n_val],
                      drawn[n_train + n_val:need], ids)


def save_background(cfg: Config, background: Background) -> None:
    # A plain dict rather than the dataclass: the file then does not depend on
    # where this class happens to live.
    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    values = {"train": background.train, "val": background.val, "test": background.test}
    if background.ids is not None:
        values["ids"] = background.ids
    with open(cfg.exp_dir / "background.pkl", "wb") as f:
        pickle.dump(values, f)


def load_background(cfg: Config) -> Background:
    path = cfg.exp_dir / "background.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run the forest first: it draws the background "
            f"and every other model reuses exactly those traces.")
    with open(path, "rb") as f:
        return Background(**pickle.load(f))


def context(cfg: Config, *, capture_ids: bool = False):
    """The split, the defense and the background: what every runner starts from."""
    split = defend(cfg, load_and_split(cfg, capture_ids=capture_ids))
    background = load_background(cfg) if cfg.is_open_world else Background()
    return split, background


def predictions_path(cfg: Config, name: str) -> Path:
    return cfg.exp_dir / f"{name}.npz"


# ---------------------------------------------------------------------------
# The forest
# ---------------------------------------------------------------------------

def fit_forest(encoder: Encoder, train: TraceSet, seed: int) -> RandomForest:
    return RandomForest(encoder.num_classes, seed).fit(encoder.forest(train.traces),
                                                       np.array(train.labels))


def out_of_fold_probs(encoder: Encoder, train: TraceSet, seed: int,
                      n_folds: int = 5) -> np.ndarray:
    """Forest probabilities on the training traces, out of fold.

    FOLD is trained on the forest's output, so on the training set that output
    must come from a forest that never saw the trace it is scoring.  Otherwise
    the forest's probabilities are near-perfect there and FOLD learns to trust
    them more than it should at test time.
    """
    from sklearn.model_selection import KFold

    X, y = encoder.forest(train.traces), np.array(train.labels)
    oof = np.zeros((len(X), encoder.num_classes), dtype=np.float32)

    folds = KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(X)
    for fold, (idx_fit, idx_score) in enumerate(folds):
        forest = RandomForest(encoder.num_classes, seed + fold).fit(X[idx_fit], y[idx_fit])
        oof[idx_score] = forest.predict_proba(X[idx_score])
    return oof


def run_forest(cfg: Config, *, return_artifacts: bool = False):
    """Fit the forest, and draw this run's background traces.

    This runs first in every experiment: it is what writes ``background.pkl``,
    so that every other model is trained and scored on exactly the same
    background traces.

    With ``return_artifacts``, return (split, background, encoder, final forest).
    """
    configure_gpu()
    split = defend(cfg, load_and_split(cfg, capture_ids=return_artifacts))
    encoder = split.encoder()

    background = Background()
    if cfg.is_open_world:
        background = draw_background(cfg, split, capture_ids=return_artifacts)
        save_background(cfg, background)
        n_test = len(background.test)
        print(f"[exp {cfg.experiment_id}] background is "
              f"{n_test / (len(split.test) + n_test):.0%} of the test set")

    print(f"[exp {cfg.experiment_id}] fitting the forest and its out-of-fold "
          f"probabilities ...")
    forest = fit_forest(encoder, split.train, cfg.experiment_id)

    def probs(traces):
        return forest.predict_proba(encoder.forest(traces))

    out = Predictions(
        test_monitored_probs=probs(split.test.traces),
        val_monitored_probs=probs(split.val.traces),
        train_out_of_fold=out_of_fold_probs(encoder, split.train, cfg.experiment_id),
    )
    if cfg.is_open_world:
        out.train_background_probs = probs(background.train)
        out.val_background_probs = probs(background.val)
        out.test_background_probs = probs(background.test)

    out.save(predictions_path(cfg, "forest"))
    print(f"[exp {cfg.experiment_id}] forest done")
    if return_artifacts:
        return split, background, encoder, forest


# ---------------------------------------------------------------------------
# TR and FOLD
# ---------------------------------------------------------------------------

def train_sequence_model(cfg: Config, name: str, encoder: Encoder,
                         train: TraceSet, val: TraceSet, forest: Optional[Predictions] = None):
    """Build and fit TR or FOLD on monitored traces only."""
    model = sequence_model(name, encoder, binary_loss_weight=BINARY_WEIGHT[name])
    fit_closed_world(
        model,
        encoder.sequence(train.traces, forest.train_out_of_fold if forest else None),
        encoder.labels(train.labels),
        encoder.sequence(val.traces, forest.val_monitored_probs if forest else None),
        encoder.labels(val.labels),
        epochs=cfg.epochs, batch_size=cfg.batch_size)
    return model


def concat_open_world(encoder: Encoder, monitored: TraceSet, background: List,
                      forest_monitored=None, forest_background=None):
    """Monitored and background traces, concatenated into one training set.

    Background traces get an all-zero website label and a zero weight on the
    website head: they are there to teach the monitored decision, and must not
    push the website head toward any site.
    """
    mon = encoder.sequence(monitored.traces, forest_monitored)
    back = encoder.sequence(background, forest_background)
    inputs = {k: np.concatenate([mon[k], back[k]]) for k in mon}

    multi, binary = encoder.labels(monitored.labels)
    multi = np.concatenate([multi, np.zeros((len(background), encoder.num_classes),
                                            np.float32)])
    binary = np.concatenate([binary, np.zeros(len(background), np.float32)])
    weights = np.concatenate([np.ones(len(monitored), np.float32),
                              np.zeros(len(background), np.float32)])
    return inputs, (multi, binary, weights)


def train_sequence_model_open_world(cfg: Config, name: str, encoder: Encoder,
                                    split: Split, background: Background,
                                    forest: Optional[Predictions] = None):
    """Build and fit TR or FOLD on monitored and background traces together."""
    rf = (forest.train_out_of_fold, forest.train_background_probs,
          forest.val_monitored_probs, forest.val_background_probs) if forest \
        else (None, None, None, None)

    train_inputs, train_labels = concat_open_world(
        encoder, split.train, background.train, rf[0], rf[1])
    val_inputs, val_labels = concat_open_world(
        encoder, split.val, background.val, rf[2], rf[3])

    model = sequence_model(name, encoder,
                           binary_loss_weight=BINARY_WEIGHT_OPEN_WORLD,
                           n1_head=cfg.n1_head and name == "fold")
    fit_open_world(model, train_inputs, train_labels, val_inputs, val_labels,
                   epochs=cfg.epochs, batch_size=cfg.batch_size)
    return model, (val_inputs, val_labels[1])


def _run_sequence_model(cfg: Config, name: str, *, return_artifacts: bool = False):
    """Train TR or FOLD in whichever scenario the configuration asks for."""
    configure_gpu()
    split, background = context(cfg, capture_ids=return_artifacts)
    encoder = split.encoder()
    forest = Predictions.load(predictions_path(cfg, "forest")) if name == "fold" else None

    print(f"[exp {cfg.experiment_id}] training {name} "
          f"({cfg.epochs} epochs, early stopping) ...")

    if not cfg.is_open_world:
        model = train_sequence_model(cfg, name, encoder, split.train, split.val, forest)
        inputs = encoder.sequence(split.test.traces,
                                  forest.test_monitored_probs if forest else None)
        probs, score = predict_sequence(model, inputs, encoder.num_classes, cfg.batch_size)
        Predictions(test_monitored_probs=probs,
                    test_monitored_score=score).save(predictions_path(cfg, name))
        if return_artifacts:
            return split, background, encoder, model
        return

    model, (val_inputs, val_binary) = train_sequence_model_open_world(
        cfg, name, encoder, split, background, forest)

    def predict(traces, forest_probs):
        return predict_sequence(model, encoder.sequence(traces, forest_probs),
                                encoder.num_classes, cfg.batch_size)

    test_probs, test_score = predict(split.test.traces,
                                     forest.test_monitored_probs if forest else None)
    back_probs, back_score = predict(background.test,
                                     forest.test_background_probs if forest else None)

    # The validation predictions are kept so that a monitored/background
    # threshold can be chosen without ever touching the test set.
    val_probs, val_score = predict_sequence(model, val_inputs, encoder.num_classes,
                                            cfg.batch_size)
    is_monitored = np.asarray(val_binary).ravel().astype(bool)

    Predictions(
        test_monitored_probs=test_probs, test_monitored_score=test_score,
        test_background_probs=back_probs, test_background_score=back_score,
        val_monitored_probs=val_probs[is_monitored],
        val_monitored_score=val_score[is_monitored],
        val_background_probs=val_probs[~is_monitored],
        val_background_score=val_score[~is_monitored],
    ).save(predictions_path(cfg, name))
    print(f"[exp {cfg.experiment_id}] {name} done")
    if return_artifacts:
        return split, background, encoder, model


def run_transformer(cfg: Config) -> None:
    """Train TR, the sequential branch on its own."""
    _run_sequence_model(cfg, "transformer")


def run_fold(cfg: Config, *, return_artifacts: bool = False):
    """Train FOLD after RF; optionally return (split, background, encoder, model)."""
    return _run_sequence_model(cfg, "fold", return_artifacts=return_artifacts)


# ---------------------------------------------------------------------------
# The models without a monitored head
# ---------------------------------------------------------------------------

def train_cnn(cfg: Config, encoder: Encoder, train: TraceSet, val: TraceSet):
    """Fit the convolutional baseline on monitored traces."""
    model = single_head_model("cnn", encoder, encoder.num_classes)
    fit(model,
        encoder.cnn(train.traces), {"multi_output": one_hot(train.labels, encoder.num_classes)},
        encoder.cnn(val.traces), {"multi_output": one_hot(val.labels, encoder.num_classes)},
        epochs=cfg.epochs, batch_size=cfg.batch_size)
    return model


def _run_single_head_model(cfg: Config, name: str, epochs: int, batch_size: int) -> None:
    """Train the convolutional baseline or Deep Fingerprinting.

    Neither has a head for the monitored decision, so in the open world they
    follow the standard protocol of the literature: background joins training
    as one extra class, and the accept/reject decision is taken afterwards by
    thresholding the confidence over the monitored classes alone.
    """
    configure_gpu()
    split, background = context(cfg)
    encoder = split.encoder()
    inputs = encoder.cnn if name == "cnn" else encoder.df

    print(f"[exp {cfg.experiment_id}] training {name} ({epochs} epochs ceiling) ...")

    num_classes = encoder.num_classes + (1 if cfg.is_open_world else 0)

    def labels(monitored_labels, background_traces):
        """Background joins the monitored websites as class N."""
        return {"multi_output": one_hot(
            list(monitored_labels) + [encoder.num_classes] * len(background_traces),
            num_classes)}

    train_traces = split.train.traces + background.train
    val_traces = split.val.traces + background.val

    model = single_head_model(name, encoder, num_classes)
    fit(model,
        inputs(train_traces), labels(split.train.labels, background.train),
        inputs(val_traces), labels(split.val.labels, background.val),
        epochs=epochs, batch_size=batch_size)

    def predict(traces):
        return model.predict(inputs(traces), batch_size=batch_size,
                             verbose=0)["multi_output"]

    out = Predictions(test_monitored_probs=predict(split.test.traces))
    if cfg.is_open_world:
        out.test_background_probs = predict(background.test)
        out.val_monitored_probs = predict(split.val.traces)
        out.val_background_probs = predict(background.val)
        out.n_monitored = encoder.num_classes
    out.save(predictions_path(cfg, name))

    # Deep Fingerprinting is the only model of a closed-world-only run, so it
    # saves the labels itself instead of waiting for evaluate.
    if not cfg.is_open_world:
        np.save(cfg.exp_dir / "y_test.npy", np.array(split.test.labels, dtype=np.int16))
    print(f"[exp {cfg.experiment_id}] {name} done")


def run_cnn(cfg: Config) -> None:
    """Train the convolutional baseline."""
    _run_single_head_model(cfg, "cnn", cfg.epochs, cfg.batch_size)


def run_df(cfg: Config) -> None:
    """Train Deep Fingerprinting."""
    _run_single_head_model(cfg, "df", DF_EPOCHS, DF_BATCH_SIZE)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluate(cfg: Config) -> None:
    """Read back every model's predictions and write ``result.json``."""
    split, background = context(cfg)
    y_test = np.array(split.test.labels)

    available = {name: Predictions.load(predictions_path(cfg, name))
                 for name in (*FAMILY, "df")
                 if predictions_path(cfg, name).exists()}

    report = {"scenario": cfg.scenario, "defense": cfg.defense,
              "n_monitored": split.num_classes, "experiment_id": cfg.experiment_id}

    if cfg.is_open_world:
        n_back = len(background.test)
        report["background_ratio"] = float(n_back / (len(split.test) + n_back))
        for name, pred in available.items():
            mon, back, mon_score, back_score = pred.views()
            report[name] = {
                # The macro metrics of the paper, over the monitored websites.
                **M.open_world(mon, back, y_test, mon_score, back_score,
                               include_background=False),
                # The monitored decision on its own.
                "detection": M.detection(mon_score, back_score),
            }
    else:
        for name, pred in available.items():
            report[name] = M.closed_world(pred.test_monitored_probs, y_test)

    cfg.exp_dir.mkdir(parents=True, exist_ok=True)
    np.save(cfg.exp_dir / "y_test.npy", y_test.astype(np.int16))
    with (cfg.exp_dir / "result.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(f"\n[exp {cfg.experiment_id}] {cfg.scenario}"
          f"{' ' + str(cfg.defense) if cfg.is_defended else ''}")
    for name in available:
        r = report[name]
        print(f"  {name:12s} accuracy {r['accuracy']:.4f}   macro-F1 {r['f1']:.4f}")


# ---------------------------------------------------------------------------
# Training the whole family on one split
# ---------------------------------------------------------------------------

@dataclass
class Family:
    """The attack and the baselines, all fitted on the same training traces.

    The two experiments that vary the *test* set rather than the training one
    -- the static defense sweep and the held-out location -- train this once
    and then predict as many times as they need.
    """
    forest: RandomForest
    transformer: "keras.Model"
    fold: "keras.Model"
    cnn: "keras.Model"

    def predict(self, encoder: Encoder, traces, batch_size: int) -> Dict[str, np.ndarray]:
        """Website probabilities from every model, on the same traces."""
        forest_probs = self.forest.predict_proba(encoder.forest(traces))
        transformer_probs, _ = predict_sequence(
            self.transformer, encoder.sequence(traces), encoder.num_classes, batch_size)
        fold_probs, _ = predict_sequence(
            self.fold, encoder.sequence(traces, forest_probs), encoder.num_classes,
            batch_size)
        cnn_probs = self.cnn.predict(encoder.cnn(traces), batch_size=batch_size,
                                     verbose=0)["multi_output"]
        return {"forest": forest_probs, "transformer": transformer_probs,
                "fold": fold_probs, "cnn": cnn_probs}


def train_family(cfg: Config, encoder: Encoder, train: TraceSet, val: TraceSet) -> Family:
    """Fit the forest, TR, FOLD and the convolutional baseline on one split."""
    print(f"[exp {cfg.experiment_id}] fitting the forest ...")
    forest = fit_forest(encoder, train, cfg.experiment_id)
    forest_out = Predictions(
        train_out_of_fold=out_of_fold_probs(encoder, train, cfg.experiment_id),
        val_monitored_probs=forest.predict_proba(encoder.forest(val.traces)))

    print(f"[exp {cfg.experiment_id}] training TR ...")
    transformer = train_sequence_model(cfg, "transformer", encoder, train, val)
    print(f"[exp {cfg.experiment_id}] training FOLD ...")
    fold = train_sequence_model(cfg, "fold", encoder, train, val, forest_out)
    print(f"[exp {cfg.experiment_id}] training the convolutional baseline ...")
    cnn = train_cnn(cfg, encoder, train, val)

    return Family(forest, transformer, fold, cnn)


# ---------------------------------------------------------------------------
# Static defense sweep: train once on clean traffic, evaluate on many defenses
# ---------------------------------------------------------------------------

def sweep_configs(defense_probs: List[float]) -> List[dict]:
    """No defense, then each mechanism alone and all three together, per ``p``."""
    configs = [{}]
    for p in defense_probs:
        configs.append({"padding": p, "dummy": p, "fragmentation": p})
        configs.append({"padding": p})
        configs.append({"dummy": p})
        configs.append({"fragmentation": p})
    return configs


def sweep_tag(config: dict) -> str:
    """The short name of a sweep cell, used for its predictions file."""
    return "_".join(f"{name}{config[name]:g}"
                    for name in ("padding", "dummy", "fragmentation")
                    if config.get(name)) or "clean"


def run_static_sweep(cfg: Config, defense_probs: List[float]) -> None:
    """Train the models once on clean traffic, then evaluate them on defended
    test sets of increasing strength.

    Because training happens once, every point of the resulting curve is the
    same attacker, which is what separates the effect of the defense on the
    traces from its effect on training.  This is also where the undefended
    reference of the paper's tables comes from: the ``clean`` cell.
    """
    configure_gpu()
    split = load_and_split(cfg)
    encoder = split.encoder()
    y_test = np.array(split.test.labels)
    family = train_family(cfg, encoder, split.train, split.val)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.out_dir / "static_sweep.csv"
    write_header = not csv_path.exists()
    preds_dir = cfg.out_dir / "preds" / f"exp_{cfg.experiment_id:03d}"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / "y_test.npy", y_test.astype(np.int16))

    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "model", "experiment_id", "padding", "dummy", "fragmentation",
            "accuracy", "macro_f1"])
        if write_header:
            writer.writeheader()

        for config in sweep_configs(defense_probs):
            defended = apply_defense(split.test.traces, config)   # {} leaves them alone
            by_model = family.predict(encoder, defended, cfg.batch_size)

            tag = sweep_tag(config)
            np.savez_compressed(preds_dir / f"{tag}.npz",
                                **{k: v.astype(np.float32)
                                   for k, v in by_model.items()})

            for name, model_probs in by_model.items():
                m = M.closed_world(model_probs, y_test)
                writer.writerow({
                    "model": name, "experiment_id": cfg.experiment_id,
                    "padding": config.get("padding", 0.0),
                    "dummy": config.get("dummy", 0.0),
                    "fragmentation": config.get("fragmentation", 0.0),
                    "accuracy": m["accuracy"], "macro_f1": m["f1"]})
            fh.flush()
            print(f"[exp {cfg.experiment_id}] [sweep] {tag} done")

    print(f"[exp {cfg.experiment_id}] [sweep] -> {csv_path}")


# ---------------------------------------------------------------------------
# Location generalization
# ---------------------------------------------------------------------------

def location_split(cfg: Config, holdout: str) -> Split:
    """Hold one client location out, or split the same pool at random.

    Each sample comes from a single capture and so from a single location, so
    the held-out location is never seen during training.  ``random`` is the
    reference the held-out numbers are read against.

    521 of the 2834 traces carry no location in the dataset.  They are never
    held out and so always stay on the training side, which is what the
    published results do.
    """
    traces, labels, locations = load_traces(cfg.dataset)
    known = sorted(set(locations))
    if holdout != "random" and holdout not in known:
        raise SystemExit(f"unknown location {holdout!r}; choose one of {known} or 'random'")

    data = list(zip(traces, labels, locations))
    rng = np.random.default_rng(cfg.seed)

    if holdout == "random":
        pool = data[:]
        rng.shuffle(pool)
        n_test, n_val = int(len(pool) * 0.25), int(len(pool) * 0.10)
        test, val, train = pool[:n_test], pool[n_test:n_test + n_val], pool[n_test + n_val:]
    else:
        test = [d for d in data if d[2] == holdout]
        pool = [d for d in data if d[2] != holdout]
        rng.shuffle(pool)
        n_val = int(len(pool) * 0.10)
        val, train = pool[:n_val], pool[n_val:]

    if not test or not train:
        raise SystemExit("empty train or test split")

    def as_set(rows):
        return TraceSet([r[0] for r in rows], [r[1] for r in rows])

    return Split(as_set(train), as_set(val), as_set(test),
                 max(labels) + 1, max(len(r[0]) for r in train))


def run_location(cfg: Config, holdout: str) -> None:
    """Train on every client location but one, test on the held-out location."""
    configure_gpu()
    split = location_split(cfg, holdout)
    encoder = split.encoder()
    y_test = np.array(split.test.labels)
    print(f"[exp {cfg.experiment_id}] holdout={holdout}: {len(split.train)} train / "
          f"{len(split.val)} val / {len(split.test)} test")

    family = train_family(cfg, encoder, split.train, split.val)
    by_model = family.predict(encoder, split.test.traces, cfg.batch_size)

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = cfg.out_dir / "location.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["model", "experiment_id",
                                                "holdout_location", "n_train",
                                                "n_test", "accuracy", "macro_f1"])
        if write_header:
            writer.writeheader()
        for name, model_probs in by_model.items():
            m = M.closed_world(model_probs, y_test)
            writer.writerow({"model": name, "experiment_id": cfg.experiment_id,
                             "holdout_location": holdout, "n_train": len(split.train),
                             "n_test": len(split.test), "accuracy": m["accuracy"],
                             "macro_f1": m["f1"]})

    preds_dir = cfg.out_dir / "preds" / f"exp_{cfg.experiment_id:03d}"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(preds_dir / f"{holdout}.npz",
                        y_test=y_test.astype(np.int16),
                        **{k: v.astype(np.float32) for k, v in by_model.items()})
    print(f"[exp {cfg.experiment_id}] holdout={holdout} done -> {csv_path}")
