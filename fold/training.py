"""Compiling and fitting the models.

Every model is trained with early stopping on the validation loss (patience 20,
best weights restored), so the epoch count is a ceiling rather than a schedule.
"""
from typing import Optional

import numpy as np
from tensorflow import keras

from fold.data import PACKET_FEATURES, SEQUENCE_FEATURES
from fold.models.cnn import build_cnn
from fold.models.df import build_df
from fold.models.transformer import build_fold, build_transformer

#: Patience of the early-stopping callback, in epochs.
PATIENCE = 20

#: Weight of the auxiliary head's loss. High enough that the encoder cannot
#: ignore it, low enough that it does not compete with the main head.
AUX_LOSS_WEIGHT = 0.4

#: Label smoothing on every multi-class loss: it reduces over-confidence, which
#: matters with roughly 18 training traces per website.
LABEL_SMOOTHING = 0.1


def _callbacks():
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                          restore_best_weights=True, mode="min")]


def _smoothed():
    return keras.losses.CategoricalCrossentropy(label_smoothing=LABEL_SMOOTHING)


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------

def sequence_model(name: str, encoder, *, binary_loss_weight: float,
                   n1_head: bool = False) -> keras.Model:
    """Build and compile TR or FOLD, sized from the encoder.

    ``binary_loss_weight`` is the weight of the monitored/background head.  In
    the open world that task is real; in the closed world every trace is
    monitored, so the head is degenerate and the weight is what keeps it from
    hurting the website decision while leaving the graph unchanged.
    """
    build = {"transformer": build_transformer, "fold": build_fold}[name]
    extra = {"n1_head": n1_head} if name == "fold" else {}
    model = build(sequence_length=encoder.target_length,
                  num_features=SEQUENCE_FEATURES,
                  num_classes=encoder.num_classes,
                  num_agg_features=3,
                  size_vocab_size=encoder.vocab_size,
                  unigram_agg_dim=encoder.histogram_bins,
                  **extra)

    if n1_head:
        # The loss goes on the single softmax over N+1 classes. Same optimiser
        # and same label smoothing as the two-head variant, so that the only
        # difference between the two arms is the structure of the outputs.
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss={"n1_output": _smoothed(), "aux_multi_output": _smoothed()},
            loss_weights={"n1_output": 1.0, "aux_multi_output": AUX_LOSS_WEIGHT},
            metrics={"n1_output": ["categorical_accuracy"]})
        return model

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss={"binary_output": "binary_crossentropy",
              "multi_output": _smoothed(),
              "aux_multi_output": _smoothed()},
        metrics={"binary_output": ["accuracy"],
                 "multi_output": ["categorical_accuracy"],
                 "aux_multi_output": ["categorical_accuracy"]},
        loss_weights={"binary_output": binary_loss_weight, "multi_output": 1.0,
                      "aux_multi_output": AUX_LOSS_WEIGHT},
        weighted_metrics={"binary_output": [], "multi_output": [],
                          "aux_multi_output": []})
    return model


def single_head_model(name: str, encoder, num_classes: int) -> keras.Model:
    """Build and compile the convolutional baseline or Deep Fingerprinting.

    ``num_classes`` is not the encoder's: in the open world these two models
    take background as one extra class, so they are built N+1 wide.  Deep
    Fingerprinting keeps the optimiser of its original implementation.
    """
    if name == "cnn":
        model = build_cnn(encoder.target_length, PACKET_FEATURES, num_classes)
        optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    else:
        model = build_df(encoder.target_length, num_classes)
        optimizer = keras.optimizers.Adamax(learning_rate=0.002, beta_1=0.9,
                                            beta_2=0.999, epsilon=1e-8)

    model.compile(optimizer=optimizer,
                  loss={"multi_output": "categorical_crossentropy"},
                  metrics={"multi_output": ["categorical_accuracy"]})
    return model


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit(model: keras.Model, train_inputs: dict, train_outputs: dict,
        val_inputs: dict, val_outputs: dict, *, epochs: int, batch_size: int,
        sample_weight: Optional[dict] = None) -> None:
    """Fit with early stopping on the validation loss."""
    model.fit(train_inputs, train_outputs, sample_weight=sample_weight,
              validation_data=(val_inputs, val_outputs),
              epochs=epochs, batch_size=batch_size,
              callbacks=_callbacks(), verbose=2)


def fit_closed_world(model: keras.Model, train_inputs: dict, train_labels,
                     val_inputs: dict, val_labels, *, epochs: int,
                     batch_size: int) -> None:
    """Fit TR or FOLD on monitored traces only.

    Every trace is a monitored visit, so all of them weigh the same, and the
    auxiliary head simply shares the website labels of the main one.
    """
    multi, binary = train_labels
    val_multi, val_binary = val_labels
    fit(model,
        train_inputs, {"binary_output": binary, "multi_output": multi,
                       "aux_multi_output": multi},
        val_inputs, {"binary_output": val_binary, "multi_output": val_multi,
                     "aux_multi_output": val_multi},
        epochs=epochs, batch_size=batch_size)


def n1_labels(multi: np.ndarray, binary: np.ndarray) -> np.ndarray:
    """One-hot over N+1 classes: monitored traces keep theirs, background gets N.

    ``multi`` is already all-zero on the background rows, which is why turning
    on one extra column is enough.
    """
    n = multi.shape[1]
    out = np.zeros((len(multi), n + 1), np.float32)
    out[:, :n] = multi
    out[np.asarray(binary).ravel() < 0.5, n] = 1.0
    return out


def fit_open_world(model: keras.Model, train_inputs: dict, train_labels,
                   val_inputs: dict, val_labels, *, epochs: int,
                   batch_size: int) -> None:
    """Fit TR or FOLD on monitored and background traces together.

    Background traces carry no website label, so they weigh zero on the website
    head and one on the monitored head, which is the head they are there to
    teach.  Under the N+1 head there is nothing to mask: background is class N
    and contributes like any other class.
    """
    multi, binary, weights = train_labels
    val_multi, val_binary, _ = val_labels
    ones = np.ones(len(binary), np.float32)

    if "n1_output" in model.output_names:
        fit(model,
            train_inputs, {"n1_output": n1_labels(multi, binary),
                           "aux_multi_output": multi},
            val_inputs, {"n1_output": n1_labels(val_multi, val_binary),
                         "aux_multi_output": val_multi},
            sample_weight={"n1_output": ones, "aux_multi_output": weights},
            epochs=epochs, batch_size=batch_size)
        return

    fit(model,
        train_inputs, {"binary_output": binary, "multi_output": multi,
                       "aux_multi_output": multi},
        val_inputs, {"binary_output": val_binary, "multi_output": val_multi,
                     "aux_multi_output": val_multi},
        sample_weight={"binary_output": ones, "multi_output": weights,
                       "aux_multi_output": weights},
        epochs=epochs, batch_size=batch_size)


def predict_sequence(model: keras.Model, inputs: dict, num_classes: int,
                     batch_size: int):
    """Predict with TR or FOLD; recover the two views under the N+1 head."""
    from fold.models.transformer import split_n1

    out = model.predict(inputs, batch_size=batch_size, verbose=0)
    if "n1_output" in out:
        return split_n1(out["n1_output"], num_classes)
    return out["multi_output"], out["binary_output"].flatten()


def configure_gpu() -> None:
    """Grow GPU memory on demand instead of reserving all of it up front."""
    import tensorflow as tf
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
