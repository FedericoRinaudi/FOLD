"""Predicting from traces with the forest and FOLD.

The encoder builds both views of each trace; the forest's probabilities feed
FOLD through ``Encoder.sequence``. Output columns follow numeric class indices.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable

import numpy as np

from fold.data import Encoder
from fold.models.forest import RandomForest
from .artifacts import _validate_encoder, _validate_forest, _validate_model


def _traces(value: Any) -> list:
    """A single ``(L, 3)`` trace or variable-length batch, as loader-style lists.

    ``[]`` is an empty batch. Individual traces must be nonempty to avoid fully
    masked attention. Packet values and order are preserved; inputs are copied.
    """
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError("Pass a trace or batch as a list, tuple, or NumPy array")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        raise ValueError("A scalar is not a trace")
    if len(value) == 0:
        return []
    first = np.asarray(value[0])
    batch = [value] if first.ndim == 1 and first.shape == (3,) else value
    result = []
    for index, trace in enumerate(batch):
        array = np.asarray(trace, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3 or array.shape[0] == 0:
            raise ValueError(f"Trace {index} must be a nonempty (packets, 3) array")
        if not np.isfinite(array).all():
            raise ValueError(f"Trace {index} contains nonfinite values")
        roles = array[:, 2]
        if np.any(array[:, 1] < 0) or np.any(roles != np.floor(roles)) or np.any((roles < 0) | (roles > 3)):
            raise ValueError(f"Trace {index} needs nonnegative sizes and integer roles 0..3")
        result.append(array.tolist())  # pad_traces pads by list concatenation
    return result


def _probabilities(values: Any, rows: int, classes: int) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != (rows, classes):
        raise ValueError(f"Expected probability shape {(rows, classes)}, got {values.shape}")
    if (not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6)):
        raise ValueError("Model returned invalid probabilities")
    return values


class FoldPredictor:
    """Trace-level predictions from trained, two-head FOLD and its forest.

    The encoder and final training-split forest must match the FOLD model.
    ``batch_size`` controls encoding and prediction chunks. Each call recomputes
    all representations, including forest probabilities, from the supplied traces.

    Inputs are a single ``(L, 3)`` trace or a batch of variable-length traces.
    A single trace returns one row; ``[]`` returns zero rows. Prediction uses
    BatchNorm and dropout in inference mode.
    """

    def __init__(self, encoder: Encoder, forest: RandomForest, model: Any,
                 batch_size: int = 16):
        _validate_encoder(encoder)
        _validate_forest(encoder, forest)
        _validate_model(encoder, model)
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        self.encoder = encoder
        self.forest = forest
        self.model = model
        self.batch_size = batch_size

    def _forest_probs(self, batch: list) -> np.ndarray:
        # The forest wrapper inserts zero columns for unseen classes.
        return _probabilities(self.forest.predict_proba(self.encoder.forest(batch)),
                              len(batch), self.encoder.num_classes)

    def predict_forest_proba(self, traces: Any) -> np.ndarray:
        """Forest probabilities, shape ``(batch, num_classes)`` in label order."""
        batch = _traces(traces)
        if not batch:
            return np.empty((0, self.encoder.num_classes), dtype=np.float32)
        return np.concatenate([self._forest_probs(batch[i:i + self.batch_size])
                               for i in range(0, len(batch), self.batch_size)])

    def predict_proba(self, traces: Any) -> np.ndarray:
        """FOLD's ``multi_output``, shape ``(batch, num_classes)`` in label order."""
        batch = _traces(traces)
        if not batch:
            return np.empty((0, self.encoder.num_classes), dtype=np.float32)
        from fold.training import predict_sequence

        results = []
        for start in range(0, len(batch), self.batch_size):
            chunk = batch[start:start + self.batch_size]
            rf_probs = self._forest_probs(chunk)
            inputs = self.encoder.sequence(chunk, forest_probs=rf_probs)
            multi, _ = predict_sequence(self.model, inputs, self.encoder.num_classes,
                                        self.batch_size)
            results.append(_probabilities(multi, len(chunk), self.encoder.num_classes))
        return np.concatenate(results)

    def _target(self, target_class: int) -> int:
        if (isinstance(target_class, (bool, np.bool_))
                or not isinstance(target_class, (int, np.integer))
                or not 0 <= target_class < self.encoder.num_classes):
            raise ValueError("target_class must be a numeric class index in [0, num_classes)")
        return int(target_class)

    def predict_target(self, traces: Any, target_class: int) -> np.ndarray:
        """Probabilities of ``target_class``, a numeric label, shape ``(batch,)``."""
        target = self._target(target_class)
        return self.predict_proba(traces)[:, target]

    def for_class(self, target_class: int) -> Callable[[Any], np.ndarray]:
        """Bind a class index to a callable returning one probability per trace."""
        return partial(self.predict_target, target_class=self._target(target_class))
