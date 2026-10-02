"""Trace-level probabilities of the original FOLD monitored/background head."""
import numpy as np

from ..end_to_end.predictor import FoldPredictor, _traces
from .data import validate_open_world


class BinaryFoldPredictor:
    """Predict P(monitored), recomputing RF inputs for every supplied trace.

    Construction uses an already loaded open-world ArtifactBundle. The composed
    FoldPredictor validates the original two-head architecture (not N+1).
    """

    def __init__(self, bundle, batch_size=16):
        validate_open_world(bundle.metadata)
        self.fold = FoldPredictor(bundle.encoder, bundle.forest, bundle.model,
                                  batch_size=batch_size)

    def predict_monitored(self, traces):
        """Return a finite probability vector of shape (number of traces,)."""
        batch = _traces(traces)
        if not batch:
            return np.empty(0, dtype=np.float64)
        results = []
        for start in range(0, len(batch), self.fold.batch_size):
            chunk = batch[start:start + self.fold.batch_size]
            rf_probs = self.fold._forest_probs(chunk)
            inputs = self.fold.encoder.sequence(chunk, forest_probs=rf_probs)
            # Same model.predict call as training.predict_sequence, retaining
            # the (N, 1) shape here so malformed binary outputs are rejected.
            outputs = self.fold.model.predict(inputs, batch_size=self.fold.batch_size, verbose=0)
            if not isinstance(outputs, dict) or "binary_output" not in outputs or "n1_output" in outputs:
                raise ValueError("Expected original two-head FOLD binary_output")
            probabilities = np.asarray(outputs["binary_output"], dtype=np.float64)
            if (probabilities.shape != (len(chunk), 1) or not np.isfinite(probabilities).all()
                    or np.any((probabilities < 0) | (probabilities > 1))):
                raise ValueError("binary_output must contain finite (N, 1) probabilities in [0, 1]")
            results.append(probabilities[:, 0])
        return np.concatenate(results)

    def predict_target(self, traces, target_class):
        """Adapter for FoldGroupExplainer; slot zero denotes P(monitored)."""
        if type(target_class) is not int or target_class != 0:
            raise ValueError("The binary adapter only explains P(monitored), slot 0")
        return self.predict_monitored(traces)
