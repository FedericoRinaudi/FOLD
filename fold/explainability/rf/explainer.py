"""Interventional TreeSHAP for the final forest's class probabilities."""
from importlib import metadata as distributions

import numpy as np

from ..core.artifacts import _class_names, _metadata, _validate_encoder, _validate_forest
from .features import forest_feature_names
from .results import ForestExplanation
from ..end_to_end.predictor import _probabilities, _traces


def _target_output(values, expected_value, shape, column, outputs):
    """Select a class from list-based or output-last TreeSHAP arrays."""
    if isinstance(values, list):
        if len(values) != outputs or any(np.asarray(v).shape != shape for v in values):
            raise ValueError("Unexpected list of TreeSHAP class outputs")
        selected = np.asarray(values[column])
    else:
        values = np.asarray(values)
        if values.shape == (*shape, outputs):
            selected = values[:, :, column]
        elif outputs == 1 and values.shape == shape:
            selected = values
        else:
            raise ValueError(f"Unsupported TreeSHAP output shape: {values.shape}")
    base = np.asarray(expected_value)
    if base.shape == (outputs,):
        base = base[column]
    elif not (outputs == 1 and base.ndim == 0):
        raise ValueError(f"Unsupported TreeSHAP base value shape: {base.shape}")
    return np.asarray(selected, dtype=np.float64), float(base)


class ForestTreeExplainer:
    """Explain one RF class using training traces as the reference distribution.

    ``background_ids`` identifies the supplied training traces. ``metadata``
    declares scenario and final-forest provenance as in the artifact bundle.
    An absent estimator class is rejected rather than assigned an attribution.
    Contributions describe the RF's derived feature space, not FOLD's output
    or necessarily realizable changes to a raw trace.

    sklearn TreeSHAP computes all fitted class outputs internally. Set
    ``allow_multiclass=True`` to accept that cost; only the selected class is
    retained, and samples are processed individually to bound output memory.
    """

    def __init__(self, encoder, forest, background_traces, *, background_ids,
                 target_class, class_names, metadata, allow_multiclass=False):
        _validate_encoder(encoder)
        _validate_forest(encoder, forest)
        self.class_names = _class_names(class_names, encoder.num_classes)
        self.metadata = _metadata(metadata)
        if (isinstance(target_class, (bool, np.bool_))
                or not isinstance(target_class, (int, np.integer))
                or not 0 <= target_class < encoder.num_classes):
            raise ValueError("target_class must be an integer in [0, num_classes)")
        columns = np.flatnonzero(forest.clf.classes_ == target_class)
        if len(columns) != 1:
            raise ValueError(f"Class {target_class} is absent from forest.clf.classes_")
        self.column = int(columns[0])
        self.outputs = len(forest.clf.classes_)
        if self.outputs > 1 and not allow_multiclass:
            raise ValueError("sklearn TreeSHAP computes all class outputs internally; "
                             "set allow_multiclass=True to accept this cost")
        self.encoder, self.forest = encoder, forest
        self.target_class = int(target_class)
        self.feature_names = forest_feature_names(encoder)
        background = _traces(background_traces)
        self.background_ids = tuple(background_ids)
        if (not background or len(background) != len(self.background_ids)
                or len(set(self.background_ids)) != len(background)
                or any(not isinstance(v, str) or not v for v in self.background_ids)):
            raise ValueError("Provide one unique trace ID per background trace")
        self.background = encoder.forest(background)
        if (self.background.shape != (len(background), len(self.feature_names))
                or not np.isfinite(self.background).all()):
            raise ValueError("Background features must be finite and match feature names")
        import shap

        # Preserve the requested reference set; Independent otherwise caps it at 100.
        reference = shap.maskers.Independent(self.background, max_samples=len(background))
        self.explainer = shap.TreeExplainer(
            forest.clf, data=reference, feature_perturbation="interventional",
            model_output="probability", feature_names=list(self.feature_names))
        reference_probs = _probabilities(forest.predict_proba(self.background),
                                         len(background), encoder.num_classes)
        self.background_probability = float(reference_probs[:, self.target_class]
                                             .mean(dtype=np.float64))

    def explain(self, traces, *, sample_ids) -> ForestExplanation:
        """Return signed RF attributions, with probability and additivity checks."""
        traces = _traces(traces)
        sample_ids = tuple(sample_ids)
        if (not traces or len(traces) != len(sample_ids)
                or len(set(sample_ids)) != len(sample_ids)
                or any(not isinstance(v, str) or not v for v in sample_ids)):
            raise ValueError("Provide one unique trace ID per explained trace")
        if set(sample_ids) & set(self.background_ids):
            raise ValueError("Explained test IDs overlap the training background")
        features = self.encoder.forest(traces)
        if (features.shape != (len(traces), len(self.feature_names))
                or not np.isfinite(features).all()):
            raise ValueError("Encoded features must be finite and match feature names")
        probabilities = _probabilities(self.forest.predict_proba(features), len(traces),
                                        self.encoder.num_classes)[:, self.target_class]
        direct = self.forest.clf.predict_proba(features)[:, self.column]
        if not np.allclose(probabilities, direct, rtol=1e-6, atol=1e-7):
            raise ValueError("RF wrapper and estimator class probabilities differ")
        selected = []
        for row in features:
            values = self.explainer.shap_values(row[None, :], check_additivity=False)
            attribution, base = _target_output(
                values, self.explainer.expected_value, (1, features.shape[1]),
                self.column, self.outputs)
            if not np.isclose(base, self.background_probability, rtol=1e-5, atol=1e-6):
                raise ValueError("TreeSHAP base differs from mean background RF probability")
            selected.append(attribution[0])
        metadata = {"experiment": self.metadata,
                    "explainer": "TreeExplainer", "feature_perturbation": "interventional",
                    "model_output": "probability", "class_column": self.column,
                    "estimator_classes": self.forest.clf.classes_.astype(int).tolist(),
                    "size_edges": self.encoder.size_edges.tolist(),
                    "internal_class_outputs": self.outputs,
                    "versions": {name: distributions.version(name)
                                 for name in ("shap", "numpy", "scikit-learn")},
                    "additivity_rtol": 1e-4, "additivity_atol": 1e-5}
        return ForestExplanation(np.asarray(selected), base, probabilities, features,
                                 sample_ids, self.background_ids, self.target_class,
                                 self.class_names, self.feature_names, metadata)
