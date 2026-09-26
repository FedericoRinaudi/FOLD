"""Per-class forest attributions, independent of models and plotting libraries."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class ForestExplanation:
    """TreeSHAP values in RF probability units for one fixed class.

    ``values`` and ``features`` have shape (samples, features). ``base_value``
    is the mean target probability on the background; ``probabilities`` has
    one entry per explained sample. IDs identify CSV traces in disjoint splits.
    """

    values: np.ndarray
    base_value: float
    probabilities: np.ndarray
    features: np.ndarray
    sample_ids: tuple[str, ...]
    background_ids: tuple[str, ...]
    target_class: int
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    metadata: dict

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=np.float64)
        self.features = np.asarray(self.features)
        self.probabilities = np.asarray(self.probabilities, dtype=np.float64)
        self.base_value = float(self.base_value)
        for name in ("sample_ids", "background_ids", "class_names", "feature_names"):
            items = tuple(getattr(self, name))
            if (not items or any(not isinstance(v, str) or not v for v in items)
                    or len(set(items)) != len(items)):
                raise ValueError(f"{name} must contain unique nonempty strings")
            setattr(self, name, items)
        shape = (len(self.sample_ids), len(self.feature_names))
        if self.values.shape != shape or self.features.shape != shape:
            raise ValueError("Attributions and feature values must match sample/feature IDs")
        if self.probabilities.shape != (shape[0],):
            raise ValueError("Expected one RF probability per sample")
        if (not np.isfinite(self.values).all() or not np.isfinite(self.features).all()
                or not np.isfinite(self.probabilities).all()
                or not 0 <= self.base_value <= 1
                or np.any((self.probabilities < 0) | (self.probabilities > 1))):
            raise ValueError("Attributions, features and probabilities must be finite and valid")
        if (isinstance(self.target_class, (bool, np.bool_))
                or not isinstance(self.target_class, (int, np.integer))
                or not 0 <= self.target_class < len(self.class_names)):
            raise ValueError("target_class must index class_names")
        self.target_class = int(self.target_class)
        if set(self.sample_ids) & set(self.background_ids):
            raise ValueError("Background and explained sample IDs must be disjoint")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary")
        json.dumps(self.metadata, allow_nan=False)
        self.check_additivity()

    def check_additivity(self, rtol=1e-4, atol=1e-5) -> None:
        """Check base plus signed contributions against the RF probability."""
        reconstructed = self.base_value + self.values.sum(axis=1)
        if not np.allclose(reconstructed, self.probabilities, rtol=rtol, atol=atol):
            error = float(np.max(np.abs(reconstructed - self.probabilities)))
            raise ValueError(f"RF SHAP additivity failed; maximum absolute error {error:g}")

    def save(self, path) -> None:
        """Write numeric arrays and JSON metadata to a new NPZ; no model is stored."""
        self.__post_init__()
        header = {"format": "fold-rf-shap", "version": 1,
                  "model": "forest", "output": "probability",
                  "target_class": self.target_class, "metadata": self.metadata}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(
                stream, values=self.values, base_value=self.base_value,
                probabilities=self.probabilities, features=self.features,
                sample_ids=np.asarray(self.sample_ids),
                background_ids=np.asarray(self.background_ids),
                class_names=np.asarray(self.class_names),
                feature_names=np.asarray(self.feature_names),
                header=json.dumps(header, allow_nan=False))

    @classmethod
    def load(cls, path) -> "ForestExplanation":
        """Read saved attributions without loading RF, TensorFlow or SHAP."""
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(data["header"].item())
            if (header.get("format") != "fold-rf-shap" or header.get("version") != 1
                    or header.get("model") != "forest" or header.get("output") != "probability"):
                raise ValueError("Unsupported forest attribution format")
            return cls(values=data["values"], base_value=data["base_value"].item(),
                       probabilities=data["probabilities"], features=data["features"],
                       sample_ids=tuple(data["sample_ids"].tolist()),
                       background_ids=tuple(data["background_ids"].tolist()),
                       target_class=header["target_class"],
                       class_names=tuple(data["class_names"].tolist()),
                       feature_names=tuple(data["feature_names"].tolist()),
                       metadata=header["metadata"])
