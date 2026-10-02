"""Safe numeric storage for grouped FOLD probability explanations."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass
class FoldExplanation:
    """Zero-padded group values, with -1 bounds outside each sample's groups."""

    values: np.ndarray
    base_values: np.ndarray
    probabilities: np.ndarray
    sample_ids: tuple[str, ...]
    reference_ids: tuple[str, ...]
    target_class: int
    class_name: str
    group_counts: np.ndarray
    bounds: np.ndarray
    unit_bounds: np.ndarray
    reference_bounds: np.ndarray
    seed: int
    metadata: dict

    def __post_init__(self):
        for name in ("sample_ids", "reference_ids"):
            items = tuple(getattr(self, name))
            if (not items or any(not isinstance(v, str) or not v for v in items)
                    or len(set(items)) != len(items)):
                raise ValueError(f"{name} must contain unique nonempty strings")
            setattr(self, name, items)
        if set(self.sample_ids) & set(self.reference_ids):
            raise ValueError("Sample and reference IDs must be disjoint")
        if (type(self.target_class) is not int or self.target_class < 0
                or type(self.seed) is not int or self.seed < 0
                or not isinstance(self.class_name, str) or not self.class_name.strip()):
            raise ValueError("Invalid target class, class name or seed")
        for name in ("values", "base_values", "probabilities"):
            value = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} must be finite")
            setattr(self, name, value)
        for name in ("group_counts", "bounds", "unit_bounds", "reference_bounds"):
            value = np.asarray(getattr(self, name))
            if value.dtype.kind not in "iu":
                raise ValueError(f"{name} must contain integer indices")
            setattr(self, name, value)
        rows = len(self.sample_ids)
        if self.values.ndim != 2 or self.values.shape[0] != rows or not 1 <= self.values.shape[1] <= 8:
            raise ValueError("Expected sample-by-group values with 1..8 columns")
        groups = self.values.shape[1]
        if (self.base_values.shape != (rows,) or self.probabilities.shape != (rows,)
                or np.any((self.base_values < 0) | (self.base_values > 1))
                or np.any((self.probabilities < 0) | (self.probabilities > 1))):
            raise ValueError("Expected one base and original probability per sample")
        if (self.group_counts.shape != (rows,) or np.any(self.group_counts < 1)
                or np.any(self.group_counts > groups)
                or self.bounds.shape != (rows, groups, 2)
                or self.unit_bounds.shape != self.bounds.shape
                or self.reference_bounds.shape != (rows, len(self.reference_ids), groups, 2)):
            raise ValueError("Group counts and bounds do not match attribution dimensions")
        for row, count in enumerate(self.group_counts):
            if np.any(self.values[row, count:] != 0):
                raise ValueError("Inactive group values must be zero")
            for spans in (self.bounds[row], self.unit_bounds[row], *self.reference_bounds[row]):
                active = spans[:count]
                if (active[0, 0] != 0 or np.any(active[:, 1] <= active[:, 0])
                        or np.any(active[1:, 0] != active[:-1, 1])
                        or np.any(spans[count:] != -1)):
                    raise ValueError("Bounds must be consecutive nonempty blocks with -1 padding")
        if not isinstance(self.metadata, dict):
            raise ValueError("metadata must be a dictionary")
        json.dumps(self.metadata, allow_nan=False)
        self.check_additivity()

    def check_additivity(self, rtol=1e-5, atol=1e-6):
        reconstructed = self.base_values + self.values.sum(axis=1)
        if not np.allclose(reconstructed, self.probabilities, rtol=rtol, atol=atol):
            error = float(np.max(np.abs(reconstructed - self.probabilities)))
            raise ValueError(f"FOLD group SHAP additivity failed; maximum absolute error {error:g}")

    def save(self, path):
        self.__post_init__()
        header = {"format": "fold-group-shap", "version": 1, "output": "multi_output",
                  "target_class": self.target_class, "class_name": self.class_name,
                  "num_groups": self.values.shape[1], "seed": self.seed, "metadata": self.metadata}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, values=self.values, base_values=self.base_values,
                                probabilities=self.probabilities, group_counts=self.group_counts,
                                bounds=self.bounds, unit_bounds=self.unit_bounds,
                                reference_bounds=self.reference_bounds,
                                sample_ids=np.asarray(self.sample_ids),
                                reference_ids=np.asarray(self.reference_ids),
                                header=json.dumps(header, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(data["header"].item())
            if (header.get("format") != "fold-group-shap" or header.get("version") != 1
                    or header.get("output") != "multi_output"):
                raise ValueError("Unsupported FOLD attribution format")
            values = data["values"]
            if values.ndim != 2 or header.get("num_groups") != values.shape[1]:
                raise ValueError("Recorded group count differs from values")
            return cls(values, data["base_values"], data["probabilities"],
                       tuple(data["sample_ids"].tolist()), tuple(data["reference_ids"].tolist()),
                       header["target_class"], header["class_name"], data["group_counts"],
                       data["bounds"], data["unit_bounds"], data["reference_bounds"],
                       header["seed"], header["metadata"])
