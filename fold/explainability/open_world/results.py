"""Safe, explicitly binary storage with population-aware trace identifiers."""
from dataclasses import dataclass
import json
from pathlib import Path
from typing import ClassVar

import numpy as np

from ..end_to_end.results import FoldExplanation
from .data import POPULATIONS, qualified_ids, validate_open_world


@dataclass
class BinaryExplanation:
    values: np.ndarray
    base_values: np.ndarray
    probabilities: np.ndarray
    sample_ids: tuple
    sample_types: tuple
    reference_ids: tuple
    reference_types: tuple
    group_counts: np.ndarray
    bounds: np.ndarray
    unit_bounds: np.ndarray
    reference_bounds: np.ndarray
    seed: int
    metadata: dict

    output: ClassVar[str] = "binary_output"
    positive_class: ClassVar[str] = "monitored"

    def _grouped(self):
        """Validation-only view; never serialize it as multiclass FOLD output."""
        return FoldExplanation(
            self.values, self.base_values, self.probabilities,
            qualified_ids(self.sample_ids, self.sample_types),
            qualified_ids(self.reference_ids, self.reference_types),
            0, "monitored", self.group_counts, self.bounds, self.unit_bounds,
            self.reference_bounds, self.seed, self.metadata)

    def __post_init__(self):
        for name in ("sample_ids", "sample_types", "reference_ids", "reference_types"):
            setattr(self, name, tuple(getattr(self, name)))
        checked = self._grouped()  # Reuse bounds, padding, finite values and additivity checks.
        for name in ("values", "base_values", "probabilities", "group_counts",
                     "bounds", "unit_bounds", "reference_bounds"):
            setattr(self, name, getattr(checked, name))
        experiment = self.metadata.get("experiment", {})
        splits = validate_open_world(experiment)
        selection = experiment.get("selection", {})
        for kind, ids, populations, split_name in (
                ("sample", self.sample_ids, self.sample_types, "test"),
                ("reference", self.reference_ids, self.reference_types, "train")):
            for population in POPULATIONS:
                selected = [tid for tid, p in zip(ids, populations) if p == population]
                if not set(selected) <= set(splits[population][split_name]):
                    raise ValueError(f"{population} {kind} IDs outside recorded {split_name} split")
                if selection.get(f"{population}_{kind}_ids") != selected:
                    raise ValueError(f"{population} {kind} IDs differ from selection provenance")
        quotas = {"monitored": (len(self.reference_ids) + 1) // 2,
                  "background": len(self.reference_ids) // 2}
        if len(self.reference_ids) < 2 or any(self.reference_types.count(p) != n for p, n in quotas.items()):
            raise ValueError("References must balance both populations, with odd extra monitored")
        eligible = selection.get("eligible_reference_counts", {})
        if any(type(eligible.get(p)) is not int or eligible[p] < n for p, n in quotas.items()):
            raise ValueError("Invalid eligible reference counts")
        if (selection.get("output") != self.output or selection.get("positive_class") != self.positive_class
                or selection.get("sampling_seed") != self.seed
                or selection.get("requested_num_groups") != self.values.shape[1]
                or np.any(self.group_counts != selection.get("effective_num_groups"))):
            raise ValueError("Binary output, seed or group counts differ from selection provenance")

    def check_additivity(self, rtol=1e-5, atol=1e-6):
        FoldExplanation.check_additivity(self, rtol=rtol, atol=atol)

    def save(self, path):
        self.__post_init__()
        header = {"format": "fold-open-world-binary-shap", "version": 1,
                  "output": self.output, "positive_class": self.positive_class,
                  "num_groups": self.values.shape[1], "seed": self.seed, "metadata": self.metadata}
        arrays = {name: np.asarray(getattr(self, name)) for name in (
            "values", "base_values", "probabilities", "group_counts", "bounds", "unit_bounds",
            "reference_bounds", "sample_ids", "sample_types", "reference_ids", "reference_types")}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays, header=json.dumps(header, allow_nan=False))

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            header = json.loads(data["header"].item())
            if (header.get("format") != "fold-open-world-binary-shap" or header.get("version") != 1
                    or header.get("output") != cls.output or header.get("positive_class") != cls.positive_class):
                raise ValueError("Unsupported open-world binary attribution format")
            arrays = {name: data[name] for name in (
                "values", "base_values", "probabilities", "group_counts", "bounds", "unit_bounds",
                "reference_bounds", "sample_ids", "sample_types", "reference_ids", "reference_types")}
            if arrays["values"].ndim != 2 or header.get("num_groups") != arrays["values"].shape[1]:
                raise ValueError("Recorded group count differs from values")
            for name in ("sample_ids", "sample_types", "reference_ids", "reference_types"):
                if arrays[name].ndim != 1 or arrays[name].dtype.kind != "U":
                    raise ValueError("Expected one-dimensional Unicode IDs and populations")
                arrays[name] = tuple(arrays[name].tolist())
            return cls(**arrays, seed=header["seed"], metadata=header["metadata"])
