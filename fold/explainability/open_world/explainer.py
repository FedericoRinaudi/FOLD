"""Composition adapter: the existing exact game, evaluated on P(monitored)."""
from ..end_to_end.explainer import FoldGroupExplainer
from .data import POPULATIONS, qualified_ids, validate_open_world
from .results import BinaryExplanation


class BinaryGroupExplainer:
    """Use an OpenWorldSelection without altering either original split."""

    def __init__(self, predictor, selection):
        self.selection = selection
        splits = validate_open_world(selection.metadata)
        info = selection.metadata["selection"]
        # The generic engine validates one ID namespace. Qualify both original
        # namespaces only in this transient adapter; retain the real manifest.
        joint = {part: [f"{p}:{tid}" for p in POPULATIONS for tid in splits[p][part]]
                 for part in ("train", "val", "test")}
        self.engine = FoldGroupExplainer(
            predictor, selection.references,
            reference_ids=qualified_ids(selection.reference_ids, selection.reference_types),
            target_class=0, class_names=("monitored",), metadata={"split_ids": joint},
            eligible_reference_count=sum(info["eligible_reference_counts"].values()),
            num_groups=info["requested_num_groups"], seed=info["sampling_seed"])

    def explain(self):
        selection = self.selection
        grouped = self.engine.explain(
            selection.samples, sample_ids=qualified_ids(selection.sample_ids, selection.sample_types))
        metadata = {**grouped.metadata, "experiment": selection.metadata,
                    "output": "binary_output", "positive_class": "monitored",
                    "game": "mean_reference_hybrid_monitored_probability"}
        return BinaryExplanation(
            grouped.values, grouped.base_values, grouped.probabilities,
            selection.sample_ids, selection.sample_types, selection.reference_ids, selection.reference_types,
            grouped.group_counts, grouped.bounds, grouped.unit_bounds, grouped.reference_bounds,
            grouped.seed, metadata)
