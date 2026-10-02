"""Exact Shapley values for consecutive groups of complete ODoH resolutions."""
from math import comb

import numpy as np

from fold.data import CLIENT_QUERY, PROXY_QUERY, PROXY_RESPONSE, TARGET_RESPONSE
from ..core.data import _ids
from .results import FoldExplanation
from .predictor import _traces


def resolution_edges(trace):
    """Locate loader cycles, including consecutive fragments/dummies of each role."""
    trace = _traces([trace])[0]
    order = (PROXY_RESPONSE, CLIENT_QUERY, PROXY_QUERY, TARGET_RESPONSE)
    roles = [int(packet[2]) for packet in trace]
    if roles[0] != order[0] or roles[-1] != order[-1]:
        raise ValueError("Trace must contain complete ODoH resolutions in loader role order")
    edges = [0]
    for i in range(1, len(roles)):
        previous, current = roles[i - 1], roles[i]
        if current != previous:
            if current != order[(order.index(previous) + 1) % len(order)]:
                raise ValueError("Invalid ODoH role order; cannot recover resolution boundaries")
            if current == order[0]:
                edges.append(i)
    return np.asarray([*edges, len(trace)], dtype=np.int64)


def group_bounds(edges, count):
    """Balanced consecutive resolution groups; larger groups come first."""
    units = len(edges) - 1
    if not 1 <= count <= units:
        raise ValueError("Group count must be positive and cannot exceed resolution count")
    width, extra = divmod(units, count)
    boundaries = np.r_[0, np.cumsum([width + (i < extra) for i in range(count)])]
    unit_bounds = np.column_stack([boundaries[:-1], boundaries[1:]])
    return np.asarray(edges)[unit_bounds], unit_bounds


def select_references(samples, training, training_ids, *, num_groups, reference_size, seed):
    """Fix run granularity from samples, then sample eligible training traces."""
    if type(num_groups) is not int or not 1 <= num_groups <= 8:
        raise ValueError("Exact explanations support num_groups from 1 to 8")
    count = min(num_groups, min(len(resolution_edges(t)) - 1 for t in samples))
    eligible = [i for i, trace in enumerate(training) if len(resolution_edges(trace)) - 1 >= count]
    if len(eligible) < reference_size:
        raise ValueError(f"Insufficient eligible references: g_run={count}, "
                         f"requested={reference_size}, eligible={len(eligible)}")
    chosen = np.random.default_rng(seed).choice(eligible, reference_size, replace=False)
    return ([training[i] for i in chosen], [training_ids[i] for i in chosen],
            {"requested_num_groups": num_groups, "effective_num_groups": count,
             "eligible_reference_count": len(eligible)})


def hybrid_trace(trace, reference, bounds, reference_bounds, mask):
    """Join present sample blocks and absent reference blocks in ordinal order."""
    if len(bounds) != len(reference_bounds) or not 0 <= mask < 2**len(bounds):
        raise ValueError("Invalid coalition or mismatched group counts")
    result = []
    for group, (sample_span, reference_span) in enumerate(zip(bounds, reference_bounds)):
        source, span = (trace, sample_span) if mask & (1 << group) else (reference, reference_span)
        result.extend([list(packet) for packet in source[int(span[0]):int(span[1])]])
    return result


def shapley_values(coalitions):
    """Exact weighted marginal differences on a binary-indexed coalition table."""
    coalitions = np.asarray(coalitions, dtype=np.float64)
    count = len(coalitions).bit_length() - 1
    if count < 1 or coalitions.shape != (2**count,) or not np.isfinite(coalitions).all():
        raise ValueError("Expected a finite value for every coalition")
    values = np.zeros(count)
    for group in range(count):
        for mask in range(2**count):
            if not mask & (1 << group):
                values[group] += (coalitions[mask | (1 << group)] - coalitions[mask]) / (
                    count * comb(count - 1, mask.bit_count()))
    return values


class FoldGroupExplainer:
    """Explain mean FOLD probability over same-reference hybrid coalitions.

    ``predictor`` is the existing FoldPredictor; tests may supply a scalar
    probability predictor with the same predict_target interface.
    """

    def __init__(self, predictor, references, *, reference_ids, target_class,
                 class_names, metadata, eligible_reference_count, num_groups=8, seed=0):
        if type(num_groups) is not int or not 1 <= num_groups <= 8:
            raise ValueError("Exact explanations support num_groups from 1 to 8")
        if type(target_class) is not int or not 0 <= target_class < len(class_names):
            raise ValueError("target_class must index class_names")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be nonnegative")
        self.reference_ids = tuple(_ids(list(reference_ids), "reference_ids"))
        self.references = _traces(references)
        if type(eligible_reference_count) is not int or eligible_reference_count < len(self.references):
            raise ValueError("Eligible reference count cannot be smaller than the selected set")
        self.eligible_reference_count = eligible_reference_count
        if len(self.references) != len(self.reference_ids):
            raise ValueError("Provide one ID per reference trace")
        split = metadata.get("split_ids", {})
        train = _ids(split.get("train"), "train")
        test = _ids(split.get("test"), "test")
        if set(train) & set(test) or not set(self.reference_ids) <= set(train):
            raise ValueError("Reference IDs must belong exclusively to the recorded training split")
        self.test_ids = set(test)
        self.predictor, self.target_class = predictor, target_class
        self.class_names, self.metadata = tuple(class_names), metadata
        self.num_groups, self.seed = num_groups, seed
        self.reference_edges = [resolution_edges(trace) for trace in self.references]

    def _predict(self, traces):
        values = np.asarray(self.predictor.predict_target(traces, self.target_class), dtype=np.float64)
        if (values.shape != (len(traces),) or not np.isfinite(values).all()
                or np.any((values < 0) | (values > 1))):
            raise ValueError("FOLD must return one finite target probability per trace")
        return values

    def explain(self, traces, *, sample_ids):
        traces = _traces(traces)
        sample_ids = tuple(_ids(list(sample_ids), "sample_ids"))
        if len(traces) != len(sample_ids) or not set(sample_ids) <= self.test_ids:
            raise ValueError("Provide one recorded test ID per sample trace")
        edges = [resolution_edges(trace) for trace in traces]
        shape = (len(traces), self.num_groups)
        values = np.zeros(shape)
        bounds = np.full((*shape, 2), -1, dtype=np.int64)
        unit_bounds = bounds.copy()
        ref_bounds = np.full((len(traces), len(self.references), self.num_groups, 2), -1, dtype=np.int64)
        count = min(self.num_groups, min(len(e) - 1 for e in edges))
        if any(len(r) - 1 < count for r in self.reference_edges):
            raise ValueError(f"Selected reference is shorter than g_run={count}; filter before sampling")
        counts = np.full(len(traces), count, dtype=np.int64)
        base = float(self._predict(self.references).mean())
        probabilities = self._predict(traces)
        for row, (trace, sample_edges, count) in enumerate(zip(traces, edges, counts)):
            count = int(count)
            bounds[row, :count], unit_bounds[row, :count] = group_bounds(sample_edges, count)
            for r, reference_edges in enumerate(self.reference_edges):
                ref_bounds[row, r, :count] = group_bounds(reference_edges, count)[0]
            coalitions = np.empty(2**count)
            coalitions[0], coalitions[-1] = base, probabilities[row]
            for mask in range(1, 2**count - 1):
                hybrids = [hybrid_trace(trace, reference, bounds[row, :count],
                                       ref_bounds[row, r, :count], mask)
                           for r, reference in enumerate(self.references)]
                coalitions[mask] = self._predict(hybrids).mean()
            values[row, :count] = shapley_values(coalitions)
        metadata = {"experiment": self.metadata, "algorithm": "exact_group_shapley_v1",
                    "game": "mean_reference_hybrid_target_probability",
                    "grouping": "balanced_resolution_blocks_larger_first",
                    "group_count": "min(requested, shortest_selected_sample_units)",
                    "requested_num_groups": self.num_groups,
                    "effective_num_groups": int(count),
                    "eligible_reference_count": self.eligible_reference_count,
                    "replacement": "same_ordinal_block_same_reference_per_coalition",
                    "bounds_unit": "zero_based_half_open_packet_and_resolution_indices",
                    "reference_lengths": [len(t) for t in self.references],
                    "sample_lengths": [len(t) for t in traces],
                    "versions": {"numpy": np.__version__}, "shap_library_used": False,
                    "additivity_rtol": 1e-5, "additivity_atol": 1e-6}
        return FoldExplanation(values, np.full(len(traces), base), probabilities,
                               sample_ids, self.reference_ids, self.target_class,
                               self.class_names[self.target_class], counts, bounds, unit_bounds,
                               ref_bounds, self.seed, metadata)
