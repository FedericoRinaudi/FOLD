"""Explicit two-population selection from recorded open-world splits."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from fold.data import load_traces
from ..core.artifacts import _digest, _metadata
from ..core.configuration import validate_split
from ..end_to_end.explainer import resolution_edges, select_references


POPULATIONS = ("monitored", "background")


def validate_open_world(metadata):
    """Require the existing versioned provenance, without inferring any split."""
    if not isinstance(metadata, dict) or metadata.get("scenario") != "open-world":
        raise ValueError("Binary explanations require an open-world bundle")
    _metadata(metadata)
    return {"monitored": validate_split(metadata.get("split_ids")),
            "background": validate_split(metadata.get("background_split_ids"))}


def qualified_ids(ids, populations):
    """Keep source namespaces disjoint even when raw IDs have the same text."""
    if len(ids) != len(populations):
        raise ValueError("Expected one population per trace ID")
    result = []
    for tid, population in zip(ids, populations):
        if population not in POPULATIONS or not isinstance(tid, str) or not tid:
            raise ValueError("Expected nonempty string IDs and monitored/background populations")
        result.append(f"{population}:{tid}")
    if len(set(result)) != len(result):
        raise ValueError("Duplicate IDs within a population")
    return tuple(result)


def _read_ids(path, kind):
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    keys = {f"{population}_{kind}_ids" for population in POPULATIONS}
    if (not isinstance(value, dict) or value.get("output") != "binary_output"
            or set(value) - keys - {"output"}):
        raise ValueError(f"Expected binary_output and population-specific {kind} IDs")
    result = {}
    for population in POPULATIONS:
        ids = value.get(f"{population}_{kind}_ids", [])
        if (not isinstance(ids, list) or any(not isinstance(tid, str) or not tid for tid in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError(f"{population} {kind} IDs must be unique nonempty strings")
        result[population] = ids
    if not any(result.values()):
        raise ValueError(f"Select at least one {kind}")
    return result


def load_background_pool(path):
    """Reconstruct exactly the pool ordering/types used by draw_background.

    pandas inference and str(group key) deliberately match experiments.py;
    the generated CSV uses integer trace_id and zero-based packet indices.
    No pool is generated or unpickled here.
    """
    import pandas as pd

    frame = pd.read_csv(path, comment="#")
    required = {"trace_id", "pkt_idx", "t", "size", "role"}
    if not required <= set(frame.columns) or frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Background pool requires trace_id, pkt_idx, t, size, role without missing values")
    if frame.duplicated(["trace_id", "pkt_idx"]).any():
        raise ValueError("Duplicate background packet indices")
    result = {}
    for tid, group in frame.sort_values(["trace_id", "pkt_idx"]).groupby("trace_id", sort=True):
        indices = group["pkt_idx"].to_numpy()
        if not np.array_equal(indices, np.arange(len(group))):
            raise ValueError("Background pkt_idx must be consecutive and zero-based")
        key = str(tid)
        if not key or key in result:
            raise ValueError("Ambiguous background trace IDs")
        trace = [[float(t), float(s), float(r)]
                 for t, s, r in zip(group["t"], group["size"], group["role"])]
        resolution_edges(trace)  # Validate complete cycles, preserving all packet values.
        result[key] = trace
    return result


@dataclass
class OpenWorldSelection:
    samples: list
    references: list
    sample_ids: tuple
    sample_types: tuple
    reference_ids: tuple
    reference_types: tuple
    metadata: dict


def select_open_world(dataset, background_pool, metadata, *, sample_ids_file,
                      num_groups=8, reference_size=16, seed=0, reference_ids_file=None):
    """Select explicit test cases and balanced eligible training references.

    Order is monitored first, then background; order within each population is
    preserved. An odd reference size gives the extra slot to monitored. At
    least two references are required so both populations are represented.
    Each population uses the existing reference sampler with the same seed.
    """
    if type(reference_size) is not int or reference_size < 2:
        raise ValueError("reference_size must be >= 2 to represent both populations")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    splits = validate_open_world(metadata)
    paths = {"monitored": Path(dataset), "background": Path(background_pool)}
    hashes = {population: _digest(path) for population, path in paths.items()}
    for population in POPULATIONS:
        if hashes[population] != splits[population]["dataset_sha256"]:
            raise ValueError(f"{population} dataset/pool hash differs from the recorded split")
    if metadata.get("dataset_sha256", hashes["monitored"]) != hashes["monitored"]:
        raise ValueError("Monitored dataset hash differs from bundle metadata")
    selected = _read_ids(sample_ids_file, "sample")
    explicit = _read_ids(reference_ids_file, "reference") if reference_ids_file else None
    for population in POPULATIONS:
        if not set(selected[population]) <= set(splits[population]["test"]):
            raise ValueError(f"{population} sample IDs must belong to its recorded test split")
        if explicit is not None and not set(explicit[population]) <= set(splits[population]["train"]):
            raise ValueError(f"{population} reference IDs must belong to its recorded training split")
    traces, _, _, ids = load_traces(dataset, return_ids=True)
    pools = {"monitored": dict(zip(ids, traces)), "background": load_background_pool(background_pool)}
    if len(pools["monitored"]) != len(ids):
        raise ValueError("Duplicate monitored trace IDs")
    for population in POPULATIONS:
        recorded = set().union(*(splits[population][part] for part in ("train", "val", "test")))
        if not recorded <= set(pools[population]):
            raise ValueError(f"Recorded {population} IDs absent from source file")
    samples = [pools[p][tid] for p in POPULATIONS for tid in selected[p]]
    quotas = {"monitored": (reference_size + 1) // 2, "background": reference_size // 2}
    references, chosen, eligible = [], {}, {}
    for population in POPULATIONS:
        train_ids = splits[population]["train"]
        try:
            drawn, drawn_ids, info = select_references(
                samples, [pools[population][tid] for tid in train_ids], train_ids,
                num_groups=num_groups, reference_size=quotas[population], seed=seed)
        except ValueError as error:
            raise ValueError(f"{population} references: {error}") from error
        eligible[population] = info["eligible_reference_count"]
        if explicit is not None:
            drawn_ids = explicit[population]
            if len(drawn_ids) != quotas[population]:
                raise ValueError(f"Explicit {population} references must match balanced quota {quotas[population]}")
            drawn = [pools[population][tid] for tid in drawn_ids]
            if any(len(resolution_edges(t)) - 1 < info["effective_num_groups"] for t in drawn):
                raise ValueError(f"Explicit {population} reference is shorter than effective groups")
        references.extend(drawn)
        chosen[population] = drawn_ids
    selection = {
        "output": "binary_output", "positive_class": "monitored",
        "sample_order": "monitored_then_background_preserving_file_order",
        "reference_policy": "balanced_extra_monitored_if_odd",
        "reference_selection": "explicit_ids" if explicit is not None else "uniform_without_replacement",
        "reference_rng": "independent_population_samplers_same_seed",
        "sampling_seed": seed, "reference_size": reference_size,
        "requested_num_groups": num_groups, "effective_num_groups": info["effective_num_groups"],
        "eligible_reference_counts": eligible,
        "dataset_sha256": hashes["monitored"], "background_pool_sha256": hashes["background"],
        "sample_ids_file_sha256": _digest(Path(sample_ids_file)),
        "reference_ids_file_sha256": _digest(Path(reference_ids_file)) if reference_ids_file else None,
        **{f"{p}_sample_ids": selected[p] for p in POPULATIONS},
        **{f"{p}_reference_ids": chosen[p] for p in POPULATIONS},
    }
    return OpenWorldSelection(
        samples, references,
        tuple(tid for p in POPULATIONS for tid in selected[p]),
        tuple(p for p in POPULATIONS for _ in selected[p]),
        tuple(tid for p in POPULATIONS for tid in chosen[p]),
        tuple(p for p in POPULATIONS for _ in chosen[p]),
        {**metadata, "selection": selection})
