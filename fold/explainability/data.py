"""Selecting traces from explicit training and test splits.

Split records contain ``dataset_sha256``, ``train`` and ``test`` (lists of CSV
trace_id strings), and optionally ``val``. They can be supplied as JSON or in
bundle metadata under ``split_ids``. The dataset hash binds IDs to file contents.
The records must describe the split used to fit the bundle's final forest.
The experiment runner's ``load_and_split`` imports TensorFlow; explicit IDs
allow RF-only analysis without reimplementing its balancing and permutation.
"""
import json
from pathlib import Path
import platform
import random

import numpy as np

from fold.data import apply_defense, load_traces
from .artifacts import _digest


def _ids(values, name):
    if (not isinstance(values, list) or not values
            or any(not isinstance(v, str) or not v for v in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{name} must contain unique, nonempty trace_id strings")
    return values


def _replay_defense(traces, index, split, metadata):
    """Replay the original defense in run order, preserving the caller's RNG."""
    from fold import data

    replay = metadata.get("defense_replay", {})
    if (replay.get("order") != ["train", "val", "test"]
            or replay.get("source_sha256") != _digest(Path(data.__file__))
            or replay.get("python_version", "").split(".")[:2] != platform.python_version().split(".")[:2]):
        raise ValueError("Defense replay requires matching source, Python and recorded order")

    def tuples(value):
        return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value

    state = random.getstate()
    result = {}
    try:
        random.setstate(tuples(replay["python_random_state"]))
        for name in replay["order"]:
            ids = split[name]
            defended = apply_defense([traces[index[tid]] for tid in ids], metadata["defense"])
            result.update(zip(ids, defended))
    finally:
        random.setstate(state)
    return result


def select_traces(dataset, metadata, *, split_file=None, seed=0,
                  background_size=32, max_samples=None, sample_ids_file=None,
                  target_class=None, reference_num_groups=None):
    """Sample background from training and explanation cases from test, without fit.

    ``seed`` controls sampling within the supplied split, not split construction.
    Explicit sample IDs retain file order; max_samples optionally limits their prefix.
    Without a selection file, max_samples defaults to 10 random test traces.
    Returns background traces, test traces, their IDs, and selection metadata.
    """
    if sample_ids_file is None and max_samples is None:
        max_samples = 10
    if background_size < 1 or (max_samples is not None and max_samples < 1) or seed < 0:
        raise ValueError("Sample counts must be positive and seed nonnegative")
    if split_file is None:
        split = metadata.get("split_ids")
    else:
        with Path(split_file).open(encoding="utf-8") as stream:
            split = json.load(stream)
        embedded = metadata.get("split_ids")
        if embedded is not None and embedded != split:
            raise ValueError("Explicit split differs from bundle split_ids")
    if not isinstance(split, dict):
        raise ValueError("Provide --split-file or bundle metadata split_ids; no split is inferred")
    path = Path(dataset)
    digest = _digest(path)
    if split.get("dataset_sha256") != digest:
        raise ValueError("Dataset does not match the split's dataset_sha256")
    if metadata.get("dataset_sha256", digest) != digest:
        raise ValueError("Dataset does not match the bundle's dataset_sha256")
    train = _ids(split.get("train"), "train")
    test = _ids(split.get("test"), "test")
    val = _ids(split["val"], "val") if "val" in split else []
    if set(train) & set(test) or set(val) & (set(train) | set(test)):
        raise ValueError("Training, validation and test IDs must be disjoint")
    sample_ids = None
    if sample_ids_file is not None:
        with Path(sample_ids_file).open(encoding="utf-8") as stream:
            selection = json.load(stream)
        if not isinstance(selection, dict):
            raise ValueError("Sample IDs file must contain a JSON object")
        sample_ids = _ids(selection.get("sample_ids"), "sample_ids")
        if "target_class" in selection and (type(selection["target_class"]) is not int
                                            or selection["target_class"] != target_class):
            raise ValueError("Sample IDs file target_class differs from --target-class")
        outside = set(sample_ids) - set(test)
        if outside:
            raise ValueError(f"Sample IDs do not belong to the recorded test split: {sorted(outside)}")
        sample_ids = sample_ids[:max_samples]
    if ((reference_num_groups is None and background_size > len(train))
            or (sample_ids is None and max_samples > len(test))):
        raise ValueError("Requested sample count exceeds its split size")
    traces, labels, _, ids = load_traces(path, return_ids=True)
    index = {tid: i for i, tid in enumerate(ids)}
    if not set(train + val + test) <= set(index):
        raise ValueError("Split contains trace IDs absent from the dataset")
    defended = (_replay_defense(traces, index, split, metadata)
                if metadata.get("scenario") == "defense" else None)

    def trace(tid):
        return defended[tid] if defended is not None else traces[index[tid]]

    rng = np.random.default_rng(seed)
    if reference_num_groups is None:
        background_ids = [train[i] for i in rng.choice(len(train), background_size, replace=False)]
    if sample_ids is None:
        sample_ids = [test[i] for i in rng.choice(len(test), max_samples, replace=False)]
    reference_info = {}
    if reference_num_groups is not None:
        from .grouped import select_references

        _, background_ids, reference_info = select_references(
            [trace(tid) for tid in sample_ids], [trace(tid) for tid in train], train,
            num_groups=reference_num_groups, reference_size=background_size, seed=seed)
    info = {"dataset_sha256": digest, "sampling_seed": seed,
            "selection": "uniform_without_replacement_from_explicit_train_test_ids",
            "split_sha256": _digest(split_file) if split_file else None,
            "train_size": len(train), "test_size": len(test),
            "sample_labels": [labels[index[tid]] for tid in sample_ids]}
    info["population"] = "monitored"
    info["defense_replayed"] = defended is not None
    info.update(reference_info)
    if sample_ids_file is not None:
        info["selection"] = "explicit_test_ids_in_file_order"
        info["sample_ids_file_sha256"] = _digest(sample_ids_file)

    return ([trace(tid) for tid in background_ids],
            [trace(tid) for tid in sample_ids], background_ids, sample_ids, info)
