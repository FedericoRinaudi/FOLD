"""Saving and loading closed-world FOLD inference bundles.

Format version 1 stores the two-head model in four files:

    manifest.json       builder parameters, class names, metadata, versions, hashes
    encoder.npz         num_classes, target_length, size_edges
    forest.joblib       the fitted RandomForest wrapper
    fold.weights.h5     Keras weights, including BatchNorm state

``class_names[c]`` names probability column ``c``. The encoder's bucket edges
retain their dtype. FOLD is rebuilt from its parameters before loading weights;
optimizer state is not stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata as distributions
import hashlib
import inspect
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence
import warnings

import joblib
import numpy as np
from sklearn.utils.validation import check_is_fitted

from fold.data import Encoder, SEQUENCE_FEATURES
from fold.models.forest import RandomForest

_FORMAT = "fold-explainability"
_VERSION = 1
_FILES = ("encoder.npz", "forest.joblib", "fold.weights.h5")


@dataclass(frozen=True)
class ArtifactBundle:
    """Loaded models and encoder, with ``class_names[c]`` naming output column c."""

    encoder: Encoder
    forest: RandomForest
    model: Any
    model_config: dict
    class_names: tuple[str, ...]
    metadata: dict


def _validate_encoder(encoder: Encoder) -> None:
    if not isinstance(encoder, Encoder):
        raise TypeError("encoder must be fold.data.Encoder")
    for name, minimum in (("num_classes", 2), ("target_length", 1)):
        value = getattr(encoder, name)
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"Encoder.{name} must be an integer >= {minimum}")
    edges = np.asarray(encoder.size_edges)
    if (edges.ndim != 1 or not len(edges) or edges.dtype.kind not in "fiu"
            or not np.isfinite(edges).all() or np.any(edges[1:] < edges[:-1])):
        raise ValueError("size_edges must be a finite, nondecreasing numeric vector")
    # Equal quantiles are valid bucket edges.


def _validate_forest(encoder: Encoder, forest: RandomForest) -> None:
    if not isinstance(forest, RandomForest):
        raise TypeError("forest must be the original RandomForest wrapper, not OOF probabilities")
    check_is_fitted(forest.clf, ["estimators_", "classes_", "n_features_in_"])
    if forest.num_classes != encoder.num_classes:
        raise ValueError("Forest and Encoder num_classes differ")
    classes = np.asarray(forest.clf.classes_)
    if (forest.clf.n_outputs_ != 1 or classes.ndim != 1 or not len(classes)
            or classes.dtype.kind not in "iuf" or not np.isfinite(classes).all()
            or np.any(classes != np.floor(classes)) or np.any(classes < 0)
            or np.any(classes >= encoder.num_classes)
            or np.any(classes[1:] <= classes[:-1])):
        raise ValueError("RF classes must be sorted unique integer indices in [0, num_classes)")
    if forest.clf.n_features_in_ != 33 + encoder.histogram_bins:
        raise ValueError("RF feature count does not match Encoder.forest")


def _validate_model(encoder: Encoder, model: Any) -> None:
    inputs = model.input
    outputs = model.output
    expected = {
        "transformer_input": (None, encoder.target_length, SEQUENCE_FEATURES),
        "agg_input": (None, 3),
        "size_vocab_input": (None, encoder.target_length),
        "unigram_agg_input": (None, encoder.histogram_bins),
        "rf_input": (None, encoder.num_classes),
    }
    if not isinstance(inputs, dict) or set(inputs) != set(expected):
        raise ValueError("Expected the named input dictionary of original build_fold")
    for name, shape in expected.items():
        if tuple(inputs[name].shape) != shape:
            raise ValueError(f"Unexpected shape for {name}: {inputs[name].shape}; expected {shape}")
        input_dtype = inputs[name].dtype
        dtype = np.dtype(getattr(input_dtype, "as_numpy_dtype", input_dtype))
        if dtype != np.dtype("int32" if name == "size_vocab_input" else "float32"):
            raise ValueError(f"Unexpected input dtype for {name}: {dtype}")
    expected_outputs = {"multi_output": encoder.num_classes,
                        "aux_multi_output": encoder.num_classes, "binary_output": 1}
    if not isinstance(outputs, dict) or set(outputs) != set(expected_outputs):
        raise ValueError("Only original two-head FOLD is supported; n1_output is not multi_output")
    for name, width in expected_outputs.items():
        if tuple(outputs[name].shape) != (None, width):
            raise ValueError(f"Unexpected output shape for {name}")


def fold_config(encoder: Encoder, **overrides: Any) -> dict:
    """The complete ``build_fold`` parameters, sized from the encoder.

    ``overrides`` replaces builder defaults. Dimensions must match the encoder
    and ``n1_head`` must be false. Requires TensorFlow without building a model.
    """
    _validate_encoder(encoder)
    from fold.models.transformer import build_fold

    signature = inspect.signature(build_fold)
    dimensions = {"sequence_length": int(encoder.target_length),
                  "num_features": SEQUENCE_FEATURES, "num_classes": int(encoder.num_classes),
                  "num_agg_features": 3, "size_vocab_size": encoder.vocab_size,
                  "unigram_agg_dim": encoder.histogram_bins}
    for key, value in dimensions.items():
        if key in overrides and overrides[key] != value:
            raise ValueError(f"{key} conflicts with the Encoder/original preprocessing")
    bound = signature.bind(**{**dimensions, **overrides})
    bound.apply_defaults()
    config = dict(bound.arguments)
    if config["n1_head"] is not False:
        raise ValueError("This bundle format supports two-head closed-world FOLD only")
    return config


def _complete_config(encoder: Encoder, config: Mapping[str, Any]) -> dict:
    resolved = fold_config(encoder, **dict(config))
    if set(config) != set(resolved):
        raise ValueError("model_config must include all builder parameters; use fold_config")
    return resolved


def _validate_architecture(model: Any, config: Mapping[str, Any]) -> None:
    """Check builder parameters against the model's named layers.

    Weight shapes alone do not identify dropout rates or attention head counts.
    """
    attributes = {
        "rf_proj_dim": ("rf_projection", "units"),
        "rf_branch_dropout": ("cross_q_dropout", "rate"),
        "gate_hidden_dim": ("gate_hidden", "units"),
        "size_vocab_size": ("size_vocab_embedding", "input_dim"),
        "size_embed_dim": ("size_vocab_embedding", "output_dim"),
        "pos_gate_hidden_dim": ("pos_encoding", "gate_hidden_dim"),
        "unigram_agg_gate_hidden_dim": ("unigram_gate_hidden", "units"),
        "d_model": ("input_projection", "units"),
        "dropout_rate": ("pre_cls_dropout", "rate"),
        "num_heads": ("cross_attn_fusion", "num_heads"),
    }
    for option, (name, attribute) in attributes.items():
        if getattr(model.get_layer(name), attribute) != config[option]:
            raise ValueError(f"model_config[{option!r}] does not match model layer {name}")
    blocks = [layer for layer in model.layers if layer.name.startswith("transformer_block_")]
    if len(blocks) != config["num_layers"]:
        raise ValueError("model_config num_layers does not match the Transformer blocks")
    for block in blocks:
        if (block.attn.num_heads != config["num_heads"]
                or block.attn.key_dim * block.attn.num_heads != config["d_model"]
                or block.ffn.layers[0].units != config["d_ff"]
                or block.drop1.rate != config["dropout_rate"]
                or block.drop2.rate != config["dropout_rate"]):
            raise ValueError(f"model_config does not match {block.name}")


def _class_names(names: Sequence[str], count: int) -> tuple[str, ...]:
    if isinstance(names, (str, bytes, Mapping)):
        raise TypeError("class_names must be a sequence in numeric label order")
    names = tuple(names)
    if (len(names) != count or any(not isinstance(n, str) or not n.strip() for n in names)
            or len(set(names)) != count):
        raise ValueError("Provide one unique nonempty class name per Encoder class")
    return names


def _metadata(value: Mapping[str, Any]) -> dict:
    result = dict(value)
    if result.get("scenario") != "closed-world" or result.get("forest_role") != "final":
        raise ValueError("metadata must declare scenario='closed-world' and forest_role='final'")
    return result


def _versions() -> dict:
    result = {"python": platform.python_version()}
    for name in ("numpy", "scikit-learn", "joblib", "keras"):
        result[name] = distributions.version(name)
    import tensorflow as tf
    result["tensorflow"] = tf.__version__
    return result


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_bundle(path: str | Path, *, encoder: Encoder, forest: RandomForest,
                model: Any, model_config: Mapping[str, Any], class_names: Sequence[str],
                metadata: Mapping[str, Any]) -> None:
    """Save the trained models and encoder to a new directory.

    ``forest`` must be fitted on the full training split, and ``model`` must hold
    the selected trained weights. Fitted-state checks cannot distinguish a final
    forest from an out-of-fold forest.

    ``model_config`` must contain the exact parameters used by ``build_fold``;
    :func:`fold_config` resolves its defaults. ``class_names`` follows numeric
    label order. ``metadata`` must declare ``scenario='closed-world'`` and
    ``forest_role='final'``; it can also record seeds, split identifiers and the
    source revision.

    The destination must not exist. The manifest is written last, after the
    model files and their hashes.
    """
    _validate_encoder(encoder)
    _validate_forest(encoder, forest)
    _validate_model(encoder, model)
    config = _complete_config(encoder, model_config)
    _validate_architecture(model, config)
    manifest = {"format": _FORMAT, "version": _VERSION,
                "model_config": config,
                "class_names": _class_names(class_names, encoder.num_classes),
                "metadata": _metadata(metadata), "versions": _versions()}
    # Check JSON compatibility before creating the destination.
    json.dumps(manifest, allow_nan=False)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    np.savez(path / "encoder.npz", num_classes=encoder.num_classes,
             target_length=encoder.target_length, size_edges=encoder.size_edges)
    joblib.dump(forest, path / "forest.joblib", compress=3)
    model.save_weights(str(path / "fold.weights.h5"))
    manifest["sha256"] = {name: _digest(path / name) for name in _FILES}
    with (path / "manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, ensure_ascii=False, allow_nan=False)


def load_bundle(path: str | Path, *, trusted: bool = False,
                strict_versions: bool = True) -> ArtifactBundle:
    """Load the forest and encoder, rebuild FOLD and restore its weights.

    No dataset or fitting is needed. Set ``trusted=True`` only for trusted
    bundles: joblib loading can execute code. Hashes check file integrity, not
    authenticity.

    Python major/minor and library versions must match the saved versions.
    ``strict_versions=False`` warns on mismatches instead of rejecting them;
    prediction equivalence then needs checking in the target environment.
    """
    if trusted is not True:
        raise ValueError("Only load bundles from a trusted source; pass trusted=True")
    path = Path(path)
    with (path / "manifest.json").open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format") != _FORMAT or manifest.get("version") != _VERSION:
        raise ValueError("Unsupported artifact bundle format/version")
    metadata = _metadata(manifest["metadata"])
    hashes = manifest["sha256"]
    if set(hashes) != set(_FILES):
        raise ValueError("Incomplete artifact file manifest")
    for name in _FILES:
        if _digest(path / name) != hashes[name]:
            raise ValueError(f"Artifact integrity check failed: {name}")
    current = _versions()
    recorded = manifest["versions"]
    mismatches = []
    for name, version in current.items():
        previous = recorded.get(name)
        same = (str(previous).split(".")[:2] == version.split(".")[:2]
                if name == "python" else previous == version)
        if not same:
            mismatches.append(f"{name}: saved={previous}, current={version}")
    if mismatches:
        message = "Artifact runtime differs: " + "; ".join(mismatches)
        if strict_versions:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    with np.load(path / "encoder.npz", allow_pickle=False) as values:
        encoder = Encoder(num_classes=values["num_classes"].item(),
                          target_length=values["target_length"].item(),
                          size_edges=values["size_edges"].copy())
    _validate_encoder(encoder)
    names = _class_names(manifest["class_names"], encoder.num_classes)
    config = _complete_config(encoder, manifest["model_config"])
    forest = joblib.load(path / "forest.joblib")
    _validate_forest(encoder, forest)
    from fold.models.transformer import build_fold

    model = build_fold(**config)
    model.load_weights(str(path / "fold.weights.h5"))
    _validate_model(encoder, model)
    _validate_architecture(model, config)
    return ArtifactBundle(encoder, forest, model, config, names, metadata)
