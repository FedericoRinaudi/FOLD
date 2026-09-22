"""Saving, loading and predicting with FOLD and its forest.

TensorFlow is imported only when building, saving, loading or predicting with
the neural model.
"""

from .artifacts import ArtifactBundle, fold_config, load_bundle, save_bundle
from .wrappers import FoldPredictor

__all__ = ["ArtifactBundle", "fold_config", "load_bundle", "save_bundle", "FoldPredictor"]
