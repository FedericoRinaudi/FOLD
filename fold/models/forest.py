"""The order-independent branch: a random forest over the aggregate statistics
and the packet-size histogram."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    """A forest over the 33 aggregate statistics plus the 64-bin size histogram.

    Parameters
    ----------
    num_classes:
        Number of monitored websites.  Used to keep the probability columns
        aligned with class indices even when a split misses a class.
    seed:
        Set to the experiment id by the runners, so a repetition's forest is
        reproducible.
    n_estimators:
        Number of trees.  More trees only reduce variance, at the cost of
        training time.
    """

    def __init__(self, num_classes: int, seed: int = 42, n_estimators: int = 500):
        self.num_classes = num_classes
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            criterion="gini",
            max_features="sqrt",      # the standard choice for classification
            bootstrap=True,
            n_jobs=1,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, always ``num_classes`` wide.

        sklearn only returns columns for the classes it saw while fitting, so
        a split missing a website would otherwise change the shape.
        """
        probs = self.clf.predict_proba(X)
        if probs.shape[1] < self.num_classes:
            full = np.zeros((len(probs), self.num_classes), dtype=np.float32)
            for j, cls in enumerate(self.clf.classes_):
                full[:, int(cls)] = probs[:, j]
            probs = full
        return probs.astype(np.float32)
