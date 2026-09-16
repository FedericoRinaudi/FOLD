"""What a model leaves behind, and how to read it back.

A run keeps every model's per-trace output rather than only its summary
metrics, so that any metric can be recomputed later without retraining.  One
file per model, inside the experiment's directory:

    <out-dir>/exp_NNN/fold.npz

The fields below are the whole on-disk format.  Which of them are present
depends on the model and the scenario: a closed-world run has only the test
fields, a model without a monitored head has no score, and the forest is the
only one that also stores its out-of-fold training probabilities, which is
what FOLD is trained on.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass
class Predictions:
    """One model's output on one experiment.

    ``*_probs`` are probabilities over the monitored websites; ``*_score`` is
    the "this trace is monitored" score, for the models that decide it with a
    head of their own.
    """
    test_monitored_probs: Optional[np.ndarray] = None
    test_monitored_score: Optional[np.ndarray] = None
    test_background_probs: Optional[np.ndarray] = None
    test_background_score: Optional[np.ndarray] = None

    val_monitored_probs: Optional[np.ndarray] = None
    val_monitored_score: Optional[np.ndarray] = None
    val_background_probs: Optional[np.ndarray] = None
    val_background_score: Optional[np.ndarray] = None

    #: Forest only: probabilities on the training traces, from folds that never
    #: saw the trace being scored.
    train_out_of_fold: Optional[np.ndarray] = None
    train_background_probs: Optional[np.ndarray] = None

    #: Set by the models whose background is an extra class rather than a
    #: separate head, so that the N+1 columns can be told apart from N.
    n_monitored: Optional[int] = None

    def save(self, path: Path) -> None:
        # Not dataclasses.asdict: it would deep-copy every array on the way out.
        present = {f.name: getattr(self, f.name) for f in fields(self)}
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **{k: v for k, v in present.items() if v is not None})

    @classmethod
    def load(cls, path: Path) -> "Predictions":
        with np.load(path, allow_pickle=False) as z:
            known = {f.name for f in fields(cls)}
            return cls(**{k: z[k] for k in z.files if k in known})

    def views(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``(monitored probs, background probs, monitored score, background score)``
        on the test split, with the three output conventions reconciled.

        * **A softmax over N+1 classes** (the convolutional baseline and Deep
          Fingerprinting): background is an explicit class of index N, so the
          probabilities have N+1 columns.  The score is 1 - P(background) and
          the website probabilities are the first N renormalised.  Reading
          those N+1 columns like the others -- argmax over all of them -- is
          wrong: when such a model correctly recognises background it answers
          N, which is not a website, and the trace would be counted as
          misclassified exactly when the model got it right.
        * **A monitored head** (TR and FOLD): the score is the head's output.
        * **No head at all** (the forest): the normalised class confidence.
        """
        from fold.metrics import confidence_score

        mon = self.test_monitored_probs
        back = self.test_background_probs

        if self.n_monitored is not None and mon.shape[1] == int(self.n_monitored) + 1:
            n = int(self.n_monitored)
            mon_score, back_score = 1.0 - mon[:, n], 1.0 - back[:, n]
            mon = mon[:, :n] / np.maximum(mon[:, :n].sum(1, keepdims=True), 1e-9)
            back = back[:, :n] / np.maximum(back[:, :n].sum(1, keepdims=True), 1e-9)
        elif self.test_monitored_score is not None:
            mon_score = np.asarray(self.test_monitored_score).ravel()
            back_score = np.asarray(self.test_background_score).ravel()
        else:
            mon_score, back_score = confidence_score(mon), confidence_score(back)

        return mon, back, mon_score, back_score
