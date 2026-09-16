"""The metrics of the paper, defined once.

Accuracy, F1, FNR and FPR are computed on the multi-class problem with the
One-vs-Rest convention: each class takes its turn as the positive one, the
others as negatives, and the per-class values are averaged (macro).  The
classes are the N monitored websites in the closed world, and those plus
background in the open world.

This is a different quantity from the *detection* rates, the fraction of
background that passes the binary gate: with about 101 classes the macro FPR
is of the order of 1e-4, while the detection FPR is tens of percentage points.
Both appear in the paper, so both are defined here, under names that cannot be
confused, and every figure and table uses these.

Models without a monitored head
-------------------------------
The forest, the convolutional baseline and Deep Fingerprinting classify among
the monitored websites and do not decide whether a trace is monitored at all.
For them the decision is taken on the maximum class probability, which is the
membership score.  FOLD and TR have a head of their own.
"""
from __future__ import annotations

import numpy as np

#: Conventional label of non-monitored traffic.
BACKGROUND = -1


def macro_ovr(y_true, y_pred, classes) -> dict:
    """Accuracy, and macro One-vs-Rest F1, FNR and FPR over ``classes``."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    f1s, fnrs, fprs = [], [], []
    for c in classes:
        pos_t = y_true == c
        pos_p = y_pred == c
        tp = int(np.sum(pos_t & pos_p))
        fp = int(np.sum(~pos_t & pos_p))
        fn = int(np.sum(pos_t & ~pos_p))
        tn = int(np.sum(~pos_t & ~pos_p))
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
        fnrs.append(fn / (tp + fn) if (tp + fn) else 0.0)
        fprs.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return {"accuracy": float(np.mean(y_true == y_pred)),
            "f1": float(np.mean(f1s)),
            "fnr": float(np.mean(fnrs)),
            "fpr": float(np.mean(fprs))}


def closed_world(probs, y_true) -> dict:
    """Closed-world metrics: N classes, no background."""
    return macro_ovr(y_true, np.argmax(probs, axis=1), np.unique(y_true))


def open_world(mon_probs, back_probs, y_true, mon_score=None, back_score=None,
               thr: float = 0.5, include_background: bool = True) -> dict:
    """Open-world metrics on the N+1 class problem.

    ``mon_score`` / ``back_score`` are the "this is monitored" scores.  When
    absent, the maximum class probability is used, which is how a model without
    a monitored head takes the decision.

    The *prediction* always has N+1 outcomes -- a trace can be sent to
    background -- but the classes being averaged over need not include it:
    with ``include_background=False`` the macro metrics describe website
    identification alone, and membership is measured separately by the ROC
    curves.  Background weighs 1/101, so the numerical difference is under 0.03
    points; the choice is one of meaning, not of value.
    """
    if mon_score is None:
        mon_score = mon_probs.max(axis=1)
    if back_score is None:
        back_score = back_probs.max(axis=1)
    mon_score = np.asarray(mon_score).ravel()
    back_score = np.asarray(back_score).ravel()

    pred_mon = np.where(mon_score > thr, np.argmax(mon_probs, axis=1), BACKGROUND)
    pred_back = np.where(back_score > thr, np.argmax(back_probs, axis=1), BACKGROUND)

    y = np.concatenate([np.asarray(y_true), np.full(len(back_score), BACKGROUND)])
    p = np.concatenate([pred_mon, pred_back])

    classes = list(np.unique(y_true))
    if include_background:
        classes = classes + [BACKGROUND]
    return macro_ovr(y, p, classes)


def detection(mon_score, back_score, thr: float = 0.5) -> dict:
    """The monitored/background decision alone, independently of the website.

    Monitored traces are the positives.  ``tpr`` is the fraction of monitored
    traces accepted, ``fpr`` the fraction of background wrongly accepted, and
    ``auc`` summarises the whole trade-off, independently of the threshold.
    """
    from sklearn.metrics import roc_auc_score

    mon_score = np.asarray(mon_score).ravel()
    back_score = np.asarray(back_score).ravel()
    tp = int(np.sum(mon_score > thr))
    fp = int(np.sum(back_score > thr))
    labels = np.r_[np.ones(len(mon_score)), np.zeros(len(back_score))]
    return {"tpr": tp / len(mon_score) if len(mon_score) else 0.0,
            "fpr": fp / len(back_score) if len(back_score) else 0.0,
            "auc": float(roc_auc_score(labels, np.r_[mon_score, back_score]))}


def confidence_score(probs: np.ndarray) -> np.ndarray:
    """Turn class probabilities into a membership score in [0, 1].

    The maximum probability, rescaled so that a uniform distribution over the
    N classes maps to 0 and certainty to 1.  This is how a model without a
    monitored head decides whether a trace belongs to the monitored set.
    """
    n = probs.shape[1]
    return np.clip((np.max(probs, axis=1) - 1.0 / n) / (1.0 - 1.0 / n), 0.0, 1.0)
