"""Confusion-matrix metrics with the definitions used in the paper's GlobalSummary."""

from __future__ import annotations

import numpy as np


def confusion(y_true, y_pred_positive) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred_positive).astype(bool)
    return {
        "TP": int(((y == 1) & p).sum()),
        "FP": int(((y == 0) & p).sum()),
        "TN": int(((y == 0) & ~p).sum()),
        "FN": int(((y == 1) & ~p).sum()),
    }


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def summarize(c: dict) -> dict:
    tp, fp, tn, fn = c["TP"], c["FP"], c["TN"], c["FN"]
    precision, recall = _div(tp, tp + fp), _div(tp, tp + fn)
    return {
        **c,
        "Accuracy": _div(tp + tn, tp + fp + tn + fn),
        "Precision": precision,
        "Recall": recall,
        "NegPrecision": _div(tn, tn + fn),
        "Specificity": _div(tn, tn + fp),
        "F1": _div(2 * precision * recall, precision + recall),
    }


def gene_level_metrics(labels, predictions) -> dict:
    """A miRNA-gene pair is called a target when its prediction (max site score) is > 0."""
    return summarize(confusion(labels, np.asarray(predictions) > 0))


def roc_auc(y_true, scores) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y_true)
    return float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else float("nan")
