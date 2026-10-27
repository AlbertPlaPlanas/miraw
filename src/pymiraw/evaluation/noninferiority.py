"""Non-inferiority test used by Gate 2 ("at least as good as the paper, within a margin").

For paired differences ``d = ours - paper`` (one per fold, or per config x test set), the model is
non-inferior on a metric if the one-sided lower confidence bound of ``mean(d)`` is above ``-margin``.
Bounds are t-based (more conservative than a bootstrap for the small n used here).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

CONFIDENCE = 0.99  # one-sided; agreed 2026-10-27 (replaces the strict "mean >= paper" rule)
MARGINS = {"Accuracy": 0.01, "F1": 0.01, "AUC": 0.005}  # largest acceptable drop per metric


@dataclass(frozen=True)
class Verdict:
    mean: float
    lower: float
    margin: float
    confidence: float
    n: int

    @property
    def non_inferior(self) -> bool:
        return self.lower > -self.margin

    @property
    def slack(self) -> float:
        """Distance between the lower bound and the limit; small values mean a fragile pass."""
        return self.lower + self.margin


def lower_bound(diffs, confidence: float = CONFIDENCE) -> float:
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2:
        raise ValueError("need at least 2 paired differences")
    return float(d.mean() - stats.t.ppf(confidence, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def non_inferiority(diffs, metric: str, confidence: float = CONFIDENCE, margins: dict | None = None) -> Verdict:
    margin = (margins or MARGINS)[metric]
    d = np.asarray(diffs, dtype=float)
    return Verdict(float(d.mean()), lower_bound(d, confidence), margin, confidence, len(d))
