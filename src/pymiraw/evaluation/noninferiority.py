"""Non-inferiority test used by Gate 2 ("at least as good as the paper, within a margin").

For paired differences ``d = ours - paper`` (one per fold, or per config x test set), the model is
non-inferior on a metric if the one-sided lower confidence bound of the location of ``d`` is above ``-margin``.

Two bounds:
* ``rank`` (default since 2026-10-28, owner decision): exact Wilcoxon signed-rank bound of the Hodges-Lehmann
  estimate (the median of the Walsh averages). A large win counts by its rank, not its size, so being much
  better than the paper on some units does not widen the bound. Needs n >= 7 at 99 % (otherwise -inf).
* ``t``: t-based bound of the mean (the rule from 2026-10-27 to 2026-10-28); large wins inflate the SD.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

CONFIDENCE = 0.99  # one-sided; agreed 2026-10-27 (replaces the strict "mean >= paper" rule)
MARGINS = {"Accuracy": 0.01, "F1": 0.01, "AUC": 0.005}  # largest acceptable drop per metric
METHOD = "rank"  # agreed 2026-10-28 (replaces "t")


@dataclass(frozen=True)
class Verdict:
    estimate: float  # Hodges-Lehmann for "rank", mean for "t"
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
    """t-based one-sided lower bound of the mean."""
    d = np.asarray(diffs, dtype=float)
    if len(d) < 2:
        raise ValueError("need at least 2 paired differences")
    return float(d.mean() - stats.t.ppf(confidence, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))


def _walsh_averages(d: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(d))
    return np.sort((d[i] + d[j]) / 2)


def hodges_lehmann(diffs) -> float:
    return float(np.median(_walsh_averages(np.asarray(diffs, dtype=float))))


def rank_lower_bound(diffs, confidence: float = CONFIDENCE) -> float:
    """Exact one-sided lower bound of the Hodges-Lehmann estimate from the signed-rank null distribution."""
    d = np.asarray(diffs, dtype=float)
    n = len(d)
    if n < 2:
        raise ValueError("need at least 2 paired differences")
    counts = np.zeros(n * (n + 1) // 2 + 1)  # number of subsets of {1..n} with each rank sum
    counts[0] = 1
    for k in range(1, n + 1):
        counts[k:] = counts[k:] + counts[:-k].copy()
    cdf = np.cumsum(counts) / 2.0**n
    c = int(np.sum(cdf <= 1 - confidence)) - 1  # largest rank sum C with P(W <= C) <= alpha
    if c < 0:
        return -np.inf  # n too small to reach this confidence
    return float(_walsh_averages(d)[c])  # the (C+1)-th smallest Walsh average


def non_inferiority(
    diffs, metric: str, confidence: float = CONFIDENCE, margins: dict | None = None, method: str = METHOD
) -> Verdict:
    margin = (margins or MARGINS)[metric]
    d = np.asarray(diffs, dtype=float)
    if method == "rank":
        return Verdict(hodges_lehmann(d), rank_lower_bound(d, confidence), margin, confidence, len(d))
    if method == "t":
        return Verdict(float(d.mean()), lower_bound(d, confidence), margin, confidence, len(d))
    raise ValueError(f"unknown method {method!r}")
