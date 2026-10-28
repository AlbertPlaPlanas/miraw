"""A-posteriori site filters (ported from ``SiteFilters``)."""

from __future__ import annotations

from pymiraw.folding.base import FoldService
from pymiraw.sites.candidate import CandidateSite

MAX_SITE_OPENING_LENGTH = 200


def site_opening_energy(folds: FoldService, utr: str, start: int, end: int, extra_open: int = 0) -> float:
    """ΔG_open: energy cost of making ``utr[start:end]`` unpaired, over a ~200 nt window."""
    chunk = MAX_SITE_OPENING_LENGTH - end + start
    sub_start = max(start - chunk // 2, 0)
    sub_end = min(end + chunk // 2, len(utr))
    sub = utr[sub_start:sub_end]
    free = folds.fold(sub).energy
    gap_start = start - sub_start - extra_open
    gap_end = end - sub_start + extra_open
    return folds.fold_with_open_gap(sub, gap_start, gap_end).energy - free


class AccessibilityEnergyFilter:
    """Remove sites whose opening energy exceeds ``threshold`` (paper configs use 10)."""

    name = "AccessibilityEnergy"

    def __init__(self, threshold: float = 10.0):
        self.threshold = threshold

    def evaluate(self, site: CandidateSite, utr: str, folds: FoldService) -> bool:
        """Return True if the site passes. Marks it filtered otherwise."""
        key = folds.opening_key(utr, site.start, site.end)
        energy = folds.opening_cache.get(key)  # saved or already computed in this process
        if energy is None:
            energy = folds.opening_cache[key] = site_opening_energy(folds, utr, site.start, site.end)
        site.properties["AccessibilityEnergy"] = energy
        if energy > self.threshold:
            site.filtered = True
            site.filtered_reason = f"Accessibility Energy:{energy}>{self.threshold}"
            return False
        return True


def get_filter(spec: str):
    parts = spec.split(";")
    if parts[0].lower() == "accessibilityenergy":
        try:
            return AccessibilityEnergyFilter(float(parts[1]))
        except (IndexError, ValueError):
            return AccessibilityEnergyFilter()
    raise ValueError(f"unsupported filter {spec!r}")
