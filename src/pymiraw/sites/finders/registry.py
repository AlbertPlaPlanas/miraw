"""Create finders from the original's type strings: ``miRAW``/``Default``, ``Personalized;s;n;k``,
``PITA``, ``TargetScan``/``TS``, ``Training`` (used when generating negative sites)."""

from __future__ import annotations

import logging

from pymiraw.sites.finders.base import CandidateSiteFinder
from pymiraw.sites.finders.default import DefaultFinder, PersonalizedFinder
from pymiraw.sites.finders.pita import PitaFinder
from pymiraw.sites.finders.targetscan import TargetScanFinder

log = logging.getLogger(__name__)


def get_site_finder(spec: str | None, max_site_length: int = 40, seed_alignment_offset: int = 5) -> CandidateSiteFinder:
    kw = {"max_site_length": max_site_length, "seed_alignment_offset": seed_alignment_offset}
    key = (spec or "").strip()
    low = key.lower()
    if low == "pita":
        return PitaFinder(**kw)
    if low in ("targetscan", "ts"):
        return TargetScanFinder(**kw)
    if low.startswith("personalized"):
        parts = key.split(";")
        start, size, minb = 1, 10, 6  # original defaults when parts are missing
        if len(parts) > 1 and parts[1]:
            start = int(parts[1])
        if len(parts) > 2 and parts[2]:
            size = int(parts[2])
        if len(parts) > 3 and parts[3]:
            minb = int(parts[3])
        return PersonalizedFinder(seed_region_start=start, seed_region_size=size, min_binds_seed=minb, **kw)
    if low not in ("default", "training", "miraw", ""):
        log.warning("unknown site finder %r: using Default", spec)
    return DefaultFinder(**kw)
