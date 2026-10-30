"""The miRAW CSSMs: ``Default`` (paper "miRAW-6-1:10") and ``Personalized;start;size;minBinds``."""

from __future__ import annotations

from miraw.folding.base import Fold
from miraw.sites.candidate import CandidateSite
from miraw.sites.finders.base import CandidateSiteFinder


class DefaultFinder(CandidateSiteFinder):
    """>= ``min_binds_seed`` pairs in the seed window (default: miRNA nt 2-10, 6 pairs)."""

    def is_candidate(self, mirna: str, site: str, fold: Fold) -> CandidateSite | None:
        if len(site) > self.max_site_length:
            return None
        if fold.energy == 0 or fold.energy > self.max_free_energy:
            return None
        if not self._binding(fold.structure):
            return None
        binds = wc = wobble = canonical = 0
        for idx, off in self._seed_scan(fold, mirna):
            if fold.structure[idx] == ")":
                binds += 1
                pair = self._pair(fold, idx)
                if len(pair) > 1 and self._wc(pair[0], pair[1]):
                    wc += 1
                    if 0 <= off < 6:
                        canonical += 1
                elif len(pair) > 1 and self._wobble(pair[0], pair[1]):
                    wobble += 1
        if binds < self.min_binds_seed:
            return None
        return CandidateSite(
            seed_region_start=1,
            seed_region_end=self.seed_region_size,
            free_energy=fold.energy,
            binds_in_seed=binds,
            wc_pairs=wc,
            wobble_pairs=wobble,
            seed_start_in_sequence=self._start_in_sequence(fold),
            canonical=canonical >= 6,
            comment="DEF: cannonical" if canonical >= 6 else "DEF: non-cannonical",
            structure=fold.structure,
        )


class PersonalizedFinder(CandidateSiteFinder):
    """Configurable seed window. Scans from ``len(utr)-len(mirna)`` including truncated windows."""

    def __init__(self, seed_region_start: int = 1, seed_region_size: int = 10, min_binds_seed: int = 6, **kw):
        min_binds_seed = min(min_binds_seed, seed_region_size)
        super().__init__(
            seed_region_start=seed_region_start,
            seed_region_size=seed_region_size,
            min_binds_seed=min_binds_seed,
            **kw,
        )

    def scan_starts(self, mirna: str, utr: str) -> list[int]:
        return list(range(len(utr) - len(mirna), -1, -1))

    def find_candidates(self, mirna, utr, folds):
        # windows near the 3' end are shorter than max_site_length: no whole-window prefetch there
        return super().find_candidates(mirna, utr, folds)

    def is_candidate(self, mirna: str, site: str, fold: Fold) -> CandidateSite | None:
        if len(site) > self.max_site_length:
            return None
        if fold.energy >= 0 or fold.energy > self.max_free_energy:
            return None
        if not self._binding(fold.structure):
            return None
        binds = wc = wobble = canonical = 0
        first, last = 99, 0
        for idx, off in self._seed_scan(fold, mirna):
            if fold.structure[idx] == ")":
                binds += 1
                pair = self._pair(fold, idx)
                if len(pair) > 1 and self._wc(pair[0], pair[1]):
                    wc += 1
                    if 0 < off < 7:
                        canonical += 1
                elif len(pair) > 1 and self._wobble(pair[0], pair[1]):
                    wobble += 1
                if first == 99:
                    first = off + self.seed_region_start
                last = off + self.seed_region_start
        if binds < self.min_binds_seed:
            return None
        return CandidateSite(
            seed_region_start=first,
            seed_region_end=last,
            free_energy=fold.energy,
            binds_in_seed=binds,
            wc_pairs=wc,
            wobble_pairs=wobble,
            seed_start_in_sequence=self._start_in_sequence(fold),
            canonical=canonical >= 6,
            comment="DEF: cannonical" if canonical >= 6 else "DEF: non-cannonical",
            structure=fold.structure,
        )
