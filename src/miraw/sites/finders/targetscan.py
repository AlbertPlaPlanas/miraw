"""TargetScan-like CSSM (paper "miRAW-TS"): canonical, centered and compensatory sites."""

from __future__ import annotations

from miraw.folding.base import Fold
from miraw.sites.candidate import CandidateSite
from miraw.sites.finders.base import CandidateSiteFinder

REQUIRED_CONSECUTIVE_COMPENSATORY = 9


class TargetScanFinder(CandidateSiteFinder):
    def __init__(self, **kw):
        super().__init__(seed_region_size=6, min_binds_seed=6, seed_region_start=1, **kw)

    def _make(self, fold, n_pairs, wc, wobble, comment, canonical=False) -> CandidateSite:
        return CandidateSite(
            seed_region_start=1,
            seed_region_end=6,
            free_energy=fold.energy,
            binds_in_seed=n_pairs,
            wc_pairs=wc,
            wobble_pairs=wobble,
            seed_start_in_sequence=self._start_in_sequence(fold),
            canonical=canonical,
            comment=comment,
            structure=fold.structure,
        )

    def is_candidate(self, mirna: str, site: str, fold: Fold) -> CandidateSite | None:
        if len(site) > self.max_site_length:
            return None
        st = fold.structure
        if not self._binding(st):
            return None
        seed_start = len(st) - len(mirna) + self.seed_region_start
        wc = wobble = n_pairs = 0
        first_pair = -1
        for idx, off in self._seed_scan(fold, mirna):
            if st[idx] in ")}":
                pair = self._pair(fold, idx)
                if len(pair) > 1:
                    if self._wobble(pair[0], pair[1]):
                        wobble += 1
                    elif self._wc(pair[0], pair[1]):
                        wc += 1
                        if first_pair == -1:
                            first_pair = off
                n_pairs += 1
        if wc >= 6:
            return self._make(fold, n_pairs, wc, wobble, "TS:Cannonical", canonical=True)
        if n_pairs <= 2:
            return None
        # centered site: >= 11 consecutive WC pairs starting at the first seed pair
        idx, consecutive, end = seed_start + first_pair, 0, False
        while not end and 0 <= idx < len(st):
            if st[idx] == ")":
                pair = self._pair(fold, idx)
                if len(pair) > 1:
                    if self._wc(pair[0], pair[1]):
                        consecutive += 1
                    else:
                        end = True
            else:
                end = True
            if idx - first_pair - seed_start >= 12:
                end = True
            idx += 1
        if consecutive >= 11:
            return self._make(fold, n_pairs, wc, wobble, "TS:Centered")
        # compensatory 3' pairing
        if (n_pairs > 4 and wobble > 0) or wc > 4:
            run = best = 0
            for idx in range(seed_start + (17 - REQUIRED_CONSECUTIVE_COMPENSATORY), len(st)):
                if st[idx] == ")":
                    pair = self._pair(fold, idx)
                    if len(pair) > 1:
                        if self._wc(pair[0], pair[1]):
                            run += 1
                            best = max(best, run)
                        if self._wobble(pair[0], pair[1]):
                            best = 0
                else:
                    run = 0
            if best >= REQUIRED_CONSECUTIVE_COMPENSATORY:
                return self._make(fold, n_pairs, wc, wobble, "TS:Compensatory")
        return None
