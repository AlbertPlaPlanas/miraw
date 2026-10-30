"""PITA-like CSSM (paper "miRAW-PITA"): 7mer-ish seeds with <= 1 wobble and >= 6 WC pairs."""

from __future__ import annotations

from miraw.folding.base import Fold
from miraw.sites.candidate import CandidateSite
from miraw.sites.finders.base import CandidateSiteFinder


class PitaFinder(CandidateSiteFinder):
    def __init__(self, **kw):
        super().__init__(seed_region_size=8, min_binds_seed=6, seed_region_start=1, **kw)

    def is_candidate(self, mirna: str, site: str, fold: Fold) -> CandidateSite | None:
        if len(site) > self.max_site_length or fold.energy == 0:
            return None
        if not self._binding(fold.structure):
            return None
        comment, canonical = "PITA: canonnical", True
        wc = wobble = n_pairs = 0
        st = fold.structure
        for idx, _ in self._seed_scan(fold, mirna):
            if st[idx] in ")}":
                pair = self._pair(fold, idx)
                if len(pair) > 1:
                    if self._wobble(pair[0], pair[1]):
                        wobble += 1
                        comment, canonical = "PITA: noncanonical", False
                    elif self._wc(pair[0], pair[1]):
                        wc += 1
                n_pairs += 1
            elif n_pairs < 6:
                return None
            else:
                break
        if wc < 6 or wobble > 1:
            return None
        return CandidateSite(
            seed_region_start=1,
            seed_region_end=n_pairs,
            free_energy=fold.energy,
            binds_in_seed=n_pairs,
            wc_pairs=wc,
            wobble_pairs=wobble,
            seed_start_in_sequence=self._start_in_sequence(fold),
            canonical=canonical,
            comment=comment,
            structure=st,
        )
