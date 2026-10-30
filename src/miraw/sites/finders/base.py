"""Candidate-site finder base class: window scan, seed alignment, overlap resolution.

Ported from ``CandidateSiteFinder.getCandidateSites`` and the ``*CandidateSiteFinder`` classes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from miraw.folding.base import Fold, FoldService
from miraw.sites.candidate import CandidateSite
from miraw.sites.structure import (
    check_binding_from_structure,
    find_binding_start_in_sequence,
    get_pair_at,
    is_watson_crick,
    is_wobble,
)


class CandidateSiteFinder(ABC):
    def __init__(
        self,
        max_site_length: int = 40,
        seed_alignment_offset: int = 5,
        win_step: int = 5,
        max_free_energy: float = 0.0,
        min_binds_seed: int = 6,
        seed_region_start: int = 1,
        seed_region_size: int = 9,
    ):
        self.max_site_length = max_site_length
        self.seed_alignment_offset = seed_alignment_offset
        self.win_step = win_step
        self.max_free_energy = max_free_energy
        self.min_binds_seed = min_binds_seed
        self.seed_region_start = seed_region_start
        self.seed_region_size = seed_region_size

    @property
    def seed_start_index(self) -> int:
        return self.max_site_length - self.seed_alignment_offset

    # -- scan -------------------------------------------------------------------------------
    def scan_starts(self, mirna: str, utr: str) -> list[int]:
        """Window start positions in the order the original visits them (3' to 5')."""
        return list(range(len(utr) - self.max_site_length, 0, -1))

    @abstractmethod
    def is_candidate(self, mirna: str, site: str, fold: Fold) -> CandidateSite | None: ...

    def _seed_scan(self, fold: Fold, mirna: str):
        """Yield (index, offset) of every position of the seed window in the co-fold structure."""
        seed_start = len(fold.structure) - len(mirna) + self.seed_region_start
        end = min(len(fold.structure), seed_start + self.seed_region_size)
        for idx in range(seed_start, end):
            yield idx, idx - seed_start

    # -- main entry point ---------------------------------------------------------------------
    def find_candidates(self, mirna: str, utr: str, folds: FoldService) -> list[CandidateSite]:
        win_step = max(1, min(self.win_step, len(mirna)))
        starts = self.scan_starts(mirna, utr)
        if starts:
            folds.prefetch_windows(mirna, utr, self.max_site_length)
        raw: list[CandidateSite] = []
        for i in starts:
            site = utr[i : i + self.max_site_length]
            cand = self.is_candidate(mirna, site, folds.cofold(site, mirna))
            if cand is not None:
                cand.mirna = mirna
                cand.start, cand.end, cand.transcript = i, i + len(site), site
                raw.append(cand)
        for cand in raw:
            if cand.binds_in_seed > 0:
                cand.align_and_extend(utr, self.max_site_length, self.seed_start_index)
        kept: list[CandidateSite] = []
        for cand in raw:
            if cand.binds_in_seed <= 0:
                continue
            if kept and kept[-1].start - cand.start < win_step:
                if kept[-1].free_energy > cand.free_energy:
                    kept[-1] = cand
            else:
                kept.append(cand)
        return kept

    # shared helpers
    @staticmethod
    def _pair(fold: Fold, idx: int) -> str:
        return get_pair_at(idx, fold.sequence, fold.structure)

    @staticmethod
    def _start_in_sequence(fold: Fold) -> int:
        return find_binding_start_in_sequence(fold.structure)

    _binding = staticmethod(check_binding_from_structure)
    _wc = staticmethod(is_watson_crick)
    _wobble = staticmethod(is_wobble)
