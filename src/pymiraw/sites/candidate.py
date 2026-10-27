"""Candidate miRNA binding site and the 'align and extend' step."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CandidateSite:
    start: int = 0  # 0-based start in the 3'UTR (after alignment)
    end: int = 0  # exclusive end
    transcript: str = ""  # site sequence (gaps as 'L')
    mirna: str = ""
    free_energy: float = 0.0
    binds_in_seed: int = 0
    wc_pairs: int = 0
    wobble_pairs: int = 0
    seed_region_start: int = 1
    seed_region_end: int = 0
    seed_start_in_sequence: int = 0
    canonical: bool = False
    comment: str = ""
    structure: str = ""
    score: float | None = None
    filtered: bool = False
    filtered_reason: str = ""
    properties: dict = field(default_factory=dict)

    def align_and_extend(self, utr: str, site_length: int, seed_start_index: int) -> None:
        """Shift the site so the seed pairs start at ``seed_start_index``; pad off-UTR with 'L'."""
        shift = seed_start_index - self.seed_start_in_sequence
        new_start = self.start - shift
        new_end = new_start + site_length
        pad_left = pad_right = 0
        if new_start < 0:
            pad_left, new_start = -new_start, 0
        if new_end > len(utr):
            pad_right, new_end = new_end - len(utr), len(utr)
        self.start, self.end = new_start, new_end
        self.seed_start_in_sequence += shift
        self.transcript = "L" * pad_left + utr[new_start:new_end] + "L" * pad_right
