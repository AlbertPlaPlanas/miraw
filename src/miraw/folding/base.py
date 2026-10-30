"""Folding backends: co-fold a site with a miRNA, fold a sequence, fold with an opened region."""

from __future__ import annotations

import hashlib
import pickle
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from miraw.sites.store import utr_key


@dataclass(frozen=True)
class Fold:
    sequence: str
    structure: str  # dot-bracket; co-folds contain '&' between the two strands
    energy: float  # kcal/mol


class FoldingBackend(Protocol):
    name: str
    version: str

    def cofold(self, site: str, mirna: str) -> Fold: ...

    def fold(self, sequence: str) -> Fold: ...

    def fold_with_open_gap(self, sequence: str, start: int, end: int) -> Fold:
        """MFE fold with positions ``start..end`` (inclusive) forbidden to pair."""
        ...


class FoldService:
    """Memoizing front for a backend, plus an optional on-disk cache of whole-UTR window scans.

    All candidate-site finders scan the same sliding windows of a 3'UTR, so folding every
    window once per (miRNA, UTR) and caching it makes the 10-config benchmark cost little more
    than a single config.
    """

    def __init__(self, backend: FoldingBackend, cache_dir: str | Path | None = None):
        self.backend = backend
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cofold: dict[tuple[str, str], Fold] = {}
        self._fold: dict[str, Fold] = {}
        # saved accessibility energies {(utr_key, start, end): ΔG_open}, see miraw.sites.store.OpeningStore
        self.opening_cache: dict[tuple[str, int, int], float] = {}
        self._utr_hash: tuple[str, str] = ("", "")  # (utr, its key): the key is reused for all sites of a pair

    def opening_key(self, utr: str, start: int, end: int) -> tuple[str, int, int]:
        """Key of the opening energy of ``utr[start:end]`` under this backend and version."""
        if self._utr_hash[0] != utr:
            self._utr_hash = (utr, utr_key(f"{self.backend.name}-{self.backend.version}", utr))
        return self._utr_hash[1], start, end

    def cofold(self, site: str, mirna: str) -> Fold:
        key = (site, mirna)
        hit = self._cofold.get(key)
        if hit is None:
            hit = self._cofold[key] = self.backend.cofold(site, mirna)
        return hit

    def fold(self, sequence: str) -> Fold:
        hit = self._fold.get(sequence)
        if hit is None:
            hit = self._fold[sequence] = self.backend.fold(sequence)
        return hit

    def fold_with_open_gap(self, sequence: str, start: int, end: int) -> Fold:
        return self.backend.fold_with_open_gap(sequence, start, end)

    def prefetch_windows(self, mirna: str, utr: str, win_len: int) -> None:
        """Fold every full-length window of ``utr`` against ``mirna`` (cached on disk if enabled)."""
        n = len(utr) - win_len + 1
        if n <= 0:
            return
        path = None
        if self.cache_dir:
            tag = f"{self.backend.name}-{self.backend.version}|{win_len}|{mirna}|{utr}"
            digest = hashlib.sha1(tag.encode()).hexdigest()
            path = self.cache_dir / digest[:2] / f"{digest}.pkl"
            if path.exists():
                try:
                    packed = pickle.loads(zlib.decompress(path.read_bytes()))
                    for i, (structure, energy) in enumerate(packed):
                        site = utr[i : i + win_len]
                        self._cofold[(site, mirna)] = Fold(f"{site}&{mirna}", structure, energy)
                    return
                except Exception:  # corrupt cache file: recompute
                    path.unlink(missing_ok=True)
        folds = [self.cofold(utr[i : i + win_len], mirna) for i in range(n)]
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{id(self)}.tmp")
            tmp.write_bytes(zlib.compress(pickle.dumps([(f.structure, f.energy) for f in folds]), 1))
            tmp.replace(path)
