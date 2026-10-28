"""On-disk store of pre-computed candidate binding sites (the ViennaRNA-dependent part of the pipeline).

Finding candidate sites means co-folding hundreds of sliding windows per (miRNA, 3'UTR) pair with
ViennaRNA, which dominates the run time. The result only depends on the miRNA, the 3'UTR, the
finder settings and the folding backend/version, never on the neural network. This module saves the
sites once and loads them on later runs, so a different model, filter or evaluation can reuse them.

Layout: a directory of ``sites-<id>.parquet`` shards (one is appended per run, existing shards are
never rewritten). Every row is one candidate site; a pair without sites is stored as a single row
with ``site_idx == -1`` so that "no sites" is distinguished from "not computed yet".
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import fields
from pathlib import Path

import pandas as pd

from pymiraw.sites.candidate import CandidateSite

# CandidateSite attributes that are written to disk. ``score``/``filtered`` are produced later by
# the model and the filters (which also fill ``properties``), so they are deliberately not stored.
_SITE_FIELDS = [
    f.name for f in fields(CandidateSite) if f.name not in ("score", "filtered", "filtered_reason", "properties")
]


def finder_fingerprint(finder) -> str:
    """Stable text identifying a finder class and all of its settings (window, seed rules, ...)."""
    settings = ",".join(f"{k}={v}" for k, v in sorted(vars(finder).items()))
    return f"{type(finder).__name__}({settings})"


def pair_key(backend_id: str, finder, mirna: str, utr: str) -> str:
    """Hash of everything the candidate sites depend on."""
    tag = f"{backend_id}|{finder_fingerprint(finder)}|{mirna}|{utr}"
    return hashlib.sha1(tag.encode()).hexdigest()


def backend_id(backend) -> str:
    """``name-version`` of a folding backend, e.g. ``vienna-bindings-2.7.2`` (part of every key)."""
    return f"{backend.name}-{backend.version}"


def _copy(s: CandidateSite) -> CandidateSite:
    """Independent copy (the pipeline mutates sites: score, filter results)."""
    return CandidateSite(**{k: getattr(s, k) for k in _SITE_FIELDS})


class SiteStore:
    """Read/append access to a directory of candidate-site shards."""

    def __init__(self, directory: str | Path):
        self.dir = Path(directory)
        self._sites: dict[str, list[CandidateSite]] = {}
        self._pending: dict[str, list[CandidateSite]] = {}  # added since the last flush
        self._load_existing()

    def _load_existing(self) -> None:
        """Read every shard into memory, grouped by pair key (later shards win on duplicates)."""
        for shard in sorted(self.dir.glob("sites-*.parquet")):
            df = pd.read_parquet(shard)
            for key, grp in df.groupby("pair_key", sort=False):
                grp = grp[grp.site_idx >= 0].sort_values("site_idx")
                self._sites[key] = [self._row_to_site(r) for r in grp.to_dict("records")]

    @staticmethod
    def _row_to_site(row: dict) -> CandidateSite:
        return CandidateSite(**{k: row[k] for k in _SITE_FIELDS})

    def __len__(self) -> int:
        return len(self._sites)

    def __contains__(self, key: str) -> bool:
        return key in self._sites

    def get(self, key: str) -> list[CandidateSite] | None:
        """Fresh copies of the stored sites of a pair, or ``None`` if the pair was never computed."""
        stored = self._sites.get(key)
        if stored is None:
            return None
        return [_copy(s) for s in stored]

    def add(self, key: str, sites: list[CandidateSite]) -> None:
        """Remember the sites of a pair in memory (call :meth:`flush` to write them out)."""
        snapshot = [_copy(s) for s in sites]
        self._sites[key] = self._pending[key] = snapshot

    @staticmethod
    def _site_to_row(key: str, idx: int, s: CandidateSite) -> dict:
        return {"pair_key": key, "site_idx": idx, **{k: getattr(s, k) for k in _SITE_FIELDS}}

    def flush(self) -> Path | None:
        """Write the pairs added since the last flush as one new shard; returns its path."""
        pending = self._pending
        if not pending:
            return None
        rows = []
        for key, sites in pending.items():
            if not sites:  # keep "this pair has no candidate sites" as a marker row
                rows.append(self._site_to_row(key, -1, CandidateSite()))
            rows += [self._site_to_row(key, i, s) for i, s in enumerate(sites)]
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"sites-{uuid.uuid4().hex[:12]}.parquet"
        tmp = path.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        tmp.replace(path)  # atomic: a half-written shard is never picked up by later loads
        self._pending = {}
        return path
