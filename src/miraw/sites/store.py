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

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from miraw.sites.candidate import CandidateSite

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
    """Read/append access to a directory of candidate-site shards.

    Shards are read into one table, but :class:`CandidateSite` objects are only built for the pairs
    that are asked for (:meth:`get`), and ``keys`` limits the read to the pairs of the current run.
    """

    def __init__(self, directory: str | Path, keys: list[str] | None = None):
        self.dir = Path(directory)
        self._added: dict[str, list[CandidateSite]] = {}  # computed in this session
        self._pending: dict[str, list[CandidateSite]] = {}  # added since the last flush
        self._rows: list[dict] = []
        self._span: dict[str, tuple[int, int]] = {}
        self._load_existing(keys)

    def _load_existing(self, keys: list[str] | None) -> None:
        """Read the shards (only ``keys`` if given) into one table sorted by pair and site index."""
        shards = sorted(self.dir.glob("sites-*.parquet"))
        if not shards:
            return
        flt = [("pair_key", "in", list(keys))] if keys is not None else None
        table = pa.concat_tables([pq.read_table(f, filters=flt) for f in shards])
        if table.num_rows == 0:
            return
        df = table.to_pandas().drop_duplicates(["pair_key", "site_idx"], keep="last")
        df = df.sort_values(["pair_key", "site_idx"], kind="stable").reset_index(drop=True)
        self._rows = df  # type: ignore[assignment]
        pk = df["pair_key"].to_numpy()
        starts = np.flatnonzero(np.r_[True, pk[1:] != pk[:-1]])
        stops = np.r_[starts[1:], len(pk)]
        self._span = {pk[a]: (a, b) for a, b in zip(starts, stops, strict=True)}

    @staticmethod
    def _row_to_site(row: dict) -> CandidateSite:
        return CandidateSite(**{k: row[k] for k in _SITE_FIELDS})

    def __len__(self) -> int:
        return len(self._span.keys() | self._added.keys())

    def __contains__(self, key: str) -> bool:
        return key in self._added or key in self._span

    def get(self, key: str) -> list[CandidateSite] | None:
        """Fresh copies of the stored sites of a pair, or ``None`` if the pair was never computed."""
        if key in self._added:
            return [_copy(s) for s in self._added[key]]
        span = self._span.get(key)
        if span is None:
            return None
        rows = self._rows.iloc[span[0] : span[1]]  # type: ignore[attr-defined]
        return [self._row_to_site(r) for r in rows.to_dict("records") if r["site_idx"] >= 0]

    def add(self, key: str, sites: list[CandidateSite]) -> None:
        """Remember the sites of a pair in memory (call :meth:`flush` to write them out)."""
        snapshot = [_copy(s) for s in sites]
        self._added[key] = self._pending[key] = snapshot

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
        path = _write_shard(self.dir, "sites", pd.DataFrame(rows))
        self._pending = {}
        return path


def _write_shard(directory: Path, prefix: str, df: pd.DataFrame) -> Path:
    """Write ``df`` as a new ``<prefix>-<id>.parquet`` shard (atomic: half-written files are never read)."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}-{uuid.uuid4().hex[:12]}.parquet"
    tmp = path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return path


# --- accessibility (opening) energies -----------------------------------------------------------
def utr_key(backend_id_: str, utr: str) -> str:
    """Short hash of a 3'UTR under a folding backend/version (the first part of an opening-energy key)."""
    return hashlib.sha1(f"{backend_id_}|{utr}".encode()).hexdigest()[:20]


class OpeningStore:
    """Saved ΔG_open values of sites, ``{(utr_key, start, end): energy}``.

    ΔG_open depends only on the 3'UTR and the site's position, not on the miRNA, finder, model or
    filter threshold, so one value serves every run. Stored next to the sites as ``opening-*.parquet``.
    """

    def __init__(self, directory: str | Path, utr_keys: list[str] | None = None):
        self.dir = Path(directory)
        self.energy: dict[tuple[str, int, int], float] = {}
        self._pending: dict[tuple[str, int, int], float] = {}
        shards = sorted(self.dir.glob("opening-*.parquet"))
        if shards:
            flt = [("utr_key", "in", list(utr_keys))] if utr_keys is not None else None
            df = pa.concat_tables([pq.read_table(f, filters=flt) for f in shards]).to_pandas()
            self.energy = {
                (k, int(a), int(b)): float(e)
                for k, a, b, e in zip(df.utr_key, df.start, df.end, df.energy, strict=True)
            }

    def add(self, key: tuple[str, int, int], energy: float) -> None:
        if key not in self.energy:
            self.energy[key] = self._pending[key] = energy

    def flush(self) -> Path | None:
        if not self._pending:
            return None
        df = pd.DataFrame(
            [(k, a, b, e) for (k, a, b), e in self._pending.items()], columns=["utr_key", "start", "end", "energy"]
        )
        self._pending = {}
        return _write_shard(self.dir, "opening", df)
