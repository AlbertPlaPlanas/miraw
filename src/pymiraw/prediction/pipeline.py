"""Gene-level prediction: find candidate sites -> encode -> score -> filter -> aggregate (max)."""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pymiraw.encoding.onehot import OneHotFlat
from pymiraw.folding.base import FoldService
from pymiraw.folding.vienna_cli import make_backend
from pymiraw.models.artifact import load_artifact
from pymiraw.models.base import SiteClassifier
from pymiraw.sites.candidate import CandidateSite
from pymiraw.sites.filters.accessibility import get_filter
from pymiraw.sites.finders.registry import get_site_finder
from pymiraw.sites.store import OpeningStore, SiteStore, backend_id, pair_key, utr_key

log = logging.getLogger(__name__)


@dataclass
class PairResult:
    gene_name: str
    ensembl_id: str
    mirna: str
    label: int | None
    prediction: float = 0.0
    max_val: float = 0.0
    min_val: float = 0.0
    pos_sites: int = 0
    neg_sites: int = 0
    removed_sites: int = 0
    sites: list[CandidateSite] = field(default_factory=list)


class Predictor:
    def __init__(self, model: SiteClassifier, encoder: OneHotFlat, finder, folds: FoldService, filters=()):
        self.model, self.encoder, self.finder, self.folds = model.eval(), encoder, finder, folds
        self.filters = list(filters)

    @torch.no_grad()
    def score_sites(self, mirna: str, sites: list[CandidateSite]) -> np.ndarray:
        if not sites:
            return np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(self.encoder.encode_batch(mirna, [s.transcript for s in sites]))
        return self.model.site_score(x).numpy()

    def predict_pair(
        self, mirna: str, utr: str, gene_name="", ensembl_id="", label=None, sites: list[CandidateSite] | None = None
    ) -> PairResult:
        """Predict one pair. ``sites`` = pre-computed candidate sites (skips the ViennaRNA site search)."""
        if sites is None:
            sites = self.finder.find_candidates(mirna, utr, self.folds)
        scores = self.score_sites(mirna, sites)
        final, removed = [], 0
        for site, score in zip(sites, scores, strict=True):
            site.score = float(score)
            s = float(score)
            if s > 0 and self.filters:
                if not all(f.evaluate(site, utr, self.folds) for f in self.filters):
                    removed += 1
                    s = 0.0
            final.append(s)
        res = PairResult(gene_name, ensembl_id, mirna, label, sites=sites, removed_sites=removed)
        if final:
            res.prediction = res.max_val = max(final)
            res.min_val = min(final)
            res.pos_sites = sum(v > 0 for v in final)
            res.neg_sites = len(final) - res.pos_sites
        return res


# --- parallel driver -------------------------------------------------------------------------
_WORKER: Predictor | None = None


def _init_worker(cfg: dict) -> None:
    global _WORKER
    torch.set_num_threads(1)
    model, encoder, _ = load_artifact(cfg["model"])
    backend = make_backend(cfg["backend"], **cfg.get("backend_args", {}))
    folds = FoldService(backend, cfg.get("cache_dir"))
    folds.opening_cache = dict(cfg.get("opening_cache") or {})  # saved ΔG_open values (read-only here)
    finder = get_site_finder(cfg["finder"], cfg["max_site_length"], cfg["seed_alignment_offset"])
    filters = [get_filter(f) for f in cfg.get("filters", [])]
    _WORKER = Predictor(model, encoder, finder, folds, filters)


def _run_row(row: tuple) -> PairResult:
    assert _WORKER is not None
    gene, ens, mirna, mirna_seq, utr, label, sites = row
    res = _WORKER.predict_pair(mirna_seq, utr, gene, ens, label, sites)
    res.mirna = mirna
    return res


def read_unified(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).rename(
        columns={"Mature_mirna_transcript": "mirna_seq", "3UTR_transcript": "utr"}
    )


def predict_file(input_file: str | Path, cfg: dict, workers: int | None = None, limit: int | None = None):
    """Run the whole pipeline on a unified file. ``cfg`` keys: model, finder, filters, backend,
    max_site_length, seed_alignment_offset, cache_dir, and optionally ``sites_dir`` (directory of saved
    candidate sites: pairs found there skip ViennaRNA, new pairs are computed and saved) and
    ``require_sites`` (error out instead of computing missing pairs). Returns the list of PairResult."""
    df = read_unified(input_file)
    if limit:
        df = df.head(limit)
    has_label = "Positive_Negative" in df
    rows = [
        (r.gene_name, r.EnsemblId, r.miRNA, r.mirna_seq, r.utr, int(r.Positive_Negative) if has_label else None)
        for r in df.itertuples()
    ]
    store, keys, opening = None, [], None
    if cfg.get("sites_dir"):
        store, keys, opening, cfg = _open_store(cfg, rows)
    # attach the saved sites (or None = compute them) as the last element of each row
    saved = [store.get(k) for k in keys] if store is not None else [None] * len(rows)
    rows = [(*r, sites) for r, sites in zip(rows, saved, strict=True)]
    missing = [i for i, r in enumerate(rows) if store is not None and r[-1] is None]
    if missing and cfg.get("require_sites"):
        raise LookupError(
            f"{len(missing)} of {len(rows)} pairs have no saved sites in {cfg['sites_dir']} "
            "(they were saved with other finder settings, backend/version, or never computed)"
        )
    if store is not None:
        log.info(
            "sites store %s: %d pairs loaded, %d to compute", cfg["sites_dir"], len(rows) - len(missing), len(missing)
        )
    workers = workers or os.cpu_count() or 1
    if workers == 1:
        _init_worker(cfg)
        results = [_run_row(r) for r in rows]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cfg,)) as ex:
            results = list(ex.map(_run_row, rows, chunksize=1))
    if store is not None:
        for i in missing:  # save what was just computed so the next run can skip it
            store.add(keys[i], results[i].sites)
        store.flush()
    if opening is not None:
        _save_opening_energies(opening, cfg, rows, results)
    return results


def _open_store(cfg: dict, rows: list[tuple]):
    """Open the sites directory for this run: returns (site store, row keys, opening-energy store, cfg).

    Only the pairs of this run are read from disk. The returned cfg carries the saved opening
    energies to the workers (the accessibility filter reuses them instead of folding).
    """
    backend = make_backend(cfg["backend"], **cfg.get("backend_args", {}))
    finder = get_site_finder(cfg["finder"], cfg["max_site_length"], cfg["seed_alignment_offset"])
    bid = backend_id(backend)
    keys = [pair_key(bid, finder, r[3], r[4]) for r in rows]
    opening = OpeningStore(cfg["sites_dir"], sorted({utr_key(bid, r[4]) for r in rows}))
    cfg = {**cfg, "opening_cache": opening.energy, "_backend_id": bid}
    return SiteStore(cfg["sites_dir"], keys), keys, opening, cfg


def _save_opening_energies(opening: OpeningStore, cfg: dict, rows: list[tuple], results: list[PairResult]) -> None:
    """Save the ΔG_open values the accessibility filter computed in this run (new ones only)."""
    for row, res in zip(rows, results, strict=True):
        key = utr_key(cfg["_backend_id"], row[4])
        for s in res.sites:
            if "AccessibilityEnergy" in s.properties:
                opening.add((key, s.start, s.end), s.properties["AccessibilityEnergy"])
    opening.flush()


def results_frame(results: list[PairResult]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "GeneName": [r.gene_name for r in results],
            "GeneId": [r.ensembl_id for r in results],
            "miRNA": [r.mirna for r in results],
            "Prediction": [r.prediction for r in results],
            "RealClass": [r.label for r in results],
            "HighestPredVal": [r.max_val for r in results],
            "LowestPredVal": [r.min_val for r in results],
            "PosSites": [r.pos_sites for r in results],
            "NegSites": [r.neg_sites for r in results],
            "RemovedSites": [r.removed_sites for r in results],
        }
    )


def sites_frame(results: list[PairResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for s in r.sites:
            rows.append(
                {
                    "GeneName": r.gene_name,
                    "miRNA": r.mirna,
                    "Start": s.start,
                    "End": s.end,
                    "Score": s.score,
                    "SeedPairs": s.binds_in_seed,
                    "WC": s.wc_pairs,
                    "Wobble": s.wobble_pairs,
                    "FreeEnergy": s.free_energy,
                    "Filtered": s.filtered,
                    "Transcript": s.transcript,
                    "Comment": s.comment,
                }
            )
    return pd.DataFrame(rows)
