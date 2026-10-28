"""Gene-level prediction: find candidate sites -> encode -> score -> filter -> aggregate (max)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from pymiraw.config import PredictConfig
from pymiraw.encoding.onehot import OneHotFlat
from pymiraw.folding.base import FoldService
from pymiraw.folding.vienna_cli import make_backend
from pymiraw.models.artifact import Model, single_thread
from pymiraw.models.base import SiteClassifier
from pymiraw.sites.candidate import CandidateSite
from pymiraw.sites.filters.accessibility import get_filter
from pymiraw.sites.finders.registry import get_site_finder
from pymiraw.sites.store import OpeningStore, SiteStore, backend_id, pair_key, utr_key

log = logging.getLogger(__name__)

PairsLike = str | Path | pd.DataFrame | Iterable


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

    @property
    def is_target(self) -> bool:
        """The miRAW call: the pair is a target when its prediction (best site score, in [-1, 1]) is > 0."""
        return self.prediction > 0


def make_folding_backend(cfg: PredictConfig):
    """The ViennaRNA backend of a run: ``bindings`` (Python package) or ``cli`` (tools in ``vienna_bin``)."""
    if cfg.vienna_bin and cfg.backend not in ("cli", "vienna-cli"):
        raise ValueError("vienna_bin is only used by backend='cli' (the ViennaRNA command-line tools)")
    return make_backend(cfg.backend, **({"bin_dir": cfg.vienna_bin} if cfg.vienna_bin else {}))


class Predictor:
    def __init__(self, model: SiteClassifier, encoder: OneHotFlat, finder, folds: FoldService, filters=()):
        self.model, self.encoder, self.finder, self.folds = model.eval(), encoder, finder, folds
        self.filters = list(filters)

    @classmethod
    def from_config(cls, cfg: PredictConfig, model: Model | None = None, opening_cache: dict | None = None):
        """Build the finder, filters and folding service of ``cfg`` around ``model`` (default: ``cfg.model``)."""
        model = model or Model.load(cfg.model)
        folds = FoldService(make_folding_backend(cfg), cfg.cache_dir)
        folds.opening_cache = dict(opening_cache or {})  # saved ΔG_open values (read-only here)
        finder = get_site_finder(cfg.finder, cfg.max_site_length, cfg.seed_alignment_offset)
        return cls(model.network, model.encoder, finder, folds, [get_filter(f) for f in cfg.filters])

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


# --- input ------------------------------------------------------------------------------------
UNIFIED_COLUMNS = {"Mature_mirna_transcript": "mirna_seq", "3UTR_transcript": "utr"}
_SYNONYMS = {"label": "Positive_Negative", "utr_seq": "utr", "mirna_name": "miRNA", "gene": "gene_name"}


def read_unified(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False).rename(columns=UNIFIED_COLUMNS)


def pairs_frame(pairs: PairsLike) -> pd.DataFrame:
    """Normalise the accepted inputs to one table with columns ``miRNA, gene_name, EnsemblId, mirna_seq, utr``
    (+ ``Positive_Negative`` when labelled).

    Accepted: the path of a unified TSV file (the original miRAW input); a DataFrame with the unified
    columns or ``mirna_seq`` and ``utr`` (optional: ``miRNA``, ``gene_name``, ``EnsemblId``, ``label``);
    or an iterable of ``(mirna_seq, utr)`` tuples or of dicts with those keys.
    """
    if isinstance(pairs, str | Path):
        df = read_unified(pairs)
    elif isinstance(pairs, pd.DataFrame):
        df = pairs.rename(columns={**UNIFIED_COLUMNS, **_SYNONYMS})
    else:
        rows = list(pairs)
        if rows and not isinstance(rows[0], dict):
            rows = [dict(zip(("mirna_seq", "utr"), r, strict=True)) for r in rows]
        df = pd.DataFrame(rows, columns=None if rows else ["mirna_seq", "utr"]).rename(columns=_SYNONYMS)
    missing = {"mirna_seq", "utr"} - set(df.columns)
    if missing:
        raise ValueError(
            f"pairs need the columns {sorted(missing)} (or Mature_mirna_transcript / 3UTR_transcript); got {list(df)}"
        )
    df = df.copy()
    for col in ("miRNA", "gene_name", "EnsemblId"):
        if col not in df:
            df[col] = ""
    df[["miRNA", "gene_name", "EnsemblId"]] = df[["miRNA", "gene_name", "EnsemblId"]].fillna("").astype(str)
    return df.reset_index(drop=True)


def _rows(df: pd.DataFrame) -> list[tuple]:
    labels = df.get("Positive_Negative")
    has_label = labels is not None and bool(labels.notna().all() and (labels.astype(str) != "").all())
    return [
        (r.gene_name, r.EnsemblId, r.miRNA, r.mirna_seq, r.utr, int(r.Positive_Negative) if has_label else None)
        for r in df.itertuples()
    ]


# --- parallel driver -------------------------------------------------------------------------
_WORKER: Predictor | None = None


def _init_worker(cfg: PredictConfig, model_path: str, opening_cache: dict | None) -> None:
    global _WORKER
    torch.set_num_threads(1)
    _WORKER = Predictor.from_config(cfg, Model.load(model_path), opening_cache)


def _predict_row(predictor: Predictor, row: tuple) -> PairResult:
    gene, ens, mirna, mirna_seq, utr, label, sites = row
    res = predictor.predict_pair(mirna_seq, utr, gene, ens, label, sites)
    res.mirna = mirna
    return res


def _run_row(row: tuple) -> PairResult:
    assert _WORKER is not None
    return _predict_row(_WORKER, row)


def _as_config(cfg: PredictConfig | dict) -> PredictConfig:
    return cfg if isinstance(cfg, PredictConfig) else PredictConfig(**cfg)


def predict_pairs(
    pairs: PairsLike,
    cfg: PredictConfig | dict | None = None,
    model: Model | None = None,
    workers: int | None = None,
    progress: bool = False,
) -> list[PairResult]:
    """Run the whole pipeline on ``pairs`` (see :func:`pairs_frame`). Returns one PairResult per pair.

    ``model`` is a loaded :class:`Model` (default: ``cfg.model``, the original weights unless set).
    With ``cfg.sites_dir`` (directory of saved candidate sites) pairs found there skip ViennaRNA and new pairs
    are computed and saved; ``cfg.require_sites`` errors out instead of computing missing pairs.
    ``workers``: processes (default ``cfg.workers``, else one per CPU, at most one per pair).
    """
    cfg = _as_config(cfg or PredictConfig())
    rows = _rows(pairs_frame(pairs))
    store, keys, opening, bid = None, [], None, ""
    if cfg.sites_dir:
        store, keys, opening, bid = _open_store(cfg, rows)
    # attach the saved sites (or None = compute them) as the last element of each row
    saved = [store.get(k) for k in keys] if store is not None else [None] * len(rows)
    rows = [(*r, sites) for r, sites in zip(rows, saved, strict=True)]
    missing = [i for i, r in enumerate(rows) if store is not None and r[-1] is None]
    if missing and cfg.require_sites:
        raise LookupError(
            f"{len(missing)} of {len(rows)} pairs have no saved sites in {cfg.sites_dir} "
            "(they were saved with other finder settings, backend/version, or never computed)"
        )
    if store is not None:
        n_saved = len(rows) - len(missing)
        log.info("sites store %s: %d pairs loaded, %d to compute", cfg.sites_dir, n_saved, len(missing))
    workers = max(1, min(workers or cfg.workers or os.cpu_count() or 1, len(rows)))
    opening_cache = opening.energy if opening is not None else None
    bar = {"total": len(rows), "disable": not progress, "unit": "pair", "desc": "pyMIRAW"}
    if workers == 1:
        predictor = Predictor.from_config(cfg, model or Model.load(cfg.model), opening_cache)
        with single_thread():
            results = [_predict_row(predictor, r) for r in tqdm(rows, **bar)]
    else:
        with tempfile.TemporaryDirectory() as tmp:
            if model is None:
                model_path = Model.load(cfg.model).source
            else:  # an in-memory model reaches the worker processes through a temporary artefact
                model_path = str(Model(model.network, model.encoder, model.meta).save(Path(tmp) / "model.pt"))
            init = (cfg, model_path, opening_cache)
            with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=init) as ex:
                results = list(tqdm(ex.map(_run_row, rows, chunksize=1), **bar))
    if store is not None:
        for i in missing:  # save what was just computed so the next run can skip it
            store.add(keys[i], results[i].sites)
        store.flush()
    if opening is not None:
        _save_opening_energies(opening, bid, rows, results)
    return results


def predict_file(
    input_file: str | Path, cfg: PredictConfig | dict, workers: int | None = None, limit: int | None = None
) -> list[PairResult]:
    """:func:`predict_pairs` on a unified file (optionally only its first ``limit`` pairs)."""
    df = read_unified(input_file)
    return predict_pairs(df.head(limit) if limit else df, cfg, workers=workers)


def _open_store(cfg: PredictConfig, rows: list[tuple]):
    """Open the sites directory for this run: returns (site store, row keys, opening-energy store, backend id).

    Only the pairs of this run are read from disk. The saved opening energies go to the workers
    (the accessibility filter reuses them instead of folding).
    """
    bid = backend_id(make_folding_backend(cfg))
    finder = get_site_finder(cfg.finder, cfg.max_site_length, cfg.seed_alignment_offset)
    keys = [pair_key(bid, finder, r[3], r[4]) for r in rows]
    opening = OpeningStore(cfg.sites_dir, sorted({utr_key(bid, r[4]) for r in rows}))
    return SiteStore(cfg.sites_dir, keys), keys, opening, bid


def _save_opening_energies(opening: OpeningStore, bid: str, rows: list[tuple], results: list[PairResult]) -> None:
    """Save the ΔG_open values the accessibility filter computed in this run (new ones only)."""
    for row, res in zip(rows, results, strict=True):
        key = utr_key(bid, row[4])
        for s in res.sites:
            if "AccessibilityEnergy" in s.properties:
                opening.add((key, s.start, s.end), s.properties["AccessibilityEnergy"])
    opening.flush()


# --- output ----------------------------------------------------------------------------------
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


@dataclass
class Predictions:
    """The result of a prediction run.

    ``genes``: one row per miRNA-gene pair (``Prediction`` in [-1, 1]; ``IsTarget`` = ``Prediction > 0``).
    ``sites``: one row per candidate site with its score. ``metrics()``: accuracy, F1... when labelled.
    ``save(folder)`` writes the original miRAW output files.
    """

    results: list[PairResult]
    config: PredictConfig
    model: str = "original miRAW"

    @cached_property
    def genes(self) -> pd.DataFrame:
        df = results_frame(self.results)
        df.insert(4, "IsTarget", df["Prediction"] > 0)
        return df

    @cached_property
    def sites(self) -> pd.DataFrame:
        return sites_frame(self.results)

    @property
    def has_labels(self) -> bool:
        return bool(self.results) and all(r.label is not None for r in self.results)

    def metrics(self) -> dict:
        """TP/FP/TN/FN, accuracy, precision, recall, specificity, F1 (a pair is a target when Prediction > 0)."""
        from pymiraw.evaluation.metrics import gene_level_metrics

        if not self.has_labels:
            raise ValueError("the input has no labels (Positive_Negative / label column)")
        return gene_level_metrics([r.label for r in self.results], [r.prediction for r in self.results])

    def save(self, folder: str | Path, metrics: bool | None = None) -> Path:
        """Write ``targetPredictionOutput.csv``, ``targetSites.csv``, ``config.yaml`` (and ``metrics.json`` when
        labelled, or when ``metrics=True``) into ``folder``."""
        out = Path(folder)
        out.mkdir(parents=True, exist_ok=True)
        results_frame(self.results).to_csv(out / "targetPredictionOutput.csv", sep="\t", index=False)
        self.sites.to_csv(out / "targetSites.csv", sep="\t", index=False)
        (out / "config.yaml").write_text(yaml.safe_dump(self.config.model_dump()))
        if metrics or (metrics is None and self.has_labels):
            (out / "metrics.json").write_text(json.dumps(self.metrics(), indent=2))
        return out

    def __len__(self) -> int:
        return len(self.results)

    def __repr__(self) -> str:
        n_t = sum(r.is_target for r in self.results)
        return (
            f"Predictions({len(self)} pairs, {n_t} predicted targets, weights: {self.model}, "
            f"finder={self.config.finder})"
        )
