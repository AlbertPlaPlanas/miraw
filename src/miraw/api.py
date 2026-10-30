"""The high-level miRAW API: one call per task. Everything is also importable from ``miraw`` directly.

>>> import miraw
>>> preds = miraw.predict(miraw.example_pairs())   # the original miRAW weights, bundled
>>> preds.genes                                         # one row per miRNA-gene pair
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from miraw.config import PredictConfig, TrainConfig
from miraw.encoding.onehot import OneHotFlat
from miraw.models.artifact import Model
from miraw.prediction.pipeline import PairResult, PairsLike, Predictions, predict_pairs

__all__ = ["cross_validate", "encode_sites", "evaluate", "load_model", "predict", "predict_pair", "train"]


def load_model(model: str | Path | Model = "original") -> Model:
    """``"original"`` (the miRAW paper weights, bundled), a ``.pt`` artefact or a DL4J ``.bin`` file."""
    return model if isinstance(model, Model) else Model.load(model)


def _config(config: PredictConfig | None, model, options: dict) -> tuple[PredictConfig, Model]:
    m = load_model(model if model is not None else (config.model if config else "original"))
    fields = {k: v for k, v in options.items() if v is not None}
    if "filters" in fields:
        fields["filters"] = [fields["filters"]] if isinstance(fields["filters"], str) else list(fields["filters"])
    base = config.model_dump() if config else {}
    ref = "original" if m.is_original else (m.source or "in-memory")
    return PredictConfig(**{**base, **fields, "model": ref, "original_weights": False}), m


def predict(
    pairs: PairsLike,
    model: str | Path | Model | None = None,
    *,
    finder: str | None = None,
    filters: list[str] | str | None = None,
    backend: str | None = None,
    vienna_bin: str | None = None,
    workers: int | None = None,
    progress: bool = True,
    config: PredictConfig | None = None,
    **options,
) -> Predictions:
    """Predict which miRNA-gene pairs are targets.

    ``pairs``: a unified TSV file (the original miRAW input), a DataFrame (``mirna_seq`` and ``utr`` columns,
    or the unified ones) or a list of ``(mirna_seq, utr)`` tuples. ``model``: ``"original"`` (default),
    a :class:`Model` or a model file. ``finder``: ``"miRAW"`` (default), ``"Personalized;1;9;7"``, ``"PITA"``,
    ``"TargetScan"``. ``filters``: e.g. ``"AccessibilityEnergy;10"``. ``backend``/``vienna_bin``: which
    ViennaRNA to use (see the README). ``config``: a full :class:`PredictConfig`; keyword arguments override it.
    Other keywords (``cache_dir``, ``sites_dir``, ``max_site_length``...) are :class:`PredictConfig` fields.
    """
    opts = dict(finder=finder, filters=filters, backend=backend, vienna_bin=vienna_bin, workers=workers, **options)
    cfg, m = _config(config, model, opts)
    results = predict_pairs(pairs, cfg, model=m, workers=cfg.workers, progress=progress)
    return Predictions(results, cfg, "original miRAW" if m.is_original else (m.source or "in-memory model"))


def predict_pair(mirna_seq: str, utr: str, model: str | Path | Model | None = None, **options) -> PairResult:
    """Predict a single miRNA (mature sequence) against a 3'UTR. ``.prediction`` is in [-1, 1],
    ``.is_target`` is ``prediction > 0`` and ``.sites`` holds the scored candidate sites."""
    return predict([(mirna_seq, utr)], model, workers=1, progress=False, **options).results[0]


def evaluate(pairs: PairsLike, model: str | Path | Model | None = None, **options) -> dict:
    """Predict labelled pairs and return the metrics (TP/FP/TN/FN, accuracy, F1...)."""
    return predict(pairs, model, **options).metrics()


def encode_sites(mirnas, sites, site_orientation: str = "old-default") -> np.ndarray:
    """One-hot encode (miRNA, site) sequence pairs into the network's input rows (as the released data)."""
    enc = OneHotFlat(site_orientation=site_orientation)
    return np.stack([enc.encode(m, s) for m, s in zip(mirnas, sites, strict=True)]).astype(np.float32)


def _xy(data, orientation: str) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(data, pd.DataFrame):
        return encode_sites(data["mirna_seq"], data["site_seq"], orientation), np.array(data["label"], np.int64)
    x, y = data
    return np.array(x, np.float32), np.array(y, np.int64)


def train(data, test=None, config: TrainConfig | None = None, **overrides) -> Model:
    """Train a site classifier and return it as a :class:`Model` (``.save(path)`` to keep it).

    ``data`` / ``test``: a DataFrame of sites (``mirna_seq``, ``site_seq``, ``label``), or ``(x, y)``
    arrays already encoded in ``config.site_orientation`` (e.g. from :func:`miraw.training.data.load_fold`).
    ``config`` defaults to :meth:`TrainConfig.paper` (the paper's regime); keywords override its fields.
    The test-set metrics are in ``model.meta["metrics"]``.
    """
    from miraw.training.trainer import train_fold

    cfg = TrainConfig(**{**asdict(config or TrainConfig.paper()), **overrides})
    x, y = _xy(data, cfg.site_orientation)
    xt, yt = _xy(test, cfg.site_orientation) if test is not None else (None, None)
    net, metrics, encoder = train_fold(cfg, x, y, xt, yt)
    return Model(net, encoder, {"metrics": metrics, "train_config": asdict(cfg)})


def cross_validate(folds_dir, config: TrainConfig | None = None, folds=range(9), seeds=(0,), out_dir=None):
    """Train on the released CV folds (``set_<k>_train.csv`` / ``set_<k>_test.csv``) and return one row of
    test metrics per (fold, seed); with ``out_dir`` the models are saved there."""
    from miraw.training.cv import run_cv

    return run_cv(folds_dir, config or TrainConfig.paper(), folds, seeds, out_dir)
