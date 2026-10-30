"""miRAW: Python/PyTorch reimplementation of the original miRAW (miRNA target prediction with deep learning).

Quick start (the original miRAW weights are bundled; nothing to download)::

    import miraw
    preds = miraw.predict("pairs.tsv")          # or a DataFrame, or [(mirna_seq, utr), ...]
    preds.genes                                    # one row per pair; IsTarget = Prediction > 0
    miraw.predict_pair("UGAGGUAGUAGGUUGUAUAGUU", utr_sequence).is_target

The names below are imported lazily, so ``import miraw`` is fast.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.3.0"

_EXPORTS = {
    # high-level API
    "predict": "miraw.api",
    "predict_pair": "miraw.api",
    "evaluate": "miraw.api",
    "load_model": "miraw.api",
    "train": "miraw.api",
    "cross_validate": "miraw.api",
    "encode_sites": "miraw.api",
    # configuration and result types
    "PredictConfig": "miraw.config",
    "load_config": "miraw.config",
    "TrainConfig": "miraw.config",
    "Model": "miraw.models.artifact",
    "Predictions": "miraw.prediction.pipeline",
    "PairResult": "miraw.prediction.pipeline",
    # bundled data
    "example_pairs": "miraw.resources",
    "example_sites": "miraw.resources",
    # extension points
    "register_model": "miraw.models",
    "SiteClassifier": "miraw.models",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
    else:  # a submodule, e.g. miraw.config or miraw.storage
        try:
            value = import_module(f"miraw.{name}")
        except ModuleNotFoundError as e:
            if e.name != f"miraw.{name}":
                raise
            raise AttributeError(f"module 'miraw' has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *__all__])
