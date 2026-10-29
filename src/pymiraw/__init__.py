"""pyMIRAW: Python/PyTorch reimplementation of miRAW (miRNA target prediction with deep learning).

Quick start (the original miRAW weights are bundled; nothing to download)::

    import pymiraw
    preds = pymiraw.predict("pairs.tsv")          # or a DataFrame, or [(mirna_seq, utr), ...]
    preds.genes                                    # one row per pair; IsTarget = Prediction > 0
    pymiraw.predict_pair("UGAGGUAGUAGGUUGUAUAGUU", utr_sequence).is_target

The names below are imported lazily, so ``import pymiraw`` is fast.
"""

from __future__ import annotations

from importlib import import_module

__version__ = "0.2.0"

_EXPORTS = {
    # high-level API
    "predict": "pymiraw.api",
    "predict_pair": "pymiraw.api",
    "evaluate": "pymiraw.api",
    "load_model": "pymiraw.api",
    "train": "pymiraw.api",
    "cross_validate": "pymiraw.api",
    "encode_sites": "pymiraw.api",
    # configuration and result types
    "PredictConfig": "pymiraw.config",
    "load_config": "pymiraw.config",
    "TrainConfig": "pymiraw.config",
    "Model": "pymiraw.models.artifact",
    "Predictions": "pymiraw.prediction.pipeline",
    "PairResult": "pymiraw.prediction.pipeline",
    # bundled data
    "example_pairs": "pymiraw.resources",
    "example_sites": "pymiraw.resources",
    # extension points
    "register_model": "pymiraw.models",
    "SiteClassifier": "pymiraw.models",
}
__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        value = getattr(import_module(_EXPORTS[name]), name)
    else:  # a submodule, e.g. pymiraw.config or pymiraw.storage
        try:
            value = import_module(f"pymiraw.{name}")
        except ModuleNotFoundError as e:
            if e.name != f"pymiraw.{name}":
                raise
            raise AttributeError(f"module 'pymiraw' has no attribute {name!r}") from None
    globals()[name] = value
    return value


def __dir__():
    return sorted([*globals(), *__all__])
