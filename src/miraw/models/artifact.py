"""Model artefacts: weights + the exact encoder/architecture they were trained with.

Loading a model with a mismatching encoder is impossible by construction: the encoder is
rebuilt from the artefact, never from the caller's configuration.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from miraw.config import resolve_model_path
from miraw.encoding.onehot import OneHotFlat
from miraw.models.base import SiteClassifier, build_model


@contextlib.contextmanager
def single_thread():
    """Run torch on one thread, then restore the caller's setting. All scoring runs this way, so scores are
    bit-identical whichever path computes them (the thread count changes the float summation order)."""
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(before)


def build_encoder(cfg: dict) -> OneHotFlat:
    cfg = dict(cfg)
    name = cfg.pop("name", "onehot_flat")
    if name != "onehot_flat":
        raise ValueError(f"unknown encoder {name!r}")
    return OneHotFlat(**cfg)


def save_artifact(path: str | Path, model: SiteClassifier, encoder: OneHotFlat, meta: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.config(),
            "encoder": encoder.config(),
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "meta": meta or {},
        },
        path,
    )


def load_artifact(path: str | Path) -> tuple[SiteClassifier, OneHotFlat, dict]:
    """Load a ``.pt`` artefact, or a DL4J ``.bin`` (assumed trained by miRAW: old-default encoding)."""
    path = Path(path)
    if path.suffix == ".bin":
        from miraw.models.dl4j_import import load_dl4j

        model, cfg = load_dl4j(path)
        site_len = (model.layer_sizes[0] - 120) // 4
        return model, OneHotFlat(30, site_len, "old-default"), {"source": "dl4j", "file": path.name}
    blob = torch.load(path, map_location="cpu", weights_only=True)  # tensors and plain data only, no code
    mcfg = dict(blob["model"])
    model = build_model(mcfg.pop("name"), **mcfg)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), build_encoder(blob["encoder"]), blob.get("meta", {})


@dataclass
class Model:
    """A site classifier together with the encoder it was trained with and its metadata.

    ``Model.load("original")`` gives the miRAW paper model (bundled with the package); any other
    argument is a ``.pt`` artefact written by :meth:`save` / ``miraw train``, or a DL4J ``.bin``.
    """

    network: SiteClassifier
    encoder: OneHotFlat
    meta: dict = field(default_factory=dict)
    source: str | None = None  # the file it was loaded from, if any

    @classmethod
    def load(cls, model: str | Path = "original") -> Model:
        path = resolve_model_path(model)
        network, encoder, meta = load_artifact(path)
        return cls(network, encoder, meta, path)

    def save(self, path: str | Path) -> Path:
        save_artifact(path, self.network, self.encoder, self.meta)
        self.source = str(path)
        return Path(path)

    @property
    def is_original(self) -> bool:
        from miraw.storage import original_weights_path

        return self.source is not None and Path(self.source) == original_weights_path()

    @torch.no_grad()
    def score_sites(self, mirna: str, sites: list[str]) -> np.ndarray:
        """miRAW site scores in [-1, 1] (``p(target) - p(no target)``) of site sequences against one miRNA."""
        if not sites:
            return np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(self.encoder.encode_batch(mirna, list(sites)))
        with single_thread():
            return self.network.eval().site_score(x).numpy()

    def __repr__(self) -> str:
        name = "original miRAW" if self.is_original else (self.source or "in-memory")
        return f"Model({name}, {self.network.config().get('name')}, site_orientation={self.encoder.site_orientation!r})"
