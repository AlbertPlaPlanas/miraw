"""Model artefacts: weights + the exact encoder/architecture they were trained with.

Loading a model with a mismatching encoder is impossible by construction: the encoder is
rebuilt from the artefact, never from the caller's configuration.
"""

from __future__ import annotations

from pathlib import Path

import torch

from pymiraw.encoding.onehot import OneHotFlat
from pymiraw.models.base import SiteClassifier, build_model


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
        from pymiraw.models.dl4j_import import load_dl4j

        model, cfg = load_dl4j(path)
        site_len = (model.layer_sizes[0] - 120) // 4
        return model, OneHotFlat(30, site_len, "old-default"), {"source": "dl4j", "file": path.name}
    blob = torch.load(path, map_location="cpu", weights_only=False)
    mcfg = dict(blob["model"])
    model = build_model(mcfg.pop("name"), **mcfg)
    model.load_state_dict(blob["state_dict"])
    return model.eval(), build_encoder(blob["encoder"]), blob.get("meta", {})
