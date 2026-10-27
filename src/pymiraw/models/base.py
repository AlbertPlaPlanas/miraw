"""Model interface and registry. New architectures register themselves by name."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

_REGISTRY: dict[str, Callable[..., SiteClassifier]] = {}


class SiteClassifier(nn.Module):
    """Maps an encoded (miRNA, site) batch to logits ``(B, 2)``: class 0 = no target, 1 = target."""

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self(x), dim=-1)

    def site_score(self, x: torch.Tensor) -> torch.Tensor:
        """miRAW site score ``p(target) - p(no target)`` in [-1, 1]; positive means target."""
        p = self.predict_proba(x)
        return p[:, 1] - p[:, 0]


def register_model(name: str):
    def deco(factory):
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        _REGISTRY[name] = factory
        return factory

    return deco


def build_model(name: str, **params) -> SiteClassifier:
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**params)


def registered_models() -> list[str]:
    return sorted(_REGISTRY)
