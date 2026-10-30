"""Loss functions matching the two options of the original (ANN.LossFunction)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def nll(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative log-likelihood of the true class (DL4J ``NEGATIVELOGLIKELIHOOD`` / MCXENT)."""
    return F.cross_entropy(logits, target)


def xent(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """DL4J ``XENT`` (binary cross entropy summed over the two softmax outputs).

    For a one-hot 2-class target this equals ``-2 * log p(true class)``: twice the NLL, which
    also doubles the effective learning rate of the original runs.
    """
    return 2.0 * F.cross_entropy(logits, target)


LOSSES = {"nll": nll, "xent": xent}


def get_loss(name: str):
    try:
        return LOSSES[name.lower()]
    except KeyError:
        raise KeyError(f"unknown loss {name!r}; choose from {sorted(LOSSES)}") from None
