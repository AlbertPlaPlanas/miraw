"""Released miRAW training data: headerless CSV rows ``label, 4*(30+40) one-hot features``.

The released features were produced with the original (reversed-site) encoding. ``reorient``
converts them to the other orientation by flipping the site block, which is exactly equivalent
to re-encoding from sequences.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from miraw.encoding.onehot import normalize_orientation

MIRNA_LEN, SITE_LEN = 30, 40


def load_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(path, delimiter=",", dtype=np.float32)
    return d[:, 1:], d[:, 0].astype(np.int64)


def reorient(x: np.ndarray, orientation: str, mirna_len: int = MIRNA_LEN, site_len: int = SITE_LEN) -> np.ndarray:
    """Released features are ``old-default``; return them in the requested orientation."""
    if normalize_orientation(orientation) == "old-default":
        return x
    out = x.copy()
    block = out[:, 4 * mirna_len :].reshape(len(x), site_len, 4)[:, ::-1, :]
    out[:, 4 * mirna_len :] = block.reshape(len(x), -1)
    return out


def load_fold(folds_dir: str | Path, k: int, orientation: str = "old-default"):
    d = Path(folds_dir)
    xtr, ytr = load_csv(d / f"set_{k}_train.csv")
    xte, yte = load_csv(d / f"set_{k}_test.csv")
    return reorient(xtr, orientation), ytr, reorient(xte, orientation), yte


def split_validation(x: np.ndarray, y: np.ndarray, frac: float, seed: int):
    """Stratified train/validation split (deterministic for a seed)."""
    rng = np.random.default_rng(seed)
    val = np.zeros(len(y), bool)
    for c in np.unique(y):
        idx = rng.permutation(np.nonzero(y == c)[0])
        val[idx[: int(round(frac * len(idx)))]] = True
    return x[~val], y[~val], x[val], y[val]


def loader(x: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)
