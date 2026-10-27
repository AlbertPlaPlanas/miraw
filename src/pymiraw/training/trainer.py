"""Train one fold and evaluate it. Checkpoints on validation loss, and refuses collapsed models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pymiraw.encoding.onehot import OneHotFlat
from pymiraw.evaluation.metrics import gene_level_metrics, roc_auc
from pymiraw.models.base import build_model
from pymiraw.training.data import loader, split_validation
from pymiraw.training.module import SiteModule

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)


class CollapsedModelError(RuntimeError):
    """The trained model predicts a single class for (almost) every input."""


@dataclass
class TrainConfig:
    model: str = "mlp"
    model_params: dict = field(default_factory=dict)
    loss: str = "xent"
    lr: float = 0.005
    momentum: float = 0.9
    batch_size: int = 50
    max_epochs: int = 10
    patience: int = 0  # 0 = no early stopping
    val_fraction: float = 0.1
    shuffle: bool = True  # False = file order every epoch, as the original (DL4J) trained
    site_orientation: str = "old-default"
    seed: int = 0
    accelerator: str = "auto"


def predict_scores(model, x: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return model.site_score(torch.from_numpy(x)).cpu().numpy()


def check_not_collapsed(scores: np.ndarray, min_minority: float = 0.01) -> None:
    pos = float((scores > 0).mean())
    if pos < min_minority or pos > 1 - min_minority:
        raise CollapsedModelError(f"model predicts {pos:.1%} positives: collapsed")


def _build(cfg: TrainConfig, in_dim: int):
    params = dict(cfg.model_params)
    if cfg.model == "mlp":  # the paper architecture unless the config overrides it
        params.setdefault("layer_sizes", (in_dim, 400, 350, 300, 150, 100, 100, 50, 50, 2))
    return build_model(cfg.model, **params)


def train_fold(cfg: TrainConfig, xtr, ytr, xte, yte, workdir):
    """Returns (best model on validation loss, metrics on the test fold, encoder)."""
    L.seed_everything(cfg.seed, workers=True)
    # val_fraction=0: train on everything and keep the epoch with the lowest training loss (the paper's regime)
    if cfg.val_fraction > 0:
        x_fit, y_fit, x_val, y_val = split_validation(xtr, ytr, cfg.val_fraction, cfg.seed)
    else:
        x_fit, y_fit, x_val, y_val = xtr, ytr, None, None
    monitor = "val_loss" if x_val is not None else "train_loss"
    model = _build(cfg, xtr.shape[1])
    module = SiteModule(model, cfg.loss, cfg.lr, cfg.momentum)
    ckpt = ModelCheckpoint(dirpath=workdir, monitor=monitor, mode="min", save_top_k=1, filename="best")
    callbacks = [ckpt] + ([EarlyStopping(monitor, patience=cfg.patience)] if cfg.patience else [])
    trainer = L.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=1,
        callbacks=callbacks,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        deterministic=False,
    )
    val_loader = loader(x_val, y_val, 1024, False) if x_val is not None else None
    trainer.fit(module, loader(x_fit, y_fit, cfg.batch_size, cfg.shuffle, cfg.seed), val_loader)
    best = SiteModule.load_from_checkpoint(ckpt.best_model_path, model=_build(cfg, xtr.shape[1]), loss=cfg.loss)
    net = best.model.cpu().eval()
    scores = predict_scores(net, xte)
    check_not_collapsed(scores)
    metrics = gene_level_metrics(yte, scores)  # same confusion definitions; site-level here
    metrics["AUC"] = roc_auc(yte, scores)
    metrics["epochs_run"] = int(trainer.current_epoch)
    encoder = OneHotFlat(site_len=(xtr.shape[1] - 120) // 4, site_orientation=cfg.site_orientation)
    return net, metrics, encoder
