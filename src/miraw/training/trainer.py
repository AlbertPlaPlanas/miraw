"""Train one fold and evaluate it. Checkpoints on validation loss, and refuses collapsed models."""

from __future__ import annotations

import logging
import tempfile
import warnings

import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from miraw.config import TrainConfig
from miraw.encoding.onehot import OneHotFlat
from miraw.evaluation.metrics import gene_level_metrics, roc_auc
from miraw.models.base import build_model
from miraw.training.data import loader, split_validation
from miraw.training.module import SiteModule

logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
logging.getLogger("lightning.fabric").setLevel(logging.ERROR)
# Lightning advice that does not apply here (small in-memory datasets, CPU or GPU by choice)
_QUIET = (".*GPU available but not used.*", ".*does not have many workers.*", ".*val_dataloader.*", ".*treespec.*")


class CollapsedModelError(RuntimeError):
    """The trained model predicts a single class for (almost) every input."""


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


def train_fold(cfg: TrainConfig, xtr, ytr, xte=None, yte=None, workdir=None):
    """Train on ``xtr, ytr`` (encoded features, labels). Returns (the best checkpoint: lowest validation loss,
    or lowest training loss when ``val_fraction`` is 0; its metrics on the test set, empty without one;
    the encoder). ``workdir`` holds the checkpoint (a temporary folder if not given)."""
    if workdir is None:
        with tempfile.TemporaryDirectory() as tmp:
            return train_fold(cfg, xtr, ytr, xte, yte, tmp)
    L.seed_everything(cfg.seed, workers=True, verbose=False)
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
    with warnings.catch_warnings():
        for pattern in _QUIET:
            warnings.filterwarnings("ignore", message=pattern)
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
    encoder = OneHotFlat(site_len=(xtr.shape[1] - 120) // 4, site_orientation=cfg.site_orientation)
    if xte is None:
        check_not_collapsed(predict_scores(net, xtr))
        return net, {"epochs_run": int(trainer.current_epoch)}, encoder
    scores = predict_scores(net, xte)
    check_not_collapsed(scores)
    metrics = gene_level_metrics(yte, scores)  # same confusion definitions; site-level here
    metrics["AUC"] = roc_auc(yte, scores)
    metrics["epochs_run"] = int(trainer.current_epoch)
    return net, metrics, encoder
