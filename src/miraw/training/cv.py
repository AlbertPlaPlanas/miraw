"""Cross-validation over the released folds, several seeds, comparison with the paper."""

from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from miraw import resources
from miraw.config import TrainConfig
from miraw.models.artifact import save_artifact
from miraw.training.data import load_fold
from miraw.training.trainer import CollapsedModelError, train_fold


def run_cv(folds_dir, cfg: TrainConfig, folds=range(9), seeds=(0,), out_dir=None) -> pd.DataFrame:
    """Train ``cfg`` on each fold x seed. Folds 0-8 match those evaluated in the paper."""
    rows = []
    for seed in seeds:
        for k in folds:
            c = TrainConfig(**{**asdict(cfg), "seed": seed})
            xtr, ytr, xte, yte = load_fold(folds_dir, k, cfg.site_orientation)
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    net, m, enc = train_fold(c, xtr, ytr, xte, yte, tmp)
                except CollapsedModelError as e:
                    rows.append({"fold": k, "seed": seed, "collapsed": True, "error": str(e)})
                    continue
            if out_dir:
                save_artifact(
                    Path(out_dir) / f"{cfg.site_orientation}_f{k}_s{seed}.pt",
                    net,
                    enc,
                    {"fold": k, "seed": seed, "loss": cfg.loss, "metrics": m, "train_config": asdict(cfg)},
                )
            rows.append({"fold": k, "seed": seed, "collapsed": False, **m})
    return pd.DataFrame(rows)


def compare_with_paper(df: pd.DataFrame) -> dict:
    """Mean accuracy/AUC over non-collapsed runs vs the paper's XENT CV (acc 0.921, AUC 0.958)."""
    paper = resources.paper_cv_models()  # re-scored with our metric code (the recorded AUC used coarse thresholds)
    ok = df[~df.collapsed]
    return {
        "runs": len(df),
        "collapsed": int(df.collapsed.sum()),
        "acc": ok.Accuracy.mean(),
        "auc": ok.AUC.mean(),
        "paper_acc": paper.Accuracy.mean(),
        "paper_auc": paper.AUC.mean(),
        "acc_diff": ok.Accuracy.mean() - paper.Accuracy.mean(),
        "auc_diff": ok.AUC.mean() - paper.AUC.mean(),
    }
