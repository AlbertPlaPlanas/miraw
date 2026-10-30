import numpy as np
import pytest

from miraw.config import TrainConfig
from miraw.training.data import loader, reorient, split_validation
from miraw.training.trainer import CollapsedModelError, check_not_collapsed


def test_unshuffled_loader_keeps_file_order():
    x = np.arange(20, dtype=np.float32).reshape(10, 2)
    y = np.arange(10)
    first = [b[1].tolist() for b in loader(x, y, 5, shuffle=False)]
    again = [b[1].tolist() for b in loader(x, y, 5, shuffle=False)]
    assert first == again == [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]


def test_shuffled_loader_is_seeded_and_permutes():
    x, y = np.zeros((50, 2), np.float32), np.arange(50)
    a = [b[1].tolist() for b in loader(x, y, 50, shuffle=True, seed=1)][0]
    b = [b[1].tolist() for b in loader(x, y, 50, shuffle=True, seed=1)][0]
    assert a == b and a != list(range(50)) and sorted(a) == list(range(50))


def test_cli_passes_shuffle_to_the_trainer(monkeypatch, tmp_path):
    """Regression: --no-shuffle once never reached TrainConfig, silently shuffling."""
    import pandas as pd
    from typer.testing import CliRunner

    from miraw import cli
    from miraw.training import cv

    seen = {}

    def fake_run_cv(folds_dir, cfg, folds, seeds, out_dir):
        seen["cfg"] = cfg
        return pd.DataFrame({"fold": [0], "seed": [0], "collapsed": [False], "Accuracy": [1.0], "AUC": [1.0]})

    monkeypatch.setattr(cv, "run_cv", fake_run_cv)
    monkeypatch.setattr(cv, "compare_with_paper", lambda df: {})
    args = ["train", str(tmp_path), "--output-dir", str(tmp_path / "o")]
    assert CliRunner().invoke(cli.app, args).exit_code == 0  # defaults: the paper regime
    assert seen["cfg"].shuffle is False and seen["cfg"].val_fraction == 0
    assert seen["cfg"] == TrainConfig.paper(seed=seen["cfg"].seed)
    assert CliRunner().invoke(cli.app, [*args, "--shuffle", "--val-fraction", "0.1"]).exit_code == 0
    assert seen["cfg"].shuffle is True and seen["cfg"].val_fraction == 0.1
    assert seen["cfg"].accelerator == "auto"
    assert CliRunner().invoke(cli.app, [*args, "--accelerator", "cpu"]).exit_code == 0
    assert seen["cfg"].accelerator == "cpu"


def test_validation_split_is_stratified_deterministic_and_keeps_order():
    y = np.array([0, 1] * 50)
    x = np.arange(100, dtype=np.float32)[:, None]
    a = split_validation(x, y, 0.2, seed=3)
    b = split_validation(x, y, 0.2, seed=3)
    assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
    xf, yf, xv, yv = a
    assert len(yv) == 20 and yv.mean() == 0.5 and np.all(np.diff(xf[:, 0]) > 0)


def test_reorient_flips_only_the_site_block():
    x = np.zeros((1, 4 * 3 + 4 * 2), np.float32)  # mirna_len 3, site_len 2
    x[0, 12:16] = [1, 0, 0, 0]  # site nt 1 = C
    x[0, 16:20] = [0, 1, 0, 0]  # site nt 2 = A
    nat = reorient(x, "natural", mirna_len=3, site_len=2)
    assert nat[0, 12:16].tolist() == [0, 1, 0, 0] and nat[0, 16:20].tolist() == [1, 0, 0, 0]
    assert np.array_equal(nat[0, :12], x[0, :12])  # miRNA block untouched
    assert np.array_equal(reorient(nat, "natural", 3, 2), x)  # flipping twice restores the original
    assert reorient(x, "old-default") is x


def test_collapse_guard():
    check_not_collapsed(np.array([1.0, -1.0] * 50))
    with pytest.raises(CollapsedModelError):
        check_not_collapsed(np.full(100, -0.9))
