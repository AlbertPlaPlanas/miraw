"""Functional tests against the original miRAW paper model and results (need the original weights)."""

import json

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from pymiraw.cli import app
from pymiraw.models.artifact import load_artifact
from pymiraw.models.dl4j_import import load_dl4j
from pymiraw.prediction.pipeline import predict_file, results_frame


def test_gate0_imported_paper_model_reproduces_paper_cv_accuracy(original_weights, fixtures):
    """Fold 6 (the paper's 'fold 7') accuracy on the first 500 test sites, pinned from the weights."""
    model, _ = load_dl4j(original_weights)
    d = np.loadtxt(fixtures / "cv_set6_test_head500.csv", delimiter=",", dtype=np.float32)
    y, x = d[:, 0].astype(int), torch.from_numpy(d[:, 1:])
    with torch.no_grad():
        s = model.site_score(x).numpy()
    acc = ((s > 0) == y).mean()
    assert acc > 0.90  # paper fold-6 accuracy over all 3500 sites is 0.9351
    # the same model, round-tripped through the pyMIRAW artefact format, gives identical scores
    m2, enc, meta = load_artifact(original_weights)
    assert enc.site_orientation == "old-default" and meta["source"] == "dl4j"
    with torch.no_grad():
        assert torch.allclose(m2.site_score(x), torch.from_numpy(s))


@pytest.fixture(scope="module")
def paper_cfg(original_weights):
    return dict(
        model=str(original_weights),
        finder="miRAW",
        filters=[],
        backend="bindings",
        max_site_length=40,
        seed_alignment_offset=5,
        cache_dir=None,
    )


def test_pipeline_matches_paper_results_on_pinned_pairs(paper_cfg, fixtures):
    got = results_frame(predict_file(fixtures / "paper_pairs.tsv", paper_cfg, workers=3))
    exp = pd.read_csv(fixtures / "paper_pairs_expected.tsv", sep="\t")
    assert list(got.GeneName) == list(exp.GeneName)
    assert ((got.Prediction > 0) == (exp.Prediction > 0)).all()  # same gene-level call
    assert np.abs(got.Prediction.values - exp.Prediction.values).max() < 0.02
    assert np.abs(got.PosSites.values - exp.PosSites.values).max() <= 1  # ViennaRNA 2.2 vs 2.7 drift


def test_cli_original_weights_flag_end_to_end(original_weights, fixtures, tmp_path, monkeypatch):
    monkeypatch.setenv("PYMIRAW_ORIGINAL_WEIGHTS", str(original_weights))
    r = CliRunner().invoke(
        app,
        [
            "evaluate",
            str(fixtures / "paper_pairs.tsv"),
            "--original-weights",
            "--output-dir",
            str(tmp_path),
            "--name",
            "t",
            "--workers",
            "2",
        ],
    )
    assert r.exit_code == 0, r.output
    assert "original miRAW" in r.output
    out = tmp_path / "t"
    assert (out / "targetPredictionOutput.csv").exists() and (out / "targetSites.csv").exists()
    m = json.loads((out / "metrics.json").read_text())
    assert m["TP"] + m["FP"] + m["TN"] + m["FN"] == 6 and m["Accuracy"] >= 0.8


def test_cli_default_model_is_the_original_weights(original_weights, fixtures, tmp_path, monkeypatch):
    monkeypatch.setenv("PYMIRAW_ORIGINAL_WEIGHTS", str(original_weights))
    r = CliRunner().invoke(
        app,
        [
            "predict",
            str(fixtures / "paper_pairs.tsv"),
            "--limit",
            "1",
            "--output-dir",
            str(tmp_path),
            "--name",
            "d",
            "--workers",
            "1",
        ],
    )
    assert r.exit_code == 0 and "original miRAW" in r.output
