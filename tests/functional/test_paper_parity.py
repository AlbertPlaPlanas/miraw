"""Functional tests against the original miRAW paper model (bundled) and the paper's results."""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch
from typer.testing import CliRunner

from miraw import resources
from miraw.cli import app
from miraw.models.artifact import load_artifact
from miraw.models.dl4j_import import load_dl4j
from miraw.prediction.pipeline import predict_file, results_frame
from miraw.storage import sha256_file
from miraw.training.data import load_csv


def _site_scores(model, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model.site_score(torch.from_numpy(x)).numpy()


def test_bundled_weights_are_the_pinned_file(original_weights):
    assert original_weights == resources.original_weights()
    assert sha256_file(original_weights) == resources.ORIGINAL_WEIGHTS_SHA256
    _, enc, meta = load_artifact(original_weights)
    assert enc.site_orientation == "old-default" and meta["source_sha256"] == resources.SOURCE_DL4J_SHA256


def test_gate0_bundled_model_on_pinned_cv_sites(original_weights, fixtures):
    """Fold 6 (the paper's 'fold 7') test sites, first 500: 470 correct (all 3500: 0.9351, test below)."""
    model, _, _ = load_artifact(original_weights)
    d = np.loadtxt(fixtures / "cv_set6_test_head500.csv", delimiter=",", dtype=np.float32)
    assert ((_site_scores(model, d[:, 1:]) > 0) == d[:, 0]).sum() == 470


def test_gate0_full_fold6_accuracy_equals_the_paper(original_weights, paper_data):
    """Gate 0: the paper's BestModelEval.txt reports 0.9351428571 on CV fold 6 for this model."""
    model, _, _ = load_artifact(original_weights)
    x, y = load_csv(paper_data / "Data/CrossVal/RelaxedPW60_14K_35K/binary/set_6_test.csv")
    assert ((_site_scores(model, x) > 0) == y).sum() == 3273  # 3273 / 3500 = 0.9351428571


def test_bundled_weights_equal_the_paper_dl4j_file(original_weights, paper_data):
    """The bundled .pt is a bit-exact conversion of NetworkModel/bestModel.bin."""
    bin_path = paper_data / "NetworkModel/bestModel.bin"
    assert sha256_file(bin_path) == resources.SOURCE_DL4J_SHA256
    dl4j, _ = load_dl4j(bin_path)
    bundled, _, _ = load_artifact(original_weights)
    a, b = dl4j.state_dict(), bundled.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


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


def test_cli_original_weights_flag_end_to_end(fixtures, tmp_path):
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


def test_cli_default_model_is_the_original_weights(fixtures, tmp_path):
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


REF_BIN = "/opt/conda/envs/miraw-ref/bin"


@pytest.mark.reference
@pytest.mark.skipif(
    not os.path.exists(f"{REF_BIN}/RNAcofold"), reason="needs the miraw-ref environment (ViennaRNA 2.2.5)"
)
def test_exact_match_with_the_original_viennarna_version(paper_cfg, fixtures):
    """With ViennaRNA 2.2.5 (the version miRAW was built against) the port reproduces the published
    per-pair results exactly: same site counts and the same scores."""
    cfg = {**paper_cfg, "backend": "cli", "vienna_bin": REF_BIN}
    got = results_frame(predict_file(fixtures / "paper_pairs.tsv", cfg, workers=3))
    exp = pd.read_csv(fixtures / "paper_pairs_expected.tsv", sep="\t")
    assert list(got.PosSites) == list(exp.PosSites) and list(got.NegSites) == list(exp.NegSites)
    assert np.abs(got.Prediction.values - exp.Prediction.values).max() < 1e-5
