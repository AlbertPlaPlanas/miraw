"""The public API (``import pymiraw``): input forms, equivalence with the file pipeline, in-memory models,
outputs, and training. Uses the bundled original weights and the bundled example data."""

import json

import numpy as np
import pandas as pd
import pytest
import torch

import pymiraw
from pymiraw.prediction.pipeline import pairs_frame, predict_file, results_frame
from pymiraw.resources import path as resource_path

torch.set_num_threads(2)


@pytest.fixture(scope="module")
def few():
    return pymiraw.example_pairs().head(4)


@pytest.fixture(scope="module")
def reference(few, tmp_path_factory):
    """The file-based pipeline (what the CLI and the gates run) on the same pairs."""
    f = tmp_path_factory.mktemp("in") / "pairs.tsv"
    few.to_csv(f, sep="\t", index=False)
    return results_frame(predict_file(f, {"model": "original"}, workers=1))


def test_public_names_resolve():
    for name in pymiraw.__all__:
        assert getattr(pymiraw, name) is not None
    with pytest.raises(AttributeError):
        pymiraw.no_such_name  # noqa: B018


def test_bundled_examples():
    pairs, sites = pymiraw.example_pairs(), pymiraw.example_sites()
    assert len(pairs) == 24 and pairs.Positive_Negative.value_counts().to_dict() == {"1": 12, "0": 12}
    assert set(sites.split) == {"train", "test"} and set(sites.label) == {0, 1}
    assert len(pymiraw.example_sites("train")) == 4000 and "split" not in pymiraw.example_sites("test")


def test_pairs_frame_accepts_files_frames_and_tuples(few):
    from_df = pairs_frame(few)
    from_file = pairs_frame(resource_path("example_pairs.tsv")).head(4)
    short = pairs_frame(few.rename(columns={"Mature_mirna_transcript": "mirna_seq", "3UTR_transcript": "utr"}))
    tuples = pairs_frame(list(zip(few.Mature_mirna_transcript, few["3UTR_transcript"], strict=True)))
    for df in (from_file, short, tuples):
        assert list(df.mirna_seq) == list(from_df.mirna_seq) and list(df.utr) == list(from_df.utr)
    assert tuples.gene_name.eq("").all() and "Positive_Negative" not in tuples
    with pytest.raises(ValueError, match="utr"):
        pairs_frame(pd.DataFrame({"mirna_seq": ["ACGU"]}))


def test_predict_equals_the_file_pipeline(few, reference):
    preds = pymiraw.predict(few, workers=1, progress=False)
    got = preds.genes.drop(columns="IsTarget")
    pd.testing.assert_frame_equal(got, reference)
    assert (preds.genes.IsTarget == (reference.Prediction > 0)).all()
    assert preds.model == "original miRAW" and preds.config.model == "original" and preds.has_labels
    assert preds.metrics()["TP"] + preds.metrics()["FN"] == int((few.Positive_Negative == "1").sum())


def test_in_memory_model_in_worker_processes_gives_the_same_result(few, reference):
    model = pymiraw.load_model()  # a Model object, sent to the workers through a temporary artefact
    preds = pymiraw.predict(few, model, workers=2, progress=False)
    np.testing.assert_array_equal(preds.genes.Prediction.to_numpy(), reference.Prediction.to_numpy())


def test_predict_pair_and_site_scores(few, reference):
    r = pymiraw.predict_pair(few.Mature_mirna_transcript.iloc[0], few["3UTR_transcript"].iloc[0])
    assert r.prediction == reference.Prediction.iloc[0] and r.is_target == (r.prediction > 0)
    model = pymiraw.load_model()
    scores = model.score_sites(few.Mature_mirna_transcript.iloc[0], [s.transcript for s in r.sites])
    assert np.array_equal(scores, np.array([s.score for s in r.sites], dtype=np.float32))
    assert np.all(np.abs(scores) <= 1)


def test_predictions_save_writes_the_original_outputs(few, tmp_path):
    preds = pymiraw.predict(few.head(2), workers=1, progress=False, finder="PITA", filters="AccessibilityEnergy;10")
    out = preds.save(tmp_path / "run")
    names = {p.name for p in out.iterdir()}
    assert names == {"targetPredictionOutput.csv", "targetSites.csv", "config.yaml", "metrics.json"}
    assert json.loads((out / "metrics.json").read_text()) == preds.metrics()
    assert preds.config.filters == ["AccessibilityEnergy;10"] and preds.config.finder == "PITA"
    assert "2 pairs" in repr(preds)


def test_unlabelled_input_has_no_metrics(few):
    preds = pymiraw.predict(few.drop(columns="Positive_Negative").head(1), workers=1, progress=False)
    assert not preds.has_labels and preds.genes.RealClass.isna().all()
    with pytest.raises(ValueError, match="labels"):
        preds.metrics()


def test_in_process_prediction_restores_torch_threads(few):
    torch.set_num_threads(3)
    pymiraw.predict(few.head(1), workers=1, progress=False)
    assert torch.get_num_threads() == 3
    torch.set_num_threads(2)


def test_encode_sites_matches_the_encoder_and_the_released_layout():
    sites = pymiraw.example_sites("test").head(5)
    x = pymiraw.encode_sites(sites.mirna_seq, sites.site_seq)
    assert x.shape == (5, 280) and x.dtype == np.float32
    nat = pymiraw.encode_sites(sites.mirna_seq, sites.site_seq, "natural")
    assert np.array_equal(x[:, :120], nat[:, :120]) and not np.array_equal(x, nat)


def test_train_from_site_sequences_and_save_load_predict(tmp_path, few):
    train, test = pymiraw.example_sites("train"), pymiraw.example_sites("test").head(200)
    model = pymiraw.train(train, test, accelerator="cpu")  # the paper's regime on 4000 sites: seconds
    assert model.meta["metrics"]["AUC"] > 0.85 and model.meta["train_config"]["max_epochs"] == 10
    assert model.meta["train_config"]["shuffle"] is False  # default = the paper's regime
    path = model.save(tmp_path / "m.pt")
    again = pymiraw.load_model(path)
    x = torch.from_numpy(pymiraw.encode_sites(test.mirna_seq, test.site_seq))
    assert torch.equal(model.network.site_score(x), again.network.site_score(x))
    preds = pymiraw.predict(few.head(1), path, workers=1, progress=False)
    assert preds.model == str(path) and preds.config.model == str(path)
