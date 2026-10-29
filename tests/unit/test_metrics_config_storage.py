import numpy as np
import pytest

from pymiraw.config import PredictConfig, from_legacy_properties, resolve_model_path
from pymiraw.evaluation.metrics import confusion, gene_level_metrics, roc_auc, summarize
from pymiraw.storage import ORIGINAL_WEIGHTS_ENV, load_manifest, original_weights_path, sha256_file


def test_confusion_and_derived_metrics_match_paper_summary_row():
    # GlobalSummary 'miRAW 6 1 10 NF' set 0: TP=510 FP=349 TN=199 FN=38
    y = np.array([1] * 548 + [0] * 548)
    pred = np.array([True] * 510 + [False] * 38 + [True] * 349 + [False] * 199)
    m = summarize(confusion(y, pred))
    assert (m["TP"], m["FP"], m["TN"], m["FN"]) == (510, 349, 199, 38)
    assert m["Accuracy"] == pytest.approx(0.646898, abs=1e-6)
    assert m["Precision"] == pytest.approx(0.593714, abs=1e-6)
    assert m["Recall"] == pytest.approx(0.930657, abs=1e-6)
    assert m["Specificity"] == pytest.approx(0.363139, abs=1e-6)
    assert m["NegPrecision"] == pytest.approx(0.839662, abs=1e-6)
    assert m["F1"] == pytest.approx(0.724947, abs=1e-6)


def test_gene_call_is_prediction_greater_than_zero():
    m = gene_level_metrics([1, 1, 0, 0], [0.5, 0.0, 0.0, -0.3])
    assert (m["TP"], m["FN"], m["FP"], m["TN"]) == (1, 1, 0, 2)


def test_zero_division_is_zero_and_auc_needs_two_classes():
    assert summarize(confusion([0, 0], [False, False]))["F1"] == 0.0
    assert np.isnan(roc_auc([1, 1], [0.1, 0.2]))
    assert roc_auc([0, 1], [0.1, 0.9]) == 1.0


def test_original_weights_flag_overrides_model():
    cfg = PredictConfig(input_file="x", model="/some/other.pt", original_weights=True)
    assert cfg.model == "original" and cfg.uses_original_weights
    assert PredictConfig(input_file="x").uses_original_weights  # default is the original weights
    assert not PredictConfig(input_file="x", model="m.pt").uses_original_weights


def test_original_weights_path_override_env(tmp_path, monkeypatch):
    f = tmp_path / "w.bin"
    f.write_bytes(b"abc")
    monkeypatch.setenv(ORIGINAL_WEIGHTS_ENV, str(f))
    assert original_weights_path() == f and resolve_model_path("original") == str(f)
    monkeypatch.setenv(ORIGINAL_WEIGHTS_ENV, str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        original_weights_path()


def test_resolve_model_path_rejects_missing_file():
    with pytest.raises(FileNotFoundError):
        resolve_model_path("/no/such/model.pt")


def test_manifest_lists_original_weights_with_checksum():
    entry = load_manifest()["files"]["original-weights"]
    assert len(entry["sha256"]) == 64 and entry["path"].endswith("bestModel.bin")


def test_sha256(tmp_path):
    p = tmp_path / "a"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_legacy_properties_parsing(tmp_path):
    p = tmp_path / "c.properties"
    p.write_text(
        "ExperimentName=Personalized_1_9_7_AE\nExperimentFolder=./R\nUnifiedFile=./u.csv\n"
        "DLModel=./ConfFiles/NetworkModel/bestModel.bin\nCandidateSiteFinder.Type=Personalized;1;9;7\n"
        "MaxSiteLength=40\nSeedAlignementOffset=5\nFilter.0=AccessibilityEnergy;10\n"
    )
    cfg = from_legacy_properties(p)
    assert cfg.finder == "Personalized;1;9;7" and cfg.filters == ["AccessibilityEnergy;10"]
    assert (cfg.max_site_length, cfg.seed_alignment_offset) == (40, 5) and cfg.uses_original_weights


def test_original_weights_default_to_the_bundled_copy(monkeypatch):
    from pymiraw import resources

    monkeypatch.delenv(ORIGINAL_WEIGHTS_ENV, raising=False)
    assert original_weights_path() == resources.original_weights() and original_weights_path().is_file()
    assert resolve_model_path("original") == str(resources.original_weights())


def test_predict_config_accepts_paths(tmp_path):
    cfg = PredictConfig(input_file=tmp_path / "a.tsv", sites_dir=tmp_path, cache_dir=tmp_path, model=tmp_path / "m.pt")
    assert cfg.sites_dir == str(tmp_path) and cfg.model == str(tmp_path / "m.pt")


def test_predict_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        PredictConfig(backend_args={"bin_dir": "/x"})  # a typo or an old key must not be silently ignored


def test_unpack_archive_flattens_and_filters(tmp_path):
    import py7zr

    from pymiraw import storage

    src = tmp_path / "src"
    for rel in ("NetworkModel/bestModel.bin", "Data/TestData/balanced10/a.csv", "Data/Other/skip.csv"):
        (src / "PLOSComb" / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / "PLOSComb" / rel).write_text("1")
    with py7zr.SevenZipFile(tmp_path / "a.7z", "w") as z:
        z.writeall(src / "PLOSComb", "PLOSComb")
    out = storage.unpack_archive(tmp_path / "a.7z", tmp_path / "dest")
    assert (out / "NetworkModel" / "bestModel.bin").is_file()
    assert (out / "Data" / "TestData" / "balanced10" / "a.csv").is_file()
    assert not (out / "Data" / "Other").exists()


def test_vienna_bin_reaches_the_pipeline_config(monkeypatch, tmp_path):
    """--vienna-bin must reach the run configuration (the README promises an exact-reproduction recipe)."""
    from typer.testing import CliRunner

    from pymiraw import api, cli

    seen = {}

    def fake_predict_pairs(pairs, cfg, model=None, workers=None, progress=False):
        seen["cfg"] = cfg
        return []

    inp = tmp_path / "in.tsv"
    inp.write_text("miRNA\tgene_name\tEnsemblId\tMature_mirna_transcript\t3UTR_transcript\n")
    monkeypatch.setattr(api, "predict_pairs", fake_predict_pairs)
    args = ["predict", str(inp), "--original-weights", "--output-dir", str(tmp_path / "o"), "--name", "x"]
    r = CliRunner().invoke(cli.app, [*args, "--backend", "cli", "--vienna-bin", "/opt/ref/bin"])
    assert r.exit_code == 0, r.output
    assert seen["cfg"].backend == "cli" and seen["cfg"].vienna_bin == "/opt/ref/bin"
    assert seen["cfg"].model == "original"
    assert CliRunner().invoke(cli.app, args).exit_code == 0 and seen["cfg"].vienna_bin is None


def test_vienna_bin_without_the_cli_backend_is_an_error():
    from pymiraw.prediction.pipeline import make_folding_backend

    with pytest.raises(ValueError, match="cli"):
        make_folding_backend(PredictConfig(vienna_bin="/opt/ref/bin"))
