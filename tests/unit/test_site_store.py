"""Saving and reloading pre-computed candidate sites (no ViennaRNA needed: scripted fake backend)."""

import pytest

from pymiraw.folding.base import FoldService
from pymiraw.prediction.pipeline import predict_file
from pymiraw.sites.finders.default import DefaultFinder, PersonalizedFinder
from pymiraw.sites.store import SiteStore, backend_id, finder_fingerprint, pair_key
from tests.unit.test_finders import MIRNA, FakeBackend, duplex_fold

UTR = "ACGU" * 5 + "G" + "ACGU" * 24


def find(finder=None):
    """Candidate sites of the test pair, found with the fake backend."""
    return (finder or DefaultFinder()).find_candidates(
        MIRNA, UTR, FoldService(FakeBackend(lambda s, m: duplex_fold(s, m, 8, -9.0)))
    )


def test_round_trip_keeps_every_site_field(tmp_path):
    sites = find()
    assert sites
    store = SiteStore(tmp_path)
    store.add("k", sites)
    assert store.flush() is not None
    loaded = SiteStore(tmp_path).get("k")  # fresh instance = read back from the parquet shard
    assert [vars(s) for s in loaded] == [vars(s) for s in sites]


def test_pair_without_sites_is_distinguished_from_unknown_pair(tmp_path):
    store = SiteStore(tmp_path)
    store.add("empty", [])
    store.flush()
    again = SiteStore(tmp_path)
    assert again.get("empty") == [] and again.get("never-computed") is None


def test_get_returns_copies_and_flush_is_incremental(tmp_path):
    store = SiteStore(tmp_path)
    store.add("a", find())
    first = store.flush()
    store.get("a")[0].score = 123.0  # mutating a returned site must not change the store
    assert store.get("a")[0].score is None
    store.add("b", find())
    second = store.flush()
    assert first != second and store.flush() is None
    assert len(SiteStore(tmp_path)) == 2


def test_key_depends_on_backend_finder_and_sequences():
    f, b = DefaultFinder(), FakeBackend(None)
    base = pair_key(backend_id(b), f, MIRNA, UTR)
    assert base == pair_key(backend_id(b), DefaultFinder(), MIRNA, UTR)
    assert base != pair_key("other-1", f, MIRNA, UTR)
    assert base != pair_key(backend_id(b), PersonalizedFinder(), MIRNA, UTR)
    assert base != pair_key(backend_id(b), DefaultFinder(max_site_length=30), MIRNA, UTR)
    assert base != pair_key(backend_id(b), f, MIRNA, UTR + "A")
    assert "min_binds_seed=6" in finder_fingerprint(f)


def test_predict_file_reuses_saved_sites(tmp_path, monkeypatch):
    """Run 1 folds and saves; run 2 (same sites_dir) gives identical sites with zero folding calls."""
    from pymiraw.models.base import SiteClassifier
    from pymiraw.prediction import pipeline

    backends = []

    def fake_make_backend(kind, **kw):
        backends.append(FakeBackend(lambda s, m: duplex_fold(s, m, 8, -9.0)))
        return backends[-1]

    class Const(SiteClassifier):
        def forward(self, x):
            import torch

            return torch.tensor([[-1.0, 1.0]]).repeat(len(x), 1)

    from pymiraw.encoding.onehot import OneHotFlat

    monkeypatch.setattr(pipeline, "make_backend", fake_make_backend)
    monkeypatch.setattr(pipeline, "load_artifact", lambda path: (Const(), OneHotFlat(30, 40), {}))
    inp = tmp_path / "pairs.tsv"
    inp.write_text(f"miRNA\tgene_name\tEnsemblId\tMature_mirna_transcript\t3UTR_transcript\ng\tG\tE\t{MIRNA}\t{UTR}\n")
    cfg = {"model": "x", "finder": "miRAW", "backend": "bindings", "max_site_length": 40,
           "seed_alignment_offset": 5, "sites_dir": str(tmp_path / "sites")}  # fmt: skip

    first = predict_file(inp, cfg, workers=1)
    calls_first = sum(b.calls for b in backends)
    assert calls_first > 0 and list((tmp_path / "sites").glob("sites-*.parquet"))

    backends.clear()
    second = predict_file(inp, cfg, workers=1)
    assert sum(b.calls for b in backends) == 0
    assert [vars(s) for s in second[0].sites] == [vars(s) for s in first[0].sites]
    assert second[0].prediction == first[0].prediction

    with pytest.raises(LookupError):  # different finder settings -> not saved -> --require-sites fails
        predict_file(inp, {**cfg, "finder": "Personalized;1;9;7", "require_sites": True}, workers=1)
