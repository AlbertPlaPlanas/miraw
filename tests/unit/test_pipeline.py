"""Gene-level aggregation with a fake backend and a fixed fake model."""

import numpy as np
import torch

from miraw.encoding.onehot import OneHotFlat
from miraw.folding.base import Fold, FoldService
from miraw.models.base import SiteClassifier
from miraw.prediction.pipeline import Predictor
from miraw.sites.filters.accessibility import AccessibilityEnergyFilter, site_opening_energy
from miraw.sites.finders.default import DefaultFinder
from tests.unit.test_finders import MIRNA, FakeBackend, duplex_fold


class ByContent(SiteClassifier):
    """Scores a site +0.9 if its (reversed) encoding has a G at the first site position else -0.9."""

    def forward(self, x):
        site_first_nt_is_g = x[:, 120 + 2] > 0
        logit = torch.where(site_first_nt_is_g, 5.0, -5.0)
        return torch.stack([-logit, logit], dim=1)


def make(filters=(), open_energy=0.0, utr_energy=-10.0):
    class B(FakeBackend):
        def fold(self, seq):
            return Fold(seq, "." * len(seq), utr_energy)

        def fold_with_open_gap(self, seq, a, b):
            return Fold(seq, "." * len(seq), utr_energy + open_energy)

    svc = FoldService(B(lambda s, m: duplex_fold(s, m, 8, -9.0)))
    return Predictor(ByContent(), OneHotFlat(30, 40), DefaultFinder(), svc, filters)


def test_gene_prediction_is_max_site_score_and_counts():
    utr = "ACGU" * 5 + "G" + "ACGU" * 24  # some windows end with G -> reversed first nt is G
    res = make().predict_pair(MIRNA, utr, "g", "e", 1)
    scores = [s.score for s in res.sites]
    assert res.prediction == res.max_val == max(scores) and res.min_val == min(scores)
    assert res.pos_sites == sum(s > 0 for s in scores) and res.pos_sites + res.neg_sites == len(scores)
    assert res.removed_sites == 0


def test_no_sites_gives_zero_prediction():
    svc = FoldService(FakeBackend(lambda s, m: Fold(s + "&" + m, "." * (len(s) + 1 + len(m)), 0.0)))
    res = Predictor(ByContent(), OneHotFlat(), DefaultFinder(), svc).predict_pair(MIRNA, "ACGU" * 25)
    assert res.prediction == 0.0 and res.sites == [] and (res.pos_sites, res.neg_sites) == (0, 0)


def test_accessibility_filter_removes_positive_sites_and_zeroes_their_score():
    utr = "G" * 100
    kept = make([AccessibilityEnergyFilter(10)], open_energy=5.0).predict_pair(MIRNA, utr)
    assert kept.removed_sites == 0 and kept.prediction > 0
    removed = make([AccessibilityEnergyFilter(10)], open_energy=15.0).predict_pair(MIRNA, utr)
    assert removed.removed_sites == removed.pos_sites + removed.removed_sites > 0
    assert removed.prediction == 0.0 and removed.pos_sites == 0  # removed sites count as negatives
    assert all(s.filtered for s in removed.sites if s.score and s.score > 0)


def test_site_opening_energy_window_arithmetic():
    seen = {}

    class B(FakeBackend):
        def fold(self, seq):
            return Fold(seq, "." * len(seq), -20.0)

        def fold_with_open_gap(self, seq, a, b):
            seen.update(len=len(seq), gap=(a, b))
            return Fold(seq, "." * len(seq), -17.0)

    svc = FoldService(B(lambda s, m: None))
    e = site_opening_energy(svc, "A" * 1000, 400, 440)  # chunk = 200-40 = 160 -> window 320..520
    assert e == 3.0 and seen["len"] == 200 and seen["gap"] == (80, 120)


def test_scores_bounded():
    s = make().score_sites(MIRNA, make().finder.find_candidates(MIRNA, "ACGU" * 25, make().folds))
    assert np.all(np.abs(s) <= 1)
