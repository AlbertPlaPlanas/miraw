import numpy as np
import pytest

from miraw.evaluation.noninferiority import (
    CONFIDENCE,
    MARGINS,
    METHOD,
    hodges_lehmann,
    lower_bound,
    non_inferiority,
    rank_lower_bound,
)


def test_agreed_settings():
    assert CONFIDENCE == 0.99 and MARGINS == {"Accuracy": 0.02, "F1": 0.02, "AUC": 0.005} and METHOD == "rank"


def test_identical_models_are_non_inferior():
    v = non_inferiority([0.0, 0.001, -0.001, 0.0, 0.002], "Accuracy", method="t")
    assert v.non_inferior and abs(v.estimate) < 0.001 and v.lower < v.estimate


def test_clearly_worse_model_fails():
    assert not non_inferiority([-0.03, -0.02, -0.04, -0.025], "F1", method="t").non_inferior


def test_noisy_small_sample_is_not_declared_non_inferior():
    # mean is above the limit but the spread is huge: the bound is below -margin
    assert not non_inferiority([0.02, -0.03, 0.01, -0.02, 0.03, -0.04], "Accuracy", method="t").non_inferior


def test_higher_confidence_gives_lower_bound():
    d = np.array([0.001, -0.004, 0.002, -0.001, 0.0, -0.003])
    assert lower_bound(d, 0.99) < lower_bound(d, 0.95) < d.mean()


def test_margin_is_per_metric_and_boundary_is_exclusive():
    d = np.array([-0.0075] * 5) + np.array([0.0, 1e-9, -1e-9, 0.0, 1e-9])  # lower bound ~ -0.0075
    assert non_inferiority(d, "Accuracy", method="t").non_inferior  # margin 0.02
    assert not non_inferiority(d, "AUC", method="t").non_inferior  # margin 0.005
    assert non_inferiority(d, "AUC", margins={"AUC": 0.0075 + 1e-3}, method="t").non_inferior


def test_needs_two_points():
    with pytest.raises(ValueError):
        lower_bound([0.1])
    with pytest.raises(ValueError):
        rank_lower_bound([0.1])


def test_slack_reports_fragility():
    v = non_inferiority(
        [-0.0099, -0.0101, -0.01, -0.0099, -0.0101], "Accuracy", confidence=0.5, margins={"Accuracy": 0.01}, method="t"
    )
    assert abs(v.slack) < 1e-3


def test_rank_bound_is_the_fourth_smallest_walsh_average_for_n9():
    # n = 9 at 99 %: P(W <= 3) = 5/512 <= 0.01 < P(W <= 4) = 7/512, so the bound is the 4th smallest Walsh average
    d = np.arange(1, 10) / 1000
    w = np.sort([(d[i] + d[j]) / 2 for i in range(9) for j in range(i, 9)])
    assert rank_lower_bound(d) == pytest.approx(w[3])
    assert hodges_lehmann(d) == pytest.approx(np.median(w))


def test_large_win_does_not_penalise_under_rank_but_does_under_t():
    base = [-0.004, -0.002, 0.0, 0.001, 0.002, 0.003, -0.001, 0.004]
    small, big = base + [0.01], base + [0.2]  # the same fold, a modest vs a huge win
    assert rank_lower_bound(big) == rank_lower_bound(small)
    assert lower_bound(big) < lower_bound(small)


def test_rank_needs_enough_units_for_the_confidence():
    assert rank_lower_bound([0.01] * 6) == -np.inf  # 1/64 > 0.01: 99 % unreachable
    assert rank_lower_bound([0.01] * 7) == pytest.approx(0.01)  # 1/128 <= 0.01
