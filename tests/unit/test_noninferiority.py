import numpy as np
import pytest

from pymiraw.evaluation.noninferiority import CONFIDENCE, MARGINS, lower_bound, non_inferiority


def test_agreed_settings():
    assert CONFIDENCE == 0.99 and MARGINS == {"Accuracy": 0.01, "F1": 0.01, "AUC": 0.005}


def test_identical_models_are_non_inferior():
    v = non_inferiority([0.0, 0.001, -0.001, 0.0, 0.002], "Accuracy")
    assert v.non_inferior and abs(v.mean) < 0.001 and v.lower < v.mean


def test_clearly_worse_model_fails():
    assert not non_inferiority([-0.03, -0.02, -0.04, -0.025], "F1").non_inferior


def test_noisy_small_sample_is_not_declared_non_inferior():
    # mean is above the limit but the spread is huge: the bound is below -margin
    assert not non_inferiority([0.02, -0.03, 0.01, -0.02, 0.03, -0.04], "Accuracy").non_inferior


def test_higher_confidence_gives_lower_bound():
    d = np.array([0.001, -0.004, 0.002, -0.001, 0.0, -0.003])
    assert lower_bound(d, 0.99) < lower_bound(d, 0.95) < d.mean()


def test_margin_is_per_metric_and_boundary_is_exclusive():
    d = np.array([-0.0075] * 5) + np.array([0.0, 1e-9, -1e-9, 0.0, 1e-9])  # lower bound ~ -0.0075
    assert non_inferiority(d, "Accuracy").non_inferior  # margin 0.01
    assert not non_inferiority(d, "AUC").non_inferior  # margin 0.005
    assert non_inferiority(d, "AUC", margins={"AUC": 0.0075 + 1e-3}).non_inferior


def test_needs_two_points():
    with pytest.raises(ValueError):
        lower_bound([0.1])


def test_slack_reports_fragility():
    v = non_inferiority([-0.0099, -0.0101, -0.01, -0.0099, -0.0101], "Accuracy", confidence=0.5)
    assert abs(v.slack) < 1e-3
