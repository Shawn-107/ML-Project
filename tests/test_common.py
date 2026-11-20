import numpy as np
import pytest

from layered_nids.common.metrics import calibration, rates, ranking, threshold_at_fpr
from layered_nids.common.schema import Evidence


def test_threshold_respects_false_alarm_budget():
    rng = np.random.default_rng(0)
    normal = rng.normal(size=10_000)
    for target in (0.01, 0.03, 0.05):
        t = threshold_at_fpr(normal, target)
        assert (normal >= t).mean() <= target + 1e-9
        assert (normal >= t).mean() >= target - 0.002          # and it is not wastefully strict


def test_threshold_handles_ties_without_exceeding_budget():
    normal = np.array([0.0] * 95 + [1.0] * 5)
    t = threshold_at_fpr(normal, 0.03)                         # 5% of normals tie at 1.0 -> must flag none
    assert (normal >= t).mean() <= 0.03


def test_threshold_zero_budget_flags_none_of_the_sample():
    normal = np.linspace(0, 1, 50)
    assert (normal >= threshold_at_fpr(normal, 0.0)).sum() == 0


def test_threshold_rejects_empty():
    with pytest.raises(ValueError):
        threshold_at_fpr([], 0.01)


def test_rates_and_ranking():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.6, 0.7, 0.9])
    r = rates(y, s, 0.65)
    assert r["recall"] == 1.0 and r["false_alarm_rate"] == 0.0 and r["precision"] == 1.0
    assert ranking(y, s)["auc"] == 1.0
    assert np.isnan(ranking(np.zeros(4), s)["auc"])             # one class only: undefined, not a fake number


def test_calibration_perfect_and_bad():
    y = np.array([0, 1] * 50)
    assert calibration(y, y.astype(float))["brier"] == 0.0
    assert calibration(y, np.full(100, 0.9))["ece"] > 0.3


def test_evidence_validation():
    e = Evidence("rules", "10.0.0.5", 1.0, 0.4, "x")
    assert e.to_dict()["host"] == "10.0.0.5"
    with pytest.raises(ValueError):
        Evidence("nope", "h", 0, 0.5, "x")
    with pytest.raises(ValueError):
        Evidence("dga", "h", 0, 1.5, "x")
