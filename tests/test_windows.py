"""Shared window-evidence rule and the L2/L4 adapters (generated data)."""
import numpy as np
import pytest
from scipy.stats import binom

from layered_nids.common.windows import window_score, windowed_evidence
from layered_nids.flows.adapter import flow_evidence


def test_window_score_matches_binomial_tail():
    assert window_score(1, 1, 0.03) == pytest.approx(0.97)
    assert window_score(100, 3, 0.03) == pytest.approx(1 - binom.sf(2, 100, 0.03))
    assert 0.35 < window_score(100, 3, 0.03) < 0.5                      # about chance: weak evidence
    assert window_score(100, 20, 0.03) > 0.99                            # far above chance: strong
    assert window_score(10, 0, 0.03) == 0.0 and window_score(0, 0, 0.03) == 0.0


def test_window_score_rejects_impossible_counts():
    with pytest.raises(ValueError):
        window_score(3, 4, 0.05)


def test_busy_but_normal_host_does_not_produce_evidence():
    rng = np.random.default_rng(0)
    n = 5000
    flagged = rng.random(n) < 0.03                                       # exactly the nominal false-alarm rate
    ts = np.sort(rng.uniform(0, 3600, n))
    ev = windowed_evidence("flows", ["h"] * n, ts, flagged, 0.03, 60.0, "flows", min_score=0.99)
    assert len(ev) <= 3                                                  # 60 windows x ~1% chance each: about 0-1 expected
    assert flagged.sum() > 100                                           # while a per-flow rule would have raised 100+ events


def test_attacker_window_is_detected_with_details_and_valid_evidence():
    ts = np.arange(30, dtype=float)
    flagged = np.array([True] * 12 + [False] * 18)
    ev = windowed_evidence("dga", ["bad"] * 30, ts, flagged, 0.01, 60.0, "domain lookups",
                           details=[f"d{i}.example" for i in range(30)])
    assert len(ev) == 1 and ev[0].layer == "dga" and ev[0].host == "bad" and ev[0].score > 0.99
    assert "12 of 30" in ev[0].reason and ev[0].detail["examples"] == ["d0.example", "d1.example", "d2.example"]


def test_hosts_and_windows_are_separate_buckets():
    ev = windowed_evidence("flows", ["a", "a", "b", "b"], [0, 100, 0, 100], [True, True, True, False],
                           0.03, 60.0, "flows", min_score=0.5)
    assert sorted((e.host, e.detail["window"]) for e in ev) == [("a", 0), ("a", 1), ("b", 0)]


def test_flow_adapter_uses_threshold_and_validates_inputs():
    scores = np.array([0.1, 0.9, 0.95, 0.2])
    ev = flow_evidence(scores, 0.8, ["h"] * 4, [0, 1, 2, 3], fpr=0.03, window_s=60,
                       attack_names=["Normal", "DoS", "DoS", "Normal"])
    assert len(ev) == 1 and ev[0].detail["flagged"] == 2
    with pytest.raises(ValueError):
        flow_evidence(scores, 0.8, ["h"] * 4, [0, 1, 2, 3], fpr=0.0)
    with pytest.raises(ValueError):
        flow_evidence(scores, 0.8, ["h"] * 4, [0, 1, 2, 3], fpr=0.03, window_s=0)


def test_burst_aware_null_is_more_forgiving_and_matches_binomial_without_bursts():
    from layered_nids.common.windows import window_score
    base = window_score(100, 10, 0.03)                                   # 10 of 100 flagged at a 3% rate: very surprising if independent
    assert window_score(100, 10, 0.03, rho=0.2) < base                   # but not if flags arrive in bursts
    assert window_score(100, 10, 0.03, rho=0.2) < window_score(100, 10, 0.03, rho=0.05) < base
    assert window_score(100, 10, 0.03, rho=1e-9) == pytest.approx(base, abs=1e-6)     # no burstiness -> the binomial answer
    assert window_score(100, 100, 0.03, rho=0.2) >= 0.99                 # a window that is ENTIRELY flagged is still strong evidence


def test_fit_overdispersion_recovers_burstiness_and_sees_none_in_independent_data():
    from layered_nids.common.windows import fit_overdispersion
    rng = np.random.default_rng(0)
    n = rng.integers(40, 200, 600)
    iid = [(int(m), int(rng.binomial(m, 0.02)), 0.0) for m in n]
    mu, rho, used = fit_overdispersion(iid)
    assert abs(mu - 0.02) < 0.004 and rho < 0.03 and used == 600
    true_rho, p = 0.15, 0.02
    s = 1 / true_rho - 1
    bursty = [(int(m), int(rng.binomial(m, rng.beta(p * s, (1 - p) * s))), 0.0) for m in n]
    mu2, rho2, _ = fit_overdispersion(bursty)
    assert 0.08 < rho2 < 0.25 and abs(mu2 - 0.02) < 0.01                  # recovers roughly the true 0.15
    assert fit_overdispersion([(3, 1, 0.0)]) == (0.01, 0.0, 0)            # too few events per window: falls back safely
