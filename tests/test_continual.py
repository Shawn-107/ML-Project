"""Continual scorer: follows slow drift, does not learn an attack block (synthetic features; the real-capture check is in anomaly/labelled_exp.py)."""
import numpy as np

from layered_nids.anomaly.continual import ContinualScorer
from layered_nids.anomaly.model import AnomalyEnsemble
from layered_nids.common.metrics import threshold_at_fpr

D = 16


def _make(rng, n, shift):
    z = rng.normal(size=(n, 4))
    W = np.random.default_rng(1).normal(size=(4, D))
    return (np.abs(z @ W + 0.2 * rng.normal(size=(n, D))) + shift).astype(np.float32)


def test_follows_drift_and_refuses_attack_block():
    rng = np.random.default_rng(0)
    kw = dict(epochs=4, seed=0)
    tr, tail, thr_rows = _make(rng, 30000, 0), _make(rng, 10000, 0), _make(rng, 10000, 0)
    ens = AnomalyEnsemble(**kw).fit(tr, tail)
    thr = threshold_at_fpr(ens.score(thr_rows), 0.01)
    X = np.vstack([_make(rng, 100000, 0)] + [_make(rng, 25000, 3 * (i + 1) / 8) for i in range(8)] + [_make(rng, 100000, 3)])
    attack = np.zeros(len(X), bool)
    a0 = len(X) - 50000
    X[a0:a0 + 15000] += 25.0
    attack[a0:a0 + 15000] = True

    static_flags = ens.score(X) >= thr
    sc = ContinualScorer(AnomalyEnsemble(**kw).fit(tr, tail), thr, window=60000, min_refit_rows=20000)
    s, t, log = sc.run(X)
    flags = s >= t
    after = slice(300000, len(X) - 50000)
    assert static_flags[after].mean() > 0.9                         # the fixed model is lost after the drift
    assert flags[after].mean() < 0.03                               # the continual one is back near the 1% target
    assert flags[:100000].mean() < 0.03                             # nothing needless while traffic is stable
    assert flags[attack].mean() > 0.95                              # the attack block is still flagged
    assert any(e["refit"] for e in log) and any("blocked" in e for e in log)


def test_veto_by_alerts_hides_the_alerted_windows_only():
    from layered_nids.anomaly.continual import vetoed_by_alerts
    from layered_nids.common.schema import Evidence

    hosts = np.array(["a", "a", "a", "a", "b"])
    ts = np.array([0.0, 100.0, 170.0, 400.0, 100.0])
    ev = [Evidence("rules", "a", 120.0, 0.9, "x")]                    # window 60 s ending at t=120, pad 60: hides [0, 180] for host a
    v = vetoed_by_alerts(hosts, ts, ev, window_s=60, pad_s=60)
    assert v.tolist() == [True, True, True, False, False]


def test_confirmed_recalibrator_follows_drift_and_never_learns_unconfirmed_stealth_attack():
    from layered_nids.anomaly.continual import ConfirmedRecalibrator

    rng = np.random.default_rng(0)
    kw = dict(epochs=4, seed=0)
    tr, tail, thr_rows = _make(rng, 30000, 0), _make(rng, 10000, 0), _make(rng, 10000, 0)
    ens = AnomalyEnsemble(**kw).fit(tr, tail)
    thr = threshold_at_fpr(ens.score(thr_rows), 0.01)
    X = np.vstack([_make(rng, 50000, 0)] + [_make(rng, 25000, 3 * (i + 1) / 8) for i in range(8)] + [_make(rng, 100000, 3)])
    attack = np.zeros(len(X), bool)
    a0 = len(X) - 60000
    X[a0:] = _make(rng, 60000, 3)                                      # same drifted normal ...
    X[a0:a0 + 30000] += 2.0                                            # ... plus a stealthy block (the score-only ContinualScorer learns most of it: 35% recall on this stream)
    attack[a0:a0 + 30000] = True
    sc = ConfirmedRecalibrator(AnomalyEnsemble(**kw).fit(tr, tail), thr, window=60000, min_rows=20000, min_gap=40000)
    s, t, log = sc.run(X, ~attack)                                     # the operator confirms everything except the attack block
    flags = s >= t
    assert flags[300000 - 25000:a0].mean() < 0.05                      # drift followed
    assert flags[attack].mean() > 0.95                                 # the stealthy block, never confirmed clean, is still flagged
    assert any(e["refit"] for e in log)
