"""Fusion simulator tests (generated pools). They check the bookkeeping, not that fusion 'works' on real networks."""
import numpy as np
import pytest

from layered_nids.fusion.accumulator import EvidenceAccumulator, FusionConfig
from layered_nids.fusion.simulate import LayerPool, make_scenario, score_alerts


def pools(seed=0):
    rng = np.random.default_rng(seed)
    mk = lambda layer, rate: LayerPool(layer, rng.beta(1, 12, 5000), rng.beta(5, 2, 5000), 0.5, 0.02, rate, "events")
    return {"dga": mk("dga", 6.0), "flows": mk("flows", 20.0)}


def test_pool_rates_are_measured_from_the_scores():
    p = pools()["flows"]
    assert 0.0 <= p.measured_fpr < 0.1 and 0.5 < p.measured_recall <= 1.0


def test_scenario_is_reproducible_and_truth_is_consistent():
    a = make_scenario(pools(), n_benign=10, n_attack=4, hours=1.0, seed=3)
    b = make_scenario(pools(), n_benign=10, n_attack=4, hours=1.0, seed=3)
    assert [(e.host, e.layer, round(e.ts, 6)) for e in a.evidence] == [(e.host, e.layer, round(e.ts, 6)) for e in b.evidence]
    assert len(a.truth) == 4 and len(a.hosts) == 14 and set(a.truth) <= set(a.hosts)
    assert all(0 < s < e <= a.duration_s for s, e in a.truth.values())


def test_attackers_produce_more_evidence_inside_their_window_than_benign_hosts():
    sc = make_scenario(pools(), n_benign=20, n_attack=8, hours=2.0, seed=1)
    inside = sum(1 for e in sc.evidence if e.host in sc.truth and sc.truth[e.host][0] <= e.ts <= sc.truth[e.host][1])
    benign = sum(1 for e in sc.evidence if e.host not in sc.truth)
    assert inside / 8 > 3 * (benign / 20 + 1e-9)


def test_score_alerts_counts_detections_and_false_alarms_separately():
    sc = make_scenario(pools(), n_benign=5, n_attack=2, hours=1.0, seed=2)
    h_att = next(iter(sc.truth))
    s, e = sc.truth[h_att]
    h_ben = next(h for h in sc.hosts if h not in sc.truth)
    from layered_nids.fusion.accumulator import Alert
    mk = lambda h, t: Alert(h, t, 0.99, "high", {"flows": 1.0}, ("x",))
    r = score_alerts([mk(h_att, s + 30), mk(h_att, s + 90), mk(h_ben, 100.0), mk(h_att, max(s - 60, 0) - 1)], sc)
    assert r["detected"] == 1 and r["median_time_to_detect_s"] == pytest.approx(30.0)
    assert r["false_alerts_total"] == 2 and r["benign_hosts_alerted_share"] == pytest.approx(1 / 5)           # 5 benign hosts, 1 alerted


def test_accumulator_runs_on_scenario_evidence_and_both_layers_beat_one_layer():
    cfg = FusionConfig(alert_p=0.99, cooldown_s=300.0)
    found = {"flows": [], "both": []}
    for seed in range(6):
        sc = make_scenario(pools(), n_benign=15, n_attack=6, hours=2.0, seed=seed)
        only = [e for e in sc.evidence if e.layer == "flows"]
        found["flows"].append(score_alerts(EvidenceAccumulator(cfg).run(only), sc)["detection_rate"])
        found["both"].append(score_alerts(EvidenceAccumulator(cfg).run(sc.evidence), sc)["detection_rate"])
    assert np.mean(found["both"]) > np.mean(found["flows"])             # DGA-only attackers are invisible to flows alone


def test_bucket_labels_follow_the_attack_interval():
    from layered_nids.fusion.combiner import bucket_labels
    keys = [("a", 0), ("a", 1), ("a", 2), ("a", 3), ("b", 1)]
    y, keep = bucket_labels(keys, {"a": (350.0, 700.0)}, bucket_s=300.0)
    assert list(y) == [0, 1, 1, 0, 0]                                   # a: buckets 1 and 2 overlap [350, 700]
    assert list(keep) == [True, True, True, False, True]                # bucket 3 is after the attack: dropped


def test_bucket_features_shape_and_trailing_window():
    from layered_nids.common.schema import Evidence
    from layered_nids.fusion.combiner import bucket_features
    ev = [Evidence("flows", "h", 10.0, 0.9, "x"), Evidence("flows", "h", 700.0, 0.9, "x")]
    X, keys = bucket_features(ev, ["h"], 1800.0, bucket_s=300.0, trail=3)
    assert X.shape == (6, 7) and keys[0] == ("h", 0)                    # 2 layers x 3 feature kinds + active count
    w = -np.log(0.1)
    # column layout: [max dga, max flows, bucket-sum dga, bucket-sum flows, trailing dga, trailing flows, active layers]
    assert X[0, 1] == pytest.approx(0.9) and X[2, 1] == pytest.approx(0.9)       # per-bucket max (flows)
    assert X[1, 1] == 0.0                                                          # nothing in bucket 1
    assert X[2, 5] == pytest.approx(2 * w)                                         # trailing sum at bucket 2 covers buckets 0..2
    assert X[5, 5] == 0.0                                                          # bucket 5: older evidence has left the window


def test_learned_combiner_beats_chance_on_simulated_scenarios():
    from layered_nids.fusion.combiner import LearnedCombiner
    train = [make_scenario(pools(), n_benign=15, n_attack=6, hours=2.0, seed=s) for s in range(6)]
    comb = LearnedCombiner().fit(train)
    test = make_scenario(pools(), n_benign=15, n_attack=6, hours=2.0, seed=99)
    res = score_alerts(comb.alerts(test, 0.9), test)
    assert res["detection_rate"] >= 0.5 and res["benign_hosts_alerted_share"] < 0.5
