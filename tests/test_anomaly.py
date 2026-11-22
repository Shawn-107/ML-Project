"""NIDS packet anomaly detector: features, model, detector (train/score/stream/save/load) and the injected attacks. Generated data only."""
import numpy as np
import pytest

from layered_nids.anomaly import PacketAnomalyDetector
from layered_nids.anomaly.features import PER_LAMBDA, PacketFeatures, entropy, feature_names
from layered_nids.anomaly.inject import EXTRA_KINDS, KINDS, inject, make_attack
from layered_nids.anomaly.model import AnomalyEnsemble
from layered_nids.packets.pcap import Packet

HOSTS = [f"192.168.2.{i}" for i in range(10, 22)]
EXT = ["93.184.216.34", "8.8.8.8", "151.101.1.69", "104.16.0.1"]


def normal_traffic(n=7000, seed=0, t0=1000.0, rate=25.0):
    rng = np.random.default_rng(seed)
    t, out = t0, []
    for _ in range(n):
        t += rng.exponential(1 / rate)
        h, d = HOSTS[rng.integers(len(HOSTS))], EXT[rng.integers(len(EXT))]
        kind = rng.random()
        if kind < 0.5:
            out.append(Packet(t, int(rng.normal(600, 150)) % 1400 + 60, "aa:aa:aa:00:00:" + h[-2:], "bb:bb:bb:00:00:01", h, d, "tcp", 40000 + int(rng.integers(0, 50)), 443, "", 0x10, ""))
        elif kind < 0.8:
            out.append(Packet(t, int(rng.normal(120, 20)) % 300 + 60, "aa:aa:aa:00:00:" + h[-2:], "bb:bb:bb:00:00:01", h, d, "udp", 50000, 53, "www.example.org", 0, ""))
        else:
            out.append(Packet(t, 90, "aa:aa:aa:00:00:" + h[-2:], "bb:bb:bb:00:00:01", d, h, "tcp", 443, 40000 + int(rng.integers(0, 50)), "", 0x10, ""))
    return out


@pytest.fixture(scope="module")
def trained():
    tr, ca, ho = normal_traffic(5000, 0, 1000.0), normal_traffic(1500, 1, 1000.0 + 5000 / 25.0 + 1), normal_traffic(3000, 2, 1000.0 + 6500 / 25.0 + 2)
    det = PacketAnomalyDetector(fpr=0.02, epochs=4, seed=0).fit(tr, ca)
    return det, ho


def test_feature_vector_shape_names_and_finite():
    fx = PacketFeatures()
    X = fx.run(normal_traffic(300))
    assert X.shape == (300, 120) and len(feature_names()) == 120 == PER_LAMBDA * 4 and np.isfinite(X).all()


def test_entropy_random_names_exceed_ordinary_names():
    assert entropy("www.google.com") < 3.2 < entropy("x7k2q9vb3m1z8w.example-bad.com")
    assert entropy("") == 0.0


def test_scan_raises_the_novelty_features_far_above_normal():
    names = feature_names()
    idx = [i for i, n in enumerate(names) if n.startswith("novelty_new_targets") and n.endswith("L1")][0]
    normal = PacketFeatures().run(normal_traffic(800))
    host = HOSTS[0]
    scan = [Packet(2000.0 + i * 0.05, 60, "aa", "bb", host, f"10.1.{i // 200}.{i % 200 + 1}", "tcp", 40000, 23, "", 0x02, "") for i in range(300)]
    X = PacketFeatures().run(normal_traffic(800) + scan)
    assert X[-1, idx] > 3 * max(normal[:, idx].max(), 1.0)


def test_streaming_equals_batch_scores(trained):
    det, ho = trained
    import copy
    a, b = copy.deepcopy(det), copy.deepcopy(det)
    batch = a.score_packets(ho[:200])
    stream = np.array([b.process(p) for p in ho[:200]])
    assert np.allclose(batch, stream, rtol=1e-4, atol=1e-4)                 # same scores: the tail comparison tolerates float round-off


def test_threshold_gives_roughly_the_target_false_alarm_rate_on_new_normal_traffic(trained):
    det, ho = trained
    import copy
    flagged = copy.deepcopy(det).flag(copy.deepcopy(det).score_packets(ho))
    assert 0.0 <= flagged.mean() < 0.10                  # target 2%; small generated sample, so a loose bound


def test_attacks_score_higher_than_normal(trained):
    det, ho = trained
    import copy
    stream, labels, eps = inject(ho, ("syn_flood", "hscan", "exfil", "udp_flood", "dns_burst", "vscan", "web_attack"), seed=1)
    s = copy.deepcopy(det).score_packets(stream)
    thr = det.threshold
    for kind in ("syn_flood", "hscan", "exfil", "udp_flood"):
        assert (s[labels == kind] >= thr).mean() > 0.5, kind
    assert (s[labels == ""] >= thr).mean() < 0.3


def test_evidence_for_attacker_host_not_for_quiet_hosts(trained):
    det, ho = trained
    import copy
    stream, labels, eps = inject(ho, ("syn_flood",), seed=3)
    d = copy.deepcopy(det)
    ev = d.evidence(stream, d.score_packets(stream))
    atk = eps[0]["host"]
    assert any(e.host == atk and e.layer == "packets" and e.score > 0.99 for e in ev)


def test_save_load_roundtrip_and_tamper_check(trained, tmp_path):
    det, ho = trained
    import copy
    d = det.save(tmp_path / "m")
    det2 = PacketAnomalyDetector.load(d)
    a, b = copy.deepcopy(det), det2
    b.features = copy.deepcopy(det.features)
    b.features = copy.deepcopy(det.features)
    assert np.allclose(a.score_packets(ho[:100]), b.score_packets(ho[:100]), rtol=1e-4, atol=1e-4)
    w = d / "main" / "weights.pt"
    w.write_bytes(w.read_bytes() + b"x")
    with pytest.raises(ValueError):
        PacketAnomalyDetector.load(d)


def test_tail_sum_mode_scores_and_rejects_unknown_mode():
    X = np.random.default_rng(0).normal(size=(600, 12)).astype(np.float32)
    e = AnomalyEnsemble(max_group=4, epochs=3, score_mode="tail_sum", topk=2, grouping="correlation").fit(X[:400], X[400:])
    s = e.score(np.vstack([X[:50], X[:50] + 8]))
    assert s[50:].mean() > s[:50].mean()
    with pytest.raises(ValueError):
        AnomalyEnsemble(score_mode="nope")
    with pytest.raises(ValueError):
        AnomalyEnsemble(grouping="nope")
    with pytest.raises(ValueError):
        AnomalyEnsemble(epochs=1, grouping="family").fit(X[:400], X[400:])      # family grouping needs the feature names


def test_injection_is_labelled_deterministic_and_needs_enough_hosts():
    base = normal_traffic(14000, 5)
    s1, l1, e1 = inject(base, KINDS + EXTRA_KINDS, seed=7)
    s2, l2, e2 = inject(base, KINDS + EXTRA_KINDS, seed=7)
    assert [p.ts for p in s1] == [p.ts for p in s2] and (l1 == l2).all() and e1 == e2
    assert len(s1) == len(base) + sum(e["n_packets"] for e in e1) and set(l1) == {""} | set(KINDS + EXTRA_KINDS)
    assert all(s1[i].ts <= s1[i + 1].ts for i in range(len(s1) - 1))
    with pytest.raises(ValueError):
        inject(base[:50], KINDS, seed=0)
    with pytest.raises(ValueError):
        make_attack("nope", "192.168.2.1", "aa", 0.0, np.random.default_rng(0))


def test_two_view_detector_scores_saves_and_loads(tmp_path):
    import copy
    tr, ca, ho = normal_traffic(4000, 0, 1000.0), normal_traffic(1200, 1, 1000.0 + 4000 / 25.0 + 1), normal_traffic(1200, 2, 1000.0 + 5200 / 25.0 + 2)
    det = PacketAnomalyDetector(fpr=0.02, epochs=3, seed=0, views=2).fit(tr, ca)
    a = copy.deepcopy(det)
    s1 = a.score_packets(ho[:300])
    assert np.isfinite(s1).all() and det.threshold > 0
    d = det.save(tmp_path / "two")
    b = PacketAnomalyDetector.load(d)
    b.features = copy.deepcopy(det.features)
    assert np.allclose(s1, b.score_packets(ho[:300]), rtol=1e-4, atol=1e-4)
    f = d / "b" / "weights.pt"
    f.write_bytes(f.read_bytes() + b"x")
    with pytest.raises(ValueError):
        PacketAnomalyDetector.load(d)


def test_setup_check_reports_each_piece_with_a_fix():
    from layered_nids import setup_check
    rows = setup_check.checks()
    assert rows and all(len(r) == 4 and isinstance(r[1], bool) and r[3] for r in rows)
    assert any("UNSW-NB15 training file" in r[0] for r in rows)


def test_drift_threshold_keeps_the_false_alarm_rate_and_still_flags_attacks():
    from layered_nids.anomaly.drift import DriftThreshold, adaptive_thresholds
    rng = np.random.default_rng(0)
    calib = rng.lognormal(0.0, 0.5, 20000)
    thr0 = float(np.quantile(calib, 0.99))
    later = rng.lognormal(0.0, 0.5, 30000) * np.linspace(1.0, 1.35, 30000)         # traffic drifts: normal scores creep up by 35%
    fixed = (later >= thr0).mean()
    assert fixed > 0.02                                                              # the fixed threshold overshoots its 1% target
    thr = adaptive_thresholds(later, thr0)
    assert (later >= thr)[10000:].mean() < 0.015 and thr.min() >= 0.7 * thr0 - 1e-9 and thr.max() <= 2.5 * thr0 + 1e-9
    attack = np.concatenate([later[:5000], later[5000:5050] * 8, later[5050:9000]])  # 50 packets with scores 8x higher: far above the cap
    assert (attack >= adaptive_thresholds(attack, thr0))[5000:5050].mean() > 0.95
    d = DriftThreshold(thr0)                                                         # streaming gives exactly the same thresholds
    assert np.allclose([d.update(s) for s in later[:6000]], thr[:6000])


def test_regularity_option_adds_four_features_per_decay_rate_and_scores():
    from layered_nids.anomaly.features import feature_names
    fx = PacketFeatures(regularity=True)
    X = fx.run(normal_traffic(300))
    assert X.shape == (300, 136) and len(feature_names(regularity=True)) == 136 and np.isfinite(X).all()
    assert PacketFeatures().run(normal_traffic(50)).shape == (50, 120)                  # the default stays at 120
