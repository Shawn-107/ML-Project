"""L4 tests. Generated data only: they check plumbing and the evaluation rules, not detection quality."""
import numpy as np
import pytest

from layered_nids.flows.data import source_feature
from layered_nids.flows.evaluate import evaluate_scores, mean_std
from layered_nids.flows.models import HGB, IsoNovelty, KMeansNovelty, LogitStacker, MultiClass


def make_xy(n=1500, seed=0):
    rng = np.random.default_rng(seed)
    fam = rng.choice(["Normal", "DoS", "Fuzzers", "Exploits"], size=n, p=[0.5, 0.2, 0.15, 0.15])
    shift = {"Normal": 0.0, "DoS": 2.0, "Fuzzers": 1.0, "Exploits": 1.5}
    X = rng.normal(size=(n, 6)) + np.array([shift[f] for f in fam])[:, None]
    return X.astype(np.float32), (fam != "Normal").astype(int), fam


def test_source_feature_mapping():
    assert source_feature("cat__proto_tcp") == "proto"
    assert source_feature("cat__service_-") == "service"
    assert source_feature("cat__state_FIN") == "state"
    assert source_feature("num__sbytes") == "sbytes"
    assert source_feature("bin__is_sm_ips_ports") == "is_sm_ips_ports"


def test_models_fit_and_score_in_unit_interval():
    X, y, fam = make_xy()
    for m in (HGB(0, max_iter=30), MultiClass("hgb", 0), MultiClass("rf", 0)):
        m.fit(X, y, cats=fam) if isinstance(m, MultiClass) else m.fit(X, y)
        s = m.score(X)
        assert s.shape == (len(X),) and (s >= 0).all() and (s <= 1).all()
        assert s[y == 1].mean() > s[y == 0].mean() + 0.2


def test_multiclass_requires_family_labels():
    X, y, _ = make_xy(200)
    with pytest.raises(ValueError):
        MultiClass("rf").fit(X)


def test_stacker_learns_to_trust_the_informative_model():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 4000)
    good = np.clip(y * 0.6 + 0.2 + rng.normal(0, 0.1, 4000), 0.01, 0.99)
    noise = rng.uniform(0.01, 0.99, 4000)
    s = LogitStacker().fit(np.column_stack([good, noise]), y)
    w, _ = s.weights
    assert abs(w[0]) > 5 * abs(w[1])


def test_novelty_scorers_rank_far_points_higher_and_kmeans_picks_k():
    rng = np.random.default_rng(2)
    normal = rng.normal(size=(2000, 5)).astype(np.float32)
    far = (rng.normal(size=(200, 5)) + 6).astype(np.float32)
    for m in (IsoNovelty(0, trees=50), KMeansNovelty(ks=(2, 4), sil_sample=500)):
        m.fit(normal)
        assert m.score(far).mean() > m.score(normal[:200]).mean()
    assert KMeansNovelty(ks=(2, 4), sil_sample=500).fit(normal).k_ in (2, 4)


def test_shift_scan_finds_the_shifted_feature_and_weights_are_sane():
    from layered_nids.flows.adversarial import groups_of, shift_scan, test_likeness_weights
    rng = np.random.default_rng(4)
    src = rng.normal(size=(1500, 4)).astype(np.float32)
    tgt = rng.normal(size=(1500, 4)).astype(np.float32)
    tgt[:, 2] += 2.5                                                     # only 'num__c' is shifted
    names = ["num__a", "num__b", "num__c", "num__d"]
    scan = shift_scan(src, tgt, names, repeats=2)
    assert scan["cv_auc"] > 0.85
    assert next(iter(scan["importance"])) == "c"
    same = shift_scan(src, rng.normal(size=(1500, 4)).astype(np.float32), names, repeats=1)
    assert same["cv_auc"] < 0.6                                          # no shift -> near chance
    w = test_likeness_weights(src, tgt)
    assert w.shape == (1500,) and abs(w.mean() - 1.0) < 1e-6 and w.min() > 0
    assert w[src[:, 2] > 1.5].mean() > w[src[:, 2] < -1.0].mean()        # rows that look like the target get more weight
    assert groups_of(["cat__proto_tcp", "cat__proto_udp", "num__dur"]) == {"proto": [0, 1], "dur": [2]}


def test_evaluate_scores_threshold_comes_from_validation_only():
    rng = np.random.default_rng(3)
    yv = rng.integers(0, 2, 6000)
    sv = np.clip(yv * 0.4 + rng.normal(0.3, 0.15, 6000), 0, 1)
    yt = rng.integers(0, 2, 6000)
    st = np.clip(yt * 0.4 + rng.normal(0.3, 0.15, 6000), 0, 1)
    cat = np.where(yt == 1, "DoS", "Normal")
    r = evaluate_scores(sv, yv, st, yt, cat)
    assert r["fpr_1pct"]["threshold"] >= r["fpr_3pct"]["threshold"]       # stricter policy -> higher threshold
    assert abs(r["fpr_3pct"]["false_alarm_rate"] - 0.03) < 0.012          # same distribution -> close to target
    assert "DoS" in r["fpr_3pct"]["flagged_by_family"]
    agg = mean_std([r, r])
    assert agg["fpr_3pct"]["recall"]["std"] == 0.0 and agg["auc_test"]["n"] == 2
