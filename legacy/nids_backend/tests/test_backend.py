import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
import pytest

from nids import Config, Detector, SchemaError, load_bundle, save_bundle, train
from nids.config import CONTINUOUS, FEATURES, LABEL
from nids.data import DataError, load_splits, validate_labelled
from nids.drift import DriftMonitor, psi
from nids.features import coerce_flows, log1p_clip
from nids.pipeline import blend
from nids.server import make_server
from nids.thresholds import f1_threshold, fpr_threshold, labelled_threshold


@pytest.fixture()
def det(bundle):
    return Detector(bundle)


def flows(df, n=5):
    return df[FEATURES].head(n).to_dict("records")


# ---- data contract / leakage ---------------------------------------------------------------------
def test_splits_are_disjoint_and_clean(splits):
    h = lambda d: set(pd.util.hash_pandas_object(d[FEATURES], index=False))
    assert not (h(splits.test) & h(splits.train)), "test rows must not appear in training"
    assert splits.train[FEATURES].duplicated().sum() == 0
    assert splits.info["n_test"] == len(splits.test) > 0


def test_validate_rejects_bad_labels(splits):
    bad = splits.val.copy()
    bad.loc[bad.index[0], LABEL] = 1 - bad.loc[bad.index[0], LABEL]
    with pytest.raises(DataError):
        validate_labelled(bad)
    with pytest.raises(DataError):
        validate_labelled(splits.val.drop(columns=["dur"]))


def test_preprocessor_fitted_on_train_only(bundle, splits):
    scaler = bundle.preprocessor.named_transformers_["num"].named_steps["scale"]
    expect = log1p_clip(splits.train[CONTINUOUS].to_numpy()).mean(0)
    assert np.allclose(scaler.mean_, expect)


# ---- input validation ------------------------------------------------------------------------------
def test_coerce_errors_and_normalisation(splits):
    f = flows(splits.val, 1)[0]
    with pytest.raises(SchemaError, match="missing"):
        coerce_flows({k: v for k, v in f.items() if k != "sbytes"})
    with pytest.raises(SchemaError, match="finite"):
        coerce_flows({**f, "dur": float("nan")})
    with pytest.raises(SchemaError, match="non-numeric"):
        coerce_flows({**f, "dur": "abc"})
    with pytest.raises(SchemaError):
        coerce_flows([])
    assert coerce_flows({**f, "proto": " TCP "}).loc[0, "proto"] == "tcp"


# ---- thresholds -------------------------------------------------------------------------------------
def test_threshold_policies():
    rng = np.random.RandomState(0)
    y = np.r_[np.zeros(5000), np.ones(5000)]
    s = np.r_[rng.beta(2, 5, 5000), rng.beta(5, 2, 5000)]
    t = fpr_threshold(s, y, 0.03)
    assert abs((s[y == 0] >= t).mean() - 0.03) < 0.005
    assert 0 < f1_threshold(s, y) < 1
    t2 = labelled_threshold(s, y, fpr_cap=0.05)
    assert (s[y == 0] >= t2).mean() <= 0.05
    with pytest.raises(ValueError):
        labelled_threshold(s[:50], y[:50])
    with pytest.raises(ValueError):
        fpr_threshold(s[y == 1], y[y == 1], 0.05)


# ---- persistence --------------------------------------------------------------------------------------
def test_save_load_roundtrip_and_integrity(bundle, splits, tmp_path):
    p = save_bundle(bundle, tmp_path / "b.joblib")
    a = Detector(bundle).predict_frame(splits.val.head(50), update_monitor=False)
    b = Detector(load_bundle(p)).predict_frame(splits.val.head(50), update_monitor=False)
    assert np.allclose(a.attack_probability, b.attack_probability)
    data = bytearray(p.read_bytes()); data[len(data) // 2] ^= 0xFF; p.write_bytes(bytes(data))
    with pytest.raises(RuntimeError, match="checksum"):
        load_bundle(p)
    (tmp_path / "b.meta.json").unlink()
    with pytest.raises(RuntimeError, match="verify"):
        load_bundle(p)
    with pytest.raises(FileNotFoundError):
        load_bundle(tmp_path / "nope.joblib")


# ---- inference contract --------------------------------------------------------------------------------
def test_detector_matches_training_pipeline(bundle, splits, det):
    """The deployed detector must give exactly the scores the training code computes."""
    val = splits.val.head(200)
    X = bundle.preprocessor.transform(val[FEATURES])
    expect = blend(bundle.ann.predict_proba(X), bundle.rf.predict_proba(X), bundle.blend_weight)
    got = det.predict_frame(val, update_monitor=False).attack_probability.to_numpy()
    assert np.allclose(expect, got)


def test_predict_single_batch_and_robustness(det, splits):
    rows = flows(splits.val, 4)
    batch = det.predict(rows)
    single = [det.predict([r])[0] for r in rows]
    assert [round(x["attack_probability"], 6) for x in batch] == [round(x["attack_probability"], 6) for x in single]
    assert {"is_attack", "attack_probability", "severity", "top_deviations"} <= set(batch[0])
    assert len(batch[0]["top_deviations"]) == 3
    weird = {**rows[0], "proto": "NEVER-SEEN", "state": "XYZ"}        # unseen categories must not crash
    assert 0 <= det.predict([weird])[0]["attack_probability"] <= 1
    upper = det.predict([{**rows[0], "proto": rows[0]["proto"].upper()}])[0]
    assert upper["attack_probability"] == pytest.approx(batch[0]["attack_probability"])
    neg = det.predict([{**rows[0], "dur": -5.0}])[0]                    # negative counters are clipped, not fatal
    assert 0 <= neg["attack_probability"] <= 1
    with pytest.raises(SchemaError):
        det.predict([{k: v for k, v in rows[0].items() if k != "dur"}])


def test_policies_and_novelty(bundle, splits):
    d = Detector(bundle)
    strict, loose = splits.val.head(400), splits.val.head(400)
    d.set_policy("fpr_1pct"); n1 = d.predict_frame(strict, update_monitor=False).is_attack.sum()
    d.set_policy("fpr_5pct"); n5 = d.predict_frame(loose, update_monitor=False).is_attack.sum()
    assert n1 <= n5, "a stricter false-alarm target must flag no more flows"
    with pytest.raises(ValueError):
        d.set_policy("does_not_exist")
    base = Detector(bundle).predict_frame(splits.val.head(400), update_monitor=False).is_attack.sum()
    nov = Detector(bundle, use_novelty=True).predict_frame(splits.val.head(400), update_monitor=False).is_attack.sum()
    assert nov >= base


def test_recalibration(bundle, splits):
    d = Detector(bundle)
    sample = splits.val.head(600)
    rep = d.recalibrate(sample, fpr_cap=0.08)
    assert rep["n_holdout"] > 0 and "holdout_new" in rep
    assert rep["applied"] == (d.info()["recalibrated"])
    d.reset_recalibration(); assert not d.info()["recalibrated"]
    with pytest.raises(ValueError):
        d.recalibrate(splits.val.head(50))
    with pytest.raises(SchemaError):
        d.recalibrate(splits.val[FEATURES].head(600))              # no label column


# ---- drift ------------------------------------------------------------------------------------------------
def test_drift_monitor():
    rng = np.random.RandomState(0)
    ref = rng.beta(2, 5, 5000)
    assert psi(ref, rng.beta(2, 5, 5000)) < 0.05
    assert psi(ref, rng.beta(5, 2, 5000)) > 0.25
    m = DriftMonitor(ref, min_samples=100)
    assert m.status()["ready"] is False
    m.update(rng.beta(5, 2, 500), np.ones(500))
    assert m.status()["drift"] == "major"


# ---- HTTP service -------------------------------------------------------------------------------------------
def _call(url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_http_service(bundle, splits):
    srv = make_server(Detector(bundle), port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        code, h = _call(base + "/health"); assert code == 200 and h["status"] == "ok"
        rows = flows(splits.val, 3)
        code, r = _call(base + "/predict", {"flows": rows}); assert code == 200 and len(r["results"]) == 3
        code, r = _call(base + "/predict", rows[0]); assert code == 200 and len(r["results"]) == 1
        assert _call(base + "/predict", raw=b"{not json")[0] == 400
        assert _call(base + "/predict", {"flows": [{"dur": 1}]})[0] == 400
        assert _call(base + "/nope")[0] == 404
        assert _call(base + "/recalibrate", {"flows": rows})[0] == 403
        assert _call(base + "/drift")[0] == 200
    finally:
        srv.shutdown(); srv.server_close()
