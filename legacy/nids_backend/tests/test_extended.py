"""Edge cases and the later backend modules: config, explanations, CLI, server, alerts, replay, experiments, reports."""
import concurrent.futures as cf
import dataclasses
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import nids.server as server_mod
from nids import Config, Detector, SchemaError, load_bundle, save_bundle, train
from nids.alerts import AlertStore
from nids.cli import main
from nids.config import ATTACK_CAT, ATTACK_FAMILIES, FEATURES, LABEL, SCHEMA_VERSION
from nids.drift import DriftMonitor
from nids.evaluate import calibration_metrics, evaluate
from nids.experiments import leave_one_attack_out, recalibration_curve, stability
from nids.explain import MAX_PATHS, shapley_contributions
from nids.replay import iter_replay
from nids.report import build_report, write_report
from nids.server import make_server

ATTACH = Path("/mnt/user-data/attachments")


def flows(df, n=5):
    return df[FEATURES].head(n).to_dict("records")


# ---- config ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    dict(val_size=0.01), dict(val_size=0.9), dict(ann_seeds=0), dict(blend_weight=1.5), dict(blend_weight=-0.1),
    dict(fpr_targets=(0.01, 0.01)), dict(fpr_targets=(0.015,)), dict(fpr_targets=(0.0,)), dict(fpr_targets=(1.0,)),
    dict(default_policy="nonsense"), dict(novelty_fpr=0.0), dict(novelty_fpr=1.0), dict(background_size=5),
])
def test_config_rejects_invalid(kw):
    with pytest.raises(ValueError):
        Config(**kw)


def test_config_defaults_and_fast():
    c = Config()
    assert c.default_policy in {f"fpr_{int(round(t * 100))}pct" for t in c.fpr_targets} | {"f1"}
    assert Config.fast().max_train_rows == 6000
    assert Config.fast(rows=1000).max_train_rows == 1000
    assert isinstance(c.to_dict(), dict) and len(FEATURES) == 34 and len(ATTACK_FAMILIES) == 9


# ---- explanations ----------------------------------------------------------------------------------------
def test_explain_completeness_and_shape(bundle, splits):
    d = Detector(bundle)
    flow = flows(splits.val, 1)[0]
    r = d.explain(flow, n_permutations=8, n_background=4, top=34)
    assert len(r["contributions"]) == 34 and r["n_paths"] == 32
    assert r["sum_contributions"] == pytest.approx(r["probability"] - r["baseline"], abs=1e-9)
    assert r["probability"] == pytest.approx(d.predict([flow])[0]["attack_probability"], abs=1e-9)
    mags = [abs(c["contribution"]) for c in r["contributions"]]
    assert mags == sorted(mags, reverse=True)
    assert {c["feature"] for c in r["contributions"]} == set(FEATURES)
    assert d.explain(flow, n_permutations=2, n_background=2, top=3)["contributions"].__len__() == 3
    assert isinstance(r["is_attack"], bool) and r["threshold"] == d.threshold
    json.dumps(r, allow_nan=False)  # JSON-safe (numpy scalars converted)


def test_explain_deterministic_and_seeded(bundle, splits):
    d, flow = Detector(bundle), flows(splits.val, 1)[0]
    a, b = d.explain(flow, seed=3), d.explain(flow, seed=3)
    assert a == b
    c = d.explain(flow, seed=4)
    assert a["contributions"] != c["contributions"] or a["baseline"] != c["baseline"]


def test_explain_finds_the_driver_of_a_simple_model():
    """With a model that only reads 'sbytes', that feature must receive all of the credit."""
    bg = pd.DataFrame([{f: (0.0 if f not in ("proto", "service", "state") else "tcp") for f in FEATURES}] * 20)
    flow = dict(bg.iloc[0]); flow["sbytes"] = 100.0
    fn = lambda df: (df["sbytes"].to_numpy(float) > 50).astype(float)
    r = shapley_contributions(fn, flow, bg, n_permutations=10, n_background=3)
    top = r["contributions"][0]
    assert top["feature"] == "sbytes" and top["contribution"] == pytest.approx(1.0)
    assert all(abs(c["contribution"]) < 1e-12 for c in r["contributions"][1:])


def test_explain_limits_and_input_errors(bundle, splits):
    d, flow = Detector(bundle), flows(splits.val, 1)[0]
    for kw in (dict(n_permutations=0), dict(n_background=0), dict(n_permutations=201), dict(n_background=51),
               dict(n_permutations=100, n_background=20)):
        with pytest.raises(ValueError):
            d.explain(flow, **kw)
    assert 100 * 20 > MAX_PATHS
    with pytest.raises(SchemaError):
        d.explain({k: v for k, v in flow.items() if k != "dur"})
    with pytest.raises(SchemaError):
        d.explain(splits.val[FEATURES].head(2))  # more than one flow
    with pytest.raises(ValueError):
        shapley_contributions(lambda df: np.zeros(len(df)), flow, bundle.background.head(0))


# ---- determinism / thread-safety ---------------------------------------------------------------------------
def test_training_is_deterministic(splits):
    cfg = Config.fast()
    a, b = train(cfg, splits), train(cfg, splits)
    x = splits.val.head(100)
    sa, sb = Detector(a).score_frame(x), Detector(b).score_frame(x)
    assert np.allclose(sa.to_numpy(), sb.to_numpy())
    assert a.thresholds == pytest.approx(b.thresholds, abs=1e-9)


def test_concurrent_predictions_match_serial(bundle, splits):
    d = Detector(bundle)
    rows = flows(splits.val, 24)
    serial = [d.predict([r])[0]["attack_probability"] for r in rows]
    with cf.ThreadPoolExecutor(8) as ex:
        par = list(ex.map(lambda r: d.predict([r])[0]["attack_probability"], rows))
    assert np.allclose(serial, par)


def test_drift_monitor_thread_safe_and_reset():
    m = DriftMonitor(np.random.RandomState(0).rand(1000), window=500, min_samples=10)
    ts = [threading.Thread(target=lambda: [m.update([0.5] * 5, [1] * 5) for _ in range(100)]) for _ in range(6)]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert m.status()["n"] == 500
    m.reset()
    assert m.status()["n"] == 0 and m.status()["ready"] is False


# ---- bundle versioning / overrides ---------------------------------------------------------------------------
def test_schema_version_mismatch_rejected(bundle, tmp_path):
    old = dataclasses.replace(bundle, meta={**bundle.meta, "schema_version": SCHEMA_VERSION - 1})
    p = save_bundle(old, tmp_path / "old.joblib")
    with pytest.raises(RuntimeError, match="schema"):
        load_bundle(p)


def test_override_roundtrip_and_validation(bundle, splits, tmp_path):
    d = Detector(bundle)
    with pytest.raises(RuntimeError):
        d.save_override(tmp_path / "o.json")
    d.recalibrate(splits.val.head(600), fpr_cap=0.08)
    if not d.info()["recalibrated"]:
        d._override = {"threshold": 0.42}
    d.save_override(tmp_path / "o.json")
    d2 = Detector(bundle); d2.load_override(tmp_path / "o.json")
    assert d2.threshold == d.threshold
    (tmp_path / "bad.json").write_text(json.dumps({"threshold": 7}))
    with pytest.raises(ValueError):
        d2.load_override(tmp_path / "bad.json")
    d2.set_policy("fpr_1pct")
    assert not d2.info()["recalibrated"]  # switching policy discards the local override


# ---- evaluation ---------------------------------------------------------------------------------------------
def test_calibration_metrics():
    y = np.array([0, 0, 1, 1])
    perfect = calibration_metrics(y, np.array([0.0, 0.0, 1.0, 1.0]))
    assert perfect["brier"] == 0 and perfect["ece"] == 0
    bad = calibration_metrics(y, np.array([1.0, 1.0, 0.0, 0.0]))
    assert bad["brier"] == 1 and bad["ece"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        calibration_metrics([], [])


def test_evaluate_contains_calibration(bundle, splits):
    r = evaluate(Detector(bundle), splits.test.head(500))
    assert 0 <= r["calibration"]["ece"] <= 1 and 0 <= r["calibration"]["brier"] <= 1
    assert r["n"] == 500 and set(r["per_class"]) <= set(ATTACK_FAMILIES) | {"Normal"}


# ---- alerts ----------------------------------------------------------------------------------------------------
def _res(att, p, sev="medium"):
    return {"is_attack": att, "attack_probability": p, "severity": sev if att else "none", "flag_reason": "classifier" if att else "",
            "policy": "fpr_3pct", "ann_score": p, "rf_score": p, "novelty_score": 0.0, "top_deviations": []}


def test_alert_store_basic_and_cap(tmp_path):
    st = AlertStore(max_alerts=5)
    n = st.add_batch([_res(True, 0.9, "high"), _res(False, 0.1), _res(True, 0.8)], ["a", "b", "c"])
    assert n == 2 and st.stats() == {"flows_seen": 3, "alerts_stored": 2, "high_severity": 1, "alert_rate": pytest.approx(2 / 3)}
    r = st.recent()
    assert [x["flow_id"] for x in r] == ["c", "a"]
    assert [x["flow_id"] for x in st.recent(min_severity="high")] == ["a"]
    with pytest.raises(ValueError):
        st.recent(min_severity="bogus")
    st.add_batch([_res(True, 0.7)] * 10)
    assert st.stats()["alerts_stored"] == 5          # oldest alerts are dropped
    assert len(st.recent(limit=1000)) == 5 and len(st.recent(limit=0)) == 1
    st.clear(); assert st.stats()["flows_seen"] == 0
    p = tmp_path / "alerts.db"
    s2 = AlertStore(p); s2.add_batch([_res(True, 0.9, "high")]); s2.close()
    assert AlertStore(p).stats()["alerts_stored"] == 1


def test_alert_store_threads():
    st = AlertStore()
    with cf.ThreadPoolExecutor(8) as ex:
        list(ex.map(lambda i: st.add_batch([_res(True, 0.9)] * 10), range(20)))
    assert st.stats()["alerts_stored"] == 200


# ---- replay ----------------------------------------------------------------------------------------------------
def test_replay(splits):
    df = splits.val.head(95)
    batches = list(iter_replay(df, batch_size=20, shuffle=False))
    assert [len(b) for b in batches] == [20, 20, 20, 20, 15]
    assert pd.concat(batches).index.equals(df.index)
    a = [b.index.tolist() for b in iter_replay(df, 20, seed=1)]
    assert a == [b.index.tolist() for b in iter_replay(df, 20, seed=1)] and a != [b.index.tolist() for b in iter_replay(df, 20, seed=2)]
    assert sorted(sum(a, [])) == sorted(df.index.tolist())
    assert len(list(iter_replay(df, 10, max_batches=2))) == 2
    assert LABEL in batches[0].columns
    for bad in (dict(batch_size=0), dict(flows_per_second=0)):
        with pytest.raises(ValueError):
            list(iter_replay(df, **{"batch_size": 5, **bad}))
    with pytest.raises(ValueError):
        list(iter_replay(df.drop(columns=["dur"])))
    assert list(iter_replay(df.head(0))) == []


def test_replay_through_detector_and_store(bundle, splits):
    d, st = Detector(bundle), AlertStore()
    for b in iter_replay(splits.val.head(100), 25, seed=0):
        st.add_batch(d.predict(b[FEATURES].to_dict("records")))
    assert st.stats()["flows_seen"] == 100


# ---- experiments ------------------------------------------------------------------------------------------------
def test_leave_one_attack_out_small(bundle, splits):
    cfg = Config.fast()
    res = leave_one_attack_out(cfg, splits, ["Generic", "Worms"], seen_detector=Detector(bundle))
    assert set(res["folds"]) == {"Generic", "Worms"}
    for f in res["folds"].values():
        assert 0 <= f["detect_classifier"] <= f["detect_combined"] <= 1 and 0 <= f["detect_when_seen"] <= 1
        assert f["false_alarm_classifier"] <= f["false_alarm_combined"] + 1e-12
    assert res["summary_mean_over_folds"]["detect_combined"] >= res["summary_mean_over_folds"]["detect_classifier"]
    with pytest.raises(ValueError, match="not in the test set"):
        leave_one_attack_out(cfg, splits, ["NotAnAttack"])
    with pytest.raises(ValueError):
        leave_one_attack_out(cfg, splits, ["Generic"], policy="f1")  # f1 is not a false-alarm-target policy
    json.dumps(res, allow_nan=False)


def test_held_out_attack_is_absent_from_training(splits):
    """The fold must really exclude the attack from every fitted component's data."""
    A = "Generic"
    assert (splits.train[ATTACK_CAT] == A).any()
    assert not (splits.train[splits.train[ATTACK_CAT] != A][ATTACK_CAT] == A).any()


def test_stability_small():
    res = stability(Config.fast(rows=2500), n_repeats=2)
    assert len(res["runs"]) == 2 and res["runs"][0]["seed"] != res["runs"][1]["seed"]
    assert 0 <= res["summary"]["test"]["accuracy"]["mean"] <= 1
    with pytest.raises(ValueError):
        stability(Config.fast(), n_repeats=1)


def test_recalibration_curve_small(bundle, splits):
    res = recalibration_curve(Detector(bundle), splits.test, sizes=(100, 400), draws=5)
    assert [r["n_labelled"] for r in res["curve"]] == [100, 400]
    assert all(0 <= r["accuracy"]["mean"] <= 1 for r in res["curve"])
    assert recalibration_curve(Detector(bundle), splits.test.head(150), sizes=(100,), draws=3)["curve"] == []
    json.dumps(res, allow_nan=False)


# ---- report ----------------------------------------------------------------------------------------------------
def test_report_generation(bundle, splits, tmp_path):
    p = save_bundle(bundle, tmp_path / "b.joblib")
    rd = tmp_path / "reports"; rd.mkdir()
    (rd / "evaluation_fpr_3pct.json").write_text(json.dumps(evaluate(Detector(bundle), splits.test.head(300)) | {
        "latency": {"single_flow_ms": 5.0, "batched_ms_per_flow": 0.1, "n": 10}}, default=float))
    (rd / "unseen_attacks.json").write_text(json.dumps(leave_one_attack_out(Config.fast(), splits, ["Worms"]), default=float))
    text = build_report(rd, p.with_suffix(".meta.json"))
    assert "Official test set" in text and "Worms" in text and "Limits" in text and "Latency" in text
    out = write_report(rd, p.with_suffix(".meta.json"))
    assert out.read_text() == text
    with pytest.raises(FileNotFoundError):
        build_report(rd, tmp_path / "missing.json")
    (rd / "stability.json").write_text("{broken")           # corrupt optional files are skipped, not fatal
    assert "Official test set" in build_report(rd, p.with_suffix(".meta.json"))


# ---- HTTP service ----------------------------------------------------------------------------------------------
def _call(url, body=None, raw=None, method=None, headers=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    h = {"Content-Type": "application/json"} if data else {}
    h.update(headers or {})
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture()
def served(bundle):
    made = []

    def start(admin=False):
        srv = make_server(Detector(bundle), port=0, allow_admin=admin)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        made.append(srv)
        return srv, f"http://127.0.0.1:{srv.server_address[1]}"
    yield start
    for s in made:
        s.shutdown(); s.server_close()


def test_server_predict_ids_alerts_stats(served, splits):
    srv, base = served()
    rows = flows(splits.test, 40)
    code, r = _call(base + "/predict", {"flows": rows, "ids": list(range(40))})
    assert code == 200 and [x["id"] for x in r["results"]] == list(range(40))
    flagged = sum(x["is_attack"] for x in r["results"])
    code, s = _call(base + "/stats"); assert code == 200 and s["flows_seen"] == 40 and s["alerts_stored"] == flagged
    code, a = _call(base + "/alerts?limit=5"); assert code == 200 and len(a["alerts"]) == min(5, flagged)
    assert _call(base + "/alerts?limit=abc")[0] == 400
    assert _call(base + "/alerts?min_severity=bogus")[0] == 400
    code, r = _call(base + "/predict", {"flows": [{**rows[0], "id": "abc"}]})
    assert r["results"][0]["id"] == "abc"
    assert _call(base + "/predict", {"flows": rows[:2], "ids": [1]})[0] == 400
    code, r = _call(base + "/predict", rows[0]); assert code == 200 and len(r["results"]) == 1


def test_server_errors(served, splits, monkeypatch):
    srv, base = served()
    rows = flows(splits.val, 3)
    assert _call(base + "/predict", raw=b"{not json")[0] == 400
    assert _call(base + "/predict", raw=b"\xff\xfe")[0] == 400
    assert _call(base + "/predict", {"flows": []})[0] == 400
    assert _call(base + "/predict", {"flows": [{"dur": 1}]})[0] == 400
    assert _call(base + "/predict", {"flows": [{**rows[0], "dur": "x"}]})[0] == 400
    assert _call(base + "/predict", {"flows": [{**rows[0], "dur": None}]})[0] == 400
    assert _call(base + "/predict", {"flows": "nope"})[0] == 400
    assert _call(base + "/predict", method="POST", raw=b"")[0] == 400
    assert _call(base + "/nope")[0] == 404 and _call(base + "/nope", {"a": 1})[0] == 404
    monkeypatch.setattr(server_mod, "MAX_FLOWS", 2)
    code, r = _call(base + "/predict", {"flows": rows}); assert code == 400 and "too many" in r["error"]
    monkeypatch.setattr(server_mod, "MAX_BODY", 50)
    code, r = _call(base + "/predict", {"flows": rows[:1]}); assert code == 400 and "too large" in r["error"]


def test_server_explain(served, splits):
    srv, base = served()
    f = flows(splits.val, 1)[0]
    code, r = _call(base + "/explain", {"flow": f, "n_permutations": 4, "n_background": 3, "top": 5})
    assert code == 200 and len(r["contributions"]) == 5 and r["sum_contributions"] == pytest.approx(r["probability"] - r["baseline"])
    assert _call(base + "/explain", {"flow": [f]})[0] == 200
    for bad in ({"flow": {"dur": 1}}, {"flow": "x"}, {"flow": f, "n_permutations": 0}, {"flow": f, "n_permutations": "a"},
                {"flow": f, "n_permutations": 100, "n_background": 50}, [f]):
        assert _call(base + "/explain", bad)[0] == 400, bad


def test_server_admin_gating_and_actions(served, splits):
    srv, base = served(admin=False)
    for path, body in (("/config", {"policy": "f1"}), ("/recalibrate", {"flows": []}), ("/reset", {"what": "alerts"})):
        assert _call(base + path, body)[0] == 403
    assert _call(base + "/health")[1]["admin_enabled"] is False
    srv2, base2 = served(admin=True)
    assert _call(base2 + "/health")[1]["admin_enabled"] is True
    code, i = _call(base2 + "/config", {"policy": "fpr_1pct", "use_novelty": True})
    assert code == 200 and i["policy"] == "fpr_1pct" and i["use_novelty"] is True
    assert _call(base2 + "/config", {"policy": "bogus"})[0] == 400
    assert _call(base2 + "/config", {"use_novelty": "yes"})[0] == 400
    assert _call(base2 + "/config", [1])[0] == 400
    lab = splits.val.head(600)[FEATURES + [LABEL]].to_dict("records")
    code, rep = _call(base2 + "/recalibrate", {"flows": lab, "fpr_cap": 0.08}); assert code == 200 and "new_threshold" in rep
    assert _call(base2 + "/recalibrate", {"flows": lab[:20]})[0] == 400
    assert _call(base2 + "/recalibrate", {"flows": [{k: v for k, v in x.items() if k != LABEL} for x in lab]})[0] == 400
    assert _call(base2 + "/reset", {"what": "recalibration"})[1] == {"reset": "recalibration"}
    assert _call(base2 + "/health")[1]["recalibrated"] is False
    _call(base2 + "/predict", {"flows": flows(splits.val, 30)})
    assert _call(base2 + "/reset", {"what": "alerts"})[0] == 200 and _call(base2 + "/stats")[1]["flows_seen"] == 0
    assert _call(base2 + "/reset", {"what": "drift"})[0] == 200 and _call(base2 + "/drift")[1]["n"] == 0
    assert _call(base2 + "/reset", {"what": "everything"})[0] == 400


def test_server_concurrent_clients(served, splits):
    srv, base = served()
    rows = flows(splits.val, 12)
    with cf.ThreadPoolExecutor(8) as ex:
        out = list(ex.map(lambda r: _call(base + "/predict", {"flows": [r]}), rows * 3))
    assert all(c == 200 for c, _ in out)
    assert _call(base + "/stats")[1]["flows_seen"] == 36


def test_server_handles_garbage_request_line(served):
    import socket
    srv, base = served()
    s = socket.create_connection(("127.0.0.1", srv.server_address[1]), timeout=5)
    s.sendall(b"GARBAGE\r\n\r\n"); s.recv(1024); s.close()
    assert _call(base + "/health")[0] == 200     # still alive afterwards


# ---- CLI ---------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def cli_env(tmp_path_factory):
    d = tmp_path_factory.mktemp("cli")
    b = d / "m" / "bundle.joblib"
    assert main(["train", "--fast", "--bundle", str(b)]) == 0
    return {"dir": d, "bundle": str(b), "reports": str(d / "reports")}


def test_cli_end_to_end(cli_env, splits, capsys):
    d, b, rep = cli_env["dir"], cli_env["bundle"], cli_env["reports"]
    assert Path(b).exists() and Path(b).with_suffix(".meta.json").exists()
    assert main(["evaluate", "--fast", "--bundle", b, "--report-dir", rep, "--policy", "fpr_1pct"]) == 0
    assert main(["evaluate", "--fast", "--bundle", b, "--report-dir", rep, "--novelty"]) == 0
    assert (Path(rep) / "evaluation_fpr_1pct.json").exists() and any("novelty" in p.name for p in Path(rep).iterdir())
    csv = d / "flows.csv"; splits.val.head(700).to_csv(csv, index=False)
    out = d / "pred.csv"
    assert main(["predict", "--bundle", b, "--input", str(csv), "--output", str(out)]) == 0
    pred = pd.read_csv(out)
    assert len(pred) == 700 and {"is_attack", "attack_probability", "severity"} <= set(pred.columns)
    capsys.readouterr()
    assert main(["explain", "--bundle", b, "--input", str(csv), "--row", "2", "--permutations", "4", "--top", "3"]) == 0
    e = json.loads(capsys.readouterr().out)
    assert len(e["contributions"]) == 3
    assert main(["explain", "--bundle", b, "--input", str(csv), "--row", "9999"]) == 1
    ov = d / "override.json"
    assert main(["recalibrate", "--bundle", b, "--input", str(csv), "--override", str(ov)]) == 0
    assert main(["recal-curve", "--fast", "--bundle", b, "--report-dir", rep, "--draws", "3"]) == 0
    assert main(["unseen", "--fast", "--bundle", b, "--report-dir", rep, "--attacks", "Worms,Shellcode"]) == 0
    assert main(["unseen", "--fast", "--bundle", b, "--report-dir", rep, "--attacks", "Nope"]) == 1
    assert main(["stability", "--fast", "--bundle", b, "--report-dir", rep, "--repeats", "1"]) == 1
    assert main(["report", "--bundle", b, "--report-dir", rep]) == 0
    text = (Path(rep) / "REPORT.md").read_text()
    assert "Worms" in text and "Local recalibration" in text


def test_cli_errors_and_convert(cli_env, tmp_path, capsys):
    b = cli_env["bundle"]
    assert main(["predict", "--bundle", b, "--input", str(tmp_path / "nope.csv"), "--output", str(tmp_path / "o.csv")]) == 1
    assert main(["predict", "--bundle", str(tmp_path / "nobundle.joblib"), "--input", "x.csv", "--output", "y.csv"]) == 1
    bad = tmp_path / "bad.csv"; pd.DataFrame({"dur": [1.0]}).to_csv(bad, index=False)
    assert main(["predict", "--bundle", b, "--input", str(bad), "--output", str(tmp_path / "o.csv")]) == 1
    assert main(["predict", "--bundle", b, "--policy", "bogus", "--input", str(bad), "--output", str(tmp_path / "o.csv")]) == 1
    txt = tmp_path / "x.txt"; txt.write_text("a")
    assert main(["convert", str(txt), str(tmp_path / "y.csv")]) == 1
    with pytest.raises(SystemExit):
        main([])
    assert "error:" in capsys.readouterr().err


# ---- built-in parquet reader vs the CSV we derived from it ---------------------------------------------------
@pytest.mark.skipif(not ATTACH.exists(), reason="original parquet attachments not present")
def test_builtin_parquet_reader_matches_csv(tmp_path):
    from nids.data import read_table
    files = list(ATTACH.glob("*/0e1017b6-test.parquet"))
    csv = Path(__file__).resolve().parent.parent / "data" / "official_training_175341.csv"
    if not files or not csv.exists():
        pytest.skip("files missing")
    a = read_table(files[0]); b = read_table(csv)
    assert a.shape == b.shape and list(a.columns) == list(b.columns)
    assert np.allclose(a["dur"].to_numpy(float), b["dur"].to_numpy(float))
    assert (a[LABEL].to_numpy() == b[LABEL].to_numpy()).all()
    assert (a["proto"].astype(str).str.lower().to_numpy() == b["proto"].astype(str).str.lower().to_numpy()).all()
    with pytest.raises(FileNotFoundError):
        read_table(tmp_path / "missing.parquet")
