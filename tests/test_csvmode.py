"""CSV mode tests: tolerant reading, format detection, scoring with bad input, scan rules, model integrity.
Small stand-in models are trained on generated data, so no real data or saved models are needed."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from nids.config import BINARY, CATEGORICAL, CONTINUOUS, FEATURES
from nids.features import build_preprocessor
from sklearn.ensemble import HistGradientBoostingClassifier

from layered_nids.common.paths import ROOT
from layered_nids.csvmode import models as M
from layered_nids.csvmode.flowrules import scan_findings
from layered_nids.csvmode.reader import CsvError, count_data_lines, detect_delimiter, read_chunks
from layered_nids.csvmode.run import process_csv
from layered_nids.csvmode.schemas import NETFLOW_FEATURES, detect_schema

ARGUS_HEADER = "StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,SrcPkts,Label"


# ------------------------------------------------------------------------------------------------ stand-in models
def unsw_frame(n, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.choice(["tcp", "udp", "icmp"], n) if c == "proto" else rng.choice(["-", "http", "dns"], n) if c == "service"
                       else rng.choice(["FIN", "CON", "INT"], n) for c in CATEGORICAL})
    for c in CONTINUOUS:
        df[c] = np.round(rng.lognormal(3, 1.5, n), 3)
    df["is_sm_ips_ports"] = rng.integers(0, 2, n)
    y = (np.log1p(df["sbytes"]) + rng.normal(0, 0.5, n) > 4.4).astype(int)
    return df[FEATURES], y.to_numpy()


def toy_unsw():
    from layered_nids.flowmodel.contract import NUMERIC
    from layered_nids.flowmodel.train import train
    df, y = unsw_frame(900)
    df = df.assign(y=y)
    bundle = train(df.iloc[:600].reset_index(drop=True), df.iloc[600:].reset_index(drop=True), fast=True)
    thr = {k: bundle.thresholds[k] for k in ("fpr_1pct", "fpr_3pct")}
    return {"kind": "unsw34", "bundle": bundle, "thresholds": thr, "ranges": M.RangeCheck.fit(df[NUMERIC].astype(float)), "features": list(NUMERIC)}


def toy_netflow():
    rng = np.random.default_rng(1)
    X = rng.lognormal(2, 1, (600, len(NETFLOW_FEATURES))).astype(np.float32)
    y = (X[:, 3] > np.median(X[:, 3])).astype(int)
    gb = HistGradientBoostingClassifier(max_iter=40, random_state=0).fit(X, y)
    s = gb.predict_proba(X)[:, 1]
    thr = {"fpr_1pct": float(np.quantile(s[y == 0], 0.99)), "fpr_3pct": float(np.quantile(s[y == 0], 0.97))}
    return {"kind": "netflow_v2", "gb": gb, "thresholds": thr, "ranges": M.RangeCheck.fit(pd.DataFrame(X, columns=NETFLOW_FEATURES).astype(float)),
            "features": list(NETFLOW_FEATURES)}


@pytest.fixture(scope="module")
def stand_ins():
    return {"unsw34": toy_unsw(), "netflow_v2": toy_netflow()}


def write(path, text, encoding="utf-8"):
    Path(path).write_bytes(text.encode(encoding))
    return path


# ------------------------------------------------------------------------------------------------ reading
def test_delimiter_detection_handles_comma_tab_semicolon_and_bom(tmp_path):
    assert detect_delimiter(write(tmp_path / "a.csv", "a,b,c\n1,2,3\n")) == ","
    assert detect_delimiter(write(tmp_path / "b.csv", "a\tb\tc\n1\t2\t3\n")) == "\t"
    assert detect_delimiter(write(tmp_path / "c.csv", "a;b;c\n1;2;3\n")) == ";"
    assert detect_delimiter(write(tmp_path / "d.csv", "﻿id,a,b\n1,2,3\n", encoding="utf-8")) == ","
    first = next(read_chunks(write(tmp_path / "e.csv", "﻿id,a,b\n1,2,3\n")))
    assert list(first.columns) == ["id", "a", "b"]                                     # the invisible byte-order mark is gone


def test_reader_strips_padding_skips_malformed_lines_and_rejects_empty_files(tmp_path):
    p = write(tmp_path / "x.csv", " Name , Value \n a , 1 \n\nb,2,EXTRA,EXTRA\n c , 3 \n")
    df = pd.concat(list(read_chunks(p)))
    assert list(df.columns) == ["name", "value"] and df["name"].tolist() == ["a", "c"] and df["value"].tolist() == ["1", "3"]
    assert count_data_lines(p) == 3                                                    # 3 non-blank data lines, only 2 readable: 1 skipped
    with pytest.raises(CsvError, match="empty"):
        next(read_chunks(write(tmp_path / "e.csv", "")))
    with pytest.raises(CsvError, match="no data rows"):
        next(read_chunks(write(tmp_path / "h.csv", "a,b,c\n")))
    with pytest.raises(CsvError, match="not found"):
        next(read_chunks(tmp_path / "missing.csv"))


# ------------------------------------------------------------------------------------------------ format detection
def test_schema_detection_and_clear_errors():
    assert detect_schema(FEATURES + ["id", "label"]) == "unsw34"
    assert detect_schema([c.upper() for c in NETFLOW_FEATURES] + ["IPV4_SRC_ADDR", "Label"]) == "netflow_v2"
    assert detect_schema(ARGUS_HEADER.split(",")) == "argus"
    with pytest.raises(CsvError, match="Closest format: unsw34") as e:
        detect_schema(FEATURES[:20])
    assert "missing 14 of 34" in str(e.value)
    with pytest.raises(CsvError, match="needs"):
        detect_schema(FEATURES[:20], forced="unsw34")
    with pytest.raises(CsvError, match="unknown schema"):
        detect_schema(FEATURES, forced="nope")


# ------------------------------------------------------------------------------------------------ ML scoring, bad input
def unsw_csv(path, n=8, sep=",", bom=False, break_rows=True):
    df, _ = unsw_frame(n, seed=5)
    df = df.astype(str).copy()
    if break_rows:
        df.loc[1, "sbytes"] = "abc"                       # text where a number belongs
        df.loc[2, "ackdat"] = ""                             # empty
        df.loc[3, "dbytes"] = "inf"                       # infinite
        df.loc[4, ["dbytes", "smean", "dload", "ackdat", "ct_src_dport_ltm", "sbytes", "dmean", "spkts"]] = ""                        # a row that is essentially empty
    text = df.to_csv(sep=sep, index=False)
    return write(path, ("﻿" if bom else "") + text)


def test_unsw_rows_with_bad_values_are_scored_flagged_and_never_dropped(tmp_path, stand_ins):
    p = unsw_csv(tmp_path / "in.csv", bom=True)
    s = process_csv(p, tmp_path / "out.csv", models=stand_ins)
    out = pd.read_csv(tmp_path / "out.csv")
    assert s["schema"] == "unsw34" and s["rows"] == 8 and len(out) == 8                  # every input row is in the output
    assert out["row_status"].tolist()[1:5] == ["warn", "warn", "warn", "unusable"]         # abc, empty, inf are warned; the empty row is unusable
    assert "unreadable" in out.loc[1, "row_notes"] and "missing" in out.loc[2, "row_notes"]
    assert out["attack_probability"].between(0, 1).all() and set(out["severity"]) <= {"none", "medium", "high"}
    assert s["row_status"]["unusable"] == 1 and any("unusable" in w for w in s["warnings"])
    assert json.loads(Path(str(tmp_path / "out.csv") + ".summary.json").read_text())["schema"] == "unsw34"


def test_tab_separated_and_comma_separated_give_the_same_answers_and_chunking_does_not_matter(tmp_path, stand_ins):
    a = process_csv(unsw_csv(tmp_path / "c.csv", n=30), tmp_path / "oc.csv", models=stand_ins)
    b = process_csv(unsw_csv(tmp_path / "t.csv", n=30, sep="\t"), tmp_path / "ot.csv", models=stand_ins)
    c = process_csv(unsw_csv(tmp_path / "c2.csv", n=30), tmp_path / "oc2.csv", models=stand_ins, chunksize=7)
    pc, pt, pk = (pd.read_csv(tmp_path / f)["attack_probability"].to_numpy() for f in ("oc.csv", "ot.csv", "oc2.csv"))
    np.testing.assert_allclose(pc, pt)
    np.testing.assert_allclose(pc, pk)
    assert a["flagged"] == b["flagged"] == c["flagged"]


def test_out_of_range_inputs_raise_a_warning_and_netflow_hosts_are_summarised(tmp_path, stand_ins):
    rng = np.random.default_rng(2)
    cols = [c.upper() for c in NETFLOW_FEATURES]
    normal = pd.DataFrame(rng.lognormal(2, 1, (40, len(cols))), columns=cols)
    extreme = pd.DataFrame(np.full((40, len(cols)), 1e12), columns=cols)
    for name, df in (("normal", normal), ("extreme", extreme)):
        df.insert(0, "IPV4_SRC_ADDR", [f"10.0.0.{i % 4}" for i in range(40)])
        df.to_csv(tmp_path / f"{name}.csv", index=False)
    ok = process_csv(tmp_path / "normal.csv", tmp_path / "n_out.csv", models=stand_ins)
    bad = process_csv(tmp_path / "extreme.csv", tmp_path / "e_out.csv", models=stand_ins)
    assert ok["schema"] == bad["schema"] == "netflow_v2"
    assert not any("unlike the data" in w for w in ok["warnings"])
    assert bad["rows_far_outside_training_range_share"] > 0.9 and any("unlike the data" in w for w in bad["warnings"])
    if ok["flagged"]:
        hosts = pd.read_csv(ok["hosts_table"])
        assert {"host", "flows", "flagged_flows", "max_probability"} <= set(hosts.columns) and hosts["flagged_flows"].sum() == ok["flagged"]


def test_errors_are_clear_not_crashes(tmp_path, stand_ins):
    with pytest.raises(CsvError, match="unrecognised"):
        process_csv(write(tmp_path / "w.csv", "a,b\n1,2\n"), tmp_path / "o.csv", models=stand_ins)
    with pytest.raises(CsvError, match="policy"):
        process_csv(unsw_csv(tmp_path / "p.csv"), tmp_path / "o.csv", policy="fpr_9pct", models=stand_ins)
    with pytest.raises(CsvError, match="not trained"):
        M.load_model("unsw34", directory=tmp_path / "no_models")


def test_model_files_are_checked_before_loading(tmp_path):
    m = toy_netflow()
    M.save_model("netflow_v2", m, {"x": 1}, directory=tmp_path)
    assert M.load_model("netflow_v2", directory=tmp_path)["kind"] == "netflow_v2"
    f = tmp_path / "netflow_v2.joblib"
    f.write_bytes(f.read_bytes() + b"tampered")
    with pytest.raises(CsvError, match="integrity"):
        M.load_model("netflow_v2", directory=tmp_path)


# ------------------------------------------------------------------------------------------------ Argus scan rules
def argus_rows():
    rows = []
    for i in range(20):                                                       # vertical scan: one source, one target, 20 ports, 10 s
        rows.append(f"2019/04/04 16:24:{10 + i // 2:02d}.{i:06d},0.1,tcp,10.0.0.5,5{i:04d},   ->,10.0.0.9,{1000 + i},S0,0,0,1,60,60,1,")
    for i in range(25):                                                       # horizontal scan: one source, one port, 25 hosts
        rows.append(f"2019/04/04 16:30:{i:02d}.000000,0.1,tcp,10.0.0.6,6{i:04d},   ->,10.0.1.{i + 1},22,S0,0,0,1,60,60,1,")
    for i in range(5):                                                        # ordinary: few ports/hosts
        rows.append(f"2019/04/04 16:31:{i:02d}.000000,0.5,tcp,10.0.0.7,7{i:04d},   ->,10.0.2.1,443,CON,0,0,6,600,300,3,")
    rows += ["not a time,0,tcp,0,0,,0,0,STA,,,0,0,0,0,", "2019/04/04 16:31:30.000000,0,man,10.0.0.8,1,  ->,notanip,80,S0,0,0,1,60,60,1,"]
    return rows


def test_scan_rules_find_scans_ignore_normal_flows_and_count_unreadable_rows(tmp_path):
    p = write(tmp_path / "a.binetflow", ARGUS_HEADER + "\n" + "\n".join(argus_rows()) + "\n")
    s = process_csv(p, tmp_path / "out.csv")
    f = pd.read_csv(s["findings_table"])
    assert s["schema"] == "argus" and s["rows"] == 52
    assert set(f["host"]) == {"10.0.0.5", "10.0.0.6"}                         # the ordinary host is not flagged
    kinds = dict(zip(f["host"], f["kind"]))
    assert kinds["10.0.0.5"] == "vertical_scan" and kinds["10.0.0.6"] == "horizontal_scan"
    assert s["rows_ignored"] == 2 and any("ignored by the rules" in w for w in s["warnings"])   # the two unreadable rows are counted
    out = pd.read_csv(tmp_path / "out.csv")
    assert out["is_attack"].sum() == 45 and len(out) == 52                    # 20 + 25 flows in findings, every row kept


@pytest.mark.skipif(not (ROOT / "reference/reference IDS/dataset/test11-portscan.binetflow").exists(), reason="reference IDS sample not present")
def test_real_reference_ids_portscan_sample_is_recognised_as_a_scan(tmp_path):
    p = ROOT / "reference/reference IDS/dataset/test11-portscan.binetflow"
    s = process_csv(p, tmp_path / "ps.csv")
    f = pd.read_csv(s["findings_table"])
    assert s["schema"] == "argus" and s["findings"] >= 2



def test_scan_rules_unit_thresholds():
    df = pd.DataFrame({"starttime": [str(i) for i in range(10)], "srcaddr": ["1.1.1.1"] * 10, "dstaddr": ["2.2.2.2"] * 10,
                       "dport": [str(1000 + i) for i in range(10)]})
    assert scan_findings(df, min_ports=15)[0].empty                           # 10 ports is below the default 15
    assert len(scan_findings(df, min_ports=10)[0]) == 1


def test_a_slow_scan_that_straddles_a_minute_boundary_is_still_found_and_browsing_ports_are_not_scans():
    # 20 ports, one per second, starting 13 s before a clock-minute boundary: fixed 60 s windows would split it 13 / 7
    base = pd.Timestamp("2019-04-04 16:24:47", tz="UTC").timestamp()
    df = pd.DataFrame({"starttime": [str(base + i) for i in range(20)], "srcaddr": ["10.8.0.69"] * 20, "dstaddr": ["14.1.11.113"] * 20,
                       "dport": [str(999 - i) for i in range(20)], "proto": ["tcp"] * 20})
    f, _ = scan_findings(df)
    assert len(f) == 1 and f.loc[0, "kind"] == "vertical_scan" and f.loc[0, "distinct"] == 20
    web = pd.DataFrame({"starttime": [str(base + i) for i in range(40)], "srcaddr": ["10.8.0.5"] * 40,
                        "dstaddr": [f"93.184.216.{i}" for i in range(40)], "dport": ["443"] * 40, "proto": ["tcp"] * 40})
    assert scan_findings(web)[0].empty                                        # 40 hosts on port 443 in 40 s is browsing, not a scan
    web["dport"] = "445"
    assert scan_findings(web)[0].loc[0, "kind"] == "horizontal_scan"          # the same on port 445 (SMB) is a scan


def test_icmp_sweep_uses_hosts_not_ports():
    base = pd.Timestamp("2019-04-04 16:24:47", tz="UTC").timestamp()
    df = pd.DataFrame({"starttime": [str(base + i) for i in range(25)], "srcaddr": ["10.8.0.69"] * 25, "dstaddr": [f"10.8.1.{i + 1}" for i in range(25)],
                       "dport": ["0x7208"] * 25, "proto": ["icmp"] * 25})
    f, facts = scan_findings(df)
    assert f["kind"].tolist() == ["icmp_sweep"] and facts["rows_without_numeric_port"] == 25


def test_saved_models_load_from_a_different_process_than_the_one_that_trained_them(tmp_path):
    """Regression: models trained by `python -m ...models train` once recorded their helper class as __main__.RangeCheck and could not be loaded."""
    import subprocess, sys
    code = ("import layered_nids.csvmode.models as mod, pandas as pd, numpy as np;"
            "from layered_nids.csvmode.schemas import NETFLOW_FEATURES;"
            "rc = mod.RangeCheck.fit(pd.DataFrame(np.ones((5, 2)), columns=['a','b']));"
            f"mod.save_model('t', {{'ranges': rc, 'features': ['a','b']}}, {{}}, directory=__import__('pathlib').Path(r'{tmp_path}'))")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-400:]
    loaded = M.load_model("t", directory=tmp_path)
    assert type(loaded["ranges"]).__module__ == "layered_nids.csvmode.models"
