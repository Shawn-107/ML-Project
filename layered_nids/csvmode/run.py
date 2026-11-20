"""Process a flow CSV end to end:  python -m layered_nids.csvmode FILE.csv [--out OUT.csv]

  1. read tolerantly (delimiter, byte-order mark, padding, malformed lines, empty files) in chunks;
  2. recognise the format from the columns (UNSW 34 features / NetFlow v2 / Argus), or stop with a clear message;
  3. ML formats: score every row with the matching model; unreadable numbers become NaN and are counted; inputs far outside the
     training range raise a warning; every row gets a status (ok / warn / unusable) and notes;
     Argus: scan rules over host and time (no ML: nothing labelled to train on);
  4. write OUT.csv (all rows + results), OUT.csv.summary.json (what happened, warnings, caveats) and, where the file has hosts, a
     per-host table of the most flagged hosts.

Nothing is dropped silently: every input row appears in the output, and every skipped line is counted in the summary.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from layered_nids.common.paths import REPORTS
from layered_nids.csvmode.flowrules import flag_rows, scan_findings
from layered_nids.csvmode.models import POLICIES, SCORERS, load_model
from layered_nids.csvmode.reader import CsvError, count_data_lines, read_chunks
from layered_nids.csvmode.schemas import SCHEMAS, detect_schema

HOST_COLUMN = {"netflow_v2": "ipv4_src_addr", "unsw34": None}
OOR_ROWS_WARN_SHARE = 0.10          # warn when more than 10% of rows have >= 3 inputs far outside the training range
CAUTION = {"netflow_v2": "trained on NF-ToN-IoT only, a dataset that is easy even for unseen hosts: treat scores on other networks as unproven",
           "unsw34": "trained on 2015 lab traffic; the official test set is distribution-shifted from training, so expect about 7% false alarms at the 3% policy"}


def _row_annotations(info: dict, n_features: int):
    bad, miss, oor = info["bad_cells"], info["missing"], info["out_of_range"]
    status = np.where(miss >= n_features / 2, "unusable", np.where((bad > 0) | (miss > 0) | (oor >= 3), "warn", "ok"))
    notes = [("; ".join(x for x in (f"{b} unreadable value(s)" if b else "", f"{m} missing" if m else "", f"{o} far outside training range" if o >= 3 else "") if x))
             for b, m, o in zip(bad, miss, oor)]
    return status, notes


def process_csv(path, out=None, schema: str | None = None, policy: str = "fpr_3pct", chunksize: int = 200_000, models: dict | None = None,
                scan_window: float = 60.0, min_ports: int = 15, min_hosts: int = 15, max_rule_rows: int = 3_000_000) -> dict:
    t0 = time.time()
    path = Path(path)
    if policy not in POLICIES:
        raise CsvError(f"unknown policy {policy!r}; choose one of {sorted(POLICIES)}")
    chunks = read_chunks(path, chunksize)
    first = next(chunks)
    kind = detect_schema(first.columns, schema)
    out = Path(out) if out else REPORTS / "csv" / f"{path.stem}_scored.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    summary = {"input": str(path), "schema": kind, "policy": policy if SCHEMAS[kind]["kind"] == "ml" else None, "output": str(out), "warnings": []}
    n_lines = count_data_lines(path)

    def all_chunks():
        yield first
        yield from chunks

    rows = 0
    if SCHEMAS[kind]["kind"] == "ml":
        model = (models or {}).get(kind) or load_model(kind)
        thr_hi, thr = model["thresholds"]["fpr_1pct"], model["thresholds"][policy]
        counts = {"flagged": 0, "high": 0, "medium": 0}
        status_counts: dict = defaultdict(int)
        oor_rows = 0
        hosts: dict = defaultdict(lambda: [0, 0, 0.0])
        host_col = HOST_COLUMN.get(kind)
        header = True
        for chunk in all_chunks():
            p, info = SCORERS[kind](chunk, model)
            flag = p >= thr
            sev = np.where(p >= thr_hi, "high", np.where(flag, "medium", "none"))
            status, notes = _row_annotations(info, len(model["features"]))
            res = chunk.copy()
            res["is_attack"], res["attack_probability"], res["severity"] = flag, np.round(p, 6), sev
            res["reason"] = np.where(flag, f"{kind} model score above the {policy} threshold", "")
            res["row_status"], res["row_notes"], res["model"], res["policy"] = status, notes, kind, policy
            res.to_csv(out, mode="a", header=header, index=False)
            header = False
            rows += len(chunk)
            counts["flagged"] += int(flag.sum())
            counts["high"] += int((sev == "high").sum())
            counts["medium"] += int((sev == "medium").sum())
            for s in status:
                status_counts[str(s)] += 1
            oor_rows += int((info["out_of_range"] >= 3).sum())
            if host_col and host_col in chunk.columns:
                for h, f, pp in zip(chunk[host_col], flag, p):
                    r = hosts[h]
                    r[0] += 1
                    r[1] += int(f)
                    r[2] = max(r[2], float(pp))
        summary.update({"rows": rows, "flagged": counts["flagged"], "flagged_share": round(counts["flagged"] / max(rows, 1), 4),
                        "severity": {"high": counts["high"], "medium": counts["medium"]}, "row_status": dict(status_counts),
                        "rows_far_outside_training_range_share": round(oor_rows / max(rows, 1), 4), "caution": CAUTION[kind]})
        if oor_rows / max(rows, 1) > OOR_ROWS_WARN_SHARE:
            summary["warnings"].append(f"{oor_rows / rows:.0%} of rows have 3 or more values far outside the training range: this file looks unlike the "
                                       f"data the {kind} model was trained on, so its scores may be unreliable")
        if status_counts.get("unusable"):
            summary["warnings"].append(f"{status_counts['unusable']} rows are unusable (half or more of their features missing or unreadable)")
        if hosts:
            top = sorted(((h, v[0], v[1], v[2]) for h, v in hosts.items() if v[1] > 0), key=lambda x: (-x[2], -x[3]))[:50]   # most flagged flows first
            hp = out.with_suffix(".hosts.csv")
            pd.DataFrame(top, columns=["host", "flows", "flagged_flows", "max_probability"]).assign(
                flagged_share=lambda d: (d["flagged_flows"] / d["flows"]).round(4)).to_csv(hp, index=False)
            summary["hosts_table"] = str(hp)
            summary["hosts_flagged"] = int(sum(1 for v in hosts.values() if v[1] > 0))
    else:
        parts, total = [], 0
        for chunk in all_chunks():
            parts.append(chunk)
            total += len(chunk)
            if total > max_rule_rows:
                raise CsvError(f"too many rows for rule mode ({total:,}+; limit {max_rule_rows:,}): split the file by time")
        df = pd.concat(parts, ignore_index=True)
        rows = len(df)
        findings, facts = scan_findings(df, window_s=scan_window, min_ports=min_ports, min_hosts=min_hosts)
        df["finding"] = flag_rows(df, findings, window_s=scan_window)
        df["is_attack"] = df["finding"] != ""
        df.to_csv(out, index=False)
        fp = out.with_suffix(".findings.csv")
        findings.to_csv(fp, index=False)
        summary.update({"rows": rows, **facts, "findings": int(len(findings)), "findings_table": str(fp),
                        "hosts_flagged": int(findings["host"].nunique()), "flows_in_findings": int(df["is_attack"].sum()),
                        "rules": {"window_s": scan_window, "min_ports": min_ports, "min_hosts": min_hosts},
                        "caution": "rule thresholds are defaults, not tuned or measured on labelled data"})
        if facts["rows_ignored"]:
            summary["warnings"].append(f"{facts['rows_ignored']} rows have an unreadable address or time and were ignored by the rules")
        if facts["rows_without_numeric_port"]:
            summary["warnings"].append(f"{facts['rows_without_numeric_port']} rows have no numeric port (ICMP ids): they take part only in the ICMP sweep rule")
        if facts["rows_with_clock_before_1990"]:
            summary["warnings"].append(f"{facts['rows_with_clock_before_1990']} rows have a timestamp before 1990 (a device clock that was never set); "
                                       f"windows inside the file still work, but dates are wrong")
    skipped = max(n_lines - rows, 0)
    summary["lines_skipped_malformed_or_unreadable"] = skipped
    if skipped:
        summary["warnings"].append(f"{skipped} line(s) of the file could not be read as rows (wrong number of fields) and were skipped")
    summary["seconds"] = round(time.time() - t0, 1)
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Score or check a flow CSV (UNSW 34 features, NetFlow v2, or Argus).")
    ap.add_argument("file")
    ap.add_argument("--out", default=None, help="output CSV (default: reports/csv/<name>_scored.csv)")
    ap.add_argument("--schema", default=None, choices=sorted(SCHEMAS), help="force the format instead of detecting it")
    ap.add_argument("--policy", default="fpr_3pct", choices=sorted(POLICIES), help="false-alarm target for the ML formats")
    ap.add_argument("--chunksize", type=int, default=200_000)
    a = ap.parse_args(argv)
    try:
        s = process_csv(a.file, a.out, a.schema, a.policy, a.chunksize)
    except CsvError as e:
        print(f"error: {e}")
        return 2
    print(f"format: {s['schema']} | rows: {s['rows']:,} | output: {s['output']}")
    if "flagged" in s:
        print(f"flagged {s['flagged']:,} ({s['flagged_share']:.1%}) at {s['policy']}: high {s['severity']['high']:,}, medium {s['severity']['medium']:,} | row status {s['row_status']}")
    else:
        print(f"scan findings: {s['findings']} on {s['hosts_flagged']} host(s); flows involved: {s['flows_in_findings']:,}")
    for w in s["warnings"]:
        print("WARNING:", w)
    print("caution:", s["caution"])
    print(f"summary: {s['output']}.summary.json ({s['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
