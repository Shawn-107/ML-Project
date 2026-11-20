"""Flow model on NetFlow-style features that can ALSO be computed from live packets.

   python -m layered_nids.flows.netflow  ->  reports/flows/netflow.json

Data: NF-ToN-IoT-v2 (NetFlow v2 features from nProbe; the file on disk from the earlier project). Question: if we restrict the
model to features a packet-to-flow builder can compute exactly from packets (so a CSV and a live capture go through the SAME model),
how much detection do we lose compared with all 40 features, and how well does it find attack families it never saw?

Protocol: a random sample of rows; exact duplicate feature rows removed; the split is BY SOURCE HOST (train / validation / test hosts
are disjoint, so the model is tested on hosts it never saw); thresholds from validation benign flows; test scored once.
Not claimed: that a live builder reproduces nProbe's flows exactly (flow timeouts and a few definitions are nProbe's own).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from layered_nids.common.metrics import calibration
from layered_nids.common.paths import REPORTS
from layered_nids.flows.evaluate import evaluate_scores
from layered_nids.flows.models import HGB

CSV = Path(r"C:\ML project\data\crossds\nfton\NF-ToN-IoT-v2-test.csv")
ID_COLS = ["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT"]
LABELS = ["Label", "Attack"]
# Computable exactly from packet headers by definition (sizes, counts, flag union, TTL, window, ICMP type, duration, protocol, dst port):
PACKET_FEATURES = [
    "PROTOCOL", "L4_DST_PORT", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "ICMP_TYPE"]


FLAG_BITS = {"FIN": 1, "SYN": 2, "RST": 4, "PSH": 8, "ACK": 16, "URG": 32}


def derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extra features computed ONLY from the packet-computable columns above (so a live builder can produce them exactly):
    the individual TCP flag bits (a tree cannot read them from the combined number), and simple ratios and rates."""
    out = pd.DataFrame(index=df.index)
    for col in ("TCP_FLAGS", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS"):
        v = df[col].to_numpy(dtype=np.int64)
        for name, bit in FLAG_BITS.items():
            out[f"{col}_{name}"] = ((v & bit) > 0).astype(np.float32)
    dur_s = np.maximum(df["FLOW_DURATION_MILLISECONDS"].to_numpy(dtype=np.float64) / 1000.0, 1e-3)
    ip, op = df["IN_PKTS"].to_numpy(dtype=np.float64), df["OUT_PKTS"].to_numpy(dtype=np.float64)
    ib, ob = df["IN_BYTES"].to_numpy(dtype=np.float64), df["OUT_BYTES"].to_numpy(dtype=np.float64)
    out["in_bytes_per_pkt"] = ib / np.maximum(ip, 1)
    out["out_bytes_per_pkt"] = ob / np.maximum(op, 1)
    out["pkts_in_over_out"] = ip / (op + 1.0)
    out["bytes_in_over_out"] = ib / (ob + 1.0)
    out["pkts_per_s"] = (ip + op) / dur_s
    out["in_bytes_per_s"] = ib / dur_s
    out["out_bytes_per_s"] = ob / dur_s
    out["one_way"] = (op == 0).astype(np.float32)
    return out


def load_sample(frac: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    keep = lambda i: i > 0 and rng.random() > frac          # row 0 is the header
    return pd.read_csv(CSV, skiprows=keep)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.2, help="share of rows to read")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS / "flows" / "netflow.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    df = load_sample(a.frac, a.seed)
    all_features = [c for c in df.columns if c not in ID_COLS + LABELS]
    n0 = len(df)
    df = df.drop_duplicates(subset=all_features + ["IPV4_SRC_ADDR"]).reset_index(drop=True)   # exact repeats from the same host
    print(f"rows read {n0:,}; after removing exact repeats {len(df):,}; attack share {df['Label'].mean():.3f}; hosts {df['IPV4_SRC_ADDR'].nunique()}", flush=True)

    groups = df["IPV4_SRC_ADDR"].to_numpy()
    tr_idx, rest = next(GroupShuffleSplit(1, train_size=0.7, random_state=a.seed).split(df, groups=groups))
    va_rel, te_rel = next(GroupShuffleSplit(1, train_size=0.5, random_state=a.seed).split(rest, groups=groups[rest]))
    va_idx, te_idx = rest[va_rel], rest[te_rel]
    assert not (set(groups[tr_idx]) & set(groups[te_idx])) and not (set(groups[tr_idx]) & set(groups[va_idx]))
    y, cat = df["Label"].to_numpy(), df["Attack"].to_numpy()
    print(f"hosts: train {len(set(groups[tr_idx]))}, validation {len(set(groups[va_idx]))}, test {len(set(groups[te_idx]))} (disjoint)", flush=True)

    out = {"rows": {"read": n0, "after_dedupe": len(df), "train": len(tr_idx), "val": len(va_idx), "test": len(te_idx)},
           "attack_counts_test": {k: int(v) for k, v in pd.Series(cat[te_idx]).value_counts().items()}, "models": {}, "loao": {}}
    derived = derived_features(df)
    variants = (("all_40_features", all_features, None), ("packet_computable_24", PACKET_FEATURES, None),
                ("packet_computable_plus_derived", PACKET_FEATURES, derived))
    for name, cols, extra in variants:
        X = df[cols].to_numpy(dtype=np.float32)
        if extra is not None:
            X = np.hstack([X, extra.to_numpy(dtype=np.float32)])
        cols = list(cols) + ([] if extra is None else list(extra.columns))
        m = HGB(a.seed, max_iter=300).fit(X[tr_idx], y[tr_idx])
        sv, st = m.score(X[va_idx]), m.score(X[te_idx])
        r = evaluate_scores(sv, y[va_idx], st, y[te_idx], cat[te_idx])
        out["models"][name] = {"n_features": len(cols), **r}
        p1, p3 = r["fpr_1pct"], r["fpr_3pct"]
        print(f"{name:24s} test AUC {r['auc_test']:.4f} | 1%: recall {p1['recall']:.3f} (false alarms {p1['false_alarm_rate']:.4f}) | "
              f"3%: recall {p3['recall']:.3f} (false alarms {p3['false_alarm_rate']:.4f}) | brier {r['calibration_test']['brier']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        print("   flagged by attack at 3%:", {k: round(v, 3) for k, v in p3["flagged_by_family"].items()}, flush=True)

    # unseen attack families, packet-computable features only: train WITHOUT family A, test on A and on benign test hosts
    X = np.hstack([df[PACKET_FEATURES].to_numpy(dtype=np.float32), derived.to_numpy(dtype=np.float32)])   # best packet-computable variant
    from layered_nids.common.metrics import threshold_at_fpr
    for fam in sorted(set(cat) - {"Benign"}):
        trm = tr_idx[cat[tr_idx] != fam]
        vam = va_idx[cat[va_idx] != fam]
        if (cat[te_idx] == fam).sum() < 20:
            continue
        m = HGB(a.seed, max_iter=200).fit(X[trm], y[trm])
        thr = threshold_at_fpr(m.score(X[vam])[y[vam] == 0], 0.03)
        s_te = m.score(X[te_idx])
        is_f, is_b = cat[te_idx] == fam, y[te_idx] == 0
        out["loao"][fam] = {"n_test": int(is_f.sum()), "detected": float((s_te[is_f] >= thr).mean()), "false_alarm": float((s_te[is_b] >= thr).mean())}
        print(f"   unseen {fam:12s} n={int(is_f.sum()):6d} detected {out['loao'][fam]['detected']:.3f}  false alarms {out['loao'][fam]['false_alarm']:.3f}", flush=True)
    out["loao_mean_detected"] = float(np.mean([v["detected"] for v in out["loao"].values()]))
    out["wall_seconds"] = round(time.time() - t0, 1)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"unseen-family mean detection (packet-computable + derived features): {out['loao_mean_detected']:.3f}\nwrote {a.out}")


if __name__ == "__main__":
    main()
