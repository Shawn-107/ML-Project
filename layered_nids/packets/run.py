"""L3 experiment on the Mirai capture.   python -m layered_nids.packets.run [--seeds 3] [--quick] [--labels FILE]

Honest framing (also written into the report):
  * Training data = the first 52.0 minutes, which the published method (Table III) defines as the clean training phase.
    That period is cut in TIME ORDER into train / threshold / held-out normal; nothing is shuffled.
  * False-alarm rate is measured on the held-out normal slice (real measurement, assuming the paper's clean period).
  * Without a per-packet label file there is NO measured recall. The 'proxy' numbers use the traffic profile (minute >= 80
    is dominated by ARP floods and ~6x more destinations) and are labelled as proxies. With --labels (UCI Mirai label file,
    one 0/1 per packet) real recall / false alarms / AUC are added.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.paths import DATA, REPORTS
from layered_nids.packets.detector import PacketDetector
from layered_nids.packets.extractor import FeatureExtractor
from layered_nids.packets.pcap import iter_packets

PCAP = DATA / "mirai" / "mirai.pcap"
CACHE = DATA / "mirai" / "cache"
TRAIN_MIN = 52.0          # published method, Table III: Mirai train phase
ATTACK_PROXY_MIN = 80.0   # from the traffic profile, NOT from labels
BIN_MIN = 8.0


def load_features(pcap: Path, limit: int | None):
    """Return (X, ts, src). Features are cached because extraction takes about a minute."""
    key = hashlib.sha256(f"{pcap.name}-{pcap.stat().st_size}-{limit}".encode()).hexdigest()[:12]
    cache = CACHE / f"features_{key}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        return z["X"], z["ts"], z["src"]
    pk = list(iter_packets(pcap, limit))
    X = FeatureExtractor().run(pk)
    ts = np.array([p.ts for p in pk])
    src = np.array([p.src for p in pk])
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(cache, X=X, ts=ts, src=src)
    return X, ts, src


def rate_feature(ts: np.ndarray, window_s: float = 1.0) -> np.ndarray:
    """Packets seen in the last `window_s` seconds (whole network). The kind of volume rule an L1 layer would use."""
    s = np.sort(ts)
    return (np.searchsorted(s, ts, side="right") - np.searchsorted(s, ts - window_s, side="right")).astype(np.float64)


class IsoForest:
    def __init__(self, seed):
        self.m = IsolationForest(n_estimators=200, random_state=seed, n_jobs=1)

    def fit(self, X):
        self.m.fit(X)
        return self

    def score(self, X):
        return -self.m.score_samples(X)


class Ens:
    def __init__(self, seed, epochs, max_group):
        self.ens = PacketDetector(seed=seed, epochs=epochs, max_group=max_group).ens

    def fit(self, X):
        self.ens.fit(X)
        return self

    def score(self, X):
        return self.ens.score(X)


def summarize(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(np.nanmean(v)) if np.isfinite(v).any() else float("nan"),
            "std": float(np.nanstd(v)) if np.isfinite(v).any() else float("nan"), "n": int(len(v))}


def evaluate(scores, fpr, sl, rel, labels, n_train_end):
    """One scored run -> metrics dict. `sl` holds slices for thr/hold."""
    thr = threshold_at_fpr(scores[sl["thr"]], fpr)
    flag = scores >= thr
    out = {"hold_false_alarm_rate": float(flag[sl["hold"]].mean())}
    bins = {}
    for a in np.arange(0, rel.max() + 1e-9, BIN_MIN):
        m = (rel >= a) & (rel < a + BIN_MIN)
        if m.any():
            bins[f"{a:g}-{a + BIN_MIN:g}min"] = float(flag[m].mean())
    out["flagged_share_by_8min"] = bins
    proxy = rel >= ATTACK_PROXY_MIN
    if proxy.any():
        out["proxy_flagged_share_min80plus"] = float(flag[proxy].mean())
        idx = np.r_[np.arange(sl["hold"].start, sl["hold"].stop), np.flatnonzero(proxy)]
        y = np.r_[np.zeros(sl["hold"].stop - sl["hold"].start), np.ones(int(proxy.sum()))]
        out["proxy_auc_hold_vs_min80plus"] = float(roc_auc_score(y, scores[idx]))
    if labels is not None:
        ex = np.arange(len(scores)) >= n_train_end
        y = labels[ex]
        out["labelled"] = {"recall": float(flag[ex][y == 1].mean()), "false_alarm_rate": float(flag[ex][y == 0].mean()),
                           "auc": float(roc_auc_score(y, scores[ex])), "n_malicious": int((y == 1).sum())}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--warmup-min", type=float, default=0.0,
                    help="minutes at the start used only to warm up the slow-decay statistics, not for training")
    ap.add_argument("--pcap", default=str(PCAP))
    ap.add_argument("--labels", default=str(DATA / "mirai" / "Mirai_labels.csv"))
    ap.add_argument("--quick", action="store_true", help="first 100k packets only, 1 seed, 3 epochs (smoke run)")
    ap.add_argument("--out", default=str(REPORTS / "packets" / "results.json"))
    a = ap.parse_args(argv)
    if a.quick:
        a.seeds, a.epochs = 1, 3

    t0 = time.time()
    X, ts, src = load_features(Path(a.pcap), 100_000 if a.quick else None)
    rel = (ts - ts.min()) / 60.0
    n = len(X)
    n52 = int(np.searchsorted(rel, TRAIN_MIN))
    w0 = int(np.searchsorted(rel, a.warmup_min))          # packets before this only warm the slow statistics up
    a1, a2 = w0 + int((n52 - w0) * 0.6), w0 + int((n52 - w0) * 0.8)
    sl = {"train": slice(w0, a1), "thr": slice(a1, a2), "hold": slice(a2, n52)}
    labels = None
    if Path(a.labels).exists():
        labels = np.loadtxt(a.labels, delimiter=",", dtype=np.int64, ndmin=1)
        if labels.ndim > 1:
            labels = labels[:, -1]
        if len(labels) != n and not a.quick:
            raise ValueError(f"label file has {len(labels)} rows, capture has {n} packets")
        if a.quick:
            labels = labels[:n]
    print(f"packets {n:,}  warm-up skipped {w0:,}  train {a1 - w0:,}  threshold {a2 - a1:,}  held-out normal {n52 - a2:,}  "
          f"labels: {'yes' if labels is not None else 'NO (proxy metrics only)'}", flush=True)

    rate = rate_feature(ts)
    results = {}

    def run_model(name, make, seeds, feats=X):
        runs = []
        for s in seeds:
            m = make(s).fit(feats[sl["train"]])
            runs.append(evaluate(m.score(feats), a.fpr, sl, rel, labels, n52))
        results[name] = runs
        far = summarize([r["hold_false_alarm_rate"] for r in runs])
        pr = summarize([r.get("proxy_flagged_share_min80plus", np.nan) for r in runs])
        print(f"{name:22s} held-out false alarms {far['mean']:.4f}+-{far['std']:.4f}   "
              f"proxy flagged share min80+ {pr['mean']:.3f}+-{pr['std']:.3f}", flush=True)

    seeds = list(range(a.seeds))
    run_model("ae_ensemble(max10)", lambda s: Ens(s, a.epochs, 10), seeds)
    run_model("single_ae(max100)", lambda s: Ens(s, a.epochs, 100), seeds)
    run_model("isolation_forest", lambda s: IsoForest(s), seeds)

    class RateRule:
        def fit(self, X): return self
        def score(self, X): return X[:, 0]
    run_model("rate_rule(pkts/s)", lambda s: RateRule(), [0], feats=rate.reshape(-1, 1))

    summary = {}
    for name, runs in results.items():
        summary[name] = {"hold_false_alarm_rate": summarize([r["hold_false_alarm_rate"] for r in runs]),
                         "proxy_flagged_share_min80plus": summarize([r.get("proxy_flagged_share_min80plus", np.nan) for r in runs]),
                         "proxy_auc_hold_vs_min80plus": summarize([r.get("proxy_auc_hold_vs_min80plus", np.nan) for r in runs])}
        if labels is not None:
            for k in ("recall", "false_alarm_rate", "auc"):
                summary[name][f"labelled_{k}"] = summarize([r["labelled"][k] for r in runs])
    report = {"data": {"packets": n, "warmup_min": a.warmup_min, "warmup_packets": w0, "train": a1 - w0,
                       "threshold": a2 - a1, "held_out_normal": n52 - a2,
                       "pcap": str(a.pcap), "target_fpr": a.fpr, "labels_used": labels is not None, "quick": a.quick,
                       "clean_period": f"first {TRAIN_MIN} min (published method, Table III), split in time order 60/20/20",
                       "proxy_definition": f"minute >= {ATTACK_PROXY_MIN:g} of the capture; chosen from the traffic profile "
                                           "(ARP share 4-7% -> 78-87%), NOT from ground-truth labels"},
              "summary": summary, "runs": results, "seconds": round(time.time() - t0, 1)}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out, f"({report['seconds']}s)")


if __name__ == "__main__":
    main()
