"""Packet-level experiment on REAL labelled captures (8 attack types, per-packet 0/1 labels): does learning only "normal" find
attacks it has never seen, and does learning known attacks help?

Designs, compared on the same held-out rows at the same false-alarm target (1%, threshold from the held-out network's own benign
validation packets):
  normal_only          our packet detector (autoencoder ensemble), trained on that capture's own benign packets before the attack only
  supervised           gradient boosting trained on the OTHER seven attack types (leave-one-attack-out), labels 0/1, no local normal
  supervised+local     the same plus the held-out capture's own benign training packets (label 0)
  both                 normal_only and supervised+local, flagged when either is alarmed (common rank scale, as in flows/openset.py)

Run:  python -m layered_nids.anomaly.labelled_exp extract     (features cached under data/labelled_iot/features; a few minutes)
      python -m layered_nids.anomaly.labelled_exp run         (writes reports/anomaly/labelled_exp.json)
      python -m layered_nids.anomaly.labelled_exp drift       (fixed versus continual model; writes reports/anomaly/labelled_drift.json)
Data folder: data/labelled_iot/ (see data/SOURCES.md). Local data; never committed.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.paths import REPORTS, ROOT

DATA = ROOT / "data" / "labelled_iot"
CACHE = DATA / "features"
ATTACKS = ["Active_Wiretap", "ARP_MitM", "Fuzzing", "Mirai", "OS_Scan", "SSDP_Flood", "SSL_Renegotiation", "SYN_DoS"]
N_BEFORE, N_AFTER = 300_000, 200_000          # benign packets before the first attack packet, and packets from there on
FA = 0.01


def _pcap(name: str) -> Path:
    for ext in (".pcap", ".pcapng"):
        p = DATA / f"{name}_pcap{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(name)


def read_labels(name: str) -> np.ndarray:
    """0/1 per packet. Two layouts occur: one value per line, or a quoted index and a value with a header line."""
    out = []
    with open(DATA / f"{name}_labels.csv") as f:
        for line in f:
            tok = line.strip().split(",")[-1].strip('"')
            if tok in ("0", "1"):
                out.append(int(tok))
    return np.asarray(out, dtype=np.int8)


def _extract_one(name: str) -> dict:
    from layered_nids.anomaly.features import PacketFeatures
    from layered_nids.packets.pcap import iter_packets

    t0 = time.time()
    y = read_labels(name)
    a = int(np.flatnonzero(y == 1)[0])
    start, end = max(0, a - N_BEFORE), min(len(y), a + N_AFTER)
    pk = list(iter_packets(_pcap(name), limit=end - start, start=start))
    if len(pk) != end - start:
        return {"name": name, "error": f"capture has {start + len(pk)} packets but the label file needs at least {end}"}
    X = PacketFeatures().run(pk)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE / f"{name}.npz", X=X, y=y[start:end], ts=np.array([p.ts for p in pk]), n_before=a - start, labels_total=len(y), start=start)
    return {"name": name, "packets": len(pk), "benign_before": a - start, "attack_in_segment": int(y[start:end].sum()), "seconds": round(time.time() - t0)}


def extract(workers: int = 4) -> None:
    with ProcessPoolExecutor(workers) as ex:
        for r in ex.map(_extract_one, ATTACKS):
            print(r, flush=True)


def _load(name: str):
    z = np.load(CACHE / f"{name}.npz")
    return z["X"], z["y"].astype(np.int8), int(z["n_before"])


def _surprise(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    ref = np.sort(ref)
    n = len(ref)
    at_least = n - np.searchsorted(ref, x * (1 - 1e-6) - 1e-12, side="left")
    return -np.log((at_least + 1.0) / (n + 1.0)) + 2.0 * np.log(np.maximum(x / (max(ref[-1], 1e-12) * (1 + 1e-6)), 1.0))


def _metrics(sv: np.ndarray, st: np.ndarray, y_test: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    from layered_nids.common.metrics import threshold_at_fpr

    thr = threshold_at_fpr(sv, FA)
    flag = st >= thr
    return {"recall": float(flag[y_test == 1].mean()), "false_alarm": float(flag[y_test == 0].mean()), "auc": float(roc_auc_score(y_test, st))}


def run(cap_pool: int = 150_000, seed: int = 0) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    from layered_nids.anomaly.features import feature_names
    from layered_nids.anomaly.model import AnomalyEnsemble

    warnings.filterwarnings("ignore")
    rng = np.random.default_rng(seed)
    data = {n: _load(n) for n in ATTACKS}
    names = feature_names()

    # per-capture split in time order. Benign packets before the first attack packet: first 70% train, next 10% tail, LAST 20% threshold
    # (the benign traffic closest to the attack: the benign stretch of these captures drifts, so an earlier slice gives a threshold the
    # later benign traffic already exceeds). Test = every packet from the first attack packet on (attacks and the benign packets among them).
    parts = {}
    for n, (X, y, nb) in data.items():
        i1, i2 = int(0.7 * nb), int(0.8 * nb)
        parts[n] = dict(tr=slice(0, i1), tail=slice(i1, i2), thr=slice(i2, nb), test_idx=np.r_[nb:len(y)])

    # a small pool of rows from each capture for supervised training (all attack rows up to half the cap, the rest benign)
    pool = {}
    for n, (X, y, nb) in data.items():
        idx_a, idx_b = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        ka = min(len(idx_a), cap_pool // 2)
        kb = min(len(idx_b), cap_pool - ka)
        sel = np.r_[rng.choice(idx_a, ka, replace=False) if ka else [], rng.choice(idx_b, kb, replace=False)].astype(int)
        pool[n] = (X[sel], y[sel])

    results, t0 = {}, time.time()
    for held in ATTACKS:
        X, y, nb = data[held]
        p = parts[held]
        Xte, yte = X[p["test_idx"]], y[p["test_idx"]]
        Xthr = X[p["thr"]]
        row = {"test_packets": int(len(yte)), "attack_packets": int((yte == 1).sum()), "benign_test_packets": int((yte == 0).sum())}

        ens = AnomalyEnsemble(seed=seed).fit(X[p["tr"]], X[p["tail"]], names=names)
        s_n_v, s_n_t = ens.score(Xthr), ens.score(Xte)
        row["normal_only"] = _metrics(s_n_v, s_n_t, yte)
        s_pre = ens.score(X[p["thr"]][::2])
        row["drift_check_normal_only"] = {"flagged_in_last_benign_stretch_when_threshold_from_first_half": float((s_pre[len(s_pre) // 2:] >= threshold_at_fpr(s_pre[:len(s_pre) // 2], FA)).mean())}

        others = [n for n in ATTACKS if n != held]
        Xp = np.vstack([pool[n][0] for n in others])
        yp = np.concatenate([pool[n][1] for n in others])
        gb = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(Xp, yp)
        s_s_v, s_s_t = gb.predict_proba(Xthr)[:, 1], gb.predict_proba(Xte)[:, 1]
        row["supervised"] = _metrics(s_s_v, s_s_t, yte)

        loc = X[p["tr"]]
        sub = rng.choice(len(loc), min(len(loc), cap_pool), replace=False)
        gb2 = HistGradientBoostingClassifier(max_iter=200, random_state=seed).fit(np.vstack([Xp, loc[sub]]), np.concatenate([yp, np.zeros(len(sub), dtype=np.int8)]))
        s_l_v, s_l_t = gb2.predict_proba(Xthr)[:, 1], gb2.predict_proba(Xte)[:, 1]
        row["supervised+local"] = _metrics(s_l_v, s_l_t, yte)

        both_v = np.maximum(_surprise(s_l_v, s_l_v), _surprise(s_n_v, s_n_v))
        both_t = np.maximum(_surprise(s_l_v, s_l_t), _surprise(s_n_v, s_n_t))
        row["both"] = _metrics(both_v, both_t, yte)
        results[held] = row
        print(f"[{time.time() - t0:5.0f}s] {held:18s} " + " | ".join(f"{k} {v['recall']:.2f}/{v['false_alarm']:.3f}" for k, v in row.items() if isinstance(v, dict) and "recall" in v), flush=True)

    designs = ["normal_only", "supervised", "supervised+local", "both"]
    summary = {d: {m: [float(np.mean([results[a][d][m] for a in ATTACKS])), float(np.std([results[a][d][m] for a in ATTACKS]))] for m in ("recall", "false_alarm", "auc")} for d in designs}
    out = {"per_attack": results, "summary_mean_std_over_attacks": summary, "settings": {"false_alarm_target": FA, "seed": seed, "benign_before": N_BEFORE, "after_onset": N_AFTER, "pool_cap_per_capture": cap_pool}}
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "anomaly").mkdir(exist_ok=True)
    (REPORTS / "anomaly" / "labelled_exp.json").write_text(json.dumps(out, indent=1))
    print("\nmean +- std over the 8 attack types (false-alarm target 1%)")
    for d in designs:
        s = summary[d]
        print(f"  {d:18s} recall {100 * s['recall'][0]:5.1f}% +- {100 * s['recall'][1]:4.1f} | false alarms {100 * s['false_alarm'][0]:5.2f}% | AUC {s['auc'][0]:.3f}")
    return out


def drift(seed: int = 0) -> dict:
    """Fixed model versus continual model (anomaly/continual.py) on the same stream. Both start from one model trained on the first half of the
    benign packets before the attack, threshold from the next 20% (1% false alarms); the stream is the last 30% of the benign stretch plus
    everything after the first attack packet. Reported: false alarms on benign packets before the attack (the drift stretch), false alarms on
    benign packets among the attack, recall on attack packets, and the refits."""
    import torch
    from sklearn.metrics import roc_auc_score

    from layered_nids.anomaly.continual import ContinualScorer
    from layered_nids.anomaly.features import feature_names
    from layered_nids.anomaly.model import AnomalyEnsemble

    warnings.filterwarnings("ignore")
    torch.set_num_threads(6)
    names, out, t0 = feature_names(), {}, time.time()
    for n in ATTACKS:
        X, y, nb = _load(n)
        i1, i2, i3 = int(0.5 * nb), int(0.6 * nb), int(0.7 * nb)
        ens = AnomalyEnsemble(seed=seed).fit(X[:i1], X[i1:i2], names=names)
        thr = threshold_at_fpr(ens.score(X[i2:i3]), FA)
        Xs, ys = X[i3:], y[i3:]
        pre = np.arange(len(ys)) < nb - i3
        sc = ContinualScorer(AnomalyEnsemble(**ens.cfg).fit(X[:i1], X[i1:i2], names=names), thr, FA, names=names)
        s_c, t_c, log = sc.run(Xs)
        s_f = ens.score(Xs)
        row = {}
        for label, s, t in (("fixed", s_f, np.full(len(ys), thr)), ("continual", s_c, t_c)):
            f = s >= t
            row[label] = {"false_alarm_before_attack": float(f[pre & (ys == 0)].mean()), "false_alarm_during_attack": float(f[~pre & (ys == 0)].mean()) if (~pre & (ys == 0)).any() else None,
                          "recall": float(f[ys == 1].mean()), "auc_during_attack": float(roc_auc_score(ys[~pre], s[~pre])) if len(np.unique(ys[~pre])) == 2 else None}
        row["refits"] = int(sum(e["refit"] for e in log))
        row["blocked_checks"] = int(sum("blocked" in e for e in log))
        out[n] = row
        print(f"[{time.time() - t0:5.0f}s] {n:18s} before-attack FA fixed {row['fixed']['false_alarm_before_attack']:.3f} -> continual {row['continual']['false_alarm_before_attack']:.3f} | "
              f"recall fixed {row['fixed']['recall']:.2f} -> continual {row['continual']['recall']:.2f} | refits {row['refits']} blocked {row['blocked_checks']}", flush=True)
    (REPORTS / "anomaly").mkdir(parents=True, exist_ok=True)
    (REPORTS / "anomaly" / "labelled_drift.json").write_text(json.dumps(out, indent=1))
    for label in ("fixed", "continual"):
        v = lambda k: np.mean([out[a][label][k] for a in ATTACKS if out[a][label][k] is not None])
        print(f"  {label:10s} mean over 8 captures: false alarms before attack {100 * v('false_alarm_before_attack'):5.1f}%, during attack {100 * v('false_alarm_during_attack'):5.1f}%, recall {100 * v('recall'):5.1f}%")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    {"extract": extract, "run": run, "drift": drift}[cmd]()
