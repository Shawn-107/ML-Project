"""Injection benchmark: our detector against our first version, on the same packets (an earlier run also compared a reference tool; its numbers are kept in benchmark_test.json).

   python -m layered_nids.anomaly.benchmark --stage validate     # choose between our own variants (validation injections)
   python -m layered_nids.anomaly.benchmark --stage test         # final numbers (different injections, 3 model seeds)

Data: the first 52 minutes of the Mirai capture (the paper's clean period), split in time order 60/20/20:
train / calibration (threshold, tail) / hold. Synthetic attacks (see inject.py) are merged into the HOLD part only.
Every model is trained on the same clean train part, calibrated to the same 1% false-alarm rate on the same calibration part,
and scored on the same attacked hold stream. Reported per attack kind (mean +- std over model seeds x injection seeds); false-alarm rate is also given for packets of hosts not involved in any attack (an attack changes the
statistics of its own host and target, so their normal packets can look odd for a good reason):
  packet recall at the 1% policy, packet AUC (attack packets vs normal hold packets), and episode detection: does the attacker
  host get a 60 s window whose flagged-packet count has probability <= 1e-6 under the normal false-alarm rate.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import binom
from sklearn.metrics import roc_auc_score

from layered_nids.anomaly.detector import PacketAnomalyDetector
from layered_nids.anomaly.inject import KINDS, inject
from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.paths import DATA, REPORTS
from layered_nids.packets.pcap import iter_packets

PCAP = DATA / "mirai" / "mirai.pcap"
CLEAN_MIN = 52.0
FPR = 0.01
ALERT_P = 1e-6
WINDOW_S = 60.0

VARIANTS = {                                    # own variants (validated, then the best is reported on the test seeds)
    "nids_v2_output_ae": dict(score_mode="output_ae"),
    "nids_v2_tail_sum_top3": dict(score_mode="tail_sum", topk=3),
    "nids_v2_tail_sum_all": dict(score_mode="tail_sum", topk=100),
    "nids_v2_tail_sum_top3_noise": dict(score_mode="tail_sum", topk=3, noise=0.05),
}


def load_clean(pcap=PCAP, minutes: float = CLEAN_MIN):
    pk = []
    t0 = None
    for p in iter_packets(pcap):
        t0 = p.ts if t0 is None else t0
        if (p.ts - t0) / 60.0 >= minutes:
            break
        pk.append(p)
    n = len(pk)
    a, b = int(n * 0.6), int(n * 0.8)
    return pk[:a], pk[a:b], pk[b:]


# --------------------------------------------------------------------------------------------------------- the models
class Own:
    def __init__(self, name, seed, **kw):
        self.name, self.det = name, PacketAnomalyDetector(fpr=FPR, seed=seed, **kw)

    def fit(self, train, calib):
        self.det.fit(train, calib)
        self.threshold = self.det.threshold

    def score(self, stream):
        d = copy.deepcopy(self.det)                         # same state after calibration for every injected stream
        return d.score_packets(stream)


class First:
    """Our first packet detector (packets/): 100 features, min-max scaling, output autoencoder."""
    name = "nids_v1"

    def __init__(self, seed):
        from layered_nids.packets.detector import PacketDetector
        from layered_nids.packets.extractor import FeatureExtractor
        self.fx, self.det = FeatureExtractor(), PacketDetector(fpr=FPR, seed=seed)

    def fit(self, train, calib):
        self.det.fit(self.fx.run(train), self.fx.run(calib))
        self.threshold = self.det.threshold

    def score(self, stream):
        fx = copy.deepcopy(self.fx)
        return self.det.score(fx.run(stream))


# --------------------------------------------------------------------------------------------------------- metrics
def episode_detected(stream, flagged, ep, fpr=FPR) -> bool:
    """Does the attacker host get a window overlapping the episode whose flagged count is improbable under the normal rate?"""
    n, k = {}, {}
    for p, f in zip(stream, flagged):
        if p.src == ep["host"] and ep["start"] - WINDOW_S <= p.ts <= ep["end"] + WINDOW_S:
            w = int(p.ts // WINDOW_S)
            n[w] = n.get(w, 0) + 1
            k[w] = k.get(w, 0) + int(f)
    w0, w1 = int(ep["start"] // WINDOW_S), int(ep["end"] // WINDOW_S)
    return any(k.get(w, 0) > 0 and binom.sf(k[w] - 1, n[w], fpr) <= ALERT_P for w in range(w0, w1 + 1))


def false_alert_windows(stream, flagged, labels, episodes, fpr=FPR) -> tuple[int, float]:
    """(number of (host, window) with improbably many flags that overlap NO injected episode of that host, hours of stream)."""
    n, k = {}, {}
    for p, f in zip(stream, flagged):
        if p.src and ":" not in p.src:
            key = (p.src, int(p.ts // WINDOW_S))
            n[key] = n.get(key, 0) + 1
            k[key] = k.get(key, 0) + int(f)
    bad = 0
    for (h, w), kk in k.items():
        if kk and binom.sf(kk - 1, n[(h, w)], fpr) <= ALERT_P:
            if not any(e["host"] == h and e["start"] - WINDOW_S <= (w + 1) * WINDOW_S and w * WINDOW_S <= e["end"] + WINDOW_S for e in episodes):
                bad += 1
    return bad, (stream[-1].ts - stream[0].ts) / 3600.0


def evaluate(model, hold, inj_seeds):
    rows = []
    for s in inj_seeds:
        stream, labels, episodes = inject(hold, KINDS, seed=s)
        scores = model.score(stream)
        flagged = scores >= model.threshold
        normal = labels == ""
        involved = {e["host"] for e in episodes} | {p.dst for p, k in zip(stream, labels) if k}          # attackers and their targets
        clean = np.array([(p.src not in involved and p.dst not in involved) for p in stream]) & normal
        res = {"seed": s, "false_alarm_rate_on_normal": float(flagged[normal].mean()),
               "false_alarm_rate_uninvolved": float(flagged[clean].mean()), "kinds": {}}
        for ep in episodes:
            m = labels == ep["kind"]
            y = np.r_[np.zeros(normal.sum()), np.ones(m.sum())]
            auc = float(roc_auc_score(y, np.r_[scores[normal], scores[m]]))
            res["kinds"][ep["kind"]] = {"packet_recall": float(flagged[m].mean()), "packet_auc": auc,
                                        "episode_detected": bool(episode_detected(stream, flagged, ep))}
        bad, hours = false_alert_windows(stream, flagged, labels, episodes)
        res["false_alert_windows"], res["hours"] = bad, hours
        rows.append(res)
    return rows


def summarise(all_rows: list) -> dict:
    out = {"false_alarm_rate_on_normal": [r["false_alarm_rate_on_normal"] for r in all_rows],
           "false_alarm_rate_uninvolved": [r["false_alarm_rate_uninvolved"] for r in all_rows],
           "false_alert_windows_per_hour": [r["false_alert_windows"] / r["hours"] for r in all_rows], "kinds": {}}
    for kind in KINDS:
        rs = [r["kinds"][kind] for r in all_rows]
        out["kinds"][kind] = {m: [x[m] for x in rs] for m in ("packet_recall", "packet_auc", "episode_detected")}
    ms = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)}
    return {"false_alarm_rate_on_normal": ms(out["false_alarm_rate_on_normal"]),
            "false_alarm_rate_uninvolved": ms(out["false_alarm_rate_uninvolved"]),
            "false_alert_windows_per_hour": ms(out["false_alert_windows_per_hour"]),
            "kinds": {k: {m: ms(np.asarray(v, float)) for m, v in d.items()} for k, d in out["kinds"].items()}}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["validate", "test"], required=True)
    ap.add_argument("--models", default=None, help="comma list; default: all variants (validate) or v1,original,<best v2> (test)")
    ap.add_argument("--model-seeds", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    train, calib, hold = load_clean()
    print(f"packets: train {len(train):,} | calibration {len(calib):,} | hold {len(hold):,}", flush=True)
    inj = [100, 101] if a.stage == "validate" else [0, 1, 2]
    mseeds = range(a.model_seeds or (1 if a.stage == "validate" else 3))
    names = a.models.split(",") if a.models else (list(VARIANTS) if a.stage == "validate" else ["nids_v1"])
    results = {}
    for name in names:
        rows, t0 = [], time.time()
        for ms in mseeds:
            if name in VARIANTS:
                m = Own(name, ms, **VARIANTS[name])
            elif name == "nids_v1":
                m = First(ms)
            else:
                raise SystemExit(f"unknown model {name}")
            m.fit(train, calib)
            rows += evaluate(m, hold, inj)
        results[name] = {"summary": summarise(rows), "runs": rows}
        k = results[name]["summary"]["kinds"]
        print(f"\n{name}  ({time.time() - t0:.0f}s, {len(rows)} runs = {len(list(mseeds))} model seeds x {len(inj)} injections)")
        print(f"  false alarms: all normal hold packets {results[name]['summary']['false_alarm_rate_on_normal']['mean']:.2%}, "
              f"packets of hosts not involved in any attack {results[name]['summary']['false_alarm_rate_uninvolved']['mean']:.2%} | "
              f"false alert windows/hour {results[name]['summary']['false_alert_windows_per_hour']['mean']:.1f}")
        for kind in KINDS:
            print(f"  {kind:10s} recall {k[kind]['packet_recall']['mean']:.2f}+-{k[kind]['packet_recall']['std']:.2f}  "
                  f"AUC {k[kind]['packet_auc']['mean']:.3f}  episode detected {k[kind]['episode_detected']['mean']:.2f}", flush=True)
    out = Path(a.out or REPORTS / "anomaly" / f"benchmark_{a.stage}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"protocol": __doc__, "stage": a.stage, "fpr": FPR, "alert_p": ALERT_P, "window_s": WINDOW_S,
                               "inj_seeds": inj, "results": results}, indent=1, default=float))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
