"""Fusion demo + ablation on the Kitsune Mirai capture (L1 rules + L3 packets).   python -m layered_nids.fusion.run

Only packets from the held-out slice onward are fused (L3 was trained on the earlier packets, so its evidence there would
be optimistic). Periods, in minutes of the capture, are DESCRIPTIVE, not labels:
  clean_holdout   from the held-out slice start to minute 52   (paper's clean phase; alerts here are false alarms)
  pre_attack      52-72                                        (steady traffic by profile; not verified clean)
  attack_like     72 onward                                    (ARP flood by profile; NOT ground truth)
Real recall needs the UCI label file. Here we only show whether layers agree and what each alert says.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

from layered_nids.common.paths import DATA, REPORTS
from layered_nids.fusion.accumulator import EvidenceAccumulator, FusionConfig
from layered_nids.packets.detector import PacketDetector, windowed_evidence
from layered_nids.packets.pcap import iter_packets
from layered_nids.packets.run import PCAP, TRAIN_MIN, load_features
from layered_nids.rules.engine import calibrate_rules, run_rules

PERIODS = (("clean_holdout", -1, 52.0), ("pre_attack", 52.0, 72.0), ("attack_like", 72.0, 1e9))


def period_of(ts_min: float) -> str:
    for name, lo, hi in PERIODS:
        if lo <= ts_min < hi:
            return name
    return "clean_holdout"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", default=str(PCAP))
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS / "fusion" / "results.json"))
    a = ap.parse_args(argv)

    pk = list(iter_packets(a.pcap))
    X, ts, _ = load_features(Path(a.pcap), None)
    t0 = ts.min()
    rel = (ts - t0) / 60.0
    n52 = int(np.searchsorted(rel, TRAIN_MIN))
    a1, a2 = int(n52 * 0.6), int(n52 * 0.8)

    det = PacketDetector(fpr=a.fpr, seed=a.seed).fit(X[:a1], X[a1:a2])
    scores = det.score(X[a2:])
    ev_pk = windowed_evidence(pk[a2:], scores, det, window_s=10.0)
    ev_rules = run_rules(pk[a2:], calibrate_rules(pk[:a2], fpr=a.fpr))
    print(f"evidence from held-out slice on: packets {len(ev_pk)}, rules {len(ev_rules)}", flush=True)

    base = FusionConfig()
    setups = {
        "rules_only": (ev_rules, base),
        "packets_only": (ev_pk, base),
        "both_min1": (ev_rules + ev_pk, base),
        "both_min2": (ev_rules + ev_pk, replace(base, min_layers=2)),
    }
    hours = {name: max((min(hi, rel.max()) - max(lo, rel[a2])) / 60.0, 1e-9) for name, lo, hi in PERIODS}
    report = {"seed": a.seed, "target_fpr": a.fpr, "setups": {}, "evidence_counts": {"packets": len(ev_pk), "rules": len(ev_rules)},
              "periods_minutes": {n: [lo, hi] for n, lo, hi in PERIODS},
              "note": "periods are descriptive (traffic profile), not ground-truth labels; no recall is measured"}
    example = None
    for name, (evidence, cfg) in setups.items():
        alerts = EvidenceAccumulator(cfg).run(evidence)
        by_p = {p: [x for x in alerts if period_of((x.ts - t0) / 60.0) == p] for p, _, _ in PERIODS}
        report["setups"][name] = {
            p: {"alerts": len(v), "alerts_per_hour": round(len(v) / hours[p], 2), "hosts": dict(Counter(x.host for x in v))}
            for p, v in by_p.items()}
        if name == "both_min2" and by_p["attack_like"]:
            x = by_p["attack_like"][0]
            example = {"host": x.host, "minute": round((x.ts - t0) / 60, 1), "probability": round(x.probability, 4),
                       "severity": x.severity, "layer_shares": {k: round(v, 3) for k, v in x.layers.items()},
                       "reasons": list(x.reasons)}
        print(f"{name:13s} " + "  ".join(f"{p}: {len(v)} alerts" for p, v in by_p.items()), flush=True)
    report["example_alert_both_min2"] = example
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
