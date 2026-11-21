"""Signature layer on the injection benchmark (see anomaly/inject.py), plus false alerts on real normal traffic.

   python -m layered_nids.signatures.benchmark

Count thresholds are calibrated on the first 80% of the 52-minute clean period of the Kitsune Mirai capture (no attack data
looked at); the last 20% is the hold stretch: false alerts are counted there without injections, then the attacks are merged in.
An attack is detected when the rules emit evidence for the attacker host in a window that overlaps the episode.
"""
from __future__ import annotations

import json
from collections import Counter

from layered_nids.anomaly.benchmark import load_clean
from layered_nids.anomaly.inject import EXTRA_KINDS, KINDS, inject
from layered_nids.common.paths import REPORTS
from layered_nids.signatures import calibrate, scan


def main():
    train, calib, hold = load_clean()
    cfg = calibrate(train + calib, fpr=0.01)
    print("calibrated thresholds:", {k: v for k, v in cfg.__dict__.items() if k.endswith("min") or k.endswith("targets") or k.endswith("packets")})
    clean_ev = scan(hold, cfg)
    clean_hours = (hold[-1].ts - hold[0].ts) / 3600.0
    by = Counter(e.detail["trail"] for e in clean_ev)
    print(f"false alerts on the clean hold stretch ({clean_hours * 60:.1f} min, real traffic): {len(clean_ev)} evidence items {dict(by)}")
    kinds = KINDS + EXTRA_KINDS
    res = {"calibrated": {k: v for k, v in cfg.__dict__.items() if not isinstance(v, tuple)}, "clean_hold_evidence": dict(by), "clean_hours": clean_hours, "kinds": {}}
    det = {k: [] for k in kinds}
    fp_inj = []
    for seed in (0, 1, 2):
        stream, labels, episodes = inject(hold, kinds, seed=seed)
        ev = scan(stream, cfg)
        for ep in episodes:
            hits = [e for e in ev if e.host == ep["host"] and ep["start"] - cfg.window_s <= e.ts <= ep["end"] + cfg.window_s]
            det[ep["kind"]].append({"detected": bool(hits), "by": sorted({e.detail["trail"] for e in hits})})
        fp_inj.append(sum(1 for e in ev if not any(e.host == ep["host"] and ep["start"] - cfg.window_s <= e.ts <= ep["end"] + cfg.window_s for ep in episodes)))
    for k in kinds:
        n = sum(d["detected"] for d in det[k])
        by_rule = Counter(r for d in det[k] for r in d["by"])
        res["kinds"][k] = {"detected": n, "of": len(det[k]), "rules": dict(by_rule)}
        print(f"  {k:12s} detected {n} of {len(det[k])} injections   by {dict(by_rule)}")
    res["evidence_outside_episodes_per_injection_run"] = fp_inj
    out = REPORTS / "signatures" / "benchmark.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float))
    print("wrote", out)


if __name__ == "__main__":
    main()
