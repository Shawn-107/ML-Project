"""Fusion ablation on the SIMULATED multi-layer scenarios (see fusion/simulate.py for what is real and what is not).

   python -m layered_nids.fusion.ablation [--flow-model legacy|mean_ann_rf_gb|stack_ann_rf_gb] [--budget 0.5]

Protocol:
- Layer pools: L2 = n-gram LR on unseen DGA families (threshold at 1% false alarms on validation benign names);
  L4 = a flow model's official-test scores (threshold at the 3% policy on validation Normal flows).
- Three disjoint sets of scenarios: CALIBRATION (choose each method's alert parameter), TRAIN (fit the learned
  combiner), TEST (report). Every method is calibrated to the SAME budget of false alerts on benign hosts per
  host-day, so detection cannot be bought with extra false alarms.
- Methods: single evidence cut-off (no accumulation), noisy-OR accumulation, accumulation needing 2 layers,
  learned logistic-regression combiner. Layer sets: DGA only, flows only, both.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.paths import REPORTS
from layered_nids.fusion.accumulator import Alert, EvidenceAccumulator, FusionConfig
from layered_nids.fusion.combiner import LearnedCombiner
from layered_nids.fusion.simulate import LayerPool, make_scenario, score_alerts

LAYER_SETS = {"dga_only": ("dga",), "flows_only": ("flows",), "both": ("dga", "flows")}
P_GRID = [1 - 10 ** -k for k in (0.5, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10)]
CUT_GRID = [0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999]
PROB_GRID = [0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9999]


def dga_pool(seed: int = 0, rate: float = 8.0) -> LayerPool:
    from layered_nids.dga.data import family_folds, load_domains
    from layered_nids.dga.models import NgramLR
    df = load_domains(max_per_family=2000, n_benign=100_000, seed=seed)
    s = next(iter(family_folds(df, 5, seed=seed)))
    m = NgramLR(seed=seed).fit(s.train["name"].tolist(), s.train["y"].to_numpy())
    sv = m.predict_proba(s.val["name"].tolist())
    st = m.predict_proba(s.test["name"].tolist())
    thr = threshold_at_fpr(sv[s.val["y"].to_numpy() == 0], 0.01)
    y = s.test["y"].to_numpy()
    return LayerPool("dga", st[y == 0], st[y == 1], thr, 0.01, rate, "domain lookups")


def flow_pool(model: str, cache: Path, rate: float = 20.0) -> LayerPool:
    from layered_nids.flows.data import load_flow_data
    from layered_nids.flows.models import LogitStacker
    d = load_flow_data()
    z = np.load(cache)
    seed = sorted({k.split("_")[0] for k in z.files})[0]
    sv = {k: z[f"{seed}_val_{k}"] for k in ("ann", "rf", "gb")}
    st = {k: z[f"{seed}_test_{k}"] for k in ("ann", "rf", "gb")}
    v1, v2 = train_test_split(np.arange(len(d.yv)), test_size=0.5, stratify=d.cv, random_state=0)
    w = d.bundle.blend_weight
    if model == "legacy":
        val, test = w * sv["ann"] + (1 - w) * sv["rf"], w * st["ann"] + (1 - w) * st["rf"]
    elif model == "mean_ann_rf_gb":
        val, test = np.mean([sv[k] for k in ("ann", "rf", "gb")], 0), np.mean([st[k] for k in ("ann", "rf", "gb")], 0)
    elif model == "stack_ann_rf_gb":
        S = lambda D: np.column_stack([D[k] for k in ("ann", "rf", "gb")])
        stk = LogitStacker().fit(S(sv)[v1], d.yv[v1])
        val, test = stk.score(S(sv)), stk.score(S(st))
    else:
        raise ValueError(model)
    thr = threshold_at_fpr(val[v2][d.yv[v2] == 0], 0.03)
    return LayerPool("flows", test[d.yte == 0], test[d.yte == 1], thr, 0.03, rate, "flows")


# ---- alerting methods: each takes (scenario, layers, param) and returns alerts ---------------------------------
def m_cutoff(sc, layers, x):
    out, last = [], {}
    for e in sorted((e for e in sc.evidence if e.layer in layers and e.score >= x), key=lambda e: e.ts):
        if e.ts - last.get(e.host, -1e18) >= 300.0:
            last[e.host] = e.ts
            out.append(Alert(e.host, e.ts, e.score, "high", {e.layer: 1.0}, (e.reason,)))
    return out


def m_accumulate(sc, layers, p, min_layers=1):
    cfg = FusionConfig(alert_p=p, min_layers=min_layers, cooldown_s=300.0)
    return EvidenceAccumulator(cfg).run([e for e in sc.evidence if e.layer in layers])


def calibrate(method, layers, grid, scenarios, budget):
    """Least strict parameter whose benign false alerts per host-day (pooled over scenarios) stay within budget."""
    for g in grid:                                   # grids run from least to most strict
        fa = np.mean([score_alerts(method(sc, layers, g), sc)["benign_false_alerts_per_host_day"] for sc in scenarios])
        if fa <= budget:
            return g, float(fa)
    return grid[-1], float("nan")                    # nothing met the budget: report the strictest, flagged by NaN


def evaluate_on(method, layers, param, scenarios):
    rs = [score_alerts(method(sc, layers, param), sc) for sc in scenarios]
    f = lambda k: {"mean": float(np.nanmean([r[k] for r in rs])), "std": float(np.nanstd([r[k] for r in rs]))}
    return {k: f(k) for k in ("detection_rate", "median_time_to_detect_s", "benign_false_alerts_per_host_day",
                              "benign_hosts_alerted_share")}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--flow-model", default="legacy")
    ap.add_argument("--budget", type=float, default=0.5, help="allowed false alerts on benign hosts per host-day")
    ap.add_argument("--n-test", type=int, default=20)
    ap.add_argument("--n-calib", type=int, default=20)
    ap.add_argument("--n-train", type=int, default=20)
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--told-measured-fpr", action="store_true",
                    help="FAIR BASELINE: tell the evidence rule each layer's locally measured false-alarm rate instead of the nominal 1%%/3%%")
    ap.add_argument("--cache", default=str(REPORTS / "flows" / "score_cache.npz"))
    ap.add_argument("--out", default=str(REPORTS / "fusion" / "ablation.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    pools = {"dga": dga_pool(), "flows": flow_pool(a.flow_model, Path(a.cache))}
    if a.told_measured_fpr:
        pools = {k: replace(p, fpr=max(p.measured_fpr, 1e-4)) for k, p in pools.items()}    # same thresholds and scores; only the rate the rule assumes changes
    pool_info = {k: {"threshold": p.threshold, "nominal_fpr": p.fpr, "measured_fpr_on_held_out": p.measured_fpr,
                     "measured_recall_on_held_out": p.measured_recall} for k, p in pools.items()}
    print("layer pools:", json.dumps(pool_info), flush=True)
    mk = lambda seeds: [make_scenario(pools, hours=a.hours, seed=s) for s in seeds]
    calib, train, test = mk(range(1000, 1000 + a.n_calib)), mk(range(2000, 2000 + a.n_train)), mk(range(a.n_test))
    comb = {name: LearnedCombiner(layers=ls).fit(train) for name, ls in LAYER_SETS.items()}
    rows = {}
    for lname, layers in LAYER_SETS.items():
        setups = {"cutoff_no_accumulation": (m_cutoff, CUT_GRID),
                  "accumulate": (m_accumulate, P_GRID),
                  "learned_combiner": (lambda sc, ls, t, c=comb[lname]: c.alerts(sc, t), PROB_GRID)}
        if len(layers) == 2:
            setups["accumulate_needs_2_layers"] = (lambda sc, ls, p: m_accumulate(sc, ls, p, 2), P_GRID)
        for mname, (method, grid) in setups.items():
            param, fa_cal = calibrate(method, layers, grid, calib, a.budget)
            res = evaluate_on(method, layers, param, test)
            rows[f"{lname}/{mname}"] = {"param": param, "calibration_false_alerts_per_host_day": fa_cal, **res}
            print(f"{lname:10s} {mname:26s} param {param:<10.6g} detect {res['detection_rate']['mean']:.2f}+-{res['detection_rate']['std']:.2f} "
                  f"ttd {res['median_time_to_detect_s']['mean']:6.0f}s  false alerts/host-day {res['benign_false_alerts_per_host_day']['mean']:.2f}",
                  flush=True)
    out = {"protocol": __doc__, "flow_model": a.flow_model, "told_measured_fpr": a.told_measured_fpr, "budget_false_alerts_per_host_day": a.budget,
           "scenarios": {"calibration": a.n_calib, "train": a.n_train, "test": a.n_test, "hours": a.hours,
                         "hosts": "40 benign + 10 attackers"},
           "pools": pool_info, "results": rows, "wall_seconds": round(time.time() - t0, 1),
           "caveat": "simulated hosts/timelines with real per-layer scores; layers independent by construction"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
