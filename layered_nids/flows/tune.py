"""Can tuning the gradient-boosting flow model help?   python -m layered_nids.flows.tune -> reports/flows/tuning.json

Protocol (declared before running): a small random search over gradient-boosting settings, scored by VALIDATION AUC only
(the legacy validation split; the official test set is not used for any choice). The best setting is then compared with
the default setting, both refitted over 3 seeds, with thresholds from validation half V2 and the test set scored ONCE.
Also reported: the learned stack (legacy ANN + forest + the tuned boosting) using the cached ANN/forest scores.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from layered_nids.common.paths import REPORTS
from layered_nids.flows.data import load_flow_data
from layered_nids.flows.evaluate import evaluate_scores, mean_std
from layered_nids.flows.models import HGB, LogitStacker

SPACE = {"lr": [0.04, 0.06, 0.08, 0.12], "max_leaf_nodes": [15, 31, 63, 127], "l2": [0.0, 1.0, 5.0, 20.0],
         "min_samples_leaf": [10, 20, 50]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--cache", default=str(REPORTS / "flows" / "score_cache.npz"))
    ap.add_argument("--out", default=str(REPORTS / "flows" / "tuning.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    d = load_flow_data()
    v1, v2 = train_test_split(np.arange(len(d.yv)), test_size=0.5, stratify=d.cv, random_state=0)
    rng = np.random.default_rng(0)
    trials = [{k: v[int(rng.integers(len(v)))] for k, v in SPACE.items()} for _ in range(a.trials)]
    default = {"lr": 0.08, "max_leaf_nodes": 31, "l2": 1.0, "min_samples_leaf": 20}
    results = []
    for i, cfg in enumerate([default] + trials):
        m = HGB(0, max_iter=400, **cfg).fit(d.Xtr, d.ytr)
        auc = float(roc_auc_score(d.yv, m.score(d.Xv)))
        results.append({"config": cfg, "val_auc": auc, "iterations": int(m.m.n_iter_)})
        print(f"{'default' if i == 0 else f'trial {i:2d}'}  val AUC {auc:.5f}  {cfg}  iters {results[-1]['iterations']}  ({time.time() - t0:.0f}s)", flush=True)
    best = max(results, key=lambda r: r["val_auc"])
    print("selected by validation AUC:", best["config"], flush=True)

    z = np.load(a.cache)
    seed_key = sorted({k.split("_")[0] for k in z.files})[0]
    ann_v, ann_t = z[f"{seed_key}_val_ann"], z[f"{seed_key}_test_ann"]
    rf_v, rf_t = z[f"{seed_key}_val_rf"], z[f"{seed_key}_test_rf"]
    runs = {"default_gb": [], "tuned_gb": [], "stack_ann_rf_default_gb": [], "stack_ann_rf_tuned_gb": []}
    for seed in range(a.seeds):
        for name, cfg in (("default_gb", default), ("tuned_gb", best["config"])):
            m = HGB(seed, max_iter=400, **cfg).fit(d.Xtr, d.ytr)
            sv, st = m.score(d.Xv), m.score(d.Xte)
            runs[name].append(evaluate_scores(sv[v2], d.yv[v2], st, d.yte, d.cte))
            S_v, S_t = np.column_stack([ann_v, rf_v, sv]), np.column_stack([ann_t, rf_t, st])
            stk = LogitStacker().fit(S_v[v1], d.yv[v1])
            runs[f"stack_ann_rf_{name}"].append(evaluate_scores(stk.score(S_v[v2]), d.yv[v2], stk.score(S_t), d.yte, d.cte))
        print(f"seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    summary = {k: mean_std(v) for k, v in runs.items()}
    out = {"protocol": __doc__, "trials": results, "selected": best, "summary_test_once": summary,
           "wall_seconds": round(time.time() - t0, 1)}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, s in summary.items():
        p = s["fpr_3pct"]
        print(f"{k:26s} AUC {s['auc_test']['mean']:.4f} | 3%: acc {p['accuracy']['mean']:.3f} recall {p['recall']['mean']:.3f} "
              f"false-alarm {p['false_alarm_rate']['mean']:.3f} | brier {s['brier']['mean']:.3f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
