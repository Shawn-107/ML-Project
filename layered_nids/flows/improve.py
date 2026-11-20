"""Step 1 of the L4 work: do a gradient-boosting model, a multi-class objective, or a LEARNED blend beat the legacy
fixed 0.5/0.5 ANN+forest blend?   python -m layered_nids.flows.improve  ->  reports/flows/improvements.json

Protocol (fixed before looking at results):
- Baseline = the legacy trained bundle (5-seed ANN + forest), unchanged.
- New models are fitted on the legacy TRAIN split only. The ANN is held fixed; the new models are repeated over 3 seeds.
- The legacy validation split is cut in two stratified halves: V1 trains the stacker, V2 sets every threshold
  (same for every candidate, so the comparison is fair). Test is scored once; nothing is chosen on it.
- Candidates are reported ALL, with their false-alarm rate; no candidate is picked by test score.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from layered_nids.common.paths import REPORTS
from layered_nids.flows.data import load_flow_data
from layered_nids.flows.evaluate import evaluate_scores, mean_std
from layered_nids.flows.models import HGB, LogitStacker, MultiClass


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default=str(REPORTS / "flows" / "improvements.json"))
    ap.add_argument("--cache", default=str(REPORTS / "flows" / "score_cache.npz"))
    a = ap.parse_args(argv)
    t0 = time.time()

    d = load_flow_data()
    b = d.bundle
    v1, v2 = train_test_split(np.arange(len(d.yv)), test_size=0.5, stratify=d.cv, random_state=0)
    base = {"ann": (b.ann.predict_proba(d.Xv), b.ann.predict_proba(d.Xte)),
            "rf": (b.rf.predict_proba(d.Xv), b.rf.predict_proba(d.Xte))}
    w = b.blend_weight

    def cands(sv, st):
        """sv/st: dict of base-model scores on val/test -> dict candidate_name -> (val_scores, test_scores, fit_on_V1?)"""
        m = lambda keys, S: np.mean([S[k] for k in keys], axis=0)
        c = {
            "ann_only": (sv["ann"], st["ann"]),
            "rf_only": (sv["rf"], st["rf"]),
            "gb_only": (sv["gb"], st["gb"]),
            "gb_multiclass_only": (sv["gbm"], st["gbm"]),
            "legacy_blend_ann_rf": (w * sv["ann"] + (1 - w) * sv["rf"], w * st["ann"] + (1 - w) * st["rf"]),
            "mean_ann_rf_gb": (m(["ann", "rf", "gb"], sv), m(["ann", "rf", "gb"], st)),
            "mean_ann_rf_gb_gbmulti": (m(["ann", "rf", "gb", "gbm"], sv), m(["ann", "rf", "gb", "gbm"], st)),
            "mean_ann_rf_gb_gbmulti_rfmulti": (m(["ann", "rf", "gb", "gbm", "rfm"], sv), m(["ann", "rf", "gb", "gbm", "rfm"], st)),
        }
        return c

    runs: dict[str, list] = {}
    stack_w = {}
    for seed in range(a.seeds):
        print(f"seed {seed}: fitting gradient boosting, multi-class gradient boosting, multi-class forest", flush=True)
        models = {"gb": HGB(seed).fit(d.Xtr, d.ytr),
                  "gbm": MultiClass("hgb", seed).fit(d.Xtr, cats=d.ctr),
                  "rfm": MultiClass("rf", seed).fit(d.Xtr, cats=d.ctr)}
        sv = {"ann": base["ann"][0], "rf": base["rf"][0], **{k: m.score(d.Xv) for k, m in models.items()}}
        st = {"ann": base["ann"][1], "rf": base["rf"][1], **{k: m.score(d.Xte) for k, m in models.items()}}
        for name, (cv_, ct_) in cands(sv, st).items():
            runs.setdefault(name, []).append(evaluate_scores(cv_[v2], d.yv[v2], ct_, d.yte, d.cte))
        for name, keys in {"stack_ann_rf_gb": ["ann", "rf", "gb"],
                           "stack_all5": ["ann", "rf", "gb", "gbm", "rfm"]}.items():
            S_v = np.column_stack([sv[k] for k in keys])
            S_t = np.column_stack([st[k] for k in keys])
            stk = LogitStacker().fit(S_v[v1], d.yv[v1])
            runs.setdefault(name, []).append(evaluate_scores(stk.score(S_v[v2]), d.yv[v2], stk.score(S_t), d.yte, d.cte))
            stack_w.setdefault(name, []).append({"inputs": keys, "weights": stk.weights[0], "bias": stk.weights[1]})
        np.savez_compressed(a.cache, **{f"seed{seed}_val_{k}": v for k, v in sv.items()},
                            **{f"seed{seed}_test_{k}": v for k, v in st.items()})
        print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    summary = {k: mean_std(v) for k, v in runs.items()}
    out = {"protocol": __doc__, "n_seeds_new_models": a.seeds,
           "n": {"train": int(len(d.ytr)), "val_v1": int(len(v1)), "val_v2": int(len(v2)), "test": int(len(d.yte))},
           "summary": summary, "stacker_weights": stack_w, "wall_seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"{'candidate':34s} {'AUC':>7s} | fpr_3pct: {'acc':>6s} {'recall':>7s} {'false-alarm':>11s} | fpr_1pct: {'recall':>7s} {'false-alarm':>11s}")
    for k, s in summary.items():
        p3, p1 = s["fpr_3pct"], s["fpr_1pct"]
        print(f"{k:34s} {s['auc_test']['mean']:7.4f} | {'':9s}{p3['accuracy']['mean']:6.3f} {p3['recall']['mean']:7.3f} "
              f"{p3['false_alarm_rate']['mean']:11.3f} | {'':9s}{p1['recall']['mean']:7.3f} {p1['false_alarm_rate']['mean']:11.3f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
