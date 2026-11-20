"""Adversarial validation for the flow classifier (improvement #1 in the project brief).

Question: the official test set is distributed differently from training. Which features give that away, and does
removing or down-weighting them make the model better on shifted data?

   python -m layered_nids.flows.adversarial  ->  reports/flows/adversarial.json

Protocol:
- The official test set is cut into halves A and B (stratified by attack family). Only the UNLABELLED FEATURES of A are
  used to learn the shift (a transductive step: it uses target-domain features, never target labels).
- Results are reported on B only, so no row is both used to choose features and scored.
- Thresholds come from validation Normal flows (half V2) as in every other L4 experiment.
- Models: random forest + gradient boosting, equal-weight mean. The legacy ANN is NOT retrained here (untested with it).
Variants: drop the k most shifted raw features (k in KS), or weight training rows by how test-like they look.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from layered_nids.common.paths import REPORTS
from layered_nids.flows.data import FlowData, load_flow_data, source_feature
from layered_nids.flows.evaluate import evaluate_scores, mean_std
from layered_nids.flows.models import HGB, RF

KS = (0, 3, 6, 10)


def groups_of(names: list) -> dict:
    g: dict = {}
    for i, n in enumerate(names):
        g.setdefault(source_feature(n), []).append(i)
    return g


def shift_scan(X_source, X_target, names, seed: int = 0, repeats: int = 3) -> dict:
    """Classifier 'source or target?' with 5-fold CV AUC, then permutation importance per RAW feature (grouped one-hot).
    AUC 0.5 = indistinguishable; the larger the AUC drop when a feature is shuffled, the more that feature gives away."""
    X = np.vstack([X_source, X_target])
    y = np.r_[np.zeros(len(X_source)), np.ones(len(X_target))]
    rng = np.random.default_rng(seed)
    groups = groups_of(names)
    oof = np.zeros(len(y))
    drops = {g: [] for g in groups}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = HGB(seed, max_iter=120).fit(X[tr], y[tr])
        oof[te] = m.score(X[te])
        base = roc_auc_score(y[te], oof[te])
        for g, cols in groups.items():
            for _ in range(repeats):
                Xp = X[te].copy()
                Xp[:, cols] = Xp[np.ix_(rng.permutation(len(te)), cols)]
                drops[g].append(base - roc_auc_score(y[te], m.score(Xp)))
    imp = {g: float(np.mean(v)) for g, v in drops.items()}
    return {"cv_auc": float(roc_auc_score(y, oof)), "importance": dict(sorted(imp.items(), key=lambda kv: -kv[1])),
            "oof_target_prob": oof[len(X_source):]}


def test_likeness_weights(X_source, X_target, seed: int = 0, clip=(0.1, 10.0)) -> np.ndarray:
    """Weight for each source row = estimated p(target|x)/p(source|x), cross-fitted so a row is never scored by a model
    that saw it, clipped and rescaled to mean 1. A standard covariate-shift correction."""
    X = np.vstack([X_source, X_target])
    y = np.r_[np.zeros(len(X_source)), np.ones(len(X_target))]
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[te] = HGB(seed, max_iter=120).fit(X[tr], y[tr]).score(X[te])
    p = np.clip(p[:len(X_source)], 1e-3, 1 - 1e-3)
    w = np.clip((p / (1 - p)) * (len(X_source) / len(X_target)), *clip)
    return w / w.mean()


def run_variant(d: FlowData, keep: np.ndarray, w, seed: int, idx_v2, idx_b) -> dict:
    Xtr, Xv, Xb = d.Xtr[:, keep], d.Xv[:, keep], d.Xte[idx_b][:, keep]
    rf, gb = RF(seed).fit(Xtr, d.ytr, w=w), HGB(seed).fit(Xtr, d.ytr, w=w)
    sv = (rf.score(Xv) + gb.score(Xv)) / 2
    sb = (rf.score(Xb) + gb.score(Xb)) / 2
    return evaluate_scores(sv[idx_v2], d.yv[idx_v2], sb, d.yte[idx_b], d.cte[idx_b])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n-source", type=int, default=40_000)
    ap.add_argument("--out", default=str(REPORTS / "flows" / "adversarial.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    d = load_flow_data()
    idx_a, idx_b = train_test_split(np.arange(len(d.yte)), test_size=0.5, stratify=d.cte, random_state=0)
    idx_v2 = train_test_split(np.arange(len(d.yv)), test_size=0.5, stratify=d.cv, random_state=0)[1]
    sub = np.random.default_rng(0).choice(len(d.Xtr), min(a.n_source, len(d.Xtr)), replace=False)

    scan = shift_scan(d.Xtr[sub], d.Xte[idx_a], d.feature_names)
    ranking = list(scan["importance"])
    print(f"train vs test-half-A: cross-validated AUC {scan['cv_auc']:.3f} (0.5 = no shift)", flush=True)
    print("most shifted raw features:", ", ".join(f"{k} ({v:.3f})" for k, v in list(scan["importance"].items())[:8]), flush=True)

    groups = groups_of(d.feature_names)
    w_all = test_likeness_weights(d.Xtr[sub], d.Xte[idx_a])
    w_full = np.ones(len(d.Xtr))
    w_full[sub] = w_all                                   # rows outside the subsample keep weight 1 (documented)
    variants = {f"drop_top{k}": (k, None) for k in KS}
    variants["reweight_test_like"] = (0, w_full)
    runs: dict = {k: [] for k in variants}
    for seed in range(a.seeds):
        for name, (k, w) in variants.items():
            dropped = {c for g in ranking[:k] for c in groups[g]}
            keep = np.array([i for i in range(d.Xtr.shape[1]) if i not in dropped])
            runs[name].append(run_variant(d, keep, w, seed, idx_v2, idx_b))
            r = runs[name][-1]["fpr_3pct"]
            print(f"seed {seed} {name:20s} test-B @fpr_3pct acc {r['accuracy']:.3f} recall {r['recall']:.3f} "
                  f"false-alarm {r['false_alarm_rate']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    out = {"protocol": __doc__, "n": {"test_A": int(len(idx_a)), "test_B": int(len(idx_b)), "source_sample": int(len(sub))},
           "shift_cv_auc": scan["cv_auc"], "importance": scan["importance"],
           "dropped_by_variant": {n: ranking[:k] for n, (k, _) in variants.items()},
           "summary_on_test_B": {k: mean_std(v) for k, v in runs.items()}, "seeds": a.seeds,
           "wall_seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
