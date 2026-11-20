"""Fold evaluation for DGA models: threshold chosen on validation benign names, then applied once to the test fold."""
from __future__ import annotations

import time

import numpy as np

from layered_nids.common.metrics import calibration, ranking, rates, threshold_at_fpr
from layered_nids.dga.data import FoldSplit
from layered_nids.dga.models import DGAModel

FPR_TARGETS = (0.01, 0.05)


def evaluate_fold(model: DGAModel, split: FoldSplit) -> dict:
    t0 = time.time()
    model.fit(split.train["name"].tolist(), split.train["y"].to_numpy(),
              val=(split.val["name"].tolist(), split.val["y"].to_numpy()))
    fit_s = time.time() - t0

    s_val = model.predict_proba(split.val["name"].tolist())
    s_test = model.predict_proba(split.test["name"].tolist())
    y_val, y_test = split.val["y"].to_numpy(), split.test["y"].to_numpy()

    out = {"fold": split.fold, "model": model.name, "fit_seconds": round(fit_s, 1),
           "n_train": len(split.train), "n_test": len(split.test), "n_test_families": len(split.test_families),
           **ranking(y_test, s_test), "policies": {}}
    if model.calibrated:
        out["calibration"] = calibration(y_test, s_test)
    for target in FPR_TARGETS:
        thr = threshold_at_fpr(s_val[y_val == 0], target)
        out["policies"][f"fpr_{int(target * 100)}pct"] = {"threshold": thr, **rates(y_test, s_test, thr)}

    # recall per unseen family at the 1% policy, so weak families are visible and not averaged away
    thr1 = out["policies"]["fpr_1pct"]["threshold"]
    flag = s_test >= thr1
    fam = split.test["family"].to_numpy()
    out["family_recall_fpr_1pct"] = {f: float(flag[fam == f].mean()) for f in split.test_families}
    return out


def summarise(folds: list[dict]) -> dict:
    """mean +- std across folds for the headline numbers."""
    def ms(vals):
        v = np.asarray(vals, dtype=float)
        return {"mean": float(np.nanmean(v)), "std": float(np.nanstd(v)), "n": int(len(v))}

    s = {"auc": ms([f["auc"] for f in folds]), "ap": ms([f["ap"] for f in folds])}
    for pol in folds[0]["policies"]:
        for k in ("recall", "false_alarm_rate"):
            s[f"{pol}.{k}"] = ms([f["policies"][pol][k] for f in folds])
    if "calibration" in folds[0]:
        s["brier"] = ms([f["calibration"]["brier"] for f in folds])
        s["ece"] = ms([f["calibration"]["ece"] for f in folds])
    fam = {}
    for f in folds:
        fam.update(f["family_recall_fpr_1pct"])
    s["family_recall_fpr_1pct_mean_over_families"] = float(np.mean(list(fam.values())))
    s["family_recall_fpr_1pct_worst5"] = dict(sorted(fam.items(), key=lambda kv: kv[1])[:5])
    return s
