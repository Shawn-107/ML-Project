"""Metrics for the flow candidates. Threshold = policy on VALIDATION Normal flows only; the official test set is used
once, to report. Everything is reported next to its false-alarm rate (never accuracy alone)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from layered_nids.common.metrics import calibration, threshold_at_fpr

TARGETS = (0.01, 0.03)
FOCUS = ("Fuzzers", "Reconnaissance")      # the two families the legacy model is weakest on


def evaluate_scores(s_val, y_val, s_test, y_test, cat_test, targets=TARGETS, calibrated: bool = True) -> dict:
    """Return headline numbers at each false-alarm policy. `s_val`/`y_val` set the threshold; test is only scored."""
    s_val, s_test = np.asarray(s_val, float), np.asarray(s_test, float)
    out = {"auc_test": float(roc_auc_score(y_test, s_test)), "auc_val": float(roc_auc_score(y_val, s_val))}
    if calibrated:
        out["calibration_test"] = calibration(y_test, s_test)
    for t in targets:
        thr = threshold_at_fpr(s_val[y_val == 0], t)
        flag = s_test >= thr
        pos, neg = y_test == 1, y_test == 0
        rec, fa = float(flag[pos].mean()), float(flag[neg].mean())
        out[f"fpr_{int(round(t * 100))}pct"] = {
            "threshold": thr, "accuracy": float((flag == pos).mean()), "recall": rec, "false_alarm_rate": fa,
            "balanced_accuracy": (rec + 1 - fa) / 2.0,
            "precision": float(pos[flag].mean()) if flag.any() else float("nan"),
            "flagged_by_family": {c: float(flag[cat_test == c].mean()) for c in sorted(set(cat_test)) if c != "Normal"},
        }
    return out


def mean_std(runs: list[dict]) -> dict:
    """Aggregate a list of evaluate_scores() dicts (e.g. several seeds) into mean +- std for the headline numbers."""
    def ms(vals):
        v = np.asarray(vals, float)
        return {"mean": float(v.mean()), "std": float(v.std()), "n": int(len(v))}

    out = {"auc_test": ms([r["auc_test"] for r in runs]), "auc_val": ms([r["auc_val"] for r in runs])}
    if "calibration_test" in runs[0]:
        out["brier"] = ms([r["calibration_test"]["brier"] for r in runs])
        out["ece"] = ms([r["calibration_test"]["ece"] for r in runs])
    for pol in [k for k in runs[0] if k.startswith("fpr_")]:
        out[pol] = {k: ms([r[pol][k] for r in runs]) for k in ("accuracy", "recall", "false_alarm_rate", "balanced_accuracy")}
        fams = runs[0][pol]["flagged_by_family"]
        out[pol]["flagged_by_family"] = {c: ms([r[pol]["flagged_by_family"][c] for r in runs]) for c in fams}
    return out
