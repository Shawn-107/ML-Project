"""Metrics shared by every layer. Same definitions as the legacy `nids.evaluate` (10 equal-width bins for ECE)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def calibration(y, prob, bins: int = 10) -> dict:
    y, prob = np.asarray(y, dtype=float), np.asarray(prob, dtype=float)
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return {"brier": float(np.mean((prob - y) ** 2)), "ece": float(ece)}


def threshold_at_fpr(scores_normal, target_fpr: float) -> float:
    """Smallest threshold t such that the share of NORMAL scores with score >= t is at most target_fpr.
    Flag a sample when score >= t. Ties are handled: the share is measured, not assumed."""
    s = np.sort(np.asarray(scores_normal, dtype=float))
    if len(s) == 0:
        raise ValueError("need at least one normal score to set a threshold")
    allowed = int(np.floor(target_fpr * len(s)))          # how many normals may be flagged
    if allowed == 0:
        return float(np.nextafter(s[-1], np.inf))           # flag nothing among the normals we saw
    t = s[len(s) - allowed]                                  # the allowed-th largest score
    if (s >= t).sum() > allowed:                             # ties push us over the budget: move just above them
        t = float(np.nextafter(t, np.inf))
    return float(t)


def rates(y, scores, threshold: float) -> dict:
    y = np.asarray(y)
    flag = np.asarray(scores) >= threshold
    pos, neg = y == 1, y == 0
    return {
        "recall": float(flag[pos].mean()) if pos.any() else float("nan"),
        "false_alarm_rate": float(flag[neg].mean()) if neg.any() else float("nan"),
        "precision": float(pos[flag].mean()) if flag.any() else float("nan"),
    }


def ranking(y, scores) -> dict:
    y = np.asarray(y)
    if y.min() == y.max():
        return {"auc": float("nan"), "ap": float("nan")}
    return {"auc": float(roc_auc_score(y, scores)), "ap": float(average_precision_score(y, scores))}
