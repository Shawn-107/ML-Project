"""Alert-threshold policies. A score >= threshold is flagged as an attack."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def fpr_threshold(scores, y, target_fpr: float) -> float:
    """Threshold so that about `target_fpr` of Normal flows (y==0) are flagged."""
    scores, y = np.asarray(scores, float), np.asarray(y)
    neg = np.sort(scores[y == 0])
    if len(neg) == 0:
        raise ValueError("need Normal flows to set a false-alarm threshold")
    k = int(np.clip((1.0 - target_fpr) * len(neg), 0, len(neg) - 1))
    return float(neg[k])


def f1_threshold(scores, y) -> float:
    p, r, t = precision_recall_curve(np.asarray(y), np.asarray(scores, float))
    f = 2 * p * r / (p + r + 1e-12)
    return float(t[int(np.argmax(f[:-1]))])


def labelled_threshold(scores, y, fpr_cap: float = 0.08, min_flows: int = 200, min_per_class: int = 20) -> float:
    """Best-accuracy threshold subject to a false-alarm cap, tuned on LABELLED flows from the target network."""
    scores, y = np.asarray(scores, float), np.asarray(y)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if len(y) < min_flows or n0 < min_per_class or n1 < min_per_class:
        raise ValueError(f"need >= {min_flows} labelled flows with >= {min_per_class} of each class "
                         f"(got {len(y)}: {n0} normal, {n1} attack)")
    cands = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401)))
    best_t, best_acc = None, -1.0
    for t in cands:
        pred = scores >= t
        fpr = (pred & (y == 0)).sum() / n0
        if fpr > fpr_cap:
            continue
        acc = (pred == (y == 1)).mean()
        if acc > best_acc:
            best_t, best_acc = float(t), float(acc)
    if best_t is None:  # even the strictest candidate exceeds the cap -> flag (almost) nothing
        best_t = float(np.nextafter(scores.max(), np.inf))
    return best_t
