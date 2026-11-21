"""Threshold tuning on LABELLED flows from the user's own network (recalibration)."""
import numpy as np


def labelled_threshold(scores, y, fpr_cap: float = 0.08, min_flows: int = 200, min_per_class: int = 20) -> float:
    """Best-accuracy threshold subject to a false-alarm cap on the labelled flows."""
    scores, y = np.asarray(scores, float), np.asarray(y)
    n0, n1 = int((y == 0).sum()), int((y == 1).sum())
    if len(y) < min_flows or n0 < min_per_class or n1 < min_per_class:
        raise ValueError(f"need >= {min_flows} labelled flows with >= {min_per_class} of each class "
                         f"(got {len(y)}: {n0} normal, {n1} attack)")
    best_t, best_acc = None, -1.0
    for t in np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 401))):
        pred = scores >= t
        if (pred & (y == 0)).sum() / n0 > fpr_cap:
            continue
        acc = float((pred == (y == 1)).mean())
        if acc > best_acc:
            best_t, best_acc = float(t), acc
    if best_t is None:                      # even the strictest candidate exceeds the cap -> flag (almost) nothing
        best_t = float(np.nextafter(scores.max(), np.inf))
    return best_t
