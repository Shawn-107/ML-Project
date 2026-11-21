"""Label-free threshold adaptation (OPTIONAL, use with care).
Idea: the alert threshold is set on a validation set. If your own traffic looks different, the false-alarm rate drifts away from the target.
A domain classifier learns how your unlabelled flows differ from the validation flows; validation Normal flows that look like your traffic
are weighted up, and the threshold is re-set on that weighted set. No labels of yours are used.
Tested (final/reports/adapt_unsw.json, adapt_shift_more.json): false alarms moved closer to the target on UNSW and on one of two CIC test sets,
did nothing on NSL-KDD and made one CIC set worse (new attack types look like a shift too). It is therefore opt-in and never applied silently."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

MIN_FLOWS = 500
MAX_TARGET = 20000


def shift_weights(Xs, Xt, seed=0):
    """-> (weights for the rows of Xs, domain-classifier AUC; 0.5 means the two sets look alike)"""
    Z = np.vstack([Xs, Xt])
    dom = np.r_[np.zeros(len(Xs)), np.ones(len(Xt))]
    idx = np.random.RandomState(seed).permutation(len(Z))
    p = np.empty(len(Z))
    half = len(Z) // 2
    for a, c in ((idx[:half], idx[half:]), (idx[half:], idx[:half])):      # 2-fold cross-fitting, so a flow is never scored by a model that saw it
        p[c] = HistGradientBoostingClassifier(max_iter=150, max_depth=5, random_state=seed).fit(Z[a], dom[a]).predict_proba(Z[c])[:, 1]
    ps = np.clip(p[:len(Xs)], 1e-4, 1 - 1e-4)
    w = ps / (1 - ps) * (len(Xs) / len(Xt))
    return np.minimum(w, np.quantile(w, 0.99)), float(roc_auc_score(dom, p))


def weighted_threshold(scores, w, target):
    """score above which a weighted share `target` of the (Normal) flows lies"""
    o = np.argsort(scores)
    cs = np.cumsum(w[o]) / w.sum()
    k = np.searchsorted(cs, 1 - target)
    return float(np.asarray(scores)[o][min(k, len(o) - 1)])
