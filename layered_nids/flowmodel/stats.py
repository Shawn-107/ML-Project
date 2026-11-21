"""Statistical building blocks used by the studies in final/ (rigor_stats.py, conformal_study.py). Pure functions, tested in tests/test_studies.py."""
import numpy as np


def q_conformal(scores, alpha):
    """Split-conformal threshold: the ceil((n+1)(1-alpha))-th smallest calibration score. A new exchangeable score exceeds it with
    probability <= alpha. Returns +inf when n is too small for that level (flag nothing)."""
    s = np.sort(np.asarray(scores, float))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return s[k - 1] if 1 <= k <= n else np.inf


def q_weighted(scores, w, alpha, w_test=1.0):
    """Weighted conformal threshold (Tibshirani et al. 2019): weighted quantile of the calibration scores with `w_test` extra mass at +infinity
    (the test point). With equal weights and w_test equal to that weight it equals q_conformal."""
    scores, w = np.asarray(scores, float), np.asarray(w, float)
    o = np.argsort(scores)
    cs = np.cumsum(w[o]) / (w.sum() + w_test)
    i = np.searchsorted(cs, 1 - alpha)
    return scores[o][i] if i < len(scores) else np.inf


def resample_indices(groups, rng, n_rows, cluster=True):
    """One bootstrap resample of row indices. cluster=True resamples whole groups (clusters of near-identical flows) with replacement,
    cluster=False resamples rows. The random calls are the same ones the studies used, so seeds reproduce their numbers."""
    if cluster:
        return np.concatenate([groups[i] for i in rng.randint(0, len(groups), len(groups))])
    return rng.randint(0, n_rows, n_rows)


def percentile_ci(values, level=95):
    lo = (100 - level) / 2
    return [float(np.nanpercentile(values, lo)), float(np.nanpercentile(values, 100 - lo))]
