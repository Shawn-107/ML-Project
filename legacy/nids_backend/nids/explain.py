"""Per-flow explanations: Monte-Carlo estimate of Shapley values for the detector's attack probability.

For one flow x and a set of typical Normal flows (the background), the Shapley value of a feature is its average
marginal effect on the attack probability when features are switched, one at a time in random order, from a
background flow's values to x's values. This is the same quantity the `shap` library estimates (we implement it
directly because it works on the full detector, including the ANN + forest blend and the one-hot encoding).

Guarantee (tested): for every sampled path the contributions telescope, so
    sum(contributions) == probability(x) - baseline        (baseline = mean probability of the background paths)
holds exactly. Individual contributions are estimates; `std_err` says how noisy each one is."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .config import CATEGORICAL, FEATURES

MAX_PERMUTATIONS, MAX_BACKGROUND, MAX_PATHS = 200, 50, 1500


def shapley_contributions(score_fn: Callable[[pd.DataFrame], np.ndarray], flow: dict | pd.Series,
                          background: pd.DataFrame, n_permutations: int = 16, n_background: int = 6,
                          seed: int = 0) -> dict:
    if not (1 <= n_permutations <= MAX_PERMUTATIONS and 1 <= n_background <= MAX_BACKGROUND):
        raise ValueError(f"n_permutations must be 1..{MAX_PERMUTATIONS}, n_background 1..{MAX_BACKGROUND}")
    if n_permutations * n_background > MAX_PATHS:
        raise ValueError(f"n_permutations * n_background must be <= {MAX_PATHS}")
    if len(background) == 0:
        raise ValueError("empty background")
    nf = len(FEATURES)
    x = pd.Series(flow)[FEATURES].to_numpy(dtype=object)
    rng = np.random.RandomState(seed)
    bg_idx = rng.choice(len(background), n_background, replace=len(background) < n_background)
    bg = background[FEATURES].to_numpy(dtype=object)[bg_idx]                         # (B, F)
    paths = n_background * n_permutations
    base = np.repeat(bg, n_permutations, axis=0)                                       # (paths, F)
    rank = np.argsort(np.argsort(rng.rand(paths, nf), axis=1), axis=1) + 1             # step at which feature j switches
    steps = np.arange(nf + 1)
    switched = rank[:, None, :] <= steps[None, :, None]                                # (paths, F+1, F)
    data = np.where(switched, x[None, None, :], base[:, None, :]).reshape(-1, nf)
    df = pd.DataFrame(data, columns=FEATURES)
    for c in FEATURES:
        if c not in CATEGORICAL:
            df[c] = df[c].astype(float)
    s = np.asarray(score_fn(df), dtype=float).reshape(paths, nf + 1)                  # scores along every path
    ahead = np.take_along_axis(s, rank, axis=1)                                       # score after switching feature j
    behind = np.take_along_axis(s, rank - 1, axis=1)                                  # score just before
    contrib = ahead - behind                                                           # (paths, F)
    mean, se = contrib.mean(0), contrib.std(0, ddof=1) / np.sqrt(paths) if paths > 1 else np.zeros(nf)
    baseline, prob = float(s[:, 0].mean()), float(s[:, -1].mean())
    items = [{"feature": f, "value": (x[i].item() if hasattr(x[i], "item") else x[i]),
              "contribution": float(mean[i]), "std_err": float(se[i])} for i, f in enumerate(FEATURES)]
    items.sort(key=lambda d: -abs(d["contribution"]))
    return {"baseline": baseline, "probability": prob, "sum_contributions": float(mean.sum()),
            "n_paths": int(paths), "contributions": items}
