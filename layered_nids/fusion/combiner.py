"""Learned combiner (project brief, L5): logistic regression over per-host, per-time-bucket summaries of the evidence.

Features for (host, bucket): for every layer, the strongest evidence score in the bucket, the evidence weight
(-log(1-score)) summed over the bucket, and summed over the trailing `trail` buckets; plus how many layers were active.
Label: 1 if the bucket overlaps the host's attack interval, 0 for buckets of benign hosts or before an attack starts;
buckets after an attack ends are dropped (ambiguous).

Labels exist only in simulation (fusion/simulate.py), so this combiner is trained and tested on SIMULATED scenarios
whose layer scores are real held-out scores. It shows whether learning beats fixed noisy-OR at the same false-alarm
budget in that setting; it is untested on real joint traffic.
"""
from __future__ import annotations

import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from layered_nids.fusion.accumulator import Alert

LAYER_ORDER = ("dga", "flows")


def bucket_features(evidence, hosts, duration_s: float, bucket_s: float = 300.0, trail: int = 3, layers=LAYER_ORDER):
    """Return (X, keys) with keys[i] = (host, bucket_index)."""
    n_b = int(math.ceil(duration_s / bucket_s))
    idx = {h: i for i, h in enumerate(hosts)}
    L = len(layers)
    mx = np.zeros((len(hosts), n_b, L))
    wsum = np.zeros((len(hosts), n_b, L))
    for e in evidence:
        if e.layer not in layers or e.host not in idx:
            continue
        b = min(int(e.ts // bucket_s), n_b - 1)
        li = layers.index(e.layer)
        mx[idx[e.host], b, li] = max(mx[idx[e.host], b, li], e.score)
        wsum[idx[e.host], b, li] += -math.log(1.0 - min(e.score, 0.999))
    cs = np.cumsum(wsum, axis=1)
    trailing = cs - np.concatenate([np.zeros((len(hosts), trail, L)), cs[:, :-trail]], axis=1)[:, :n_b]
    active = ((trailing > 0).sum(axis=2, keepdims=True)).astype(float)
    X = np.concatenate([mx, wsum, trailing, active], axis=2).reshape(len(hosts) * n_b, -1)
    keys = [(h, b) for h in hosts for b in range(n_b)]
    return X, keys


def bucket_labels(keys, truth: dict, bucket_s: float = 300.0):
    """(y, keep_mask) per key. See module docstring for the labelling rule."""
    y = np.zeros(len(keys))
    keep = np.ones(len(keys), bool)
    for i, (h, b) in enumerate(keys):
        if h not in truth:
            continue
        s, e = truth[h]
        lo, hi = b * bucket_s, (b + 1) * bucket_s
        if hi > s and lo < e:
            y[i] = 1.0
        elif lo >= e:
            keep[i] = False
    return y, keep


class LearnedCombiner:
    def __init__(self, C: float = 1.0, bucket_s: float = 300.0, trail: int = 3, layers=LAYER_ORDER):
        self.bucket_s, self.trail, self.layers = bucket_s, trail, tuple(layers)
        self.scaler = StandardScaler()
        self.lr = LogisticRegression(C=C, max_iter=2000, class_weight="balanced")

    def fit(self, scenarios) -> "LearnedCombiner":
        Xs, ys = [], []
        for sc in scenarios:
            X, keys = bucket_features(sc.evidence, sc.hosts, sc.duration_s, self.bucket_s, self.trail, self.layers)
            y, keep = bucket_labels(keys, sc.truth, self.bucket_s)
            Xs.append(X[keep])
            ys.append(y[keep])
        X, y = np.vstack(Xs), np.concatenate(ys)
        self.lr.fit(self.scaler.fit_transform(X), y)
        return self

    def probabilities(self, sc):
        X, keys = bucket_features(sc.evidence, sc.hosts, sc.duration_s, self.bucket_s, self.trail, self.layers)
        return self.lr.predict_proba(self.scaler.transform(X))[:, 1], keys

    def alerts(self, sc, threshold: float, cooldown_s: float = 300.0) -> list:
        p, keys = self.probabilities(sc)
        last: dict = {}
        out = []
        for i in np.argsort([b for _, b in keys], kind="stable"):
            h, b = keys[i]
            if p[i] >= threshold:
                t = (b + 1) * self.bucket_s
                if t - last.get(h, -1e18) >= cooldown_s:
                    last[h] = t
                    out.append(Alert(h, t, float(p[i]), "high", {l: 0.0 for l in self.layers}, ("learned combiner",)))
        return out
