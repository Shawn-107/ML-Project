"""Leak-safe train/validation split: K-Means groups the flows and WHOLE clusters go to validation, so a near-copy of a training
flow cannot sit in validation. Thresholds set there give false alarms close to the target (owner's earlier project)."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from layered_nids.flowmodel.kmeans_np import KMeansNP


class Prep:
    """One-hot for categoricals, signed log + standard scaling for numbers. Fitted on the rows given."""

    def __init__(self, cat, num):
        self.cat, self.num = list(cat), list(num)

    def _n(self, df):
        x = df[self.num].to_numpy(float)
        return np.sign(x) * np.log1p(np.abs(x))

    def _c(self, df):
        return df[self.cat].astype(str).apply(lambda s: s.str.strip().str.lower())

    def fit(self, df):
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(self._c(df)) if self.cat else None
        self.sc = StandardScaler().fit(self._n(df))
        return self

    def transform(self, df):
        parts = [self.sc.transform(self._n(df))]
        if self.oh is not None:
            parts.insert(0, self.oh.transform(self._c(df)))
        return np.hstack(parts).astype(np.float32)


def grouped_split(df, cat, num, label="y", frac=0.2, k=60, seed=0):
    """Returns (train, val). `label` is the 0/1 column."""
    rng = np.random.RandomState(seed)
    Z = Prep(cat, num).fit(df).transform(df)
    km = KMeansNP(k, seed=seed).fit(Z[rng.permutation(len(Z))[:20000]])
    g = km.predict(Z)
    inval, tot = np.zeros(len(df), bool), 0
    for c in rng.permutation(k):
        m = g == c
        if tot + m.sum() > frac * len(df) * 1.15:
            continue
        inval |= m
        tot += int(m.sum())
        if tot >= frac * len(df):
            break
    return df[~inval].reset_index(drop=True), df[inval].reset_index(drop=True)
