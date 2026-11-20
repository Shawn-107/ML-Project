"""Candidate improvements for the flow classifier. Each is a small wrapper with fit / score (attack score in [0, 1])
so they can be blended, stacked and compared on identical splits.

  HGB            - gradient boosting (sklearn HistGradientBoosting), binary attack/normal.
  MultiClass     - 10-way classifier (Normal + 9 attack families) collapsed to attack score = 1 - P(Normal).
  LogitStacker   - logistic regression over the logit of several model scores (a LEARNED blend instead of 0.5/0.5).
  IsoNovelty     - isolation forest fitted on Normal flows only; high score = unlike Normal.
  KMeansNovelty  - distance to the nearest K-Means centre of Normal flows (k chosen by silhouette, syllabus Unit 3).
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score


def _logit(p, eps: float = 1e-4):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class HGB:
    name = "gb"

    def __init__(self, seed: int = 0, max_iter: int = 300, lr: float = 0.08, l2: float = 1.0,
                 max_leaf_nodes: int = 31, min_samples_leaf: int = 20):
        self.m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=lr, l2_regularization=l2,
                                                max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                                early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                                random_state=seed)

    def fit(self, X, y, cats=None, w=None):
        self.m.fit(X, y, sample_weight=w)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class RF:
    """Same settings as the legacy forest (150 trees, min leaf 2, balanced classes), but accepts sample weights."""
    name = "rf"

    def __init__(self, seed: int = 0, trees: int = 150):
        self.m = RandomForestClassifier(trees, min_samples_leaf=2, class_weight="balanced", n_jobs=2, random_state=seed)

    def fit(self, X, y, cats=None, w=None):
        self.m.fit(X, y, sample_weight=w)
        return self

    def score(self, X):
        return self.m.predict_proba(X)[:, 1]


class MultiClass:
    """kind='hgb' or 'rf'. Trained on the attack family labels; the attack score is 1 - P(Normal)."""

    def __init__(self, kind: str = "hgb", seed: int = 0, normal: str = "Normal"):
        self.kind, self.normal = kind, normal
        self.name = f"{kind}_multi"
        if kind == "hgb":
            self.m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, l2_regularization=1.0,
                                                    early_stopping=True, validation_fraction=0.1,
                                                    n_iter_no_change=15, random_state=seed)
        elif kind == "rf":
            self.m = RandomForestClassifier(150, min_samples_leaf=2, class_weight="balanced_subsample",
                                            n_jobs=2, random_state=seed)
        else:
            raise ValueError(kind)

    def fit(self, X, y=None, cats=None):
        if cats is None:
            raise ValueError("MultiClass needs the attack-family labels (cats)")
        self.m.fit(X, cats)
        self.i_normal = list(self.m.classes_).index(self.normal)
        return self

    def score(self, X):
        return 1.0 - self.m.predict_proba(X)[:, self.i_normal]

    def family_proba(self, X):
        return dict(zip(self.m.classes_, self.m.predict_proba(X).T))


class LogitStacker:
    """Learned combination of base-model scores. Fit on rows the base models did NOT train on."""
    name = "stack"

    def __init__(self, C: float = 1.0):
        self.m = LogisticRegression(C=C, max_iter=1000)

    def fit(self, S, y):
        self.m.fit(_logit(S), y)
        return self

    def score(self, S):
        return self.m.predict_proba(_logit(S))[:, 1]

    @property
    def weights(self):
        return self.m.coef_.ravel().tolist(), float(self.m.intercept_[0])


class IsoNovelty:
    name = "isoforest"

    def __init__(self, seed: int = 0, trees: int = 200):
        self.m = IsolationForest(n_estimators=trees, max_samples=4096, random_state=seed, n_jobs=2)

    def fit(self, X_normal):
        self.m.fit(X_normal)
        return self

    def score(self, X):
        return -self.m.score_samples(X)          # higher = more anomalous


class KMeansNovelty:
    name = "kmeans"

    def __init__(self, ks=(4, 8, 16, 32), seed: int = 0, sil_sample: int = 5000):
        self.ks, self.seed, self.sil_sample = ks, seed, sil_sample
        self.k_ = None

    def fit(self, X_normal):
        rng = np.random.default_rng(self.seed)
        sub = X_normal[rng.choice(len(X_normal), min(self.sil_sample, len(X_normal)), replace=False)]
        best = (-2.0, None)
        self.silhouettes_ = {}
        for k in self.ks:
            km = MiniBatchKMeans(k, random_state=self.seed, n_init=3, batch_size=2048).fit(X_normal)
            s = silhouette_score(sub, km.predict(sub))
            self.silhouettes_[k] = float(s)
            if s > best[0]:
                best = (s, km)
        self.k_, self.km = best[1].n_clusters, best[1]
        return self

    def score(self, X):
        return self.km.transform(X).min(axis=1)
