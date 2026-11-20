"""Model wrappers: ANN ensemble, random forest, normal-traffic autoencoder (novelty)."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier, MLPRegressor


class AnnEnsemble:
    """Average of several MLPs trained with different seeds (more stable than a single network)."""

    def __init__(self, hidden=(128, 64), alpha=1e-4, seeds=5, max_iter=40, base_seed=0):
        self.hidden, self.alpha, self.seeds, self.max_iter, self.base_seed = hidden, alpha, seeds, max_iter, base_seed
        self.models_: list[MLPClassifier] = []

    def fit(self, X, y):
        self.models_ = [
            MLPClassifier(hidden_layer_sizes=self.hidden, alpha=self.alpha, max_iter=self.max_iter,
                          early_stopping=True, n_iter_no_change=5, random_state=self.base_seed + s).fit(X, y)
            for s in range(self.seeds)
        ]
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Attack probability, shape (n,)."""
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models_], axis=0)


class ForestModel:
    def __init__(self, trees=150, min_leaf=2, seed=0):
        self.model = RandomForestClassifier(trees, min_samples_leaf=min_leaf, class_weight="balanced",
                                            n_jobs=2, random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def set_single_row_mode(self):
        """Thread start-up costs dominate for one-row calls; use one job when serving."""
        self.model.n_jobs = 1


class NoveltyAutoencoder:
    """Trained on Normal traffic only; a high reconstruction error means 'unlike normal traffic'."""

    def __init__(self, hidden=(64, 16, 64), max_iter=30, seed=0):
        self.net = MLPRegressor(hidden_layer_sizes=hidden, max_iter=max_iter, early_stopping=True,
                                n_iter_no_change=4, random_state=seed)

    def fit(self, X_normal):
        self.net.fit(X_normal, X_normal)
        return self

    def score(self, X) -> np.ndarray:
        return ((self.net.predict(X) - X) ** 2).mean(axis=1)
