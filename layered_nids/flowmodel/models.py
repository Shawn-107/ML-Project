"""The model: gradient boosting (HistGradientBoosting). Settings were chosen on the held-out-cluster validation
(final/reports/leakfree_unsw.json); in the comparison of all candidates it was the best single model across datasets."""
from sklearn.ensemble import HistGradientBoostingClassifier


def make_hgb(seed=0, fast=False):
    return HistGradientBoostingClassifier(max_iter=30 if fast else 300, learning_rate=0.05, max_leaf_nodes=15,
                                          l2_regularization=5.0, min_samples_leaf=20 if fast else 100,
                                          class_weight="balanced", early_stopping=True, random_state=seed)


def p1(model, X):
    return model.predict_proba(X)[:, 1]
