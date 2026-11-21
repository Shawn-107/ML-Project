"""Train the flow model on the cleaned UNSW-NB15 official training data (legacy `nids` splits), cluster-grouped validation.

   python -m layered_nids.flowmodel.fit        # saves models/csv/unsw_flow_model.joblib (+ .meta.json with SHA-256)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from layered_nids.flowmodel.bundle import DEFAULT_PATH, save_bundle
from layered_nids.flowmodel.contract import FEATURES
from layered_nids.flowmodel.split import grouped_split
from layered_nids.flowmodel.train import train


def load_frames():
    """(train, validation, test) frames: legacy cleaned splits; train+validation are re-split by whole K-Means clusters."""
    from nids.config import BINARY, CATEGORICAL, CONTINUOUS, Config
    from nids.data import load_splits
    sp = load_splits(Config())
    for d in (sp.train, sp.val, sp.test):
        d["y"] = d["label"]
    tr, va = grouped_split(pd.concat([sp.train, sp.val], ignore_index=True), CATEGORICAL, CONTINUOUS + BINARY)
    return tr, va, sp.test.reset_index(drop=True)


def fit_and_save(path=DEFAULT_PATH, seed: int = 0):
    t0 = time.time()
    tr, va, te = load_frames()
    b = train(tr, va, seed=seed)
    save_bundle(b, path)
    from layered_nids.flowmodel.models import p1
    p = p1(b.model, b.prep.transform(te[FEATURES]))
    y = te["y"].to_numpy()
    print(f"trained in {time.time() - t0:.0f}s on {len(tr):,} rows (validation {len(va):,}); test AUC {roc_auc_score(y, p):.4f}")
    for name, thr in b.thresholds.items():
        f = p >= thr
        print(f"  {name}: threshold {thr:.3f} | accuracy {(f == (y == 1)).mean():.3f} recall {f[y == 1].mean():.3f} false alarms {f[y == 0].mean():.3f}")
    return b


if __name__ == "__main__":
    fit_and_save()
