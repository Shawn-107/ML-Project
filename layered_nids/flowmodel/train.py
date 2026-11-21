"""Training: fit on the training split, set thresholds on the held-out-cluster validation split (never on test data)."""
import datetime as dt
import platform

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import roc_auc_score

from .bundle import Bundle
from .contract import FEATURES, NUMERIC, POLICIES, DEFAULT_POLICY, NOVELTY_TARGET, SCHEMA_VERSION
from .features import Preprocessor
from .kmeans_np import KMeansNP, silhouette
from .models import make_hgb, p1


def fpr_threshold(scores, y, target):
    neg = np.sort(np.asarray(scores)[np.asarray(y) == 0])
    return float(neg[int(np.clip((1 - target) * len(neg), 0, len(neg) - 1))])


def jitter(df, sigma, seed):
    """multiplicative log-normal measurement noise on the numeric inputs (robustness augmentation)"""
    out = df.copy()
    out[NUMERIC] = out[NUMERIC].to_numpy(float) * np.exp(np.random.RandomState(seed).normal(0, sigma, (len(out), len(NUMERIC))))
    return out


VOLUME = ["sbytes", "dbytes", "spkts", "dload"]    # traffic-volume inputs (packet sizes stay the same when a network is busier or quieter)


def scale_volume(df, factor):
    out = df.copy()
    out[VOLUME] = out[VOLUME].to_numpy(float) * factor
    return out


def train(tr: pd.DataFrame, va: pd.DataFrame, seed=0, fast=False, label="y", noise=None, volume=None) -> Bundle:
    """tr/va: frames with the FEATURES columns (lower-case categoricals) and a 0/1 label column.
    noise : optional tuple of sigmas; for each, a jittered copy of half the training rows is added to the fitting data.
    volume: optional tuple of factors; for each, a copy of half the training rows with the volume inputs multiplied by it is added."""
    pre = Preprocessor().fit(tr[FEATURES])
    extra = [jitter(tr.sample(frac=0.5, random_state=i), s, i) for i, s in enumerate(noise or ())]
    extra += [scale_volume(tr.sample(frac=0.5, random_state=100 + i), f) for i, f in enumerate(volume or ())]
    fit_df = pd.concat([tr] + extra, ignore_index=True) if extra else tr
    Xtr, Xv = pre.transform(fit_df[FEATURES]), pre.transform(va[FEATURES])
    ytr, yv = fit_df[label].to_numpy(), va[label].to_numpy()
    model = make_hgb(seed, fast).fit(Xtr, ytr)
    sv = p1(model, Xv)
    thresholds = {p: fpr_threshold(sv, yv, t) for p, t in POLICIES.items()}
    # K-Means novelty: a separate "unusual traffic" flag, not part of the alert decision
    Zn = Xtr[ytr == 0]
    Zn = Zn[np.random.RandomState(seed).permutation(len(Zn))[:6000]]
    sil = {k: silhouette(Zn, KMeansNP(k, seed=seed).fit(Zn).predict(Zn), n=1500) for k in ((3,) if fast else (4, 8, 12))}
    km = KMeansNP(max(sil, key=sil.get), seed=seed).fit(Zn)
    nov_thr = fpr_threshold(km.distance(Xv), yv, NOVELTY_TARGET)
    meta = dict(schema_version=SCHEMA_VERSION, model="gradient boosting (HistGradientBoosting)", trained_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                python=platform.python_version(), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
                features=FEATURES, default_policy=DEFAULT_POLICY, n_train=len(tr), n_val=len(va), fast=fast,
                noise_augmentation=list(noise) if noise else None, volume_augmentation=list(volume) if volume else None, n_trees=int(model.n_iter_),
                kmeans_k=int(km.k), kmeans_silhouette={int(k): float(v) for k, v in sil.items()},
                val_auc=float(roc_auc_score(yv, sv)), train_auc=float(roc_auc_score(ytr[:30000], p1(model, Xtr[:30000]))))
    bg = tr[tr[label] == 0][FEATURES].sample(min(200, int((tr[label] == 0).sum())), random_state=seed).reset_index(drop=True)
    ref = dict(X=Xv.astype(np.float32), normal=(yv == 0), scores=sv.astype(np.float32))   # validation reference for Detector.adapt (no labels are kept beyond the normal flag)
    return Bundle(pre, model, km, thresholds, nov_thr, bg, meta, ref)
