"""Training pipeline and model-bundle persistence."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from .config import FEATURES, LABEL, MODEL_DIR, SCHEMA_VERSION, Config, policy_name
from .data import Splits, load_splits
from .evaluate import binary_metrics
from .features import build_preprocessor, dense, normal_profile
from .models import AnnEnsemble, ForestModel, NoveltyAutoencoder
from .thresholds import f1_threshold, fpr_threshold

log = logging.getLogger(__name__)
DEFAULT_BUNDLE = MODEL_DIR / "nids_bundle.joblib"


def blend(ann_scores, rf_scores, w: float):
    return w * np.asarray(ann_scores) + (1.0 - w) * np.asarray(rf_scores)


@dataclass
class Bundle:
    preprocessor: object
    ann: AnnEnsemble
    rf: ForestModel
    autoencoder: NoveltyAutoencoder
    blend_weight: float
    thresholds: dict
    novelty_threshold: float
    reference_scores: np.ndarray
    profile: dict
    config: dict
    meta: dict
    background: pd.DataFrame  # sample of Normal training flows (raw features) used as the explanation baseline


def train(cfg: Config | None = None, splits: Splits | None = None) -> Bundle:
    cfg = cfg or Config()
    splits = splits or load_splits(cfg)
    tr, val = splits.train, splits.val
    log.info("fitting preprocessor on %d training rows", len(tr))
    pre = build_preprocessor()
    Xtr, Xv = pre.fit_transform(tr[FEATURES]), pre.transform(val[FEATURES])  # fitted on TRAIN only
    ytr, yv = tr[LABEL].to_numpy(), val[LABEL].to_numpy()

    log.info("training ANN ensemble (%d seeds)", cfg.ann_seeds)
    ann = AnnEnsemble(cfg.ann_hidden, cfg.ann_alpha, cfg.ann_seeds, cfg.ann_max_iter, cfg.seed).fit(Xtr, ytr)
    log.info("training random forest")
    rf = ForestModel(cfg.rf_trees, cfg.rf_min_leaf, cfg.seed).fit(Xtr, ytr)
    sv = blend(ann.predict_proba(Xv), rf.predict_proba(Xv), cfg.blend_weight)

    thresholds = {policy_name(t): fpr_threshold(sv, yv, t) for t in cfg.fpr_targets}
    thresholds["f1"] = f1_threshold(sv, yv)
    if cfg.default_policy not in thresholds:
        raise ValueError(f"default_policy {cfg.default_policy!r} not in {sorted(thresholds)}")

    log.info("training novelty autoencoder on Normal flows")
    ae = NoveltyAutoencoder(cfg.ae_hidden, cfg.ae_max_iter, cfg.seed).fit(dense(Xtr)[ytr == 0])
    novelty_threshold = fpr_threshold(ae.score(dense(Xv)), yv, cfg.novelty_fpr)

    rng = np.random.RandomState(cfg.seed)
    ref = sv[rng.choice(len(sv), min(5000, len(sv)), replace=False)]
    val_metrics = binary_metrics(yv, sv >= thresholds[cfg.default_policy])
    meta = {
        "schema_version": SCHEMA_VERSION,
        "trained_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "sklearn": sklearn.__version__,
        "numpy": np.__version__, "pandas": pd.__version__,
        "data_info": splits.info, "validation_metrics": val_metrics,
        "default_policy": cfg.default_policy, "features": FEATURES,
    }
    normal_tr = tr[tr[LABEL] == 0]
    background = normal_tr[FEATURES].sample(min(cfg.background_size, len(normal_tr)), random_state=cfg.seed)
    return Bundle(pre, ann, rf, ae, cfg.blend_weight, thresholds, novelty_threshold, ref,
                  normal_profile(normal_tr), cfg.to_dict(), meta, background.reset_index(drop=True))


# ---- persistence ----------------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _meta_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def save_bundle(bundle: Bundle, path: str | Path = DEFAULT_BUNDLE) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    joblib.dump(bundle, tmp, compress=3)
    tmp.replace(path)  # atomic: a crash never leaves a half-written model
    meta = dict(bundle.meta, sha256=_sha256(path), thresholds=bundle.thresholds,
                novelty_threshold=bundle.novelty_threshold)
    _meta_path(path).write_text(json.dumps(meta, indent=2, default=float))
    return path


def load_bundle(path: str | Path = DEFAULT_BUNDLE, verify: bool = True) -> Bundle:
    """SECURITY: bundles are pickles. Only load files you trained yourself. The sha256 check detects
    accidental corruption or a file swapped after training; it is not a signature."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model bundle not found: {path} (run `python -m nids train` first)")
    mp = _meta_path(path)
    if verify:
        if not mp.exists():
            raise RuntimeError(f"missing {mp.name}; cannot verify bundle integrity (pass verify=False to skip)")
        expected = json.loads(mp.read_text()).get("sha256")
        if expected != _sha256(path):
            raise RuntimeError("bundle checksum mismatch: file is corrupted or was modified after training")
    bundle = joblib.load(path)
    if bundle.meta.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"bundle schema v{bundle.meta.get('schema_version')} != code schema v{SCHEMA_VERSION}")
    if bundle.meta.get("sklearn") != sklearn.__version__:
        log.warning("bundle trained with scikit-learn %s, running %s; retrain if results look odd",
                    bundle.meta.get("sklearn"), sklearn.__version__)
    return bundle
