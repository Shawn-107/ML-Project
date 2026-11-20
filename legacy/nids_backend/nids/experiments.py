"""Reproducible experiments: unseen-attack test, repeated-run stability, recalibration sample-size curve."""
from __future__ import annotations

import dataclasses
import logging
from typing import Callable

import numpy as np
import pandas as pd

from .config import ATTACK_CAT, FEATURES, LABEL, NORMAL, Config, policy_name
from .data import Splits, load_splits
from .evaluate import binary_metrics, evaluate
from .features import build_preprocessor, dense
from .inference import Detector
from .models import AnnEnsemble, ForestModel, NoveltyAutoencoder
from .pipeline import blend, train
from .thresholds import fpr_threshold, labelled_threshold

log = logging.getLogger(__name__)


def _fpr_target(cfg: Config, policy: str) -> float:
    for t in cfg.fpr_targets:
        if policy_name(t) == policy:
            return t
    raise ValueError(f"policy {policy!r} is not a false-alarm-target policy of this config")


def leave_one_attack_out(cfg: Config, splits: Splits, attacks: list[str] | None = None,
                         seen_detector: Detector | None = None, policy: str | None = None,
                         progress: Callable[[str], None] | None = None) -> dict:
    """For each attack family A: retrain WITHOUT A (preprocessing, models, thresholds and novelty model all
    see no A flow), then measure how many official-test flows of A are caught. Compare with a model that
    was trained on A ('seen', optional) and with the autoencoder novelty layer."""
    policy = policy or cfg.default_policy
    target = _fpr_target(cfg, policy)
    say = progress or (lambda m: log.info(m))
    present = sorted(set(splits.test[ATTACK_CAT]) - {NORMAL})
    attacks = attacks or present
    unknown = [a for a in attacks if a not in present]
    if unknown:
        raise ValueError(f"attack families not in the test set: {unknown}; available: {present}")
    normal_te = splits.test[splits.test[LABEL] == 0]
    folds = {}
    for A in attacks:
        say(f"held-out attack: {A}")
        tr, va = splits.train[splits.train[ATTACK_CAT] != A], splits.val[splits.val[ATTACK_CAT] != A]
        te_a = splits.test[splits.test[ATTACK_CAT] == A]
        pre = build_preprocessor()
        Xtr, Xva = pre.fit_transform(tr[FEATURES]), pre.transform(va[FEATURES])
        ann = AnnEnsemble(cfg.ann_hidden, cfg.ann_alpha, cfg.loao_seeds, cfg.ann_max_iter, cfg.seed + 100).fit(Xtr, tr[LABEL])
        rf = ForestModel(cfg.rf_trees, cfg.rf_min_leaf, cfg.seed).fit(Xtr, tr[LABEL])
        thr = fpr_threshold(blend(ann.predict_proba(Xva), rf.predict_proba(Xva), cfg.blend_weight), va[LABEL], target)
        ae = NoveltyAutoencoder(cfg.ae_hidden, cfg.ae_max_iter, cfg.seed).fit(dense(Xtr)[tr[LABEL].to_numpy() == 0])
        nthr = fpr_threshold(ae.score(dense(Xva)), va[LABEL], cfg.novelty_fpr)

        def flags(df):
            X = pre.transform(df[FEATURES])
            return blend(ann.predict_proba(X), rf.predict_proba(X), cfg.blend_weight) >= thr, ae.score(dense(X)) >= nthr

        c_a, n_a = flags(te_a)
        c_n, n_n = flags(normal_te)
        fold = {"n_test_attack": int(len(te_a)), "detect_classifier": float(c_a.mean()), "detect_novelty": float(n_a.mean()),
                "detect_combined": float((c_a | n_a).mean()), "false_alarm_classifier": float(c_n.mean()),
                "false_alarm_combined": float((c_n | n_n).mean())}
        if seen_detector is not None:
            sc = seen_detector.score_frame(te_a)["blend_score"].to_numpy()
            fold["detect_when_seen"] = float((sc >= seen_detector.bundle.thresholds[policy]).mean())
        folds[A] = fold
    keys = ["detect_classifier", "detect_novelty", "detect_combined", "false_alarm_classifier", "false_alarm_combined"]
    summary = {k: float(np.mean([f[k] for f in folds.values()])) for k in keys}
    if seen_detector is not None:
        summary["detect_when_seen"] = float(np.mean([f["detect_when_seen"] for f in folds.values()]))
    return {"policy": policy, "false_alarm_target": target, "novelty_false_alarm_target": cfg.novelty_fpr,
            "ann_seeds_per_fold": cfg.loao_seeds, "folds": folds, "summary_mean_over_folds": summary}


def stability(cfg: Config, n_repeats: int = 3, policy: str | None = None,
              progress: Callable[[str], None] | None = None) -> dict:
    """Retrain with different seeds / validation splits and report mean and spread of the headline metrics."""
    if n_repeats < 2:
        raise ValueError("n_repeats must be >= 2")
    say = progress or (lambda m: log.info(m))
    runs = []
    for r in range(n_repeats):
        say(f"repeat {r + 1}/{n_repeats}")
        c = dataclasses.replace(cfg, seed=cfg.seed + r)
        sp = load_splits(c)
        bundle = train(c, sp)
        det = Detector(bundle, policy=policy)
        t = evaluate(det, sp.test)
        runs.append({"seed": c.seed, "validation": bundle.meta["validation_metrics"],
                     "test": {k: t[k] for k in ("accuracy", "balanced_accuracy", "f1", "recall", "false_alarm_rate")}})
    keys = ("accuracy", "f1", "recall", "false_alarm_rate")
    summ = {part: {k: {"mean": float(np.mean([x[part][k] for x in runs])), "std": float(np.std([x[part][k] for x in runs]))}
                   for k in keys} for part in ("validation", "test")}
    return {"policy": policy or cfg.default_policy, "n_repeats": n_repeats, "runs": runs, "summary": summ}


def recalibration_curve(detector: Detector, test: pd.DataFrame, sizes=(50, 100, 200, 500, 1000, 3000),
                        draws: int = 30, fpr_cap: float = 0.08, seed: int = 0) -> dict:
    """Accuracy on the remaining target flows when the threshold is tuned on n labelled target flows."""
    s = detector.score_frame(test)["blend_score"].to_numpy()
    y = test[LABEL].to_numpy().astype(int)
    base = binary_metrics(y, s >= detector.bundle.thresholds[detector.policy])
    rng = np.random.RandomState(seed)
    rows = []
    for n in sizes:
        if n >= len(y) - 100:
            continue
        accs, recs, fars = [], [], []
        for _ in range(draws):
            ix = rng.choice(len(y), n, replace=False)
            try:
                t = labelled_threshold(s[ix], y[ix], fpr_cap, min_flows=min(n, 200), min_per_class=5)
            except ValueError:
                continue
            rest = np.setdiff1d(np.arange(len(y)), ix)
            m = binary_metrics(y[rest], s[rest] >= t)
            accs.append(m["accuracy"]); recs.append(m["recall"]); fars.append(m["false_alarm_rate"])
        if accs:
            rows.append({"n_labelled": int(n), "draws_used": len(accs),
                         "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs))},
                         "recall": {"mean": float(np.mean(recs)), "std": float(np.std(recs))},
                         "false_alarm_rate": {"mean": float(np.mean(fars)), "std": float(np.std(fars))}})
    return {"policy": detector.policy, "fpr_cap": fpr_cap, "no_target_labels": {
        k: base[k] for k in ("accuracy", "recall", "false_alarm_rate")}, "curve": rows}
