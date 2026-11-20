"""Metrics and reports."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ATTACK_CAT, FEATURES, LABEL, NORMAL


def binary_metrics(y, pred) -> dict:
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    rec = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); prec = tp / max(tp + fp, 1)
    return dict(n=len(y), accuracy=(tp + tn) / max(len(y), 1), balanced_accuracy=(rec + spec) / 2,
                f1=2 * prec * rec / max(prec + rec, 1e-12), precision=prec, recall=rec,
                false_alarm_rate=1 - spec, tp=tp, tn=tn, fp=fp, fn=fn)


def per_class_flag_rate(attack_cat, pred) -> dict:
    """Share of flows flagged as attack, per attack family. For 'Normal' this is the false-alarm rate."""
    df = pd.DataFrame({"c": np.asarray(attack_cat), "p": np.asarray(pred).astype(int)})
    g = df.groupby("c").p.agg(["size", "mean"])
    return {k: {"n": int(r["size"]), "flagged": float(r["mean"])} for k, r in g.iterrows()}


def calibration_metrics(y, prob, bins: int = 10) -> dict:
    """Brier score and expected calibration error of attack probabilities (equal-width bins)."""
    y, prob = np.asarray(y).astype(float), np.asarray(prob).astype(float)
    if len(y) == 0:
        raise ValueError("no flows to evaluate")
    idx = np.minimum((prob * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - prob[m].mean())
    return {"brier": float(np.mean((prob - y) ** 2)), "ece": float(ece)}


def evaluate(detector, df: pd.DataFrame) -> dict:
    """Evaluate a Detector on a labelled table (e.g. the official test set)."""
    out = detector.predict_frame(df, update_monitor=False)
    pred = out["is_attack"].to_numpy()
    res = binary_metrics(df[LABEL], pred)
    res["calibration"] = calibration_metrics(df[LABEL], out["attack_probability"].to_numpy())
    res["per_class"] = per_class_flag_rate(df[ATTACK_CAT], pred)
    res["policy"], res["threshold"], res["use_novelty"] = detector.policy, detector.threshold, detector.use_novelty
    return res


def benchmark_latency(detector, df: pd.DataFrame, n: int = 200) -> dict:
    rows = [r for r in df[FEATURES].head(n).to_dict("records")]
    detector.predict(rows[:5])  # warm-up
    t = time.perf_counter()
    for r in rows:
        detector.predict([r])
    single = (time.perf_counter() - t) / len(rows) * 1000
    t = time.perf_counter()
    detector.predict(rows)
    batch = (time.perf_counter() - t) / len(rows) * 1000
    return {"single_flow_ms": single, "batched_ms_per_flow": batch, "n": len(rows)}


def save_report(report: dict, name: str, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / name
    p.write_text(json.dumps(report, indent=2, default=float))
    return p
