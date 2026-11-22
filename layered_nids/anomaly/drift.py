"""Drift-aware alert threshold: keeps the false-alarm rate near its target when traffic slowly changes after calibration.

A fixed threshold is the (1 - fpr) quantile of the CALIBRATION scores. Real traffic drifts, so later normal packets score higher and the
real false-alarm rate creeps above the target (measured: 1.7-3.5% against a 1% target). This module re-estimates the threshold as scoring goes on:
it tracks the (1 - fpr) quantile of RECENT scores, but

  - only scores below `cap` times the current threshold count as "recent normal" (an attack's scores are far above and cannot drag the threshold up);
  - the threshold stays between `lo` and `hi` times the calibrated one (a sustained attack cannot talk the detector into accepting itself);
  - nothing changes until `min_recent` normal-looking scores exist.

Packets are processed in order; the threshold used for packet i only depends on packets before i, so this works for live scoring too.
"""
from __future__ import annotations

import numpy as np


def adaptive_thresholds(scores, thr0: float, fpr: float = 0.01, window: int = 4000, every: int = 200, cap: float = 3.0,
                        lo: float = 0.7, hi: float = 2.5, min_recent: int = 800) -> np.ndarray:
    """One threshold per score (same length). `thr0` is the calibrated threshold."""
    scores = np.asarray(scores, dtype=np.float64)
    out = np.empty(len(scores))
    recent: list = []
    cur = float(thr0)
    for i, s in enumerate(scores):
        out[i] = cur
        if s < cap * cur:
            recent.append(s)
        if len(recent) >= min_recent and i % every == 0:
            cur = float(np.clip(np.quantile(recent[-window:], 1.0 - fpr), lo * thr0, hi * thr0))
    return out


class DriftThreshold:
    """The same rule one score at a time (for streaming): `update(score)` returns the threshold that applies to that score."""

    def __init__(self, thr0: float, fpr: float = 0.01, window: int = 4000, every: int = 200, cap: float = 3.0, lo: float = 0.7, hi: float = 2.5,
                 min_recent: int = 800):
        self.thr0, self.fpr, self.window, self.every, self.cap, self.lo, self.hi, self.min_recent = thr0, fpr, window, every, cap, lo, hi, min_recent
        self.cur = float(thr0)
        self.recent: list = []
        self.i = 0

    def update(self, score: float) -> float:
        thr = self.cur
        if score < self.cap * self.cur:
            self.recent.append(score)
            if len(self.recent) > 2 * self.window:
                del self.recent[:-self.window]
        if len(self.recent) >= self.min_recent and self.i % self.every == 0:
            self.cur = float(np.clip(np.quantile(self.recent[-self.window:], 1.0 - self.fpr), self.lo * self.thr0, self.hi * self.thr0))
        self.i += 1
        return thr
