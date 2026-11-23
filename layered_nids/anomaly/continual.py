"""Continual packet scoring: the detector refits itself on recent clean traffic when the traffic drifts, with guards so attacks cannot teach it.

Why: the normal traffic of a real network changes (new devices, changed habits, a different time of day). A model and threshold fitted once
then flag more and more normal packets. A threshold correction alone (anomaly/drift.py, bounded 0.7-2.5x) cannot follow a score that grows 30-fold.

Method (time order, chunk by chunk; a chunk is scored ONLY by a model fitted on earlier data, so nothing leaks backwards):
  1. Score the chunk with the current model and threshold; remember the rows (the last `window` rows are kept).
  2. Drift check on the window, re-scored by the current model: flagged share above `drift_factor` x the target false-alarm rate = out of date.
  3. Guard: rows scoring below `slack` x the threshold are "clean enough to learn from". If fewer than `min_clean` of the window are clean, the
     excess is far from anything the model knows (an attack in progress, not slow drift): NO refit, the old model keeps flagging.
  4. Refit on the clean rows (oldest 70% train, next 15% tail, newest 15% sets the threshold at the target false-alarm rate on the newest traffic).
Known limit: an attack that scores below `slack` x the threshold for a long time is learned as normal (slow poisoning). Volumetric attacks score
hundreds of times above and are never learned.
"""
from __future__ import annotations

import numpy as np

from ..common.metrics import threshold_at_fpr
from .model import AnomalyEnsemble


class ContinualScorer:
    def __init__(self, ens: AnomalyEnsemble, threshold: float, fpr: float = 0.01, names: list | None = None, window: int = 100_000,
                 slack: float = 3.0, min_clean: float = 0.8, drift_factor: float = 3.0, min_refit_rows: int = 20_000):
        self.ens, self.thr, self.fpr, self.names = ens, float(threshold), fpr, names
        self.window, self.slack, self.min_clean, self.drift_factor, self.min_refit_rows = window, slack, min_clean, drift_factor, min_refit_rows
        self.buf: np.ndarray | None = None
        self.seen = 0
        self.log: list[dict] = []

    def _refit(self, Xc: np.ndarray) -> tuple[AnomalyEnsemble, float]:
        n = len(Xc)
        a, b = int(0.70 * n), int(0.85 * n)
        new = AnomalyEnsemble(**self.ens.cfg).fit(Xc[:a], Xc[a:b], names=self.names)
        return new, threshold_at_fpr(new.score(Xc[b:]), self.fpr)

    def process(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Score one chunk of feature rows (time order). Returns (scores, thresholds in effect for each row); then updates the model if needed."""
        X = np.asarray(X, dtype=np.float32)
        scores = self.ens.score(X)
        thr_used = np.full(len(X), self.thr)
        self.buf = X if self.buf is None else np.vstack([self.buf, X])[-self.window:]
        self.seen += len(X)
        if len(self.buf) >= self.min_refit_rows:
            sb = self.ens.score(self.buf)
            flagged = float((sb >= self.thr).mean())
            clean = sb < self.slack * self.thr
            share = float(clean.mean())
            entry = {"at_row": self.seen, "flagged_share": round(flagged, 4), "clean_share": round(share, 4), "threshold": self.thr, "refit": False}
            if flagged > self.drift_factor * self.fpr and share >= self.min_clean and clean.sum() >= self.min_refit_rows:
                self.ens, new_thr = self._refit(self.buf[clean])
                entry.update(refit=True, new_threshold=new_thr)
                self.thr = new_thr
            elif flagged > self.drift_factor * self.fpr:
                entry["blocked"] = "too many rows far above the threshold (attack-like), not learned"
            self.log.append(entry)
        return scores, thr_used

    def run(self, X: np.ndarray, chunk: int = 25_000) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """Whole feature matrix in time order -> (scores, thresholds in effect per row, refit log)."""
        S, T = [], []
        for a in range(0, len(X), chunk):
            s, t = self.process(X[a:a + chunk])
            S.append(s), T.append(t)
        return np.concatenate(S), np.concatenate(T), self.log
