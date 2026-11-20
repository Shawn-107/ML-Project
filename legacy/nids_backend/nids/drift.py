"""Heuristic monitoring of the model's score distribution on live traffic.

Honest limitation: in our experiments the official test set was clearly harder than the training data
(accuracy fell from ~91% to ~77% at the default cutoff) yet per-feature PSI stayed below 0.1 and the score
PSI was 0.087, so this monitor would NOT have flagged that shift. Treat it as a cheap sanity check and rely on
periodic labelled samples (Detector.recalibrate) for real assurance."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np


def psi(reference, current, bins: int = 10, eps: float = 1e-4) -> float:
    reference, current = np.asarray(reference, float), np.asarray(current, float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, edges)[0] / len(reference) + eps
    b = np.histogram(current, edges)[0] / len(current) + eps
    return float(((b - a) * np.log(b / a)).sum())


class DriftMonitor:
    def __init__(self, reference_scores, window: int = 2000, min_samples: int = 200):
        self.reference = np.asarray(reference_scores, float)
        self.scores: deque = deque(maxlen=window)
        self.flags: deque = deque(maxlen=window)
        self.min_samples = min_samples
        self._lock = threading.Lock()

    def update(self, scores, flags) -> None:
        with self._lock:
            self.scores.extend(np.asarray(scores, float).tolist())
            self.flags.extend(np.asarray(flags).astype(int).tolist())

    def reset(self) -> None:
        with self._lock:
            self.scores.clear(); self.flags.clear()

    def status(self) -> dict:
        with self._lock:
            n, scores, flags = len(self.scores), list(self.scores), list(self.flags)
        if n < self.min_samples:
            return {"ready": False, "n": n, "needed": self.min_samples}
        p = psi(self.reference, scores)
        level = "major" if p > 0.25 else "moderate" if p > 0.10 else "none"
        return {"ready": True, "n": n, "score_psi": p, "drift": level, "alert_rate": float(np.mean(flags))}
