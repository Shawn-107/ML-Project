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


# ====================================================================================================================================
# Confirmed-clean recalibration: the better fix. The score cannot tell drift from a stealthy attack, so the model learns ONLY from rows that
# something OUTSIDE the score says are clean: windows an operator approved, minus every packet of a host that another layer alerted on.
# ====================================================================================================================================
def vetoed_by_alerts(hosts, ts, evidence, window_s: float = 60.0, pad_s: float = 60.0) -> np.ndarray:
    """True for packets that must NOT be learned from: the packet's host has Evidence (any layer) whose window covers it: Evidence is stamped at the end of its
    window, so packets from window_s + pad_s before the stamp to pad_s after it are hidden. `hosts`/`ts` are per packet (hosts from PacketAnomalyDetector.internal_host), `evidence` a list of common.schema.Evidence."""
    hosts, ts = np.asarray(hosts), np.asarray(ts, dtype=float)
    out = np.zeros(len(ts), dtype=bool)
    by_host: dict = {}
    for e in evidence:
        by_host.setdefault(e.host, []).append(e.ts)
    for h, times in by_host.items():
        m = hosts == h
        if not m.any():
            continue
        et = np.sort(np.asarray(times, dtype=float))
        t = ts[m]
        lo = np.searchsorted(et, t - pad_s, side="left")                 # an evidence time e hides packets in [e - window_s - pad_s, e + pad_s]
        hi = np.searchsorted(et, t + window_s + pad_s, side="right")      # i.e. packet time t is hidden when some e lies in [t - pad_s, t + window_s + pad_s]
        out[m] = hi > lo
    return out


class ConfirmedRecalibrator:
    """Refits on confirmed-clean rows only. `process(X, clean)` scores the chunk with the current model, then adds the rows marked clean to a
    buffer (the newest `window` of them). The model is out of date when more than `drift_factor` x the target false-alarm rate of the CONFIRMED-CLEAN
    buffer is flagged (a false-alarm rate measured on rows known to be clean), at least `min_gap` rows after the previous refit. A refit trains on
    the buffer (oldest 70%, tail 15%, newest 15% sets the threshold at the target rate). An attack is never learned unless someone confirms it clean."""

    def __init__(self, ens: AnomalyEnsemble, threshold: float, fpr: float = 0.01, names: list | None = None, window: int = 60_000,
                 min_rows: int = 20_000, min_gap: int = 50_000, drift_factor: float = 3.0):
        self.ens, self.thr, self.fpr, self.names = ens, float(threshold), fpr, names
        self.window, self.min_rows, self.min_gap, self.drift_factor = window, min_rows, min_gap, drift_factor
        self.buf: np.ndarray | None = None
        self.since = min_gap                      # rows since the last refit
        self.log: list[dict] = []
        self.seen = 0

    def process(self, X: np.ndarray, clean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        clean = np.asarray(clean, dtype=bool)
        scores = self.ens.score(X)
        thr_used = np.full(len(X), self.thr)
        if clean.any():
            add = X[clean]
            self.buf = add if self.buf is None else np.vstack([self.buf, add])[-self.window:]
        self.since += len(X)
        self.seen += len(X)
        if self.buf is not None and len(self.buf) >= self.min_rows and self.since >= self.min_gap:
            flagged = float((self.ens.score(self.buf) >= self.thr).mean())
            entry = {"at_row": self.seen, "flagged_share_of_clean": round(flagged, 4), "threshold": self.thr, "refit": False}
            if flagged > self.drift_factor * self.fpr:
                n = len(self.buf)
                a, b = int(0.70 * n), int(0.85 * n)
                new = AnomalyEnsemble(**self.ens.cfg).fit(self.buf[:a], self.buf[a:b], names=self.names)
                self.thr = threshold_at_fpr(new.score(self.buf[b:]), self.fpr)
                self.ens, self.since = new, 0
                entry.update(refit=True, new_threshold=self.thr)
            self.log.append(entry)
        return scores, thr_used

    def run(self, X: np.ndarray, clean: np.ndarray, chunk: int = 25_000) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        S, T = [], []
        for a in range(0, len(X), chunk):
            s, t = self.process(X[a:a + chunk], clean[a:a + chunk])
            S.append(s), T.append(t)
        return np.concatenate(S), np.concatenate(T), self.log
