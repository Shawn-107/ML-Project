"""L3 detector: autoencoder ensemble + a threshold policy + evidence output.

Policy = a target false-alarm rate on held-out NORMAL packets (same idea as the L4 policies). The threshold is the
score that at most `fpr` of those packets exceed. Fit and threshold use different normal data.

Evidence score (what fusion sees): 1 - p_hat, where p_hat is the share of held-out normal packets scoring at least as
high as this packet (add-one smoothed). So 0.99 means "only about 1% of normal packets look this odd". It is a
rank-based calibration against normal traffic, not a probability of attack.
"""
from __future__ import annotations

import numpy as np

from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.schema import Evidence

from .ensemble import AutoencoderEnsemble


class PacketDetector:
    def __init__(self, fpr: float = 0.01, **ensemble_kw):
        self.fpr = fpr
        self.ens = AutoencoderEnsemble(**ensemble_kw)
        self.threshold: float | None = None
        self._val_sorted: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, X_val_normal: np.ndarray) -> "PacketDetector":
        self.ens.fit(X_train)
        s = self.ens.score(X_val_normal)
        self._val_sorted = np.sort(s)
        self.threshold = threshold_at_fpr(s, self.fpr)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return self.ens.score(X)

    def flag(self, X: np.ndarray) -> np.ndarray:
        return self.score(X) >= self.threshold

    def evidence_scores(self, scores: np.ndarray) -> np.ndarray:
        """Map raw anomaly scores to [0, 1] against the held-out normal scores."""
        v = self._val_sorted
        at_least = len(v) - np.searchsorted(v, scores, side="left")      # normals scoring >= s
        p_hat = (at_least + 1.0) / (len(v) + 1.0)
        return np.clip(1.0 - p_hat, 0.0, 1.0)


def windowed_evidence(packets, scores: np.ndarray, detector: PacketDetector, window_s: float = 10.0) -> list[Evidence]:
    """One Evidence per (source host, time window) that contains at least one flagged packet.

    Window score = 1 - p, where p = P(Binomial(n, fpr) >= flagged): the chance that NORMAL traffic would flag at least
    this many of the n packets in the window. This matters: with a 1% per-packet false-alarm rate, a window of 57
    packets has a 43% chance of one stray flag, so 'one flagged packet' is weak evidence, while 5 of 60 is strong.
    (Packets in a window are not independent, so p is optimistic; the held-out clean period is where that is checked.)
    A flood of 10,000 anomalous packets in one window is one piece of evidence with a count, not 10,000 alerts."""
    from scipy.stats import binom

    ev = detector.evidence_scores(scores)
    flagged = scores >= detector.threshold
    buckets: dict = {}
    for p, e, f in zip(packets, ev, flagged):
        key = (p.src, int(p.ts // window_s))
        b = buckets.setdefault(key, [0, 0, 0.0])
        b[0] += 1
        if f:
            b[1] += 1
            b[2] = max(b[2], float(e))
    out = []
    for (host, w), (n, nf, best) in buckets.items():
        if nf:
            p_win = float(binom.sf(nf - 1, n, detector.fpr))
            out.append(Evidence(
                layer="packets", host=host, ts=w * window_s, score=float(np.clip(1.0 - p_win, 0.0, 1.0 - 1e-9)),
                reason=f"{nf} of {n} packets from {host} were unusual for normal traffic (window {window_s:g}s, "
                       f"chance under normal traffic {p_win:.2g})",
                detail={"flagged": nf, "packets": n, "window_s": window_s, "p_window": p_win,
                        "max_packet_evidence": best}))
    return sorted(out, key=lambda e: e.ts)
