"""Turn many noisy per-event flags into one piece of Evidence per (host, time window).

Why: at a 3% false-alarm policy, 3 of every 100 normal flows are flagged. If every flag became evidence, a busy host
would reach an alert just by being busy. So a window's evidence asks "is the number of flags higher than chance?":

    p_value = P( Binomial(n_events, fpr) >= n_flagged )          score = 1 - p_value

One flagged flow among one event scores 1 - fpr (strong); 3 flagged among 100 at fpr 3% scores about 0.42 (weak).
The same formula is used by L3 (packets/detector.py), so layers are comparable.

Limits: events are treated as independent, and `fpr` is the nominal rate. If the real false-alarm rate on a network
is higher than the nominal one (as measured for L4 on the shifted UNSW test set: 7.5% vs 3%), scores are too high.
Pass the measured rate when you have one.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.stats import betabinom, binom

from layered_nids.common.schema import Evidence

MAX_SCORE = 0.999


def window_score(n_events: int, n_flagged: int, fpr: float, rho: float = 0.0) -> float:
    """Evidence score of one window. rho = 0 is the plain binomial null (flags independent). rho > 0 is a beta-binomial null
    with intra-window correlation rho: flags on normal traffic arrive in bursts, so the same count is less surprising."""
    if n_events <= 0 or n_flagged <= 0:
        return 0.0
    if n_flagged > n_events:
        raise ValueError("n_flagged cannot exceed n_events")
    if rho > 0:
        s = 1.0 / rho - 1.0                                    # alpha + beta of the Beta prior
        p_value = betabinom.sf(n_flagged - 1, n_events, fpr * s, (1.0 - fpr) * s)
    else:
        p_value = binom.sf(n_flagged - 1, n_events, fpr)
    return float(min(MAX_SCORE, 1.0 - p_value))


def window_counts(hosts, ts, flagged, window_s: float) -> dict:
    """(host, window index) -> [events, flagged, last timestamp]."""
    bucket: dict = defaultdict(lambda: [0, 0, -1e300])
    for h, t, f in zip(hosts, ts, flagged):
        b = bucket[(h, int(t // window_s))]
        b[0] += 1
        b[1] += bool(f)
        b[2] = max(b[2], float(t))
    return bucket


def fit_overdispersion(counts, min_events: int = 5) -> tuple[float, float, int]:
    """Estimate (mean flag rate mu, intra-window correlation rho, windows used) from windows of NORMAL traffic.
    Method of moments for a beta-binomial: rho = sum(n_i (f_i - mu)^2 / (mu (1 - mu)) - 1) / sum(n_i - 1). rho is clipped to
    [0, 0.99]; 0 means the flags look independent. Windows with fewer than `min_events` events are ignored."""
    rows = [(n, k) for n, k, _ in counts if n >= min_events]
    if not rows:
        return 0.01, 0.0, 0
    n = np.array([r[0] for r in rows], float)
    k = np.array([r[1] for r in rows], float)
    mu = float(np.clip(k.sum() / n.sum(), 1e-4, 0.5))
    f = k / n
    rho = float(np.sum(n * (f - mu) ** 2 / (mu * (1 - mu)) - 1.0) / np.sum(n - 1.0))
    return mu, float(np.clip(rho, 0.0, 0.99)), len(rows)


def windowed_evidence(layer: str, hosts, ts, flagged, fpr: float, window_s: float, noun: str,
                      min_score: float = 0.5, details=None, rho: float = 0.0) -> list[Evidence]:
    """hosts/ts/flagged are parallel sequences, one entry per event. Returns Evidence for windows scoring >= min_score.
    `details` (optional, parallel) is a list of short strings; the strongest few are put in the evidence detail."""
    if not (0.0 < fpr < 1.0):
        raise ValueError("fpr must be in (0, 1)")
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    bucket: dict = defaultdict(lambda: [0, 0, -1e300, []])
    for i, (h, t, f) in enumerate(zip(hosts, ts, flagged)):
        b = bucket[(h, int(t // window_s))]
        b[0] += 1
        b[1] += bool(f)
        b[2] = max(b[2], float(t))
        if f and details is not None and len(b[3]) < 3:
            b[3].append(details[i])
    out = []
    for (h, w), (n, k, last_t, ex) in sorted(bucket.items(), key=lambda kv: (kv[0][1], str(kv[0][0]))):
        s = window_score(n, k, fpr, rho)
        if s >= min_score:
            out.append(Evidence(layer, str(h), last_t, s, f"{k} of {n} {noun} flagged in {window_s:g}s (about {n * fpr:.1f} expected by chance{', burst-aware' if rho > 0 else ''})",
                                {"window": w, "events": n, "flagged": k, "examples": ex}))
    return out
