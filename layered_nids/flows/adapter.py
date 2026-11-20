"""L4 -> Evidence. The UNSW-NB15 feature set has NO addresses and NO timestamps, so the caller must supply `hosts`
and `ts` (for replay they are assigned by the simulator, and that is a simulation, not capture).

Scores are any attack score in [0, 1] (legacy blend, or a candidate from improve.py) and the threshold from a
validation policy. Evidence is built per (host, window) with the shared binomial rule.
"""
from __future__ import annotations

import numpy as np

from layered_nids.common.windows import windowed_evidence


def flow_evidence(scores, threshold: float, hosts, ts, fpr: float, window_s: float = 60.0, min_score: float = 0.5,
                  attack_names=None):
    scores = np.asarray(scores, float)
    flagged = scores >= threshold
    details = None
    if attack_names is not None:
        details = [f"{a} flow (score {s:.2f})" for a, s in zip(attack_names, scores)]
    return windowed_evidence("flows", hosts, ts, flagged, fpr, window_s, "flows", min_score, details=details)
