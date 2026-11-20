"""L2 -> Evidence. A host's DNS queries (name, time) become window evidence via the shared binomial rule.

The model and threshold come from a fitted DGAModel and the validation policy (e.g. 1% false alarms on benign names).
Only the name without its TLD is scored, exactly as in training.
"""
from __future__ import annotations

import numpy as np

from layered_nids.common.windows import windowed_evidence
from layered_nids.dga.data import strip_tld


def dga_evidence(model, threshold: float, hosts, ts, domains, fpr: float, window_s: float = 60.0, min_score: float = 0.5):
    names = [strip_tld(d) for d in domains]
    scores = np.asarray(model.predict_proba(names))
    flagged = scores >= threshold
    return windowed_evidence("dga", hosts, ts, flagged, fpr, window_s, "domain lookups", min_score,
                             details=[f"{d} (score {s:.2f})" for d, s in zip(domains, scores)])
