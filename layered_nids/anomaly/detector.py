"""NIDS packet anomaly detector: features + autoencoder ensemble + threshold policy + per-host evidence.

    det = PacketAnomalyDetector(fpr=0.01).fit(train_packets, calibration_packets)   # normal traffic only, in time order
    score = det.process(packet)                                                     # one packet at a time (live)
    scores = det.score_packets(packets)                                             # or a whole batch
    evidence = det.evidence(packets, scores, window_s=60.0)                         # per (host, window) Evidence

Threshold policy: the score that at most `fpr` of CALIBRATION packets (normal, held out from training) exceed.
Evidence per (host, window) asks whether the number of flagged packets is higher than chance (binomial or, when the
calibration windows show bursty false alarms, beta-binomial); see common/windows.py.
Feature statistics start empty after load(); call `warm(packets)` with recent normal traffic before scoring live packets.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from layered_nids.anomaly.features import PacketFeatures, feature_names
from layered_nids.anomaly.model import AE, AnomalyEnsemble, Prep
from layered_nids.common.hosts import pick_internal
from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.windows import fit_overdispersion, window_counts, windowed_evidence


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PacketAnomalyDetector:
    def __init__(self, fpr: float = 0.01, window_s: float = 60.0, **model_kw):
        self.fpr = fpr
        self.window_s = window_s
        self.model_kw = model_kw
        self.features = PacketFeatures()
        self.ens: AnomalyEnsemble | None = None
        self.threshold: float | None = None
        self.mu: float = fpr
        self.rho: float = 0.0

    # ------------------------------------------------------------------------------------------------ host rule
    @staticmethod
    def internal_host(p) -> str:
        """The internal host a packet belongs to (common/hosts.py rule), or '' when it has none."""
        return pick_internal(p.src, p.dst)

    # ------------------------------------------------------------------------------------------------ training
    def fit(self, train_packets, calibration_packets) -> "PacketAnomalyDetector":
        train, calib = list(train_packets), list(calibration_packets)
        X_tr = self.features.run(train)
        X_ca = self.features.run(calib)                                    # statistics keep running: no cold start at calibration
        half = len(X_ca) // 2
        tail_part, thr_part = X_ca[:half], X_ca[half:]                     # the empirical tail and the threshold use different data
        self.ens = AnomalyEnsemble(**self.model_kw).fit(X_tr, tail_part)
        s_thr = self.ens.score(thr_part)
        self.threshold = threshold_at_fpr(s_thr, self.fpr)
        s_all = self.ens.score(X_ca)                                       # burstiness of false alarms on normal windows
        hosts = [self.internal_host(p) for p in calib]
        ts = np.array([p.ts for p in calib])
        keep = np.array([h != "" for h in hosts])
        flagged = s_all >= self.threshold
        counts = window_counts(np.array(hosts)[keep], ts[keep], flagged[keep], self.window_s).values()
        self.mu, self.rho, _ = fit_overdispersion(counts)
        return self

    # ------------------------------------------------------------------------------------------------ scoring
    def _need(self):
        if self.ens is None:
            raise RuntimeError("call fit() or load() first")

    def process(self, p) -> float:
        """Score one packet; packets must arrive in capture order."""
        self._need()
        return float(self.ens.score(self.features.update(p).reshape(1, -1))[0])

    def warm(self, packets) -> None:
        """Feed normal packets to the statistics without scoring (after load, or to refresh state)."""
        for p in packets:
            self.features.update(p)

    def score_packets(self, packets) -> np.ndarray:
        self._need()
        return self.ens.score(self.features.run(packets))

    def flag(self, scores: np.ndarray) -> np.ndarray:
        return np.asarray(scores) >= self.threshold

    def evidence(self, packets, scores, window_s: float | None = None, min_score: float = 0.5, burst_aware: bool = True):
        """Evidence (layer 'packets') per (internal host, window). Packets with no internal host are ignored."""
        w = window_s or self.window_s
        hosts = np.array([self.internal_host(p) for p in packets])
        keep = hosts != ""
        ts = np.array([p.ts for p in packets])
        flagged = np.asarray(scores) >= self.threshold
        rho = self.rho if burst_aware else 0.0
        return windowed_evidence("packets", hosts[keep], ts[keep], flagged[keep], max(self.mu, 1e-4) if rho > 0 else self.fpr,
                                 w, "packets", min_score, rho=rho)

    # ------------------------------------------------------------------------------------------------ save / load
    def save(self, directory) -> Path:
        self._need()
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        e = self.ens
        torch.save({"nets": [n.state_dict() for n in e.nets],
                    "out_net": e.out_net.state_dict() if hasattr(e, "out_net") else None}, d / "weights.pt")
        arrays = {"tails": getattr(e, "tails", np.zeros(0)), "r_hi": e.r_hi.numpy() if hasattr(e, "r_hi") else np.zeros(0)}
        np.savez(d / "arrays.npz", **arrays)
        meta = {"fpr": self.fpr, "window_s": self.window_s, "threshold": self.threshold, "mu": self.mu, "rho": self.rho,
                "cfg": e.cfg, "groups": [g.tolist() for g in e.groups], "prep": e.prep.state(),
                "n_features": self.features.n_features, "feature_names": feature_names(),
                "sha256": {"weights.pt": _sha(d / "weights.pt"), "arrays.npz": _sha(d / "arrays.npz")}}
        (d / "meta.json").write_text(json.dumps(meta))
        return d

    @classmethod
    def load(cls, directory) -> "PacketAnomalyDetector":
        d = Path(directory)
        meta = json.loads((d / "meta.json").read_text())
        for name, digest in meta["sha256"].items():
            if _sha(d / name) != digest:
                raise ValueError(f"{name} failed its integrity check (changed since it was saved)")
        det = cls(meta["fpr"], meta["window_s"], **{k: v for k, v in meta["cfg"].items()})
        e = AnomalyEnsemble(**meta["cfg"])
        e.prep = Prep.from_state(meta["prep"])
        e.groups = [np.asarray(g) for g in meta["groups"]]
        w = torch.load(d / "weights.pt", weights_only=True)                # tensors only: no pickled code
        hr = meta["cfg"]["hidden_ratio"]
        e.nets = []
        for g, sd in zip(e.groups, w["nets"]):
            net = AE(len(g), hr)
            net.load_state_dict(sd)
            net.eval()
            e.nets.append(net)
        a = np.load(d / "arrays.npz")
        if w["out_net"] is not None:
            e.out_net = AE(len(e.groups), hr)
            e.out_net.load_state_dict(w["out_net"])
            e.out_net.eval()
            e.r_hi = torch.as_tensor(a["r_hi"])
        else:
            e.tails = a["tails"]
        det.ens, det.threshold, det.mu, det.rho = e, meta["threshold"], meta["mu"], meta["rho"]
        return det
