"""Packet layer of the combined pipeline: our own packet anomaly detector -> per-host evidence.

Steps: (1) turn every packet into 100 damped-statistics features (packets/extractor.py) and train the autoencoder ensemble
(packets/ensemble.py) on the first `train_n` packets, which must be clean traffic; (2) set the alert cut-off from a
CALIBRATION window right after training, assumed normal; (3) turn flagged packets into per-(host, time window) evidence with
the shared binomial rule in common/windows.py.

Limits: the learning window and the calibration window are ASSUMED benign. A capture that is already under attack while the
detector learns teaches it that attack as normal. Scores are per packet; the host is the private-address end of the packet.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from layered_nids.common.hosts import pick_internal
from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.windows import fit_overdispersion, window_counts, windowed_evidence
from layered_nids.packets.ensemble import AutoencoderEnsemble
from layered_nids.packets.extractor import FeatureExtractor
from layered_nids.packets.pcap import iter_packets

HOST_RULE = "v2"      # bump when internal_host changes: cached host columns are rebuilt, scores are not re-computed
DEFAULT_TRAIN = 55_000


def internal_host(p) -> str:
    """The internal host a packet belongs to: its source if that is a real private unicast address, else its destination if
    that is, else '' (link-layer-only traffic, 0.0.0.0 DHCP, broadcast/multicast, purely external traffic)."""
    return pick_internal(p.src, p.dst)


def score_packets(packets, train_n: int = DEFAULT_TRAIN, seed: int = 0, max_group: int = 10, epochs: int = 12,
                  progress_every: int = 0) -> dict:
    """Features for every packet, ensemble trained on the first `train_n`, scores for the rest (0.0 for the learning part)."""
    fx = FeatureExtractor()
    rows, ts, hosts = [], [], []
    t0 = time.time()
    for i, p in enumerate(packets):
        rows.append(fx.update(p))
        ts.append(p.ts)
        hosts.append(internal_host(p))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  features: {i + 1:,} packets, {(i + 1) / (time.time() - t0):.0f} packets/s", flush=True)
    X = np.asarray(rows, dtype=np.float32).reshape(-1, fx.n_features)
    scores = np.zeros(len(X), dtype=np.float64)
    if len(X) > train_n:
        ens = AutoencoderEnsemble(max_group=max_group, epochs=epochs, seed=seed).fit(X[:train_n])
        scores[train_n:] = ens.score(X[train_n:])
    return {"scores": scores, "ts": np.asarray(ts, dtype=np.float64), "hosts": np.asarray(hosts, dtype=str), "train_n": train_n}


def score_pcap(pcap, cache=None, train_n: int = DEFAULT_TRAIN, limit: int | None = None, seed: int = 0,
               progress_every: int = 100_000) -> dict:
    """Score a pcap, reusing `cache` (an .npz) only if it was made with the same settings and the same file size."""
    pcap = Path(pcap)
    key = {"size": pcap.stat().st_size, "train_n": train_n, "limit": -1 if limit is None else limit, "seed": seed}
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if all(k in z.files and int(z[k]) == v for k, v in key.items()):
            hosts = z["hosts"]
            if "host_rule" not in z.files or str(z["host_rule"]) != HOST_RULE:
                hosts = np.asarray([internal_host(p) for p in iter_packets(pcap, limit)], dtype=str)
                np.savez_compressed(cache, scores=z["scores"], ts=z["ts"], hosts=hosts, host_rule=HOST_RULE, **key)
            return {"scores": z["scores"], "ts": z["ts"], "hosts": hosts, "train_n": train_n, "cached": True}
    data = score_packets(iter_packets(pcap, limit), train_n, seed, progress_every=progress_every)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, scores=data["scores"], ts=data["ts"], hosts=data["hosts"], host_rule=HOST_RULE, **key)
    return data


class StreamScorer:
    """Packet-by-packet use for live capture: the first `train_n` packets are learned from (score 0.0), then each packet gets
    its anomaly score as it arrives. Same detector as score_packets; features are updated one packet at a time."""

    def __init__(self, train_n: int = DEFAULT_TRAIN, seed: int = 0, max_group: int = 10, epochs: int = 12):
        self.train_n, self.seed, self.max_group, self.epochs = train_n, seed, max_group, epochs
        self.fx = FeatureExtractor()
        self._buf: list = []
        self.ens: AutoencoderEnsemble | None = None
        self.n = 0

    @property
    def ready(self) -> bool:
        return self.ens is not None

    def process(self, p) -> float:
        x = self.fx.update(p)
        self.n += 1
        if self.ens is None:
            self._buf.append(x)
            if len(self._buf) >= self.train_n:
                self.ens = AutoencoderEnsemble(max_group=self.max_group, epochs=self.epochs, seed=self.seed).fit(
                    np.asarray(self._buf, dtype=np.float32))
                self._buf = []
            return 0.0
        return float(self.ens.score(x.astype(np.float32)[None, :])[0])


def calibrate_cutoff(scores, train_n: int, n_calibration: int, fpr: float) -> tuple[float, int]:
    """Cut-off at `fpr` on the n_calibration packets right after training. Returns (cutoff, index where scoring starts)."""
    start = train_n
    window = np.asarray(scores[start:start + n_calibration])
    if len(window) < 100:
        raise ValueError(f"need at least 100 packets after training to calibrate, got {len(window)}")
    return threshold_at_fpr(window, fpr), start + len(window)


def packet_evidence(data: dict, cutoff: float, first_index: int, fpr: float, window_s: float = 60.0,
                    min_score: float = 0.5, rho: float = 0.0):
    """Evidence for packets from `first_index` on. A packet is 'flagged' when its score is at least the cut-off.
    rho > 0 uses the burst-aware (beta-binomial) null, with `fpr` as the mean flag rate (see common/windows.py)."""
    hosts = data["hosts"][first_index:]
    keep = hosts != ""
    s = data["scores"][first_index:][keep]
    return windowed_evidence("packets", hosts[keep], data["ts"][first_index:][keep], s >= cutoff, fpr,
                             window_s, "packets", min_score, rho=rho)


def fit_null(data: dict, cutoff: float, first_index: int, lo_ts: float, hi_ts: float, window_s: float = 60.0,
             min_events: int = 5) -> tuple[float, float, int]:
    """Fit the burst-aware null on windows of (assumed) normal traffic between lo_ts and hi_ts.
    Returns (mean flag rate mu, intra-window correlation rho, windows used)."""
    hosts, ts = data["hosts"][first_index:], data["ts"][first_index:]
    keep = (hosts != "") & (ts >= lo_ts) & (ts < hi_ts)
    flagged = data["scores"][first_index:][keep] >= cutoff
    return fit_overdispersion(window_counts(hosts[keep], ts[keep], flagged, window_s).values(), min_events)
