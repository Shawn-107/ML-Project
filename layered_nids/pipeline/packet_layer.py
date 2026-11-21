"""Packet layer of the combined pipeline: our own packet anomaly detector (layered_nids/anomaly) -> per-host evidence.

Steps: (1) train the detector on the first `train_n` packets, which must be clean traffic; (2) set the alert cut-off from a
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
from layered_nids.anomaly import PacketAnomalyDetector
from layered_nids.anomaly.features import PacketFeatures
from layered_nids.packets.pcap import iter_packets

HOST_RULE = "v2"      # bump when internal_host changes: cached host columns are rebuilt, scores are not re-computed
DEFAULT_TRAIN = 55_000
DEFAULT_CALIB = 15_000


def internal_host(p) -> str:
    """The internal host a packet belongs to: its source if that is a real private unicast address, else its destination if
    that is, else '' (link-layer-only traffic, 0.0.0.0 DHCP, broadcast/multicast, purely external traffic)."""
    return pick_internal(p.src, p.dst)


def score_packets(packets, train_n: int = DEFAULT_TRAIN, calib_n: int = DEFAULT_CALIB, seed: int = 0, fpr: float = 0.01,
                  progress_every: int = 0) -> dict:
    """The detector is trained on the first `train_n` packets and calibrated on the next `calib_n`; scores are 0.0 for the
    training part, the calibration window's own scores follow, then every later packet is scored in capture order."""
    packets = list(packets)
    ts = np.asarray([p.ts for p in packets], dtype=np.float64)
    hosts = np.asarray([internal_host(p) for p in packets], dtype=str)
    scores = np.zeros(len(packets), dtype=np.float64)
    if len(packets) > train_n + calib_n:
        t0 = time.time()
        det = PacketAnomalyDetector(fpr=fpr, seed=seed).fit(packets[:train_n], packets[train_n:train_n + calib_n])
        X = PacketFeatures().run(packets[:train_n + calib_n])                      # same features as inside fit(): their calibration scores
        scores[train_n:train_n + calib_n] = det.ens.score(X[train_n:])
        scores[train_n + calib_n:] = det.score_packets(packets[train_n + calib_n:])
        if progress_every:
            print(f"  detector: {len(packets):,} packets in {time.time() - t0:.0f} s", flush=True)
    return {"scores": scores, "ts": ts, "hosts": hosts, "train_n": train_n}


def score_pcap(pcap, cache=None, train_n: int = DEFAULT_TRAIN, limit: int | None = None, seed: int = 0, calib_n: int = DEFAULT_CALIB,
               progress_every: int = 100_000) -> dict:
    """Score a pcap, reusing `cache` (an .npz) only if it was made with the same settings and the same file size."""
    pcap = Path(pcap)
    key = {"size": pcap.stat().st_size, "train_n": train_n, "calib_n": calib_n, "limit": -1 if limit is None else limit, "seed": seed, "detector": 2}
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if all(k in z.files and int(z[k]) == v for k, v in key.items()):
            hosts = z["hosts"]
            if "host_rule" not in z.files or str(z["host_rule"]) != HOST_RULE:
                hosts = np.asarray([internal_host(p) for p in iter_packets(pcap, limit)], dtype=str)
                np.savez_compressed(cache, scores=z["scores"], ts=z["ts"], hosts=hosts, host_rule=HOST_RULE, **key)
            return {"scores": z["scores"], "ts": z["ts"], "hosts": hosts, "train_n": train_n, "cached": True}
    data = score_packets(iter_packets(pcap, limit), train_n, calib_n, seed, progress_every=progress_every)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, scores=data["scores"], ts=data["ts"], hosts=data["hosts"], host_rule=HOST_RULE, **key)
    return data


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
