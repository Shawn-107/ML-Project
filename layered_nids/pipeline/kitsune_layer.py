"""Kitsune layer of the combined pipeline: real Kitsune scores per packet -> per-host evidence.

Steps: (1) score every packet with the vendored real Kitsune (cached to disk, because pure Python runs at about
700 packets/s); (2) set the alert cut-off from a CALIBRATION window right after training, assumed to be normal traffic
(the same idea as Kitsune's own example, which uses a benign stretch of the capture); (3) turn flagged packets into
per-(host, time window) evidence with the shared binomial rule in common/windows.py.

Limits: the calibration window is ASSUMED benign. A capture that is already under attack while Kitsune learns will
teach it that attack as normal. Scores are per packet; the host is the private-address end of the packet.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from layered_nids.common.hosts import pick_internal
from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.windows import fit_overdispersion, window_counts, windowed_evidence
from layered_nids.packets.pcap import iter_packets
from layered_nids.realtools.kitsune.stream import KitsuneStream


HOST_RULE = "v2"      # bump when internal_host changes: cached host columns are rebuilt, scores are not re-computed


def internal_host(p) -> str:
    """The internal host a packet belongs to: its source if that is a real private unicast address, else its destination if
    that is, else '' (no host: link-layer-only traffic, 0.0.0.0 DHCP, broadcast/multicast, purely external traffic).
    Packets with no host take no part in per-host evidence."""
    return pick_internal(p.src, p.dst)


def score_packets(packets, fm_grace: int = 5000, ad_grace: int = 50000, progress_every: int = 0) -> dict:
    """Run the real Kitsune over an iterable of Packet records. Scores are 0.0 during the learning phase."""
    ks = KitsuneStream(fm_grace=fm_grace, ad_grace=ad_grace)
    scores, ts, hosts = [], [], []
    t0 = time.time()
    for i, p in enumerate(packets):
        scores.append(ks.process(p))
        ts.append(p.ts)
        hosts.append(internal_host(p))
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  kitsune: {i + 1:,} packets, {(i + 1) / (time.time() - t0):.0f} packets/s", flush=True)
    return {"scores": np.asarray(scores, dtype=np.float64), "ts": np.asarray(ts, dtype=np.float64),
            "hosts": np.asarray(hosts, dtype=str), "fm": fm_grace, "ad": ad_grace}


def score_pcap(pcap, cache=None, fm_grace: int = 5000, ad_grace: int = 50000, limit: int | None = None,
               progress_every: int = 100_000) -> dict:
    """Score a pcap, reusing `cache` (an .npz) only if it was made with the same settings and the same file size."""
    pcap = Path(pcap)
    key = {"size": pcap.stat().st_size, "fm": fm_grace, "ad": ad_grace, "limit": -1 if limit is None else limit}
    if cache and Path(cache).exists():
        z = np.load(cache, allow_pickle=False)
        if all(int(z[k]) == v for k, v in key.items()):
            hosts = z["hosts"]
            if "host_rule" not in z.files or str(z["host_rule"]) != HOST_RULE:
                # the expensive part (Kitsune scores) is still valid; only the cheap host column is rebuilt (parsing only)
                hosts = np.asarray([internal_host(p) for p in iter_packets(pcap, limit)], dtype=str)
                np.savez_compressed(cache, scores=z["scores"], ts=z["ts"], hosts=hosts, host_rule=HOST_RULE, **key)
            return {"scores": z["scores"], "ts": z["ts"], "hosts": hosts, "fm": fm_grace, "ad": ad_grace, "cached": True}
    data = score_packets(iter_packets(pcap, limit), fm_grace, ad_grace, progress_every)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, scores=data["scores"], ts=data["ts"], hosts=data["hosts"], host_rule=HOST_RULE, **key)
    return data


def calibrate_cutoff(scores, fm_grace: int, ad_grace: int, n_calibration: int, fpr: float) -> tuple[float, int]:
    """Cut-off at `fpr` on the n_calibration packets right after training. Returns (cutoff, index where scoring starts)."""
    start = fm_grace + ad_grace + 1
    window = np.asarray(scores[start:start + n_calibration])
    if len(window) < 100:
        raise ValueError(f"need at least 100 packets after training to calibrate, got {len(window)}")
    return threshold_at_fpr(window, fpr), start + len(window)


def kitsune_evidence(data: dict, cutoff: float, first_index: int, fpr: float, window_s: float = 60.0,
                     min_score: float = 0.5, rho: float = 0.0):
    """Evidence for packets from `first_index` on. A packet is 'flagged' when its score is at least the cut-off.
    rho > 0 uses the burst-aware (beta-binomial) null, with `fpr` as the mean flag rate (see common/windows.py)."""
    hosts = data["hosts"][first_index:]
    keep = hosts != ""                                   # packets with no internal host take no part (see internal_host)
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
