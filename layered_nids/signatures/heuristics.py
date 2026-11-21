"""NIDS behaviour heuristics: counting rules over time windows and a regularity test. No machine learning here; this layer is
the transparent baseline the ML layers are compared with.

  scan          one host contacts many different (destination, port) pairs in a window          (horizontal or vertical scan)
  syn_flood     one host sends very many TCP SYN-without-ACK packets to ONE destination
  udp_flood     one host sends very many UDP packets to ONE destination
  arp_flood     one host sends very many ARP packets
  icmp_sweep    one host sends ICMP packets to many different destinations (ping sweep)
  dns_burst     one host sends very many DNS queries
  dns_random    one host asks for many long, random-looking names (high character entropy): DGA malware or DNS tunnelling
  beaconing     a host contacts the same external (destination, port) at very regular intervals

A rule never raises an alert by itself: it returns Evidence (see common/schema.py). Count rules score 0.5 at the threshold and
rise to 1.0 at twice the threshold.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from layered_nids.anomaly.features import entropy
from layered_nids.common.hosts import pick_internal, valid_internal

SYN, ACK = 0x02, 0x10


@dataclass
class HeuristicConfig:
    window_s: float = 60.0
    scan_min_targets: int = 40          # distinct (destination, port, protocol) per host-window
    flood_min_packets: int = 600        # packets to a single destination per host-window (10 per second)
    arp_min: int = 200
    icmp_sweep_min: int = 15            # distinct destinations hit with ICMP by one host in a window
    dns_min: int = 150                  # queries per host-window
    dns_random_min: int = 25            # random-looking names per host-window
    dns_random_entropy: float = 3.4     # bits per character of the name without dots (random 12-letter labels give about 3.3-3.6)
    dns_random_min_len: int = 14
    beacon_min_events: int = 8
    beacon_cv: float = 0.10             # std/mean of the gaps between contacts
    beacon_min_gap_s: float = 5.0
    beacon_max_gap_s: float = 900.0
    beacon_ignore_ports: tuple = (53, 123, 137, 138, 1900, 5353, 67, 68)   # name service, time, discovery, DHCP: regular by design


def count_score(value: float, threshold: float) -> float:
    return float(min(1.0, 0.5 + 0.5 * (value - threshold) / threshold))


@dataclass
class _Window:
    targets: set = field(default_factory=set)
    per_dst_syn: dict = field(default_factory=lambda: defaultdict(int))
    per_dst_udp: dict = field(default_factory=lambda: defaultdict(int))
    arp: int = 0
    icmp_dsts: set = field(default_factory=set)
    dns: int = 0
    random_names: int = 0
    random_examples: list = field(default_factory=list)


def window_features(packets, window_s: float) -> dict:
    """(host, window index) -> _Window of counters. Host = the internal end of the packet."""
    out: dict = defaultdict(_Window)
    for p in packets:
        h = pick_internal(p.src, p.dst)
        if not h or h != p.src and p.proto != "arp":          # count what a host SENDS (inbound floods are about the target, not the sender)
            continue
        w = out[(h, int(p.ts // window_s))]
        if p.proto == "arp":
            w.arp += 1
        elif p.proto == "icmp":
            w.icmp_dsts.add(p.dst)
        elif p.proto in ("tcp", "udp"):
            w.targets.add((p.dst, p.dport, p.proto))
            if p.proto == "tcp" and p.flags & SYN and not p.flags & ACK:
                w.per_dst_syn[p.dst] += 1
            elif p.proto == "udp":
                w.per_dst_udp[p.dst] += 1
        if p.qname:
            w.dns += 1
            name = p.qname.replace(".", "")
            if len(name) >= 14 and entropy(p.qname) >= 3.4:
                w.random_names += 1
                if len(w.random_examples) < 3:
                    w.random_examples.append(p.qname)
    return out


def calibrate(clean_packets, fpr: float = 0.01, window_s: float = 60.0) -> HeuristicConfig:
    """Raise each count threshold to the (1 - fpr) quantile of CLEAN host-windows when that is above the default.
    Uses clean traffic only; no attack data is looked at. Never lowers a default."""
    cfg = HeuristicConfig(window_s=window_s)
    ws = window_features(clean_packets, window_s)

    def q(values, default):
        v = np.asarray(list(values), float)
        return max(default, int(math.floor(np.quantile(v, 1 - fpr))) + 1) if len(v) else default

    cfg.scan_min_targets = q((len(w.targets) for w in ws.values()), cfg.scan_min_targets)
    cfg.flood_min_packets = q((max(list(w.per_dst_syn.values()) + list(w.per_dst_udp.values()) + [0]) for w in ws.values()), cfg.flood_min_packets)
    cfg.arp_min = q((w.arp for w in ws.values()), cfg.arp_min)
    cfg.icmp_sweep_min = q((len(w.icmp_dsts) for w in ws.values()), cfg.icmp_sweep_min)
    cfg.dns_min = q((w.dns for w in ws.values()), cfg.dns_min)
    cfg.dns_random_min = q((w.random_names for w in ws.values()), cfg.dns_random_min)
    return cfg


def beaconing(packets, cfg: HeuristicConfig):
    """Yield (host, dst, dport, mean gap, cv, n, first ts, last ts) for destinations contacted at very regular intervals.
    Only contacts a host STARTS count (TCP SYN, or any UDP packet it sends), so replies and long transfers do not look periodic."""
    seen: dict = defaultdict(list)
    for p in packets:
        if p.proto not in ("tcp", "udp") or not valid_internal(p.src) or valid_internal(p.dst) or p.dport in cfg.beacon_ignore_ports:
            continue
        if p.proto == "tcp" and not (p.flags & SYN and not p.flags & ACK):
            continue
        seen[(p.src, p.dst, p.dport)].append(p.ts)
    for (h, d, port), ts in seen.items():
        if len(ts) < cfg.beacon_min_events:
            continue
        g = np.diff(np.asarray(ts))
        g = g[g > 0.5]                                             # collapse bursts of the same contact
        if len(g) < cfg.beacon_min_events - 1:
            continue
        mean = float(g.mean())
        cv = float(g.std() / mean) if mean > 0 else 1.0
        if cfg.beacon_min_gap_s <= mean <= cfg.beacon_max_gap_s and cv <= cfg.beacon_cv:
            yield h, d, port, mean, cv, len(g) + 1, ts[0], ts[-1]
