"""L1: blocklist and heuristic rules. No machine learning; this is the baseline every other layer is compared with.

Rules (all counted per source host in tumbling time windows; thresholds are plain config, not tuned on attack data):
  scan      one host contacts at least `scan_min_targets` distinct (destination, port) pairs in the window
  arp_flood one host sends at least `arp_min` ARP packets in the window
  dns_burst one host sends at least `dns_min` DNS queries in the window
  blocklist the host talks to a blocklisted IP, or queries a blocklisted domain

Every rule returns Evidence (score in [0, 1]); a rule never raises an alert by itself. For count rules the score is
0.5 at the threshold and rises to 1.0 at twice the threshold.
"""
from __future__ import annotations

import ipaddress
from collections import defaultdict
from dataclasses import dataclass

from layered_nids.common.schema import Evidence
from layered_nids.packets.pcap import Packet

from .blocklist import Blocklists

_PRIVATE = [ipaddress.ip_network(n) for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16")]


def is_internal(ip: str) -> bool:
    try:
        a = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(a in n for n in _PRIVATE) if a.version == 4 else (a.is_private or a.is_link_local)


@dataclass
class RuleConfig:
    window_s: float = 60.0
    scan_min_targets: int = 30
    arp_min: int = 200
    dns_min: int = 200


def _count_score(value: int, threshold: int) -> float:
    return min(1.0, 0.5 + 0.5 * (value - threshold) / threshold)


def window_counts(packets, window_s: float):
    """Per (host, window): distinct targets, ARP packets, DNS queries. Shared by run_rules and calibrate_rules."""
    targets, arp, dns = defaultdict(set), defaultdict(int), defaultdict(int)
    for p in packets:
        w = int(p.ts // window_s)
        if p.proto == "arp":
            arp[(p.src, w)] += 1
        elif p.proto in ("tcp", "udp"):
            targets[(p.src, w)].add((p.dst, p.dport, p.proto))
        if p.qname:
            dns[(p.src, w)] += 1
    return targets, arp, dns


def calibrate_rules(clean_packets, fpr: float = 0.01, window_s: float = 60.0, floor: int = 5) -> RuleConfig:
    """Set each count threshold so that at most `fpr` of CLEAN host-windows would fire (floor avoids silly values).

    Uses clean traffic only, the same policy idea as the other layers; no attack data is looked at."""
    import numpy as np
    targets, arp, dns = window_counts(clean_packets, window_s)

    def thr(counts):
        v = np.array(list(counts), dtype=float)
        return max(floor, int(np.floor(np.quantile(v, 1.0 - fpr))) + 1) if len(v) else floor

    return RuleConfig(window_s=window_s, scan_min_targets=thr(len(t) for t in targets.values()),
                      arp_min=thr(arp.values()), dns_min=thr(dns.values()))


def run_rules(packets, cfg: RuleConfig | None = None, blocklists: Blocklists | None = None) -> list[Evidence]:
    cfg = cfg or RuleConfig()
    targets, arp, dns = window_counts(packets, cfg.window_s)
    hits = {}                        # (host, window, kind, indicator) -> list of feed names
    for p in packets:
        w = int(p.ts // cfg.window_s)
        if blocklists is not None:
            if p.qname:
                f = blocklists.feeds_for_domain(p.qname)
                if f:
                    hits[(p.src, w, "domain", p.qname)] = f
            if p.proto != "arp":
                for host, other in ((p.src, p.dst), (p.dst, p.src)):
                    if is_internal(host) and not is_internal(other):
                        f = blocklists.feeds_for_ip(other)
                        if f:
                            hits[(host, w, "ip", other)] = f

    out: list[Evidence] = []
    ts = lambda w: w * cfg.window_s
    for (host, w), t in targets.items():
        if len(t) >= cfg.scan_min_targets:
            out.append(Evidence("rules", host, ts(w), _count_score(len(t), cfg.scan_min_targets),
                                f"{host} contacted {len(t)} distinct targets in {cfg.window_s:g}s (scan-like)",
                                {"rule": "scan", "targets": len(t)}))
    for (host, w), n in arp.items():
        if n >= cfg.arp_min:
            out.append(Evidence("rules", host, ts(w), _count_score(n, cfg.arp_min),
                                f"{host} sent {n} ARP packets in {cfg.window_s:g}s (ARP flood/sweep)",
                                {"rule": "arp_flood", "packets": n}))
    for (host, w), n in dns.items():
        if n >= cfg.dns_min:
            out.append(Evidence("rules", host, ts(w), _count_score(n, cfg.dns_min),
                                f"{host} sent {n} DNS queries in {cfg.window_s:g}s",
                                {"rule": "dns_burst", "queries": n}))
    for (host, w, kind, what), feeds in hits.items():
        out.append(Evidence("rules", host, ts(w), Blocklists.confidence(len(feeds)),
                            f"{host} {'queried' if kind == 'domain' else 'talked to'} blocklisted {what} "
                            f"(feeds: {', '.join(sorted(feeds))})",
                            {"rule": "blocklist", "kind": kind, "indicator": what, "feeds": sorted(feeds)}))
    return sorted(out, key=lambda e: (e.ts, e.host))
