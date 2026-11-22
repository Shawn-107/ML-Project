"""NIDS packet features, version 2: one 120-number vector per packet describing the packet AND the recent behaviour of the
traffic it belongs to. Written from the maths of damped incremental statistics (see packets/stats.py), with extra
behaviour features that target what the size/timing statistics alone cannot see: scans, TCP-flag abuse and DNS abuse.

Each group below is tracked at every decay rate in `lambdas` (default 4: half-lives of 0.23 s, 0.7 s, 6.9 s and 69 s).
Per decay rate there are 30 numbers:

  host      (3)  size of packets sent by the source host: weight, mean, std
  channel   (7)  size of packets src->dst, paired with dst->src: weight, mean, std, magnitude, radius, covariance, correlation
  jitter    (3)  gap between consecutive packets src->dst: weight, mean, std
  socket    (7)  as channel, per (ip, port, protocol) pair
  novelty   (3)  NEW in v2. Per source host: decayed count of never-before-seen (destination, port, protocol) targets, the
                 same as a share of recent packets, and decayed count of never-before-seen destination addresses.
                 A scan or a worm spreading is a burst of "new" targets even when every packet looks ordinary.
  mix       (7)  NEW in v2. Per source host, as shares of recent packets: TCP, UDP, other protocols (ARP, ICMP...),
                 SYN-without-ACK, RST; plus the decayed count of DNS queries and their mean name entropy
                 (random-looking names, as made by DGA malware or DNS tunnelling, have high entropy).
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from layered_nids.packets.pcap import Packet
from layered_nids.packets.stats import Stream1D, StreamPair, StreamTable

DEFAULT_LAMBDAS = (3.0, 1.0, 0.1, 0.01)
PER_LAMBDA = 30
_EPS = 1e-9
_MAX_TARGETS = 4096                      # per source host; the set is cleared when it fills (memory bound)

_SYN, _ACK, _RST = 0x02, 0x10, 0x04


def entropy(name: str) -> float:
    """Shannon entropy (bits per character) of a DNS name without dots; 0 for the empty name."""
    s = name.replace(".", "")
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


class _Novelty:
    """Never-seen-before targets per source host (damped)."""
    __slots__ = ("lam", "w", "n_tgt", "n_dst", "t", "seen_tgt", "seen_dst")

    def __init__(self, lam):
        self.lam = lam
        self.w = np.zeros_like(lam)
        self.n_tgt = np.zeros_like(lam)
        self.n_dst = np.zeros_like(lam)
        self.t = None
        self.seen_tgt: set = set()
        self.seen_dst: set = set()

    def update(self, p: Packet, t: float):
        if self.t is not None and t > self.t:
            g = np.exp2(-self.lam * (t - self.t))
            self.w *= g
            self.n_tgt *= g
            self.n_dst *= g
        if self.t is None or t > self.t:
            self.t = t
        tgt = (p.dst, p.dport, p.proto)
        new_t = tgt not in self.seen_tgt
        new_d = p.dst not in self.seen_dst
        if len(self.seen_tgt) >= _MAX_TARGETS:
            self.seen_tgt.clear()
        if len(self.seen_dst) >= _MAX_TARGETS:
            self.seen_dst.clear()
        self.seen_tgt.add(tgt)
        self.seen_dst.add(p.dst)
        self.w += 1.0
        self.n_tgt += new_t
        self.n_dst += new_d
        return self.n_tgt.copy(), self.n_tgt / np.maximum(self.w, _EPS), self.n_dst.copy()


class _Regular:
    """Regularity of the gaps between a host's own connection starts towards one destination port (damped): recent count, mean gap,
    std of the gaps and their ratio. A beacon phones home on a clock; ordinary browsing does not. Only new contacts update it
    (TCP SYN without ACK, any UDP packet), so replies and long transfers do not look regular."""
    __slots__ = ("lam", "w", "ls", "ss", "t", "last")

    def __init__(self, lam):
        self.lam = lam
        self.w = np.zeros_like(lam)
        self.ls = np.zeros_like(lam)
        self.ss = np.zeros_like(lam)
        self.t = None
        self.last = None

    def update(self, t: float, contact: bool):
        if self.t is not None and t > self.t:
            g = np.exp2(-self.lam * (t - self.t))
            self.w *= g
            self.ls *= g
            self.ss *= g
        if self.t is None or t > self.t:
            self.t = t
        if contact:
            if self.last is not None and t - self.last > 0.5:
                gap = t - self.last
                self.w += 1.0
                self.ls += gap
                self.ss += gap * gap
            if self.last is None or t - self.last > 0.5:
                self.last = t
        w = np.maximum(self.w, _EPS)
        mean = self.ls / w
        std = np.sqrt(np.abs(self.ss / w - mean * mean))
        return self.w.copy(), mean, std, np.where(mean > _EPS, std / np.maximum(mean, _EPS), 0.0)


class _Mix:
    """Protocol, TCP-flag and DNS mix per source host (damped counters)."""
    __slots__ = ("lam", "c", "t")
    # rows: 0 packets, 1 tcp, 2 udp, 3 other, 4 syn-without-ack, 5 rst, 6 dns queries, 7 sum of dns name entropy

    def __init__(self, lam):
        self.lam = lam
        self.c = np.zeros((8, len(lam)))
        self.t = None

    def update(self, p: Packet, t: float):
        if self.t is not None and t > self.t:
            self.c *= np.exp2(-self.lam * (t - self.t))
        if self.t is None or t > self.t:
            self.t = t
        c = self.c
        c[0] += 1.0
        if p.proto == "tcp":
            c[1] += 1.0
            f = p.flags
            if f & _SYN and not f & _ACK:
                c[4] += 1.0
            if f & _RST:
                c[5] += 1.0
        elif p.proto == "udp":
            c[2] += 1.0
        else:
            c[3] += 1.0
        if p.qname:
            c[6] += 1.0
            c[7] += entropy(p.qname)
        n = np.maximum(c[0], _EPS)
        return c[1] / n, c[2] / n, c[3] / n, c[4] / n, c[5] / n, c[6].copy(), c[7] / np.maximum(c[6], _EPS)


def feature_names(lambdas=DEFAULT_LAMBDAS, regularity: bool = False) -> list[str]:
    one = ("weight", "mean", "std")
    two = one + ("magnitude", "radius", "cov", "corr")
    groups = (("host", one), ("channel", two), ("jitter", one), ("socket", two),
              ("novelty", ("new_targets", "new_target_share", "new_destinations")),
              ("mix", ("tcp_share", "udp_share", "other_share", "syn_share", "rst_share", "dns_count", "dns_entropy")))
    if regularity:
        groups = groups + (("regular", ("weight", "mean_gap", "std_gap", "cv")),)
    return [f"{g}_{s}_L{l:g}" for g, stats in groups for s in stats for l in lambdas]


class PacketFeatures:
    """Stateful extractor: call `update(packet)` for packets in capture order; returns the feature vector for that packet."""

    def __init__(self, lambdas=DEFAULT_LAMBDAS, max_streams: int = 100_000, idle_s: float = 1800.0, regularity: bool = False):
        self.lam = np.asarray(lambdas, dtype=np.float64)
        lam = self.lam
        self.hosts = StreamTable(lambda: Stream1D(lam), max_streams, idle_s)
        self.channels = StreamTable(lambda: StreamPair(lam), max_streams, idle_s)
        self.jitter = StreamTable(lambda: Stream1D(lam), max_streams, idle_s)
        self.sockets = StreamTable(lambda: StreamPair(lam), max_streams, idle_s)
        self.novelty = StreamTable(lambda: _Novelty(lam), max_streams, idle_s)
        self.mix = StreamTable(lambda: _Mix(lam), max_streams, idle_s)
        self._last: dict = {}
        self.regularity = regularity
        self.regular = StreamTable(lambda: _Regular(lam), max_streams, idle_s) if regularity else None
        self.n_features = (PER_LAMBDA + (4 if regularity else 0)) * len(self.lam)

    @staticmethod
    def _pair(a, b):
        return ((a, b), 0) if a <= b else ((b, a), 1)

    def update(self, p: Packet) -> np.ndarray:
        t, x = float(p.ts), float(p.size)
        host = self.hosts.get(p.src, t).update(x, t)
        key, side = self._pair(p.src, p.dst)
        chan = self.channels.get(key, t).update(side, x, t)
        dkey = (p.src, p.dst)
        prev = self._last.get(dkey)
        self._last[dkey] = t
        if len(self._last) > 200_000:
            for k in list(self._last)[:100_000]:
                del self._last[k]
        jit = self.jitter.get(dkey, t).update(0.0 if prev is None else max(t - prev, 0.0), t)
        a, b = (p.src, p.sport, p.proto), (p.dst, p.dport, p.proto)
        skey, sside = self._pair(a, b)
        sock = self.sockets.get(skey, t).update(sside, x, t)
        nov = self.novelty.get(p.src, t).update(p, t)
        mix = self.mix.get(p.src, t).update(p, t)
        if self.regularity:
            contact = (p.proto == "tcp" and bool(p.flags & _SYN) and not p.flags & _ACK) or p.proto == "udp"
            reg = self.regular.get((p.src, p.dst, p.dport, p.proto), t).update(t, contact)
            return np.concatenate([*host, *chan, *jit, *sock, *nov, *mix, *reg])
        return np.concatenate([*host, *chan, *jit, *sock, *nov, *mix])

    def run(self, packets) -> np.ndarray:
        rows = [self.update(p) for p in packets]
        return np.asarray(rows, dtype=np.float32).reshape(-1, self.n_features)
