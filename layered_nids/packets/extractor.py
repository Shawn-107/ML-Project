"""Per-packet feature extraction: one 100-number vector for every packet, from damped statistics of four streams.

Streams (each tracked at every lambda in `lambdas`, default 5 of them):
  host     size of packets sent by the source host                       -> weight, mean, std            (3)
  channel  size of packets src->dst IP, paired with dst->src              -> + magnitude, radius, cov, corr (7)
  jitter   gap in seconds between consecutive packets src->dst            -> weight, mean, std            (3)
  socket   size of packets src ip:port -> dst ip:port, paired with reverse -> 7
Total per lambda = 20, so 5 lambdas give 100 features. The vector describes the packet AND the recent behaviour of
the traffic it belongs to, so a scan, a flood or a beacon changes the numbers even if each packet looks ordinary.
"""
from __future__ import annotations

import numpy as np

from .pcap import Packet
from .stats import DEFAULT_LAMBDAS, Stream1D, StreamPair, StreamTable

STAT_1D = ("weight", "mean", "std")
STAT_2D = STAT_1D + ("magnitude", "radius", "cov", "corr")


def feature_names(lambdas=DEFAULT_LAMBDAS) -> list[str]:
    names = []
    for stream, stats in (("host", STAT_1D), ("channel", STAT_2D), ("jitter", STAT_1D), ("socket", STAT_2D)):
        for stat in stats:
            names += [f"{stream}_{stat}_L{l:g}" for l in lambdas]
    return names


class FeatureExtractor:
    def __init__(self, lambdas=DEFAULT_LAMBDAS, max_streams: int = 100_000, idle_s: float = 1800.0):
        self.lam = np.asarray(lambdas, dtype=np.float64)
        f = self.lam
        self.hosts = StreamTable(lambda: Stream1D(f), max_streams, idle_s)
        self.channels = StreamTable(lambda: StreamPair(f), max_streams, idle_s)
        self.jitter = StreamTable(lambda: Stream1D(f), max_streams, idle_s)
        self.sockets = StreamTable(lambda: StreamPair(f), max_streams, idle_s)
        self._last_seen: dict = {}                       # jitter needs the previous packet time per direction
        self.n_features = 20 * len(self.lam)

    @staticmethod
    def _pair_key(a, b):
        """Canonical key for a conversation plus which side `a` is (0 or 1)."""
        return ((a, b), 0) if a <= b else ((b, a), 1)

    def update(self, p: Packet) -> np.ndarray:
        t, x = p.ts, float(p.size)
        host = self.hosts.get(p.src, t).update(x, t)

        key, side = self._pair_key(p.src, p.dst)
        chan = self.channels.get(key, t).update(side, x, t)

        dkey = (p.src, p.dst)
        prev = self._last_seen.get(dkey)
        gap = 0.0 if prev is None else max(t - prev, 0.0)
        self._last_seen[dkey] = t
        if len(self._last_seen) > 200_000:               # bound memory: keep the most recent half
            for k in list(self._last_seen)[:100_000]:
                del self._last_seen[k]
        jit = self.jitter.get(dkey, t).update(gap, t)

        a, b = (p.src, p.sport, p.proto), (p.dst, p.dport, p.proto)
        skey, sside = self._pair_key(a, b)
        sock = self.sockets.get(skey, t).update(sside, x, t)

        return np.concatenate([*host, *chan, *jit, *sock])

    def run(self, packets, limit: int | None = None) -> np.ndarray:
        rows = []
        for i, p in enumerate(packets):
            if limit is not None and i >= limit:
                break
            rows.append(self.update(p))
        return np.asarray(rows, dtype=np.float32).reshape(-1, self.n_features)
