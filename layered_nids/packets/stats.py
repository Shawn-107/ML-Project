"""Damped incremental statistics: the 'damped statistics' idea, written from the maths in the published method (own code).

A stream summarises everything seen so far, but old packets count less and less. With decay rate lambda and a
gap dt since the stream was last updated, every sum is first multiplied by gamma = 2 ** (-lambda * dt), then the new
value x is added:

    w  <- gamma * w  + 1          (how many packets are 'recent'; the effective window)
    LS <- gamma * LS + x
    SS <- gamma * SS + x * x
    mean = LS / w          std = sqrt(|SS / w - mean^2|)

A lambda of 5 forgets fast (half-life 0.2 s), 0.01 forgets slowly (half-life 100 s). Tracking several lambdas gives
the model both a short-term and a long-term picture of the same traffic. All lambdas are held in one numpy vector.

Two directions of one conversation (a->b sizes x, b->a sizes y) also share a residual sum SR, which gives
magnitude, radius, covariance and correlation between the two directions.
"""
from __future__ import annotations

import numpy as np

DEFAULT_LAMBDAS = (5.0, 3.0, 1.0, 0.1, 0.01)
_EPS = 1e-12


class Stream1D:
    """One damped stream. `update(x, t)` returns (weight, mean, std), each a vector with one entry per lambda."""

    __slots__ = ("lam", "w", "ls", "ss", "t")

    def __init__(self, lam: np.ndarray):
        self.lam = lam
        self.w = np.zeros_like(lam)
        self.ls = np.zeros_like(lam)
        self.ss = np.zeros_like(lam)
        self.t = None                       # time of the last update; None until the first one

    def decay_to(self, t: float) -> None:
        if self.t is not None and t > self.t:
            g = np.exp2(-self.lam * (t - self.t))
            self.w *= g
            self.ls *= g
            self.ss *= g
        if self.t is None or t > self.t:
            self.t = t                      # out-of-order packets (t < self.t) are added without decaying

    def update(self, x: float, t: float):
        self.decay_to(t)
        self.w += 1.0
        self.ls += x
        self.ss += x * x
        return self.moments()

    def moments(self):
        w = np.maximum(self.w, _EPS)
        mean = self.ls / w
        std = np.sqrt(np.abs(self.ss / w - mean * mean))
        return self.w.copy(), mean, std


class StreamPair:
    """The two directions of one conversation. Side 0 and side 1 are the two stream owners (e.g. A->B and B->A).

    `update(side, x, t)` updates that side and returns 7 vectors (one entry per lambda):
    weight, mean, std of the updated side, then magnitude, radius, covariance and correlation of the pair."""

    __slots__ = ("lam", "s", "last", "sr", "t")

    def __init__(self, lam: np.ndarray):
        self.lam = lam
        self.s = (Stream1D(lam), Stream1D(lam))
        self.last = [0.0, 0.0]              # last value seen on each side
        self.sr = np.zeros_like(lam)        # damped sum of residual products
        self.t = None

    def update(self, side: int, x: float, t: float):
        me, other = self.s[side], self.s[1 - side]
        if self.t is not None and t > self.t:
            self.sr *= np.exp2(-self.lam * (t - self.t))
        if self.t is None or t > self.t:
            self.t = t
        w, mean, std = me.update(x, t)
        other.decay_to(t)
        _, o_mean, o_std = other.moments()
        r_me = x - mean
        r_other = (self.last[1 - side] - o_mean) if other.t is not None else 0.0
        self.sr += r_me * r_other
        self.last[side] = x
        wsum = np.maximum(w + other.w, _EPS)
        magnitude = np.sqrt(mean * mean + o_mean * o_mean)
        radius = np.sqrt(std * std + o_std * o_std)
        cov = self.sr / wsum
        denom = std * o_std
        corr = np.where(denom > _EPS, cov / np.maximum(denom, _EPS), 0.0)
        return w, mean, std, magnitude, radius, cov, np.clip(corr, -1.0, 1.0)


class StreamTable:
    """Keyed collection of streams with a size cap. Streams idle for `idle_s` seconds are dropped (their statistics
    would have decayed to nothing anyway, apart from the slowest lambda)."""

    def __init__(self, factory, max_items: int = 100_000, idle_s: float = 1800.0, purge_every: int = 20_000):
        self.factory = factory
        self.items: dict = {}
        self.max_items, self.idle_s, self.purge_every = max_items, idle_s, purge_every
        self._n = 0

    def get(self, key, t: float):
        self._n += 1
        if self._n % self.purge_every == 0:
            self._purge(t)
        obj = self.items.get(key)
        if obj is None:
            if len(self.items) >= self.max_items:
                self._purge(t, force=True)
            obj = self.items[key] = self.factory()
        return obj

    def _purge(self, t: float, force: bool = False) -> None:
        cutoff = t - self.idle_s
        keep = {k: v for k, v in self.items.items() if (v.t if hasattr(v, "t") else None) is None or v.t >= cutoff}
        if force and len(keep) >= self.max_items:                 # still full: drop the oldest half
            ordered = sorted(keep.items(), key=lambda kv: kv[1].t if kv[1].t is not None else 0.0)
            keep = dict(ordered[len(ordered) // 2:])
        self.items = keep

    def __len__(self) -> int:
        return len(self.items)
