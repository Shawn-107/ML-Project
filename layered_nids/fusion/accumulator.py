"""L5 evidence accumulation (own code): per host, evidence from every layer adds up and fades.

Each Evidence has a score s in [0, 1]. It is turned into a weight  -log(1 - s)  (so s = 0.5 -> 0.69, s = 0.99 -> 4.6).
Weights from independent pieces of evidence ADD, which is the same as combining them with noisy-OR:

    p_host = 1 - exp(-sum_i  w_layer(i) * decay_i * -log(1 - s_i))

Old weight halves every `half_life_s` seconds. The sum is kept per layer so every alert can say which layers
contributed and why. An alert fires when p_host reaches `alert_p` (and at least `min_layers` layers contributed),
then stays quiet for `cooldown_s` unless its severity goes up.

Limits (stated, not hidden): noisy-OR assumes the pieces of evidence are independent. Two layers watching the same
scanner are NOT independent, so p is optimistic when layers agree. The `layer_weight` setting and the `min_layers`
rule are the knobs; whether they help is measured in the ablation, not assumed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from layered_nids.common.schema import LAYERS, Evidence

_MAX_S = 0.999            # a score of exactly 1 would be infinite weight


@dataclass
class FusionConfig:
    half_life_s: float = 300.0
    alert_p: float = 0.95
    severities: tuple = ((0.999, "critical"), (0.99, "high"), (0.97, "medium"), (0.0, "low"))   # checked in order
    layer_weight: dict = field(default_factory=lambda: {l: 1.0 for l in LAYERS})
    min_layers: int = 1
    cooldown_s: float = 60.0
    keep_reasons: int = 3


@dataclass(frozen=True)
class Alert:
    host: str
    ts: float
    probability: float
    severity: str
    layers: dict          # layer -> its share of the total weight, 0..1
    reasons: tuple        # the strongest pieces of evidence, as readable sentences


class _Host:
    __slots__ = ("a", "t", "recent", "last_alert_ts", "last_sev")

    def __init__(self):
        self.a = {l: 0.0 for l in LAYERS}      # decayed weight per layer
        self.t = None
        self.recent: list = []                  # (weight, ts, reason) of recent evidence, for explanations
        self.last_alert_ts = -math.inf
        self.last_sev = -1


class EvidenceAccumulator:
    def __init__(self, cfg: FusionConfig | None = None):
        self.cfg = cfg or FusionConfig()
        self.hosts: dict[str, _Host] = {}
        self._rank = {name: i for i, (_, name) in enumerate(reversed(self.cfg.severities))}

    def _decay(self, h: _Host, t: float) -> float:
        if h.t is not None and t > h.t:
            g = 0.5 ** ((t - h.t) / self.cfg.half_life_s)
            for l in h.a:
                h.a[l] *= g
        if h.t is None or t > h.t:
            h.t = t
        return h.t

    def probability(self, host: str, t: float | None = None) -> float:
        """Current p for a host (decayed to time t if given)."""
        h = self.hosts.get(host)
        if h is None:
            return 0.0
        g = 1.0 if t is None or h.t is None or t <= h.t else 0.5 ** ((t - h.t) / self.cfg.half_life_s)
        total = sum(self.cfg.layer_weight[l] * h.a[l] for l in h.a) * g
        return 1.0 - math.exp(-total)

    def _severity(self, p: float) -> str:
        for floor, name in self.cfg.severities:
            if p >= floor:
                return name
        return self.cfg.severities[-1][1]

    def add(self, ev: Evidence) -> Alert | None:
        c = self.cfg
        h = self.hosts.setdefault(ev.host, _Host())
        t = self._decay(h, ev.ts)
        w = -math.log(1.0 - min(ev.score, _MAX_S))
        h.a[ev.layer] += w
        h.recent.append((w * c.layer_weight[ev.layer], ev.ts, ev.reason))
        if len(h.recent) > 50:
            h.recent = sorted(h.recent, key=lambda r: -r[0])[:25]
        parts = {l: c.layer_weight[l] * h.a[l] for l in h.a}
        total = sum(parts.values())
        p = 1.0 - math.exp(-total)
        active = sum(1 for v in parts.values() if v > 0.05)
        if p < c.alert_p or active < c.min_layers:
            return None
        sev = self._severity(p)
        rank = self._rank[sev]
        if t - h.last_alert_ts < c.cooldown_s and rank <= h.last_sev:
            return None
        h.last_alert_ts, h.last_sev = t, rank
        reasons = tuple(r for _, _, r in sorted(h.recent, key=lambda r: -r[0])[:c.keep_reasons])
        return Alert(ev.host, t, p, sev, {l: v / total for l, v in parts.items() if v > 0}, reasons)

    def run(self, evidence) -> list[Alert]:
        """Feed evidence in time order and collect alerts. Evidence is sorted by timestamp first."""
        return [a for e in sorted(evidence, key=lambda e: e.ts) if (a := self.add(e)) is not None]
