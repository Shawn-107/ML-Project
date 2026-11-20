"""Multi-layer fusion SIMULATION built from REAL held-out layer scores.

What is real: every score a layer emits in this simulation is drawn (with replacement) from that layer's own held-out
test scores (L2: unseen-family DGA names vs benign names; L4: official UNSW test flows). So each layer's false-alarm
rate and detection rate are the measured ones, including the L4 distribution shift.

What is simulated (and is therefore an assumption, not a result about real networks): which host emits what, when
attacks start and stop, how many events per minute, and that layers fire independently of each other.
No public dataset has the same hosts seen by DNS, flows and packets together.

Because layers are independent by construction here, combining them is flattered. Treat gains as an upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from layered_nids.common.schema import Evidence
from layered_nids.common.windows import windowed_evidence


@dataclass
class LayerPool:
    """Real held-out scores for one layer. `fpr` is what the evidence rule is told; `measured_fpr` is what the layer
    really does on normal events (they differ under distribution shift - that mismatch is part of the experiment)."""
    layer: str
    normal: np.ndarray
    attack: np.ndarray
    threshold: float
    fpr: float
    rate_per_min: float            # typical events per minute for one host
    noun: str = "events"

    @property
    def measured_fpr(self) -> float:
        return float((self.normal >= self.threshold).mean())

    @property
    def measured_recall(self) -> float:
        return float((self.attack >= self.threshold).mean())


@dataclass
class Scenario:
    evidence: list
    truth: dict                    # host -> (start_s, end_s) for attackers; benign hosts are absent
    hosts: list
    duration_s: float
    attack_kinds: dict = field(default_factory=dict)


ATTACK_KINDS = {                    # layer -> share of that layer's events that are attack events while the attack is on
    "dga_malware": {"dga": 0.40, "flows": 0.03},
    "scanner": {"flows": 0.40},
    "full_compromise": {"dga": 0.30, "flows": 0.30},
}


def make_scenario(pools: dict, n_benign: int = 40, n_attack: int = 10, hours: float = 2.0, window_s: float = 60.0,
                  attack_minutes=(15, 40), kinds=None, min_score: float = 0.5, seed: int = 0) -> Scenario:
    rng = np.random.default_rng(seed)
    kinds = dict(kinds or ATTACK_KINDS)      # layers absent from `pools` simply emit nothing (their share is ignored)
    duration = hours * 3600.0
    hosts = [f"10.0.0.{i + 1}" for i in range(n_benign + n_attack)]
    attackers = list(rng.choice(hosts, n_attack, replace=False))
    truth, host_kind = {}, {}
    kind_names = sorted(kinds)
    for i, h in enumerate(attackers):
        length = rng.uniform(*attack_minutes) * 60
        start = rng.uniform(0.1 * duration, duration - length - 60)
        truth[h] = (start, start + length)
        host_kind[h] = kind_names[i % len(kind_names)]
    ev: list = []
    for layer, pool in pools.items():
        H, T, F = [], [], []
        for h in hosts:
            mult = float(rng.lognormal(0.0, 0.5))                       # some hosts are much busier than others
            counts = rng.poisson(pool.rate_per_min * mult, int(duration // 60))
            n = int(counts.sum())
            if n == 0:
                continue
            ts = np.repeat(np.arange(len(counts)) * 60.0, counts) + rng.uniform(0, 60, n)
            is_att = np.zeros(n, bool)
            if h in truth:
                share = kinds[host_kind[h]].get(layer, 0.0)
                inside = (ts >= truth[h][0]) & (ts <= truth[h][1])
                is_att = inside & (rng.random(n) < share)
            s = np.where(is_att, rng.choice(pool.attack, n), rng.choice(pool.normal, n))
            H += [h] * n
            T.append(ts)
            F.append(s >= pool.threshold)
        if H:
            ev += windowed_evidence(layer, H, np.concatenate(T), np.concatenate(F), pool.fpr, window_s, pool.noun, min_score)
    return Scenario(ev, truth, hosts, duration, host_kind)


def score_alerts(alerts: list, sc: Scenario, grace_s: float = 300.0) -> dict:
    """Host-level outcomes. An alert on an attacker counts as a detection only inside [start, end + grace]; an alert
    on a benign host, or on an attacker before it started, is a false alarm."""
    detected, ttd, false_alerts = set(), [], 0
    false_hosts = set()
    for a in sorted(alerts, key=lambda a: a.ts):
        if a.host in sc.truth:
            s, e = sc.truth[a.host]
            if s <= a.ts <= e + grace_s:
                if a.host not in detected:
                    detected.add(a.host)
                    ttd.append(a.ts - s)
                continue
            if a.ts < s:
                false_alerts += 1
                false_hosts.add(a.host)
        else:
            false_alerts += 1
            false_hosts.add(a.host)
    n_att = len(sc.truth)
    n_ben = len(sc.hosts) - n_att
    benign_false = sum(1 for a in alerts if a.host not in sc.truth)
    return {"attackers": n_att, "detected": len(detected), "detection_rate": len(detected) / max(n_att, 1),
            "median_time_to_detect_s": float(np.median(ttd)) if ttd else float("nan"),
            "false_alerts_total": false_alerts,
            "benign_false_alerts_per_host_day": benign_false / max(n_ben, 1) / (sc.duration_s / 86400.0),
            "benign_hosts_alerted_share": len({a.host for a in alerts if a.host not in sc.truth}) / max(n_ben, 1)}
