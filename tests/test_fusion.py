"""L5 accumulator tests (generated evidence only)."""
import math

import pytest

from layered_nids.common.schema import Evidence
from layered_nids.fusion.accumulator import EvidenceAccumulator, FusionConfig


def ev(layer, host, ts, score, reason="r"):
    return Evidence(layer, host, ts, score, reason)


def test_probability_matches_noisy_or_for_two_independent_scores():
    acc = EvidenceAccumulator(FusionConfig(alert_p=2.0))                 # never alert; just accumulate
    acc.add(ev("rules", "h", 0.0, 0.6))
    acc.add(ev("packets", "h", 0.0, 0.7))
    assert acc.probability("h") == pytest.approx(1 - (1 - 0.6) * (1 - 0.7))


def test_weak_evidence_alone_does_not_alert_but_adds_up():
    acc = EvidenceAccumulator(FusionConfig(alert_p=0.95, half_life_s=1e9, cooldown_s=0))
    alerts = [acc.add(ev("packets", "h", i, 0.5)) for i in range(8)]
    first = next(i for i, a in enumerate(alerts) if a is not None)
    assert first >= 4 and alerts[0] is None                              # 1 - 0.5^n >= 0.95 needs n >= 5


def test_old_evidence_fades():
    acc = EvidenceAccumulator(FusionConfig(half_life_s=100.0, alert_p=2.0))
    acc.add(ev("rules", "h", 0.0, 0.9))
    p0 = acc.probability("h", 0.0)
    p1 = acc.probability("h", 100.0)
    assert p1 == pytest.approx(1 - math.exp(-(-math.log(0.1)) * 0.5)) and p1 < p0


def test_hosts_are_independent_and_alert_carries_reasons_and_layer_shares():
    acc = EvidenceAccumulator(FusionConfig(cooldown_s=0))
    assert acc.add(ev("rules", "a", 0.0, 0.99, "scan-like")) is not None
    assert acc.probability("b") == 0.0
    a = acc.add(ev("packets", "a", 1.0, 0.99, "odd packets"))
    assert a.host == "a" and a.severity in ("high", "critical") and set(a.layers) == {"rules", "packets"}
    assert abs(sum(a.layers.values()) - 1.0) < 1e-9 and "odd packets" in a.reasons


def test_min_layers_requires_corroboration():
    acc = EvidenceAccumulator(FusionConfig(min_layers=2, cooldown_s=0))
    assert acc.add(ev("rules", "h", 0.0, 0.999)) is None                 # one layer, however strong
    assert acc.add(ev("packets", "h", 1.0, 0.5)) is not None             # a second layer joins


def test_cooldown_suppresses_repeats_but_not_escalation():
    acc = EvidenceAccumulator(FusionConfig(cooldown_s=1000.0, half_life_s=1e9))
    first = acc.add(ev("rules", "h", 0.0, 0.96))
    assert first is not None and acc.add(ev("rules", "h", 1.0, 0.1)) is None      # same severity, inside cooldown
    esc = acc.add(ev("packets", "h", 2.0, 0.999))
    assert esc is not None and esc.severity in ("high", "critical") and esc.probability > first.probability


def test_layer_weight_can_silence_a_layer():
    acc = EvidenceAccumulator(FusionConfig(layer_weight={"rules": 0.0, "dga": 1, "packets": 1, "flows": 1}, alert_p=0.5))
    assert acc.add(ev("rules", "h", 0.0, 0.999)) is None


def test_run_sorts_by_time_and_out_of_order_evidence_is_safe():
    acc = EvidenceAccumulator(FusionConfig(cooldown_s=0))
    alerts = acc.run([ev("rules", "h", 50.0, 0.99), ev("packets", "h", 10.0, 0.99)])
    assert [a.ts for a in alerts] == sorted(a.ts for a in alerts)
    acc2 = EvidenceAccumulator()
    acc2.add(ev("rules", "h", 10.0, 0.5))
    acc2.add(ev("rules", "h", 9.0, 0.5))                                  # late arrival: added without decay, no crash
    assert 0.0 < acc2.probability("h") < 1.0
