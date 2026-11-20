"""L3 tests on generated data only (no capture needed), so they run anywhere."""
import struct

import numpy as np
import pytest

from layered_nids.common.schema import Evidence
from layered_nids.packets.detector import PacketDetector, windowed_evidence
from layered_nids.packets.ensemble import AutoencoderEnsemble, correlation_groups
from layered_nids.packets.extractor import FeatureExtractor, feature_names
from layered_nids.packets.pcap import Packet, iter_packets, parse_ethernet, write_pcap
from layered_nids.packets.stats import DEFAULT_LAMBDAS, Stream1D, StreamPair, StreamTable


# ---------------------------------------------------------------- damped statistics (hand-computed)
def test_stream_matches_hand_computation():
    lam = np.array([1.0])
    s = Stream1D(lam)
    s.update(10.0, 0.0)                      # w=1 LS=10 SS=100
    w, mean, std = s.update(20.0, 1.0)       # gamma = 2^-1 = 0.5: w=1.5 LS=0.5*10+20=25 SS=0.5*100+400=450
    assert w[0] == pytest.approx(1.5) and mean[0] == pytest.approx(25 / 1.5)
    assert std[0] == pytest.approx(np.sqrt(abs(450 / 1.5 - (25 / 1.5) ** 2)))     # = 4.714


def test_old_packets_fade_faster_with_bigger_lambda():
    fast, slow = Stream1D(np.array([5.0])), Stream1D(np.array([0.01]))
    for st in (fast, slow):
        st.update(100.0, 0.0)
    w_fast, _, _ = fast.update(0.0, 10.0)
    w_slow, _, _ = slow.update(0.0, 10.0)
    assert w_fast[0] < 1.001 and w_slow[0] > 1.9            # fast forgot the first packet, slow still holds it


def test_pair_correlation_positive_for_tracking_directions_and_in_range():
    pair = StreamPair(np.array([0.1]))
    rng = np.random.default_rng(0)
    t, last = 0.0, None
    for i in range(300):
        v = 100 + 50 * np.sin(i / 10)
        pair.update(0, v + rng.normal(0, 1), t)
        out = pair.update(1, v + rng.normal(0, 1), t + 0.01)
        t += 0.1
    corr = out[6][0]
    assert -1.0 <= corr <= 1.0 and corr > 0.5


def test_out_of_order_packet_does_not_blow_up():
    s = Stream1D(np.array(DEFAULT_LAMBDAS))
    s.update(1.0, 10.0)
    w, m, sd = s.update(1.0, 9.9)
    assert np.all(np.isfinite(w)) and np.all(np.isfinite(sd))


def test_stream_table_purges_idle_streams():
    tab = StreamTable(lambda: Stream1D(np.array([1.0])), max_items=10, idle_s=5.0, purge_every=1)
    tab.get("a", 0.0).update(1.0, 0.0)
    tab.get("b", 100.0)
    assert "a" not in tab.items


# ---------------------------------------------------------------- packet parsing
def _eth_ipv4_tcp(src="10.0.0.1", dst="10.0.0.2", sport=1234, dport=80, payload=b"x" * 10):
    import socket
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + 20 + len(payload), 0, 0, 64, 6, 0,
                     socket.inet_aton(src), socket.inet_aton(dst))
    tcp = struct.pack("!HHIIBBHHH", sport, dport, 0, 0, 0x50, 0x02, 8192, 0, 0)
    return bytes.fromhex("aabbccddeeff112233445566") + b"\x08\x00" + ip + tcp + payload


def test_parse_ethernet_ipv4_tcp():
    p = parse_ethernet(_eth_ipv4_tcp(), 1.5, 70)
    assert (p.src, p.dst, p.proto, p.sport, p.dport, p.size, p.ts) == ("10.0.0.1", "10.0.0.2", "tcp", 1234, 80, 70, 1.5)


def test_parse_short_frame_keeps_alignment():
    assert parse_ethernet(b"\x00" * 5, 0.0, 60).proto == "eth"


def test_pcap_round_trip_keeps_order_and_wire_length(tmp_path):
    frames = [(100.0 + i, _eth_ipv4_tcp(sport=1000 + i)) for i in range(5)]
    f = tmp_path / "t.pcap"
    write_pcap(f, frames)
    got = list(iter_packets(f))
    assert [g.sport for g in got] == [1000 + i for i in range(5)]
    assert [g.ts for g in got] == pytest.approx([100.0 + i for i in range(5)])
    assert len(list(iter_packets(f, limit=2))) == 2


# ---------------------------------------------------------------- features and the ensemble
def synth_traffic(n, seed, attack=False):
    """Steady chatter between 4 hosts; with attack=True one host floods small packets at many destinations."""
    rng = np.random.default_rng(seed)
    pk, t = [], 0.0
    for i in range(n):
        t += rng.exponential(0.05)
        if attack and rng.random() < 0.7:
            pk.append(Packet(t, 60, "", "", "10.0.0.66", f"10.0.1.{rng.integers(1, 250)}", "tcp",
                             int(rng.integers(1024, 60000)), 23))
        else:
            a, b = rng.choice(4, 2, replace=False)
            pk.append(Packet(t, int(rng.normal(500, 40)), "", "", f"10.0.0.{a + 1}", f"10.0.0.{b + 1}", "tcp", 40000 + a, 443))
    return pk


def test_feature_vector_shape_names_and_finiteness():
    fx = FeatureExtractor()
    X = fx.run(synth_traffic(300, 0))
    assert X.shape == (300, 100) and len(feature_names()) == 100 and np.isfinite(X).all()


def test_correlation_groups_cover_every_feature_once_and_respect_max_size():
    X = np.random.default_rng(0).normal(size=(500, 37))
    X[:, 1] = X[:, 0] * 2 + 0.01
    groups = correlation_groups(X, 8)
    flat = np.concatenate(groups)
    assert sorted(flat) == list(range(37)) and max(len(g) for g in groups) <= 8
    assert any({0, 1} <= set(g) for g in groups)             # the correlated pair shares a group


def test_ensemble_scores_unseen_attack_higher_and_warmup_matters():
    """Measured on this generator: with no warm-up discarded, held-out false alarms were ~49% against a 2% target;
    after discarding 3,000 packets (about 1.5 half-lives of the slowest decay rate) they were ~4%. The slow-decay
    statistics start at zero and drift for minutes, so the start of a capture is not steady state."""
    Xn = FeatureExtractor().run(synth_traffic(12000, 1))
    Xa = FeatureExtractor().run(synth_traffic(1500, 2, attack=True))

    def held_out_far(warm):
        Z = Xn[warm:]
        det = PacketDetector(fpr=0.02, epochs=8, seed=0).fit(Z[:3000], Z[3000:4500])
        return det, det.flag(Z[4500:]).mean()

    det, far_warm = held_out_far(3000)
    _, far_cold = held_out_far(0)
    assert far_warm < 0.08                                     # target 2%; generous slack for non-stationary traffic
    assert far_cold > 2 * far_warm                             # the warm-up effect is large and must stay visible
    assert np.median(det.score(Xa)) > np.median(det.score(Xn[7500:]))
    assert det.flag(Xa).mean() > 0.3


def test_evidence_scores_are_calibrated_against_normal_and_in_range():
    fx = FeatureExtractor()
    Xn = fx.run(synth_traffic(4000, 3))
    det = PacketDetector(fpr=0.05, epochs=5, seed=1).fit(Xn[:2000], Xn[2000:3000])
    s = det.score(Xn[3000:])
    ev = det.evidence_scores(s)
    assert ev.min() >= 0.0 and ev.max() <= 1.0
    assert np.all(np.diff(ev[np.argsort(s)]) >= -1e-12)         # monotone in the raw score
    assert det.evidence_scores(np.array([1e9]))[0] > 0.99


def test_windowed_evidence_groups_by_host_and_window_and_is_valid_schema():
    fx = FeatureExtractor()
    base = synth_traffic(4000, 4)
    X = fx.run(base)
    det = PacketDetector(fpr=0.05, epochs=5, seed=2).fit(X[:2000], X[2000:3000])
    pk = synth_traffic(800, 5, attack=True)
    Xa = FeatureExtractor().run(pk)
    ev = windowed_evidence(pk, det.score(Xa), det, window_s=5.0)
    assert ev and all(isinstance(e, Evidence) and e.layer == "packets" and 0 <= e.score <= 1 for e in ev)
    assert len({(e.host, e.ts) for e in ev}) == len(ev)         # one evidence per host-window
    assert any(e.host == "10.0.0.66" for e in ev)


def test_window_evidence_is_a_binomial_p_value_not_just_one_flagged_packet():
    from scipy.stats import binom

    class Det:                                                  # minimal stand-in: threshold 1.0, fpr 5%
        fpr, threshold = 0.05, 1.0
        def evidence_scores(self, s):
            return np.full(len(s), 0.95)

    def window(n, n_flagged):
        pk = [Packet(float(i) * 0.01, 60, "", "", "h", "d", "tcp", 1, 2) for i in range(n)]
        sc = np.array([2.0] * n_flagged + [0.0] * (n - n_flagged))
        return windowed_evidence(pk, sc, Det(), window_s=100.0)[0]

    weak, strong = window(60, 1), window(60, 12)
    assert weak.score == pytest.approx(1 - binom.sf(0, 60, 0.05)) and weak.score < 0.1     # 1 of 60 at 5%: ~95% by chance
    assert strong.score > 0.999 and strong.detail["flagged"] == 12
