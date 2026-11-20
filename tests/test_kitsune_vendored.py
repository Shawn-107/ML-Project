"""The vendored Kitsune must behave exactly like the original. Synthetic checks run anywhere; the parity check runs the
original's saved output (tests/data/kitsune_golden_8000.npz, made from the untouched upstream code) against ours on the
real Mirai capture, and is skipped when the cloned repo is not on disk."""
import zipfile
from pathlib import Path

import numpy as np
import pytest

from layered_nids.common.paths import ROOT
from layered_nids.packets.pcap import Packet, iter_packets
from layered_nids.realtools.kitsune.stream import KitsuneStream, kitsune_fields

GOLDEN = Path(__file__).parent / "data" / "kitsune_golden_8000.npz"
MIRAI_ZIP = ROOT / "reference" / "Kitsune-py" / "mirai.zip"


def pk(proto, src="10.0.0.1", dst="10.0.0.2", sport=1234, dport=80, ts=1.0, size=100):
    return Packet(ts, size, "aa:aa:aa:aa:aa:01", "aa:aa:aa:aa:aa:02", src, dst, proto, sport, dport)


def test_field_mapping_follows_upstream_scapy_branch():
    smac, dmac = "aa:aa:aa:aa:aa:01", "aa:aa:aa:aa:aa:02"
    assert kitsune_fields(pk("tcp")) == (0, smac, dmac, "10.0.0.1", "1234", "10.0.0.2", "80", 100)
    assert kitsune_fields(pk("udp", src="fe80::1", dst="fe80::2"))[0] == 1                 # IPv6
    assert kitsune_fields(pk("arp")) == (0, smac, dmac, "10.0.0.1", "arp", "10.0.0.2", "arp", 100)
    assert kitsune_fields(pk("icmp")) == (0, smac, dmac, "10.0.0.1", "icmp", "10.0.0.2", "icmp", 100)
    assert kitsune_fields(pk("ip58", src="fe80::1", dst="fe80::2")) == (1, smac, dmac, "fe80::1", "", "fe80::2", "", 100)
    assert kitsune_fields(pk("icmp", src="fe80::1", dst="fe80::2")) == (1, smac, dmac, "fe80::1", "", "fe80::2", "", 100)  # ICMPv6 is not 'icmp'
    assert kitsune_fields(pk("ip2")) == (0, smac, dmac, "10.0.0.1", "", "10.0.0.2", "", 100)   # IGMP: IPs, no ports
    t = kitsune_fields(pk("eth", src=smac, dst=dmac))                                      # no IP layer: MACs stand in
    assert np.isnan(t[0]) and t[3] == smac and t[5] == dmac and t[4] == "" and t[6] == ""


def test_stream_learns_then_scores_and_flags_a_change():
    rng = np.random.default_rng(0)
    ks = KitsuneStream(fm_grace=200, ad_grace=800)
    scores = []
    for i in range(3000):
        if i < 2500:                                                  # steady, ordinary-looking traffic
            p = pk("tcp", src=f"10.0.0.{1 + i % 5}", dst="10.0.0.9", sport=40000 + i % 50, dport=80,
                   ts=i * 0.01, size=int(rng.integers(60, 120)))
        else:                                                         # sudden flood of tiny packets to many ports
            p = pk("tcp", src="10.0.0.66", dst=f"10.0.0.{1 + i % 5}", sport=5555, dport=1 + i % 900,
                   ts=i * 0.0005, size=40)
        scores.append(ks.process(p))
        if i == 999:
            assert not ks.ready                                       # 1000 packets seen: still the last training packet to come
        if i == 1000:
            assert ks.ready                                           # packet 1000 was the last to train; the next is scored
    s = np.array(scores)
    assert (s[:1001] == 0.0).all() and (s[1001:] > 0).all()
    assert np.median(s[2600:]) > 3 * np.median(s[1500:2400])          # the flood scores far above normal


@pytest.mark.skipif(not GOLDEN.exists() or not MIRAI_ZIP.exists(), reason="golden file or cloned Kitsune repo not present")
def test_vendored_matches_original_on_the_mirai_capture(tmp_path):
    g = np.load(GOLDEN)
    n, fm, ad = int(g["n"]), int(g["fm"]), int(g["ad"])
    with zipfile.ZipFile(MIRAI_ZIP) as z:
        z.extract("mirai.pcap", tmp_path)
    ks = KitsuneStream(fm_grace=fm, ad_grace=ad)
    sums, rmse, first = [], [], []
    for i, p in enumerate(iter_packets(tmp_path / "mirai.pcap", limit=n)):
        x, s = ks.process_with_features(p)
        sums.append(float(np.sum(x)))
        rmse.append(s)
        if i < 600:
            first.append(np.asarray(x, dtype=np.float64))
    assert len(rmse) == n
    np.testing.assert_allclose(np.vstack(first), g["first_features"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(sums, g["feature_sum"], rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(rmse, g["rmse"], rtol=1e-6, atol=1e-9)
    assert (np.asarray(rmse)[:fm + ad + 1] == 0).all() and (np.asarray(rmse)[fm + ad + 1:] > 0).all()
