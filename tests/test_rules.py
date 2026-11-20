"""L1 tests on generated packets (no capture, no real blocklist data)."""
import struct

import pytest

from layered_nids.packets.pcap import Packet, dns_qname, parse_ethernet
from layered_nids.rules.blocklist import Blocklists
from layered_nids.rules.engine import RuleConfig, is_internal, run_rules


def pkt(ts, src, dst, proto="tcp", dport=80, sport=40000, qname=""):
    return Packet(ts, 60, "", "", src, dst, proto, sport, dport, qname)


def test_scan_rule_fires_only_above_threshold():
    cfg = RuleConfig(window_s=60, scan_min_targets=10)
    scan = [pkt(i * 0.1, "10.0.0.5", f"10.0.1.{i}") for i in range(12)]
    quiet = [pkt(i * 0.1, "10.0.0.6", "10.0.1.1", dport=443) for i in range(500)]       # many packets, ONE target
    ev = run_rules(scan + quiet, cfg)
    assert [e.host for e in ev] == ["10.0.0.5"] and ev[0].detail["rule"] == "scan"
    assert 0.5 <= ev[0].score <= 1.0


def test_count_score_is_half_at_threshold_and_one_at_double():
    cfg = RuleConfig(arp_min=10)
    at = [pkt(i, "10.0.0.9", "10.0.0.1", proto="arp", dport=0) for i in range(10)]
    dbl = [pkt(i, "10.0.0.8", "10.0.0.1", proto="arp", dport=0) for i in range(20)]
    s = {e.host: e.score for e in run_rules(at + dbl, cfg)}
    assert s["10.0.0.9"] == pytest.approx(0.5) and s["10.0.0.8"] == pytest.approx(1.0)


def test_blocklist_ip_domain_and_feed_agreement(tmp_path):
    (tmp_path / "feedA.txt").write_text("# comment\n203.0.113.9\nevil.example\n")
    (tmp_path / "feedB.txt").write_text("203.0.113.9  # also here\n")
    bl = Blocklists.from_files([tmp_path / "feedA.txt", tmp_path / "feedB.txt"])
    pk = [pkt(1, "10.0.0.5", "203.0.113.9"), pkt(2, "10.0.0.7", "8.8.8.8", proto="udp", dport=53, qname="x.sub.evil.example"),
          pkt(3, "10.0.0.8", "198.51.100.1")]
    ev = {e.host: e for e in run_rules(pk, RuleConfig(), bl)}
    assert set(ev) == {"10.0.0.5", "10.0.0.7"}
    assert ev["10.0.0.5"].score == pytest.approx(0.55) and ev["10.0.0.7"].score == pytest.approx(0.4)   # 2 feeds vs 1 feed
    assert "feedA" in ev["10.0.0.7"].reason


def test_blocklist_attributes_evidence_to_the_internal_host_in_both_directions():
    bl = Blocklists({"f": {"203.0.113.9"}})
    ev = run_rules([pkt(1, "203.0.113.9", "10.0.0.5")], RuleConfig(), bl)             # inbound from a bad IP
    assert [e.host for e in ev] == ["10.0.0.5"]


def test_calibrate_rules_keeps_clean_false_alarms_near_target_and_still_catches_a_scan():
    import numpy as np
    from layered_nids.rules.engine import calibrate_rules
    rng = np.random.default_rng(0)
    clean = []
    for m in range(200):                                                  # 200 minutes of quiet traffic, 10 hosts
        for h in range(10):
            for _ in range(int(rng.integers(3, 12))):
                clean.append(pkt(m * 60 + rng.random() * 59, f"10.0.0.{h + 1}", f"10.0.1.{int(rng.integers(1, 12))}"))
    cfg = calibrate_rules(clean, fpr=0.01)
    fired = {(e.host, int(e.ts // 60)) for e in run_rules(clean, cfg)}
    assert len(fired) / (200 * 10) <= 0.02                                # target 1% of clean host-windows
    scan = [pkt(5000 + i * 0.05, "10.0.0.99", f"10.0.2.{i % 250}", dport=1000 + i) for i in range(300)]
    assert any(e.host == "10.0.0.99" for e in run_rules(scan, cfg))


def test_confidence_formula():
    assert [Blocklists.confidence(n) for n in (0, 1, 2, 5, 10)] == pytest.approx([0.0, 0.4, 0.55, 1.0, 1.0])


def test_is_internal():
    assert is_internal("192.168.1.4") and is_internal("10.2.3.4") and not is_internal("8.8.8.8") and not is_internal("garbage")


def _dns_query(name):
    q = b"".join(bytes([len(l)]) + l.encode() for l in name.split(".")) + b"\x00" + struct.pack("!HH", 1, 1)
    return struct.pack("!HHHHHH", 0x1234, 0x0100, 1, 0, 0, 0) + q


def test_dns_qname_parsing_and_ignores_responses_and_garbage():
    assert dns_qname(_dns_query("WWW.Example.COM")) == "www.example.com"
    resp = bytearray(_dns_query("a.com")); resp[2] |= 0x80
    assert dns_qname(bytes(resp)) == "" and dns_qname(b"\x00" * 5) == "" and dns_qname(_dns_query("a.com")[:15]) == ""


def test_udp_dns_packet_gets_qname_from_the_frame():
    import socket
    payload = _dns_query("evil.example")
    udp = struct.pack("!HHHH", 5353, 53, 8 + len(payload), 0) + payload
    ip = struct.pack("!BBHHHBBH4s4s", 0x45, 0, 20 + len(udp), 0, 0, 64, 17, 0,
                     socket.inet_aton("10.0.0.5"), socket.inet_aton("8.8.8.8"))
    frame = bytes.fromhex("aabbccddeeff112233445566") + b"\x08\x00" + ip + udp
    p = parse_ethernet(frame, 1.0, len(frame))
    assert (p.proto, p.dport, p.qname) == ("udp", 53, "evil.example")
