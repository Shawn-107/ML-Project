"""TLS server names, the popular-domain whitelist and the DNS-tunnel rule."""
import struct

import numpy as np

from layered_nids.packets.pcap import Packet, tls_sni
from layered_nids.signatures import scan
from layered_nids.signatures.trails import Trails

H = "192.168.2.50"


def client_hello(name: str) -> bytes:
    nb = name.encode()
    sn = b"\x00" + struct.pack("!H", len(nb)) + nb
    ext = struct.pack("!HH", 0, len(sn) + 2) + struct.pack("!H", len(sn)) + sn
    ext += struct.pack("!HH", 0x0010, 5) + b"\x00\x03\x02h2"                              # another extension after the name
    body = (b"\x03\x03" + b"\x11" * 32 + b"\x00" + struct.pack("!H", 4) + b"\x13\x01\x00\x2f" + b"\x01\x00"
            + struct.pack("!H", len(ext)) + ext)
    hs = b"\x01" + struct.pack("!I", len(body))[1:] + body
    return b"\x16\x03\x01" + struct.pack("!H", len(hs)) + hs


def pk(ts, dst="93.184.216.34", sni="", qname="", proto="tcp", dport=443):
    return Packet(float(ts), 200, "aa", "bb", H, dst, proto, 40000, dport, qname, 0x18, "", sni)


def test_tls_sni_is_read_from_a_client_hello():
    assert tls_sni(client_hello("Login.Example.org")) == "login.example.org"
    assert tls_sni(client_hello("a.b")[:60]) == ""                                       # cut off: no crash, no name
    assert tls_sni(b"") == "" and tls_sni(b"GET / HTTP/1.1\r\n") == "" and tls_sni(b"\x16\x03\x01" + b"\xff" * 80) == ""


def test_known_bad_name_in_https_is_matched_and_whitelist_wins(tmp_path):
    t = Trails.from_entries(["evil.example", "cdn.popular.example"])
    hits = scan([pk(1, sni="x.evil.example")], trails=t)
    assert len(hits) == 1 and hits[0].detail["signature"] is True and hits[0].detail["trail"] == "evil.example"
    wl = tmp_path / "wl.txt"
    wl.write_text("# popular\npopular.example\n1,other.example\n")
    assert t.load_whitelist(wl) == 2
    assert not scan([pk(1, sni="cdn.popular.example")], trails=t)                        # whitelisted parent: ignored
    assert scan([pk(2, sni="x.evil.example")], trails=t)                                   # others still match


def test_random_tls_names_and_dns_tunnel_rule():
    rng = np.random.default_rng(3)
    names = ["".join(rng.choice(list("abcdefghijklmnopqrstuvwxyz0123456789"), 16)) + ".example.net" for _ in range(60)]
    ev = scan([pk(i * 0.5, sni=n) for i, n in enumerate(names)])
    assert "dns_random" in {e.detail["trail"] for e in ev}
    long_q = ["".join(rng.choice(list("abcdef0123456789"), 62)) + ".t.example.com" for _ in range(15)]
    ev2 = scan([pk(i * 0.5, dst="8.8.8.8", proto="udp", dport=53, qname=q) for i, q in enumerate(long_q)])
    assert "dns_tunnel" in {e.detail["trail"] for e in ev2}
    assert not scan([pk(i, sni="www.example.org") for i in range(30)])                  # ordinary names: nothing
