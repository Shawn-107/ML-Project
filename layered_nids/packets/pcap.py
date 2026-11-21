"""Read packets from a capture file into small records (own parser; scapy is used only to read the file container).

Supported: classic pcap (tested on the Mirai example capture and on files written by our tests) with Ethernet link
type; IPv4, IPv6, ARP; TCP and UDP ports. pcapng goes through scapy's reader and has NOT been tested here.
Anything else (other link types, truncated headers) yields a record with the fields it could read and zeros elsewhere,
so packet numbering stays aligned with any label file.
"""
from __future__ import annotations

import socket
import struct
from pathlib import Path
from typing import Iterator, NamedTuple

LINKTYPE_ETHERNET = 1


class Packet(NamedTuple):
    ts: float          # seconds (capture clock)
    size: int          # length on the wire, bytes (not the truncated capture length)
    src_mac: str
    dst_mac: str
    src: str           # source IP, or MAC if the frame has no IP/ARP layer
    dst: str
    proto: str         # 'tcp', 'udp', 'icmp', 'arp', 'ipN' (IP protocol number N), 'eth'
    sport: int
    dport: int
    qname: str = ""    # DNS query name for UDP queries to port 53 (lower-case, no trailing dot), else ""
    flags: int = 0     # TCP flag byte (SYN = 0x02, ACK = 0x10, ...), 0 for other protocols
    httpreq: str = ""  # "METHOD host path" for the first line of a plain HTTP request (at most 300 characters), else ""
    sni: str = ""      # server name (lower-case) of a TLS ClientHello, so encrypted web traffic still shows WHERE it goes, else ""


def tls_sni(payload: bytes) -> str:
    """Server name from a TLS ClientHello that starts this TCP payload, or ''. Bounds-checked; anything unusual gives ''."""
    try:
        if len(payload) < 48 or payload[0] != 0x16 or payload[1] != 0x03 or payload[5] != 0x01:      # handshake record, handshake type ClientHello
            return ""
        i = 5 + 4 + 2 + 32                                                                             # record header, handshake header, version, random
        i += 1 + payload[i]                                                                            # session id
        i += 2 + struct.unpack("!H", payload[i:i + 2])[0]                                              # cipher suites
        i += 1 + payload[i]                                                                            # compression methods
        end = i + 2 + struct.unpack("!H", payload[i:i + 2])[0]
        i += 2
        end = min(end, len(payload))
        while i + 4 <= end:
            etype, elen = struct.unpack("!HH", payload[i:i + 4])
            i += 4
            if etype == 0 and elen >= 5:                                                               # server_name
                nlen = struct.unpack("!H", payload[i + 3:i + 5])[0]
                name = payload[i + 5:i + 5 + nlen]
                if payload[i + 2] == 0 and 0 < nlen <= 253 and len(name) == nlen:
                    return name.decode("ascii", "replace").lower().strip(".")
                return ""
            i += elen
    except (IndexError, struct.error):
        return ""
    return ""


def dns_qname(payload: bytes) -> str:
    """First question name of a DNS query, or '' if this is not a well-formed query (responses are ignored)."""
    if len(payload) < 13 or payload[2] & 0x80 or struct.unpack("!H", payload[4:6])[0] < 1:
        return ""
    labels, i = [], 12
    while i < len(payload):
        n = payload[i]
        if n == 0:
            return ".".join(labels).lower()
        if n & 0xC0 or i + 1 + n > len(payload):          # compression pointer or truncated: give up
            return ""
        labels.append(payload[i + 1:i + 1 + n].decode("ascii", "replace"))
        i += 1 + n
    return ""


_HTTP_METHODS = (b"GET ", b"POST ", b"HEAD ", b"PUT ", b"DELETE ", b"OPTIONS ", b"PATCH ", b"CONNECT ")


def http_request(payload: bytes) -> str:
    """"METHOD host path" from the start of a TCP payload that looks like an HTTP request, else ''. Only the first 600 bytes
    are looked at; a request split over several packets is not reassembled."""
    if not payload.startswith(_HTTP_METHODS):
        return ""
    head = payload[:600].split(b"\r\n\r\n", 1)[0]
    lines = head.split(b"\r\n")
    parts = lines[0].split(b" ")
    if len(parts) < 2:
        return ""
    host = ""
    for ln in lines[1:]:
        if ln[:5].lower() == b"host:":
            host = ln[5:].strip().decode("ascii", "replace").lower()
            break
    return f"{parts[0].decode('ascii', 'replace')} {host} {parts[1].decode('ascii', 'replace')}"[:300]


def _mac(b: bytes) -> str:
    return b.hex(":")


def parse_ethernet(buf: bytes, ts: float, wirelen: int) -> Packet:
    if len(buf) < 14:
        return Packet(ts, wirelen, "", "", "", "", "eth", 0, 0)
    dmac, smac = _mac(buf[0:6]), _mac(buf[6:12])
    et = struct.unpack("!H", buf[12:14])[0]
    off = 14
    if et == 0x8100 and len(buf) >= 18:                    # one VLAN tag
        et = struct.unpack("!H", buf[16:18])[0]
        off = 18
    if et == 0x0800 and len(buf) >= off + 20:              # IPv4
        ihl = (buf[off] & 0x0F) * 4
        p = buf[off + 9]
        src, dst = socket.inet_ntoa(buf[off + 12:off + 16]), socket.inet_ntoa(buf[off + 16:off + 20])
        return _l4(buf, off + ihl, p, ts, wirelen, smac, dmac, src, dst)
    if et == 0x86DD and len(buf) >= off + 40:              # IPv6 (extension headers are not walked)
        p = buf[off + 6]
        src = socket.inet_ntop(socket.AF_INET6, buf[off + 8:off + 24])
        dst = socket.inet_ntop(socket.AF_INET6, buf[off + 24:off + 40])
        return _l4(buf, off + 40, p, ts, wirelen, smac, dmac, src, dst)
    if et == 0x0806 and len(buf) >= off + 28:              # ARP: sender and target protocol addresses
        src, dst = socket.inet_ntoa(buf[off + 14:off + 18]), socket.inet_ntoa(buf[off + 24:off + 28])
        return Packet(ts, wirelen, smac, dmac, src, dst, "arp", 0, 0)
    return Packet(ts, wirelen, smac, dmac, smac, dmac, "eth", 0, 0)


_PROTO = {6: "tcp", 17: "udp", 1: "icmp", 58: "icmp"}


def _l4(buf, off, p, ts, wirelen, smac, dmac, src, dst) -> Packet:
    name = _PROTO.get(p, f"ip{p}")
    sport = dport = flags = 0
    if p in (6, 17) and len(buf) >= off + 4:
        sport, dport = struct.unpack("!HH", buf[off:off + 4])
    qname = dns_qname(buf[off + 8:]) if p == 17 and dport == 53 else ""
    httpreq = sni = ""
    if p == 6 and len(buf) >= off + 20:
        flags = buf[off + 13]
        payload = buf[off + ((buf[off + 12] >> 4) * 4):]
        httpreq = http_request(payload)
        if payload[:1] == b"":
            sni = tls_sni(payload)
    return Packet(ts, wirelen, smac, dmac, src, dst, name, sport, dport, qname, flags, httpreq, sni)


def iter_packets(path: str | Path, limit: int | None = None) -> Iterator[Packet]:
    """Yield Packet records in file order."""
    from scapy.utils import RawPcapNgReader, RawPcapReader

    p = Path(path)
    with p.open("rb") as f:
        magic = f.read(4)
    ng = magic == b"\x0a\x0d\x0d\x0a"
    reader = RawPcapNgReader(str(p)) if ng else RawPcapReader(str(p))
    n = 0
    try:
        if not ng and getattr(reader, "linktype", LINKTYPE_ETHERNET) != LINKTYPE_ETHERNET:
            raise ValueError(f"unsupported link type {reader.linktype} (only Ethernet)")
        for buf, meta in reader:
            if ng:
                res = getattr(meta, "tsresol", 1_000_000) or 1_000_000
                ts = ((meta.tshigh << 32) | meta.tslow) / res
                wl = meta.wirelen
            else:
                ts = meta.sec + meta.usec / 1e6
                wl = meta.wirelen
            yield parse_ethernet(buf, ts, wl or len(buf))
            n += 1
            if limit is not None and n >= limit:
                break
    finally:
        reader.close()


def write_pcap(path: str | Path, packets: list[tuple[float, bytes]]) -> None:
    """Minimal classic-pcap writer (Ethernet, microsecond timestamps). Used by tests and demos."""
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, LINKTYPE_ETHERNET))
        for ts, data in packets:
            sec = int(ts)
            f.write(struct.pack("<IIII", sec, int(round((ts - sec) * 1e6)), len(data), len(data)))
            f.write(data)
