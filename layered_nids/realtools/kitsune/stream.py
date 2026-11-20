"""Packet-by-packet use of the real Kitsune (AfterImage statistics + KitNET autoencoders).

Upstream reads a whole file; this feeds one parsed packet at a time, so the same code can run on a pcap or on a live
stream of packets. It calls upstream's own `netStat.updateGetStats` and `KitNET.process`; only the packet-field
mapping below is ours, copied from the scapy branch of upstream's FeatureExtractor.get_next_vector so that the
features are identical (checked by tests/test_kitsune_vendored.py against the original on the Mirai capture).

Phases (KitNET's, unchanged): the first `fm_grace` packets learn which features go together; the next `ad_grace`
packets train the autoencoders; `process` returns 0.0 for all of these. After that it returns the reconstruction
error (RMSE) of each packet. A high score means "unlike the traffic seen while learning". Learning must therefore
happen on clean traffic.
"""
from __future__ import annotations

import numpy as np

from layered_nids.realtools.kitsune.kitnet.kitnet import KitNET
from layered_nids.realtools.kitsune.netstat import netStat

_BIG = 100000000000        # upstream's FeatureExtractor passes this for both host and session limits


def kitsune_fields(p):
    """Our Packet record -> the 8 values Kitsune's feature code takes:
    (IPtype, srcMAC, dstMAC, srcIP, srcProto, dstIP, dstProto, frame_len).
    `p` needs: size, src_mac, dst_mac, src, dst, proto, sport, dport (see packets/pcap.py:Packet)."""
    proto = p.proto
    if proto in ("tcp", "udp"):
        return (1 if ":" in p.src else 0, p.src_mac, p.dst_mac, p.src, str(p.sport), p.dst, str(p.dport), p.size)
    if proto == "arp":                                    # upstream: ports 'arp', addresses = the ARP sender/target IPs
        return (0, p.src_mac, p.dst_mac, p.src, "arp", p.dst, "arp", p.size)
    if proto == "icmp":                                   # IPv4 ICMP only (upstream tests scapy's ICMP layer)
        if ":" in p.src:                                  # our parser also calls ICMPv6 'icmp'; upstream treats it as plain IPv6
            return (1, p.src_mac, p.dst_mac, p.src, "", p.dst, "", p.size)
        return (0, p.src_mac, p.dst_mac, p.src, "icmp", p.dst, "icmp", p.size)
    if proto.startswith("ip"):                            # other IP protocols, incl. ICMPv6: IPs kept, no ports
        return (1 if ":" in p.src else 0, p.src_mac, p.dst_mac, p.src, "", p.dst, "", p.size)
    return (np.nan, p.src_mac, p.dst_mac, p.src_mac, "", p.dst_mac, "", p.size)   # no IP: MAC addresses stand in


class KitsuneStream:
    def __init__(self, max_ae: int = 10, fm_grace: int = 5000, ad_grace: int = 50000,
                 learning_rate: float = 0.1, hidden_ratio: float = 0.75):
        self.nstat = netStat(np.nan, _BIG, _BIG)
        self.n_features = len(self.nstat.getNetStatHeaders())
        self.kitnet = KitNET(self.n_features, max_ae, fm_grace, ad_grace, learning_rate, hidden_ratio)
        self.fm_grace, self.ad_grace = fm_grace, ad_grace
        self.n_seen = 0

    @property
    def ready(self) -> bool:
        """True once learning is finished: the NEXT process() call returns a real score."""
        return self.n_seen > self.fm_grace + self.ad_grace

    def features(self, p) -> np.ndarray:
        ip_type, smac, dmac, sip, sproto, dip, dproto, size = kitsune_fields(p)
        return self.nstat.updateGetStats(ip_type, smac, dmac, sip, sproto, dip, dproto, int(size), float(p.ts))

    def process_with_features(self, p):
        """(feature vector, score) for one packet. The score is 0.0 while still learning."""
        x = self.features(p)
        self.n_seen += 1
        return x, float(self.kitnet.process(x))

    def process(self, p) -> float:
        """Score one packet (0.0 while still learning). Packets must arrive in capture order."""
        return self.process_with_features(p)[1]
