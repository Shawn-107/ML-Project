"""NIDS signature and heuristic layer (layer name "rules" in the evidence schema). Own implementation, no machine learning.

    from layered_nids.signatures import scan
    evidence = scan(packets)                 # list of common.schema.Evidence, sorted by time

Every Evidence carries detail["signature"]: True for a pattern or blocklist hit (a known-bad indicator), False for a behaviour
heuristic (counts or regularity, which can fire on odd but harmless traffic); and detail["trail"], the key of the rule that fired.
"""
from __future__ import annotations

from collections import defaultdict

from layered_nids.common.hosts import pick_internal
from layered_nids.common.schema import Evidence
from layered_nids.signatures.heuristics import HeuristicConfig, beaconing, calibrate, count_score, window_features
from layered_nids.signatures.patterns import SIGNATURES, match_request

__all__ = ["scan", "HeuristicConfig", "calibrate", "SIGNATURES", "match_request"]
SignatureConfig = HeuristicConfig


def scan(packets, cfg: HeuristicConfig | None = None, blocklists=None, trails=None) -> list[Evidence]:
    cfg = cfg or HeuristicConfig()
    packets = list(packets)
    out: list[Evidence] = []
    ts = lambda w: w * cfg.window_s

    # ---- known-bad indicators (signature = True) --------------------------------------------------------------------------
    hits: dict = {}
    for p in packets:
        w = int(p.ts // cfg.window_s)
        h = pick_internal(p.src, p.dst)
        if not h:
            continue
        for name, sev in match_request(p.httpreq):
            key = (h, w, "sig", name)
            e = hits.setdefault(key, {"n": 0, "score": sev, "first": p, "name": name})
            e["n"] += 1
        if trails is not None:
            for indicator, kind in trails.match(p):
                hits.setdefault((h, w, "trail", indicator), {"n": 0, "score": 0.85, "first": p, "name": indicator, "tkind": kind})["n"] += 1
        if blocklists is not None:
            if p.qname:
                f = blocklists.feeds_for_domain(p.qname)
                if f:
                    hits.setdefault((h, w, "domain", p.qname), {"n": 0, "score": blocklists.confidence(len(f)), "first": p, "name": f"blocklisted domain {p.qname}", "feeds": sorted(f)})["n"] += 1
            if p.proto != "arp":
                other = p.dst if h == p.src else p.src
                f = blocklists.feeds_for_ip(other)
                if f:
                    hits.setdefault((h, w, "ip", other), {"n": 0, "score": blocklists.confidence(len(f)), "first": p, "name": f"blocklisted address {other}", "feeds": sorted(f)})["n"] += 1
    info = trails.describe({e["name"] for (_, _, kind, _), e in hits.items() if kind == "trail"}) if trails is not None and hits else {}
    for (h, w, kind, _), e in hits.items():
        p = e["first"]
        what = f"{e['name']}" + (f" (feeds: {', '.join(e['feeds'])})" if "feeds" in e else "")
        if kind == "trail":
            what = f"known-bad {e['tkind']} {e['name']}" + (f" ({info[e['name']]})" if info.get(e["name"]) else "")
        req = f" request '{p.httpreq[:160]}'" if kind == "sig" and p.httpreq else ""
        score, is_sig = float(e["score"]), True
        if kind == "trail" and (info.get(e["name"]) or "").rstrip().endswith("(suspicious)"):
            score, is_sig = 0.3, False        # "suspicious" indicators include public DNS servers and mass scanners: weak evidence, not a signature hit
        out.append(Evidence("rules", h, ts(w), score, f"{h}: {what}{req}" + (f" ({e['n']} times)" if e["n"] > 1 else ""),
                            {"signature": is_sig, "trail": e["name"], "kind": kind, "count": e["n"], "src": p.src, "dst": p.dst,
                             **({"request": p.httpreq[:300]} if p.httpreq else {})}))

    # ---- behaviour heuristics (signature = False) -------------------------------------------------------------------------
    for (h, w), c in window_features(packets, cfg.window_s).items():
        if len(c.targets) >= cfg.scan_min_targets:
            out.append(Evidence("rules", h, ts(w), count_score(len(c.targets), cfg.scan_min_targets),
                                f"{h} contacted {len(c.targets)} distinct targets in {cfg.window_s:g}s (scan-like)",
                                {"signature": False, "trail": "scan", "targets": len(c.targets)}))
        for kind, per_dst, noun in (("syn_flood", c.per_dst_syn, "TCP connection attempts"), ("udp_flood", c.per_dst_udp, "UDP packets")):
            if per_dst:
                dst, n = max(per_dst.items(), key=lambda kv: kv[1])
                if n >= cfg.flood_min_packets:
                    out.append(Evidence("rules", h, ts(w), count_score(n, cfg.flood_min_packets),
                                        f"{h} sent {n} {noun} to {dst} in {cfg.window_s:g}s (flood-like)",
                                        {"signature": False, "trail": kind, "packets": n, "target": dst}))
        if c.arp >= cfg.arp_min:
            out.append(Evidence("rules", h, ts(w), count_score(c.arp, cfg.arp_min), f"{h} sent {c.arp} ARP packets in {cfg.window_s:g}s (ARP flood or sweep)",
                                {"signature": False, "trail": "arp_flood", "packets": c.arp}))
        if c.dns >= cfg.dns_min:
            out.append(Evidence("rules", h, ts(w), count_score(c.dns, cfg.dns_min), f"{h} sent {c.dns} DNS queries in {cfg.window_s:g}s",
                                {"signature": False, "trail": "dns_burst", "queries": c.dns}))
        if c.random_names >= cfg.dns_random_min:
            out.append(Evidence("rules", h, ts(w), count_score(c.random_names, cfg.dns_random_min),
                                f"{h} asked for {c.random_names} random-looking names in {cfg.window_s:g}s (e.g. {c.random_examples[0]})",
                                {"signature": False, "trail": "dns_random", "names": c.random_names, "examples": c.random_examples}))
    for h, dst, port, gap, cv, n, t0, t1 in beaconing(packets, cfg):
        out.append(Evidence("rules", h, t1, 0.6, f"{h} contacted {dst}:{port} {n} times about every {gap:.0f}s (spread {cv:.0%}): beacon-like",
                            {"signature": False, "trail": "beaconing", "dst": dst, "port": port, "mean_gap_s": gap, "cv": cv, "events": n, "first": t0}))
    return sorted(out, key=lambda e: (e.ts, e.host, e.reason))
