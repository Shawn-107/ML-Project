"""Rules layer of the combined pipeline: our own signature and heuristic rules on the packets -> per-host evidence.

Uses layered_nids.signatures.scan when that package exists (pattern signatures, blocklists, heuristics); until then it falls
back to the simpler count rules in layered_nids.rules (scan, ARP flood, DNS burst), which are all heuristics.
Evidence.detail["signature"] is True for a pattern or blocklist hit and False for a heuristic; detail["trail"] is the key.
"""
from __future__ import annotations

from pathlib import Path

from layered_nids.common.hosts import pick_internal
from layered_nids.common.paths import DATA
from layered_nids.common.schema import Evidence

try:                                                     # the newer rules package, when present
    from layered_nids.signatures import scan as _scan
except ImportError:                                      # pragma: no cover - depends on what is installed
    _scan = None


def is_signature(e: Evidence) -> bool:
    return e.layer == "rules" and bool(e.detail.get("signature"))


TRAILS_FILE = DATA / "trails" / "trails.csv"                 # local indicator data (never committed); optional popular-domain list next to it
WHITELIST_FILE = DATA / "trails" / "whitelist.txt"


def load_default_trails():
    """The local indicator file, if the user put one in data/trails/ (with an optional whitelist.txt); otherwise None."""
    if not TRAILS_FILE.exists():
        return None
    from layered_nids.signatures.trails import Trails
    t = Trails.load(TRAILS_FILE)
    if WHITELIST_FILE.exists():
        t.load_whitelist(WHITELIST_FILE)
    return t


def rules_evidence(packets, blocklists=None, trails="auto", external_pcap=None) -> list[Evidence]:
    """Rule evidence for the packets. `trails`: "auto" loads data/trails/trails.csv when it exists, None skips indicator matching, or pass a Trails object.
    `external_pcap`: path of the same capture; when given and an external sensor is configured (see signatures/external.py), its events are added."""
    packets = list(packets)
    if _scan is not None:
        tr = load_default_trails() if isinstance(trails, str) and trails == "auto" else trails
        ev = list(_scan(packets, blocklists=blocklists, trails=tr))
        if external_pcap is not None:
            from layered_nids.signatures import external
            if external.configured():
                ev += external.run_external_sensor(external_pcap)
        return ev
    from layered_nids.rules.engine import RuleConfig, calibrate_rules, run_rules
    cfg = calibrate_rules(packets[: max(len(packets) // 4, 1000)])           # thresholds from the first quarter (assumed clean)
    out = []
    for e in run_rules(packets, cfg, blocklists):
        host = pick_internal(e.host, "")
        if not host:
            continue
        sig = e.detail.get("rule") == "blocklist"
        out.append(Evidence(e.layer, host, e.ts, e.score, e.reason, {**e.detail, "signature": sig,
                                                                       "trail": e.detail.get("indicator", e.detail.get("rule"))}))
    return out
