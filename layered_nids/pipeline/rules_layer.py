"""Rules layer of the combined pipeline: our own signature and heuristic rules on the packets -> per-host evidence.

Uses layered_nids.signatures.scan when that package exists (pattern signatures, blocklists, heuristics); until then it falls
back to the simpler count rules in layered_nids.rules (scan, ARP flood, DNS burst), which are all heuristics.
Evidence.detail["signature"] is True for a pattern or blocklist hit and False for a heuristic; detail["trail"] is the key.
"""
from __future__ import annotations

from layered_nids.common.hosts import pick_internal
from layered_nids.common.schema import Evidence

try:                                                     # the newer rules package, when present
    from layered_nids.signatures import scan as _scan
except ImportError:                                      # pragma: no cover - depends on what is installed
    _scan = None


def is_signature(e: Evidence) -> bool:
    return e.layer == "rules" and bool(e.detail.get("signature"))


def rules_evidence(packets, blocklists=None) -> list[Evidence]:
    packets = list(packets)
    if _scan is not None:
        return list(_scan(packets, blocklists=blocklists))
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
