"""Recognise which kind of flow file this is, from its column names alone.

  unsw34      the UNSW-NB15 feature set the earlier project's model uses (34 columns)
  netflow_v2  NetFlow v2 features as in the NF-* datasets (40 feature columns, plus addresses and ports)
  argus       Argus flow files (the Argus flow format): StartTime, SrcAddr, DstAddr, Dport, TotPkts, TotBytes, ...

Names are compared lower-case. Extra columns (labels, ids) are fine; missing required columns give a clear error naming the closest match.
"""
from __future__ import annotations

from nids.config import FEATURES as UNSW_FEATURES

from layered_nids.csvmode.reader import CsvError

NETFLOW_FEATURES = [c.lower() for c in (
    "L4_DST_PORT", "PROTOCOL", "L7_PROTO", "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS", "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN", "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT", "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE", "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE")]
assert len(NETFLOW_FEATURES) == 40

ARGUS_REQUIRED = ["starttime", "srcaddr", "dstaddr", "dport", "totpkts", "totbytes"]

SCHEMAS = {
    "unsw34": {"required": list(UNSW_FEATURES), "kind": "ml"},
    "netflow_v2": {"required": NETFLOW_FEATURES, "kind": "ml"},
    "argus": {"required": ARGUS_REQUIRED, "kind": "rules"},
}


def detect_schema(columns, forced: str | None = None) -> str:
    """Return the schema name, or raise CsvError explaining the closest match and what is missing."""
    cols = {str(c).strip().lower() for c in columns}
    if forced:
        if forced not in SCHEMAS:
            raise CsvError(f"unknown schema {forced!r}; choose one of {sorted(SCHEMAS)}")
        missing = [c for c in SCHEMAS[forced]["required"] if c not in cols]
        if missing:
            raise CsvError(f"the file does not have the columns {forced} needs; missing: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
        return forced
    best, best_missing, best_frac = None, None, 2.0
    for name, s in SCHEMAS.items():
        missing = [c for c in s["required"] if c not in cols]
        if not missing:
            return name
        frac = len(missing) / len(s["required"])                # closest = the smallest SHARE of required columns missing
        if frac < best_frac:
            best, best_missing, best_frac = name, missing, frac
    raise CsvError(f"unrecognised flow file. Closest format: {best} (missing {len(best_missing)} of {len(SCHEMAS[best]['required'])} required "
                   f"columns, for example {best_missing[:6]}). Known formats: {sorted(SCHEMAS)}")
