"""Rule-based findings for flow files that carry a host and a time (Argus format, the Argus flow format). No machine learning:
there is no labelled Argus data here to train on, and a clear rule is better than an unmeasurable model.

  vertical_scan    one source touches at least `min_ports` different destination ports on ONE destination within `window_s` seconds
  horizontal_scan  one source touches the SAME (uncommon) destination port on at least `min_hosts` different destinations within the window
  icmp_sweep       one source sends ICMP to at least `min_hosts` different destinations within the window (a ping sweep)

Windows SLIDE (any `window_s`-second stretch counts), so a slow scan that straddles a clock-minute boundary is still found.
Ports 53, 80, 123 and 443 are excluded from the horizontal rule: touching many hosts on those ports is ordinary browsing and DNS.
Thresholds (15 ports, 15 hosts, 60 s) are our defaults, not tuned on labelled data; change them to suit the network.
Rows whose address or time cannot be read are ignored by the rules and counted, never guessed; rows with a non-numeric port
(ICMP ids such as 0x7208) take part only in the ICMP rule.
"""
from __future__ import annotations

import ipaddress
from collections import defaultdict

import numpy as np
import pandas as pd

COMMON_PORTS = {53, 80, 123, 443}
FINDING_COLUMNS = ["host", "kind", "window_start", "distinct", "target", "severity"]


def _valid_ip_mask(series: pd.Series) -> pd.Series:
    ok = {}
    for v in series.unique():
        try:
            ipaddress.ip_address(v)
            ok[v] = True
        except ValueError:
            ok[v] = False
    return series.map(ok).astype(bool)


def parse_times(series: pd.Series) -> pd.Series:
    """Epoch seconds from numbers or date strings (UTC); NaN where unreadable."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.9:
        return num.astype(float)
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    return (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()


def max_distinct_in_window(times: np.ndarray, items: np.ndarray, window_s: float) -> tuple[int, float]:
    """The largest number of distinct `items` seen in any window of `window_s` seconds, and the start time of the first such window.
    `times` must be sorted ascending. Two pointers, linear time."""
    counts: dict = defaultdict(int)
    best, best_start, left, distinct = 0, float(times[0]) if len(times) else 0.0, 0, 0
    for right in range(len(times)):
        it = items[right]
        if counts[it] == 0:
            distinct += 1
        counts[it] += 1
        while times[right] - times[left] > window_s:
            old = items[left]
            counts[old] -= 1
            if counts[old] == 0:
                distinct -= 1
            left += 1
        if distinct > best:
            best, best_start = distinct, float(times[left])
    return best, best_start


def scan_findings(df: pd.DataFrame, ts_col: str = "starttime", src_col: str = "srcaddr", dst_col: str = "dstaddr",
                  port_col: str = "dport", proto_col: str = "proto", window_s: float = 60.0, min_ports: int = 15,
                  min_hosts: int = 15) -> tuple[pd.DataFrame, dict]:
    """Return (findings, facts). findings columns: host, kind, window_start (epoch s), distinct, target, severity."""
    t = parse_times(df[ts_col])
    ok = _valid_ip_mask(df[src_col]) & _valid_ip_mask(df[dst_col]) & t.notna()
    port = pd.to_numeric(df[port_col], errors="coerce")
    proto = df[proto_col].str.lower() if proto_col in df.columns else pd.Series("", index=df.index)
    is_icmp = proto.str.startswith("icmp") | proto.eq("ipv6-icmp")
    facts = {"rows": int(len(df)), "rows_used_by_rules": int(ok.sum()), "rows_ignored": int((~ok).sum()),
             "rows_without_numeric_port": int((ok & port.isna()).sum()), "rows_with_clock_before_1990": int((t[ok] < 631152000).sum())}
    d = pd.DataFrame({"src": df[src_col][ok], "dst": df[dst_col][ok], "port": port[ok], "t": t[ok], "icmp": is_icmp[ok]}).sort_values("t")
    out = []

    def add(host, kind, start, n, target, threshold):
        out.append({"host": host, "kind": kind, "window_start": start, "distinct": int(n), "target": target,
                    "severity": "high" if n >= 3 * threshold else "medium"})

    tcpudp = d[d["port"].notna() & ~d["icmp"]]
    # vertical: per (src, dst), distinct destination ports
    for (src, dst), g in tcpudp.groupby(["src", "dst"], sort=False):
        if g["port"].nunique() >= min_ports:
            n, start = max_distinct_in_window(g["t"].to_numpy(), g["port"].to_numpy(), window_s)
            if n >= min_ports:
                add(src, "vertical_scan", start, n, dst, min_ports)
    # horizontal: per (src, uncommon port), distinct destinations
    uncommon = tcpudp[~tcpudp["port"].astype("int64").isin(COMMON_PORTS)]
    for (src, p), g in uncommon.groupby(["src", "port"], sort=False):
        if g["dst"].nunique() >= min_hosts:
            n, start = max_distinct_in_window(g["t"].to_numpy(), g["dst"].to_numpy(), window_s)
            if n >= min_hosts:
                add(src, "horizontal_scan", start, n, f"port {int(p)}", min_hosts)
    # ICMP sweep: per src, distinct ICMP destinations
    for src, g in d[d["icmp"]].groupby("src", sort=False):
        if g["dst"].nunique() >= min_hosts:
            n, start = max_distinct_in_window(g["t"].to_numpy(), g["dst"].to_numpy(), window_s)
            if n >= min_hosts:
                add(src, "icmp_sweep", start, n, "many hosts", min_hosts)
    return pd.DataFrame(out, columns=FINDING_COLUMNS), facts


def flag_rows(df: pd.DataFrame, findings: pd.DataFrame, ts_col="starttime", src_col="srcaddr", dst_col="dstaddr", port_col="dport",
              proto_col="proto", window_s: float = 60.0) -> pd.Series:
    """A short text per flow row that took part in a finding ('' for the others)."""
    notes = pd.Series("", index=df.index)
    if findings.empty:
        return notes
    t = parse_times(df[ts_col])
    port = pd.to_numeric(df[port_col], errors="coerce")
    proto = df[proto_col].str.lower() if proto_col in df.columns else pd.Series("", index=df.index)
    icmp = proto.str.startswith("icmp") | proto.eq("ipv6-icmp")
    for r in findings.itertuples():
        base = (df[src_col] == r.host) & (t >= r.window_start) & (t <= r.window_start + window_s)
        if r.kind == "vertical_scan":
            m, what = base & (df[dst_col] == r.target) & port.notna() & ~icmp, "ports"
        elif r.kind == "horizontal_scan":
            m, what = base & (port == int(r.target.split()[-1])) & ~icmp, "hosts"
        else:
            m, what = base & icmp, "hosts"
        notes[m] = f"{r.kind}: {r.distinct} distinct {what} ({r.target})"
    return notes
