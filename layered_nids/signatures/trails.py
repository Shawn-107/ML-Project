"""NIDS indicator matching: known-bad addresses, address:port pairs, domains, URL paths and name patterns from a LOCAL indicator file.

The indicator data is NOT part of this repository and is not redistributed: put the file in data/trails/ (git-ignored) yourself and
respect the data's own terms (the community "trails" data is free for defending your own systems and for research and education; it must
not be built into a product or service for third parties). File format: CSV lines `indicator,description,source`; lines starting with
`#` are comments. Indicator kinds, told apart by their shape:

  1.2.3.4 or 1.2.3.4/        an address                          (either end of a packet)
  1.2.3.4:8080               an address and a destination port
  example.bad                a domain (it also matches every sub-domain: a.b.example.bad)
  /path/file.ext             a URL path (matched on the HTTP request path, without the query)
  host.example/path          a host plus path
  helplog[0-9]{3,4}\\.tk      a regular expression (anything with brackets, parentheses, braces or a backslash)

`Trails.match(packet)` returns the hits for ONE packet as [(indicator, kind)]; `describe` adds the descriptions. The matcher keeps only sets in memory
(about 1.9 million indicators need roughly 400 MB); descriptions are looked up from the file lazily, only for indicators that hit.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from urllib.parse import urlsplit

_IP = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}/?$")
_IPPORT = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}:\d{1,5}$")
_REGEX_CHARS = set("[](){}\\^$|*+?")


def _clean_ip(s: str) -> str:
    return s.rstrip("/")


class Trails:
    def __init__(self):
        self.ips: set = set()
        self.ip_ports: set = set()
        self.domains: set = set()
        self.paths: set = set()           # '/dir/file.ext'
        self.host_paths: set = set()      # 'host/dir/file'
        self.regexes: list = []           # (compiled, source)
        self.path: Path | None = None
        self._info: dict = {}
        self.whitelist: set = set()       # popular domains whose matches are ignored (see load_whitelist)

    def __len__(self):
        return len(self.ips) + len(self.ip_ports) + len(self.domains) + len(self.paths) + len(self.host_paths) + len(self.regexes)

    @classmethod
    def load(cls, path) -> "Trails":
        t = cls()
        t.path = Path(path)
        with open(t.path, encoding="utf-8", errors="replace", newline="") as f:
            for row in csv.reader(l for l in f if not l.startswith("#")):
                if row:
                    t._add(row[0].strip())
        return t

    @classmethod
    def from_entries(cls, entries) -> "Trails":
        t = cls()
        for e in entries:
            t._add(e)
        return t

    def load_whitelist(self, path) -> int:
        """Ignore indicator matches on these domains and all their sub-domains. One domain per line, '#' starts a comment; a CSV with the domain
        in the last column (rank,domain) also works. Returns how many domains are loaded."""
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                d = line.split("#", 1)[0].strip().split(",")[-1].strip().lower().strip(".")
                if d and "." in d and " " not in d:
                    self.whitelist.add(d)
        return len(self.whitelist)

    def _whitelisted(self, name: str) -> bool:
        parts = name.strip(".").lower().split(".")
        return any(".".join(parts[i:]) in self.whitelist for i in range(len(parts) - 1))

    def _add(self, s: str) -> None:
        if not s:
            return
        if _IP.match(s):
            self.ips.add(_clean_ip(s))
        elif _IPPORT.match(s):
            self.ip_ports.add(s)
        elif "/" not in s and any(c in _REGEX_CHARS for c in s):          # names with brackets, braces, parentheses...: patterns
            try:
                self.regexes.append((re.compile(s, re.IGNORECASE), s))
            except re.error:
                pass
        elif s.startswith("/"):
            self.paths.add(s.split("?", 1)[0].lower())
        elif "/" in s:
            self.host_paths.add(s.split("?", 1)[0].lower().rstrip("/"))
        else:
            self.domains.add(s.lower().strip("."))

    # ------------------------------------------------------------------------------------------------ matching
    def _domain_hit(self, name: str):
        name = name.strip(".").lower()
        parts = name.split(".")
        for i in range(len(parts) - 1):                 # a.b.c.d -> a.b.c.d, b.c.d, c.d (never the bare TLD)
            cand = ".".join(parts[i:])
            if cand in self.domains:
                return cand
        return None

    def match(self, p) -> list[tuple[str, str, str]]:
        """Hits for one Packet: [(indicator, kind)] with kind in ip, ip:port, domain, path, host+path, pattern."""
        hits = []
        for ip in (p.dst, p.src):
            if ip in self.ips:
                hits.append((ip, "ip"))
        if self.ip_ports and f"{p.dst}:{p.dport}" in self.ip_ports:
            hits.append((f"{p.dst}:{p.dport}", "ip:port"))
        names = []
        if p.qname:
            names.append(p.qname)
        if p.httpreq:
            parts = p.httpreq.split(" ", 2)
            host = parts[1].lower() if len(parts) > 1 else ""
            raw_path = parts[2] if len(parts) > 2 else ""
            path = urlsplit(raw_path).path.lower() if raw_path.startswith("/") else raw_path.split("?", 1)[0].lower()
            if host:
                names.append(host.split(":")[0])
            if path in self.paths:
                hits.append((path, "path"))
            hp = f"{host}{path}".rstrip("/")
            if hp in self.host_paths:
                hits.append((hp, "host+path"))
        if p.sni:
            names.append(p.sni)
        for n in names:
            if self.whitelist and self._whitelisted(n):
                continue
            d = self._domain_hit(n)
            if d:
                hits.append((d, "domain"))
            if self.regexes:
                for rx, src in self.regexes:
                    if rx.search(n):
                        hits.append((src, "pattern"))
                        break
        return hits

    def describe(self, indicators) -> dict:
        """Descriptions for the given indicators (one pass over the file; only called for the few that hit)."""
        want = {i for i in indicators if i not in self._info}
        if want and self.path:
            with open(self.path, encoding="utf-8", errors="replace", newline="") as f:
                for row in csv.reader(l for l in f if not l.startswith("#")):
                    if row and len(row) > 1:
                        key = row[0].strip()
                        k = _clean_ip(key) if _IP.match(key) else key.lower().strip(".") if "/" not in key else key.split("?", 1)[0].lower().rstrip("/")
                        for cand in (k, key):
                            if cand in want:
                                self._info[cand] = row[1]
                                want.discard(cand)
                        if not want:
                            break
        return {i: self._info.get(i, "") for i in indicators}
