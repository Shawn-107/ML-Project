"""Blocklists supplied by the user (IPs and domains). No threat-intel data is bundled with this project.

File format: one entry per line, '#' starts a comment. An entry is an IP address or a domain name. Each file is one
named feed. Confidence follows the idea documented for Maltrail (more independent feeds -> more confidence), with our
own numbers: 0.4 for one feed, +0.15 for each extra feed, capped at 1.0. It measures agreement between feeds, not
the chance of an attack.
"""
from __future__ import annotations

import ipaddress
from pathlib import Path


def _clean(line: str) -> str:
    return line.split("#", 1)[0].strip().lower()


class Blocklists:
    def __init__(self, feeds: dict[str, set[str]] | None = None):
        self.feeds: dict[str, set[str]] = {k: set(v) for k, v in (feeds or {}).items()}

    @classmethod
    def from_files(cls, paths) -> "Blocklists":
        feeds = {}
        for p in paths:
            p = Path(p)
            feeds[p.stem] = {e for e in map(_clean, p.read_text(encoding="utf-8", errors="replace").splitlines()) if e}
        return cls(feeds)

    def feeds_for_ip(self, ip: str) -> list[str]:
        try:
            ip = str(ipaddress.ip_address(ip))
        except ValueError:
            return []
        return [n for n, s in self.feeds.items() if ip in s]

    def feeds_for_domain(self, name: str) -> list[str]:
        """Exact match or any parent domain (a.b.evil.com matches evil.com)."""
        name = name.strip(".").lower()
        parts = name.split(".")
        parents = {".".join(parts[i:]) for i in range(len(parts) - 1)} | {name}
        return [n for n, s in self.feeds.items() if parents & s]

    @staticmethod
    def confidence(n_feeds: int) -> float:
        return 0.0 if n_feeds <= 0 else min(1.0, 0.4 + 0.15 * (n_feeds - 1))
