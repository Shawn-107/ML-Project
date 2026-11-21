"""NIDS web-attack signatures: patterns matched against the HTTP request line (method, host, path) of a packet.

Each signature is (name, severity, regular expression). The request path is percent-decoded twice and lower-cased before
matching, so `%2e%2e%2f` and `%252e%252e%252f` are caught as well as `../`. Severity is the evidence score a hit gets.

These are plain, auditable patterns written for this project from the well-known shapes of the attacks. They only see the
request line of a plain HTTP request (no encrypted traffic, no request bodies, no headers), so they miss many attacks;
a hit means "this request looks like X", not "the attack succeeded".
"""
from __future__ import annotations

import re
from urllib.parse import unquote_plus

SIGNATURES = (
    ("directory traversal", 0.85, r"(?:\.\./|\.\.\\|\.\.%2f|%2e%2e)"),
    ("sensitive file access", 0.85, r"(?:/etc/(?:passwd|shadow|hosts)|/proc/self/|boot\.ini|win\.ini|\.ssh/id_rsa|\.aws/credentials)"),
    ("sql injection", 0.85, r"(?:union\s+(?:all\s+)?select|\bor\s+1\s*=\s*1|'\s*or\s*'1'\s*=\s*'1|information_schema|sleep\s*\(\s*\d|benchmark\s*\(|;\s*drop\s+table|xp_cmdshell)"),
    ("cross-site scripting", 0.80, r"(?:<\s*script|javascript:|onerror\s*=|onload\s*=|<\s*img[^>]+src\s*=)"),
    ("command injection", 0.90, r"(?:;\s*(?:cat|ls|id|wget|curl|nc|bash|sh)\b|\|\s*(?:cat|ls|id|wget|curl|nc|bash|sh)\b|`[^`]+`|\$\([^)]+\)|&&\s*(?:cat|wget|curl))"),
    ("remote file inclusion", 0.85, r"(?:=\s*(?:https?|ftp)://|php://(?:input|filter)|data://text|expect://)"),
    ("shellshock", 0.95, r"\(\)\s*\{\s*[:;]"),
    ("web shell or admin probe", 0.75, r"(?:/(?:cmd|shell|c99|r57|webshell)\.(?:php|asp|aspx|jsp)|/wp-admin/setup-config|/phpmyadmin|/manager/html|/\.env\b|/\.git/|wp-config\.php)"),
    ("log4j lookup", 0.95, r"\$\{\s*(?:jndi|env|sys|lower|upper)\s*:"),
    ("router/IoT exploit path", 0.85, r"(?:/cgi-bin/.*(?:\.\./|;|\|)|/boaform/|/setup\.cgi|/goform/|/hnap1|/gponform/|/tmUnblock\.cgi|/shell\?|/ctrlt/deviceupgrade)"),
)
_COMPILED = [(n, s, re.compile(rx, re.IGNORECASE)) for n, s, rx in SIGNATURES]


def decode(path: str) -> str:
    """Percent-decode twice (attackers double-encode to slip past simple filters) and lower-case."""
    return unquote_plus(unquote_plus(path)).lower()


def match_request(httpreq: str) -> list[tuple[str, float]]:
    """Signatures matched by one request line "METHOD host path". Returns [(name, severity)]; empty if none."""
    if not httpreq:
        return []
    parts = httpreq.split(" ", 2)
    path = decode(parts[2] if len(parts) > 2 else parts[-1])
    return [(n, s) for n, s, rx in _COMPILED if rx.search(path)]
