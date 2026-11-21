r"""Optional: use an EXTERNAL sensor program for what is hard or slow to recreate here, and read its event log as NIDS evidence.

Why: a mature sensor has years of curated indicator data, filtering rules (popular-domain whitelist, minimum number of agreeing feeds),
extra behaviour rules, TLS and user-agent fingerprints, and it reads packets about 10,000 times faster than Python can. We do not copy
any of its code: it runs as a separate program (inside WSL2 on this machine, offline) and we only parse the text lines it writes.

Event-log line:  "<time>" <sensor> <src_ip> <src_port> <dst_ip> <dst_port> <proto> <type> <trail> "<info>" <reference>
e.g.  "2018-10-25 05:48:11.704154" s1 192.168.2.112 59290 61.188.37.216 8000 TCP IPORT 61.188.37.216:8000 "potential periodic beaconing (suspicious)" (heuristic)

Scores are our choice (the sensor gives text only): an indicator hit scores 0.8 and counts as a signature; a behaviour finding scores 0.6
and does not. When replaying a pcap the sensor writes the packet's own timestamps (read as UTC). Settings, if you use it:
  NIDS_SENSOR_WSL_ROOT   folder of the sensor inside WSL2, e.g. /home/you/sensor     (it must contain bin/, its config file and a pcaps/ folder)
  NIDS_SENSOR_SHARE      the same folder as seen from Windows, e.g. \\wsl.localhost\Ubuntu\home\you\sensor
  NIDS_SENSOR_BIN        program name inside bin/ (default: the only file there)
  NIDS_SENSOR_CONFIG     config file name inside that folder (default: sensor.conf)
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from layered_nids.common.hosts import pick_internal
from layered_nids.common.schema import Evidence

INDICATOR_SCORE = 0.8
BEHAVIOUR_SCORE = 0.6

_LINE = re.compile(
    r'^"(?P<time>[^"]+)"\s+(?P<sensor>\S+)\s+(?P<src>\S+)\s+(?P<sport>\S+)\s+(?P<dst>\S+)\s+(?P<dport>\S+)\s+'
    r'(?P<proto>\S+)\s+(?P<type>\S+)\s+(?P<trail>\S+)\s+"(?P<info>[^"]*)"\s*(?P<ref>.*)$')


def parse_events(text: str) -> tuple[list[dict], int]:
    """Parse event-log text. Returns (events, number of non-blank lines that did not parse)."""
    events, bad = [], 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = _LINE.match(line)
        if not m:
            bad += 1
            continue
        d = m.groupdict()
        try:
            d["ts"] = datetime.strptime(d["time"], "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            bad += 1
            continue
        events.append(d)
    return events, bad


def to_evidence(events: list[dict]) -> list[Evidence]:
    """One Evidence (layer 'rules') per event, about the internal end of the conversation; events with no internal host are skipped."""
    out = []
    for e in events:
        host = pick_internal(e["src"], e["dst"])
        if not host:
            continue
        behaviour = "(heuristic)" in e["ref"]
        out.append(Evidence("rules", host, e["ts"], BEHAVIOUR_SCORE if behaviour else INDICATOR_SCORE, f'{e["info"]} [{e["trail"]}]',
                            {"signature": not behaviour, "trail": e["trail"], "type": e["type"], "proto": e["proto"], "src": e["src"], "dst": e["dst"],
                             "dport": e["dport"], "source": "external sensor"}))
    return out


def configured() -> bool:
    return bool(os.environ.get("NIDS_SENSOR_WSL_ROOT") and os.environ.get("NIDS_SENSOR_SHARE"))


def run_external_sensor(pcap, timeout: int = 900) -> list[Evidence]:
    """Run the external sensor offline on a pcap and return its events as evidence. Raises RuntimeError if it is not set up or fails."""
    if not configured():
        raise RuntimeError("external sensor not configured: set NIDS_SENSOR_WSL_ROOT and NIDS_SENSOR_SHARE (see layered_nids/signatures/external.py)")
    root, share = os.environ["NIDS_SENSOR_WSL_ROOT"].rstrip("/"), Path(os.environ["NIDS_SENSOR_SHARE"])
    conf = os.environ.get("NIDS_SENSOR_CONFIG", "sensor.conf")
    binary = os.environ.get("NIDS_SENSOR_BIN") or next(p.name for p in sorted((share / "bin").iterdir()) if p.is_file())
    pcap = Path(pcap)
    for sub in ("pcaps", "logs"):
        if not (share / sub).is_dir():             # (mkdir(exist_ok=True) fails on a network-share path even when the folder exists)
            (share / sub).mkdir()
    target = share / "pcaps" / pcap.name
    shutil.copyfile(pcap, target)                  # the Linux side cannot read Windows drives (they are not mounted)
    try:
        cmd = f"cd {root} && rm -f logs/*.log && ./bin/{binary} -c {root}/{conf} --offline -q -r {root}/pcaps/{pcap.name}"
        r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "sh", "-c", cmd], capture_output=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"external sensor failed (exit {r.returncode}): {r.stderr.decode('utf-8', 'ignore')[:300]}")
        text = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in sorted((share / "logs").glob("*.log")) if p.name != "error.log")
    finally:
        target.unlink(missing_ok=True)             # keep no extra copy of the capture
    events, _ = parse_events(text)
    return to_evidence(events)
