"""Maltrail layer of the combined pipeline: the REAL Maltrail sensor's event log -> per-host evidence.

Maltrail event-log line (from its own source, sensor/src/event.rs):
  "<time>" <sensor> <src_ip> <src_port> <dst_ip> <dst_port> <proto> <type> <trail> "<info>" <reference>
e.g.  "2018-10-25 05:48:11.704154" mtlab 192.168.2.112 59290 61.188.37.216 8000 TCP IPORT 61.188.37.216:8000 "potential periodic beaconing (suspicious)" (heuristic)

Scores are OUR choice (Maltrail itself gives only text and a reference): a blocklist hit scores TRAIL_SCORE, a heuristic
finding scores HEURISTIC_SCORE. They are not calibrated probabilities; the learned combiner can re-weight them when
labelled data exists. Time: when replaying a pcap the sensor writes the packet's own timestamp; in live mode it writes
local time, which this module does not convert.

The sensor is a Linux program: run_sensor_wsl() runs the checksum-verified Linux build inside WSL2 (see NEXT_STEPS.md).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from layered_nids.common.hosts import pick_internal
from layered_nids.common.schema import Evidence

TRAIL_SCORE = 0.8
HEURISTIC_SCORE = 0.6

_LINE = re.compile(
    r'^"(?P<time>[^"]+)"\s+(?P<sensor>\S+)\s+(?P<src>\S+)\s+(?P<sport>\S+)\s+(?P<dst>\S+)\s+(?P<dport>\S+)\s+'
    r'(?P<proto>\S+)\s+(?P<type>\S+)\s+(?P<trail>\S+)\s+"(?P<info>[^"]*)"\s*(?P<ref>.*)$')

WSL_ROOT = "/home/shawn/work/maltrail"
SHARE_ROOT = Path(r"\\wsl.localhost\Ubuntu\home\shawn\work\maltrail")


def parse_log(text: str) -> tuple[list[dict], int]:
    """Parse event-log text. Returns (events, number_of_lines_that_did_not_parse). Blank lines are ignored."""
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


def events_to_evidence(events: list[dict]) -> list[Evidence]:
    """One Evidence per event. Host = the internal end (source if it is a real private unicast address, else destination);
    events with no internal host are skipped, the same rule as the Kitsune layer."""
    out = []
    for e in events:
        host = pick_internal(e["src"], e["dst"])
        if not host:
            continue
        heuristic = "(heuristic)" in e["ref"]
        score = HEURISTIC_SCORE if heuristic else TRAIL_SCORE
        out.append(Evidence("rules", host, e["ts"], score, f'Maltrail: {e["info"]} [{e["trail"]}]',
                            {"type": e["type"], "trail": e["trail"], "ref": e["ref"], "proto": e["proto"],
                             "src": e["src"], "dst": e["dst"], "dport": e["dport"]}))
    return out


def run_sensor_wsl(pcap, share_root: Path = SHARE_ROOT, wsl_root: str = WSL_ROOT, timeout: int = 600) -> str:
    """Run the real Linux sensor offline on `pcap` inside WSL2 and return its event-log text.
    Copies the pcap into the Linux folder first (the sensor cannot read Windows drives; they are not mounted)."""
    pcap = Path(pcap)
    pcaps = share_root / "pcaps"
    logs = share_root / "logs"
    pcaps.mkdir(parents=True, exist_ok=True)
    target = pcaps / pcap.name
    shutil.copyfile(pcap, target)
    cmd = (f"cd {wsl_root} && rm -f logs/*.log && MALTRAIL_ROOT={wsl_root} "
           f"./bin/maltrail-sensor -c {wsl_root}/maltrail.conf --offline -q -r {wsl_root}/pcaps/{pcap.name}")
    r = subprocess.run(["wsl", "-d", "Ubuntu", "--", "sh", "-c", cmd], capture_output=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"Maltrail sensor failed (exit {r.returncode}): {r.stderr.decode('utf-8', 'ignore')[:300]}")
    text = "".join(p.read_text(encoding="utf-8", errors="ignore") for p in sorted(logs.glob("*.log"))
                   if p.name != "error.log")
    target.unlink(missing_ok=True)                  # keep no extra copy of the capture
    return text
