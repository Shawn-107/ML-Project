"""Live capture from the dashboard: a limited capture of THIS computer's traffic, analysed by the rules layer, shown under "Live captures".

How it works (one capture at a time):
  1. the dashboard starts layered_nids.pipeline.live in a separate process (Windows shows its own approval prompt: nothing is captured until
     somebody clicks Yes; if nobody does, the job ends with a message after the time limit);
  2. the saved pcap is read and the rules layer (signatures and heuristics) runs on it;
  3. events and the alerts from signature hits are appended to local/live_events.jsonl and local/live_alerts.jsonl, marked source "live";
  4. the pcap is deleted, also when something fails. Only the findings stay; "Clear live results" removes them.
What it does NOT do: the packet anomaly model is not used (it needs about 70,000 clean packets of THIS network to learn, so a short capture
cannot feed it), so there are no anomaly-based alerts; heuristic findings are events only. Nothing leaves the computer.
Safety: hard limits below; on a network Windows calls Public (or when that is unknown) a capture needs an explicit acknowledgement,
because other people's broadcast traffic can be on the wire; the dashboard itself is localhost-only and needs a signed-in session.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from layered_nids.common.paths import ROOT

MAX_SECONDS = 120
MAX_PACKETS = 200_000
DEFAULT_SECONDS = 20
APPROVAL_WAIT_S = 180                         # extra time allowed for the Windows approval prompt
NET_NAMES = {0: "Public", 1: "Private", 2: "Domain"}


def network_profiles() -> list[dict]:
    """[{name, category}] of the connected networks as Windows reports them ('Public', 'Private', 'Domain'); [] when unknown or not Windows."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command",
                              "Get-NetConnectionProfile | Select-Object Name,NetworkCategory | ConvertTo-Json -Compress"],
                             capture_output=True, text=True, timeout=15).stdout.strip()
        rows = json.loads(out) if out else []
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return []
    rows = [rows] if isinstance(rows, dict) else rows
    res = []
    for r in rows:
        c = r.get("NetworkCategory")
        res.append({"name": str(r.get("Name", "?")), "category": NET_NAMES.get(c, str(c)) if isinstance(c, int) else str(c)})
    return res


def _capture_subprocess(path: Path, seconds: float, max_packets: int) -> dict:
    cmd = [sys.executable, "-m", "layered_nids.pipeline.live", "--seconds", str(seconds), "--max-packets", str(max_packets), "--out", Path(path).name]
    try:
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=seconds + APPROVAL_WAIT_S)
    except subprocess.TimeoutExpired:
        raise RuntimeError("no capture finished in time: was the Windows approval prompt answered?") from None
    if r.returncode != 0:
        raise RuntimeError("the capture failed: was the Windows approval declined, or is Npcap missing?")
    m = re.search(r"captured ([\d,]+) packets in ([\d.]+) s", r.stdout)
    return {"packets": int(m.group(1).replace(",", "")) if m else 0, "seconds": float(m.group(2)) if m else float(seconds)}


def _analyse(path: Path) -> list:
    from layered_nids.packets.pcap import iter_packets
    from layered_nids.pipeline.rules_layer import rules_evidence
    return rules_evidence(list(iter_packets(path)))


def _signature_alerts(evidence: list, cooldown_s: float = 300.0) -> list[dict]:
    """A signature hit alerts directly, once per host and rule per cooldown (the same policy as pipeline/run.py fuse_policy)."""
    out, last = [], {}
    for e in sorted((e for e in evidence if e.layer == "rules" and e.detail.get("signature")), key=lambda e: e.ts):
        key = (e.host, e.detail.get("trail"))
        if e.ts - last.get(key, -1e300) >= cooldown_s:
            last[key] = e.ts
            out.append({"host": e.host, "ts": e.ts, "probability": float(e.score), "severity": "high", "layers": {"rules": 1.0}, "reasons": [e.reason],
                        "setup": "live_rules_signature_hits"})
    return out


class CaptureManager:
    def __init__(self, state: Path, capture=None, analyse=None, network=None):
        self.state = Path(state)
        self._capture, self._analyse, self._network = capture or _capture_subprocess, analyse or _analyse, network or network_profiles
        self._lock = threading.Lock()
        self._job: dict = {"state": "idle"}
        self._net_cache: tuple[float, list] = (0.0, [])
        self.alerts_file, self.events_file = self.state / "live_alerts.jsonl", self.state / "live_events.jsonl"

    # ------------------------------------------------------------------------------------------------ status
    def _profiles(self) -> list[dict]:
        t, cached = self._net_cache
        if time.time() - t > 20:
            cached = self._network()
            self._net_cache = (time.time(), cached)
        return cached

    def needs_acknowledgement(self) -> bool:
        p = self._profiles()
        return not p or any(x["category"] != "Private" for x in p)

    def status(self) -> dict:
        with self._lock:
            job = dict(self._job)
        return {"job": job, "networks": self._profiles(), "needs_acknowledgement": self.needs_acknowledgement(),
                "limits": {"max_seconds": MAX_SECONDS, "max_packets": MAX_PACKETS, "default_seconds": DEFAULT_SECONDS},
                "live_alerts": self._count(self.alerts_file), "live_events": self._count(self.events_file)}

    @staticmethod
    def _count(p: Path) -> int:
        return sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0

    # ------------------------------------------------------------------------------------------------ actions
    def start(self, seconds, max_packets=None, acknowledged: bool = False) -> dict:
        try:
            seconds = float(seconds)
            max_packets = int(max_packets if max_packets is not None else MAX_PACKETS)
        except (TypeError, ValueError):
            raise ValueError("seconds and max_packets must be numbers") from None
        if not (0 < seconds <= MAX_SECONDS):
            raise ValueError(f"seconds must be between 1 and {MAX_SECONDS}")
        if not (0 < max_packets <= MAX_PACKETS):
            raise ValueError(f"max_packets must be between 1 and {MAX_PACKETS:,}")
        if self.needs_acknowledgement() and not acknowledged:
            raise PermissionError("this network is not marked Private (or could not be checked): confirm that you may capture here")
        with self._lock:
            if self._job.get("state") in ("capturing", "analysing"):
                raise RuntimeError("a capture is already running")
            job = {"id": uuid.uuid4().hex[:12], "state": "capturing", "seconds": seconds, "started": time.time(), "message": "Waiting for the Windows approval, then capturing ..."}
            self._job = job
            snapshot = dict(job)                                         # taken before the worker can change it
        threading.Thread(target=self._run, args=(snapshot, max_packets), daemon=True).start()
        return snapshot

    def clear(self) -> dict:
        with self._lock:
            if self._job.get("state") in ("capturing", "analysing"):
                raise RuntimeError("a capture is running")
        removed = 0
        for p in (self.alerts_file, self.events_file):
            if p.exists():
                p.unlink()
                removed += 1
        return {"removed_files": removed}

    # ------------------------------------------------------------------------------------------------ worker
    def _set(self, **kw):
        with self._lock:
            self._job.update(kw)

    def _run(self, job: dict, max_packets: int) -> None:
        from layered_nids.pipeline.live import LIVE_DIR
        path = LIVE_DIR / f"dashboard_{job['id']}.pcap"
        try:
            info = self._capture(path, job["seconds"], max_packets)
            self._set(state="analysing", message=f"Captured {info['packets']:,} packets. Analysing ...", packets=info["packets"])
            ev = self._analyse(path)
            alerts = _signature_alerts(ev)
            self.state.mkdir(parents=True, exist_ok=True)
            with self.events_file.open("a", encoding="utf-8") as f:
                for e in sorted(ev, key=lambda e: e.ts):
                    d = e.to_dict()
                    d["detail"] = {**d["detail"], "source": "live", "capture": job["id"]}
                    f.write(json.dumps(d, default=float) + "\n")
            with self.alerts_file.open("a", encoding="utf-8") as f:
                for a in alerts:
                    f.write(json.dumps({**a, "source": "live", "capture": job["id"]}, default=float) + "\n")
            sig = sum(1 for e in ev if e.detail.get("signature"))
            self._set(state="done", finished=time.time(), events=len(ev), alerts=len(alerts), signature_hits=sig, packets=info["packets"], captured_seconds=info["seconds"],
                      message="Done. The capture file was deleted; only the findings were kept.")
        except Exception as e:                                           # a message for the page, never a stack trace
            self._set(state="failed", finished=time.time(), message=str(e) if isinstance(e, (RuntimeError, ValueError, OSError)) else "the capture or analysis failed")
        finally:
            try:
                Path(path).unlink(missing_ok=True)
            except OSError:
                pass
