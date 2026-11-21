"""NIDS dashboard: CSV upload. Runs CSV mode (`python -m layered_nids.csvmode`) on an uploaded flow file in a separate process.

Safety: the file is only READ as data, saved under a random name, deleted after analysis, size-capped, processed with a time limit.
Downloads are limited to files made here (random 32-hex id plus fixed kind names). Flagged hosts also become dashboard events.
"""
from __future__ import annotations

import csv
import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
MAX_BYTES = 100 * 1024 * 1024
TIMEOUT_S = 900
POLICIES = ("fpr_1pct", "fpr_3pct")
KINDS = {"scored": ".csv", "findings": ".findings.csv", "hosts": ".hosts.csv"}
_ID = re.compile(r"\A[0-9a-f]{32}\Z")


def _prune(directory: Path, max_age_s: float = 7 * 86400) -> None:
    now = time.time()
    for p in directory.glob("*"):
        try:
            if p.is_file() and now - p.stat().st_mtime > max_age_s:
                p.unlink()
        except OSError:
            pass


def _events(summary: dict, base: Path, now: float) -> list[dict]:
    out = []
    f = Path(str(base) + KINDS["findings"])
    if f.exists():
        with f.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    ts = float(r["window_start"])
                except (KeyError, ValueError):
                    ts = now
                ts = ts if ts > 631152000 else now                        # a clock before 1990 is not a date: use the upload time
                out.append({"layer": "rules", "host": r["host"], "ts": ts, "score": 0.7, "reason": f"{r['kind']}: {r['distinct']} distinct targets (target {r['target']})",
                            "detail": {"signature": False, "trail": r["kind"], "source": "csv upload"}})
    h = Path(str(base) + KINDS["hosts"])
    if h.exists():
        with h.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                out.append({"layer": "flows", "host": r["host"], "ts": now, "score": min(0.99, float(r["max_probability"])),
                            "reason": f"{summary['schema']} model flagged {r['flagged_flows']} of {r['flows']} flows", "detail": {"signature": False, "trail": "flow model", "source": "csv upload"}})
    return out


def handle_upload(raw: bytes, filename: str, policy: str, state_dir: Path, events_file: Path) -> dict:
    """Returns the summary dict or {"error": "..."}; never raises for bad input."""
    if policy not in POLICIES:
        return {"error": f"unknown policy '{policy}'"}
    if not raw:
        return {"error": "the file is empty"}
    if len(raw) > MAX_BYTES:
        return {"error": f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB; split it"}
    up = Path(state_dir) / "uploads"
    up.mkdir(parents=True, exist_ok=True)
    _prune(up)
    uid = secrets.token_hex(16)
    src, out = up / f"{uid}.input", up / f"{uid}.csv"
    src.write_bytes(raw)
    env = dict(os.environ, LOKY_MAX_CPU_COUNT=os.environ.get("LOKY_MAX_CPU_COUNT", "12"), PYTHONUTF8="1")
    try:
        proc = subprocess.run([sys.executable, "-m", "layered_nids.csvmode", str(src), "--out", str(out), "--policy", policy],
                              cwd=str(PROJECT), env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"error": f"processing took longer than {TIMEOUT_S // 60} minutes and was stopped"}
    finally:
        try:
            src.unlink()
        except OSError:
            pass
    sp = Path(str(out) + ".summary.json")
    if proc.returncode != 0 or not sp.exists():
        lines = (proc.stdout + proc.stderr).strip().splitlines()
        first = next((m for m in lines if m.startswith("error:")), lines[-1] if lines else "processing failed")
        return {"error": first.replace("error:", "").strip()[:400]}
    s = json.loads(sp.read_text(encoding="utf-8"))
    keep = ("schema", "policy", "rows", "flagged", "flagged_share", "severity", "row_status", "hosts_flagged", "findings", "flows_in_findings",
            "warnings", "caution", "seconds", "lines_skipped_malformed_or_unreadable")
    res = {k: s[k] for k in keep if k in s}
    res["id"] = uid
    res["filename"] = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "upload.csv")[:80]
    res["downloads"] = [k for k, ext in KINDS.items() if Path(str(out)[:-4] + ext).exists()]
    ev = _events(s, out.with_suffix(""), time.time())
    if ev:
        events_file.parent.mkdir(parents=True, exist_ok=True)
        with events_file.open("a", encoding="utf-8", newline="\n") as f:
            for e in ev:
                f.write(json.dumps(e) + "\n")
    res["events_added"] = len(ev)
    return res


def result_file(state_dir: Path, uid: str, kind: str):
    if not _ID.match(uid or "") or kind not in KINDS:
        return None
    p = Path(state_dir) / "uploads" / f"{uid}{KINDS[kind]}"
    return (p, f"nids-{kind}-{uid[:8]}.csv") if p.is_file() else None
