"""NIDS dashboard: CSV upload. Runs the project's CSV mode on an uploaded flow file and returns a small summary.

Called by core/httpd.py (routes /nids/csv/upload and /nids/csv/download), only for a signed-in session.

Safety:
- the file is only READ as data by layered_nids.csvmode (never executed); it is saved under a random name in uploads/
- size is capped, the format is detected from the columns, bad rows are reported, not trusted
- the processing runs in a separate Python process with a time limit
- downloads are limited to files this module created (random 32-hex id + fixed kind names)
- flagged hosts are also written to the dashboard's event log (sensor "nids-csv") so they show up in the event table
"""
from __future__ import annotations

import json
import os
import re
import secrets
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parents[1]                      # repository root: tools/<this folder>/../..
UPLOADS = ROOT / "uploads"
MAX_BYTES = 100 * 1024 * 1024
TIMEOUT_S = 900
POLICIES = ("fpr_1pct", "fpr_3pct")
KINDS = {"scored": ".csv", "findings": ".findings.csv", "hosts": ".hosts.csv"}
_ID = re.compile(r"\A[0-9a-f]{32}\Z")
SENSOR = "nids-csv"


def _prune(max_age_s: float = 7 * 86400) -> None:
    now = time.time()
    for p in UPLOADS.glob("*"):
        try:
            if p.is_file() and now - p.stat().st_mtime > max_age_s:
                p.unlink()
        except OSError:
            pass


def _event(ts: float, host: str, info: str) -> str:
    when = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) + ".000000"
    info = info.replace('"', "'")
    return f'"{when}" {SENSOR} {host} 0 0.0.0.0 0 TCP IP {host} "{info}" (nids)'


def _write_events(result: dict, base: Path, log_dir: Path) -> int:
    import csv
    now = time.time()
    lines = []
    findings = Path(str(base) + KINDS["findings"])
    hosts = Path(str(base) + KINDS["hosts"])
    if findings.exists():
        with findings.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                try:
                    ts = float(r["window_start"])
                except (KeyError, ValueError):
                    ts = now
                ts = ts if ts > 631152000 else now                      # a clock before 1990 is not a date: use the upload time
                lines.append((ts, _event(ts, r["host"], f"{r['severity']} CSV rule finding: {r['kind']}, {r['distinct']} distinct targets (target {r['target']})")))
    if hosts.exists():
        with hosts.open(encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                lines.append((now, _event(now, r["host"], f"{result['schema']} model flagged {r['flagged_flows']} of {r['flows']} flows (max probability {float(r['max_probability']):.3f})")))
    by_day: dict = {}
    for ts, line in lines:
        by_day.setdefault(time.strftime("%Y-%m-%d", time.gmtime(ts)), []).append(line)
    log_dir.mkdir(parents=True, exist_ok=True)
    for day, ls in by_day.items():
        with (log_dir / f"{day}.log").open("a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(ls) + "\n")
    return len(lines)


def handle_upload(raw: bytes, filename: str, policy: str, log_dir) -> dict:
    """Returns {"id", "schema", "rows", ...} or {"error": "..."}; never raises for bad input."""
    if policy not in POLICIES:
        return {"error": f"unknown policy '{policy}'"}
    if not raw:
        return {"error": "the file is empty"}
    if len(raw) > MAX_BYTES:
        return {"error": f"the file is larger than {MAX_BYTES // (1024 * 1024)} MB; split it"}
    UPLOADS.mkdir(parents=True, exist_ok=True)
    _prune()
    uid = secrets.token_hex(16)
    src = UPLOADS / f"{uid}.input"
    out = UPLOADS / f"{uid}.csv"
    src.write_bytes(raw)
    env = dict(os.environ, LOKY_MAX_CPU_COUNT=os.environ.get("LOKY_MAX_CPU_COUNT", "12"), PYTHONUTF8="1")
    try:
        proc = subprocess.run([sys.executable, "-m", "layered_nids.csvmode", str(src), "--out", str(out), "--policy", policy],
                              cwd=str(PROJECT), env=env, capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"error": f"processing took longer than {TIMEOUT_S // 60} minutes and was stopped"}
    finally:
        try:
            src.unlink()                                                       # keep only the results, not the upload
        except OSError:
            pass
    summary_path = Path(str(out) + ".summary.json")
    if proc.returncode != 0 or not summary_path.exists():
        msg = (proc.stdout + proc.stderr).strip().splitlines()
        first = next((m for m in msg if m.startswith("error:")), msg[-1] if msg else "processing failed")
        return {"error": first.replace("error:", "").strip()[:400]}
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    keep = ("schema", "policy", "rows", "flagged", "flagged_share", "severity", "row_status", "hosts_flagged", "findings",
            "flows_in_findings", "warnings", "caution", "seconds", "lines_skipped_malformed_or_unreadable")
    res = {k: s[k] for k in keep if k in s}
    res["id"] = uid
    res["filename"] = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "upload.csv")[:80]
    res["downloads"] = [k for k, ext in KINDS.items() if Path(str(out)[:-4] + ext).exists()]
    res["events_added"] = _write_events(s, out.with_suffix(""), Path(log_dir))
    return res


def result_file(uid: str, kind: str):
    """(path, download name) for a result created by handle_upload, or None. Id and kind are strictly validated."""
    if not _ID.match(uid or "") or kind not in KINDS:
        return None
    p = UPLOADS / f"{uid}{KINDS[kind]}"
    return (p, f"nids-{kind}-{uid[:8]}.csv") if p.is_file() else None
