"""Show our fused alerts in Maltrail's own web dashboard by writing them in Maltrail's event-log format.

  python -m layered_nids.pipeline.to_dashboard --alerts reports/pipeline/mirai.alerts.jsonl --out tools/dashboard/logs

Maltrail's log line:  "<time>" <sensor> <src_ip> <src_port> <dst_ip> <dst_port> <proto> <type> <trail> "<info>" <reference>
Our alerts are machine-learning findings about a HOST, not network events, so some fields are placeholders: the source is the host, the
ports are 0, the destination is 0.0.0.0 and the protocol column says TCP only because the dashboard needs a value. The trail is the host, so
the dashboard groups alerts per host. The reference is "(nids)" so they are easy to tell apart from Maltrail's own events.
Times are UTC; the log file is named after the alert's date, as Maltrail does.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SENSOR = "nids"


def alert_to_line(a: dict, sensor: str = SENSOR) -> str:
    when = datetime.fromtimestamp(a["ts"], timezone.utc)
    layers = "+".join(sorted(a.get("layers") or {})) or "fusion"
    reason = (a.get("reasons") or [layers])[0].replace('"', "'")
    info = f'{a["severity"]} ML alert ({layers}), p={a["probability"]:.9f}: {reason}'
    return f'"{when.strftime("%Y-%m-%d %H:%M:%S.%f")}" {sensor} {a["host"]} 0 0.0.0.0 0 TCP IP {a["host"]} "{info}" (nids)'


def write_logs(alerts_path, out_dir) -> dict:
    """Append the alerts to <out_dir>/<YYYY-MM-DD>.log, one file per UTC day. Returns {file name: lines written}."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    by_day = defaultdict(list)
    for line in Path(alerts_path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            a = json.loads(line)
            by_day[datetime.fromtimestamp(a["ts"], timezone.utc).strftime("%Y-%m-%d")].append(alert_to_line(a))
    for day, lines in by_day.items():
        with (out_dir / f"{day}.log").open("a", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(lines) + "\n")
    return {f"{d}.log": len(v) for d, v in by_day.items()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--alerts", required=True)
    ap.add_argument("--out", required=True, help="Maltrail LOG_DIR")
    a = ap.parse_args(argv)
    print(write_logs(a.alerts, a.out))


if __name__ == "__main__":
    main()
