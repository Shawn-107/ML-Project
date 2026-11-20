"""AlertStore: a small SQLite log of scored flows so a dashboard can show recent alerts and running counts."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS flows (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    flow_id TEXT,
    is_attack INTEGER NOT NULL,
    probability REAL NOT NULL,
    severity TEXT NOT NULL,
    reason TEXT,
    policy TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_flows_attack_id ON flows(is_attack, id);
"""


class AlertStore:
    """Thread-safe. `path=':memory:'` keeps everything in RAM. Only flows that raised an alert keep their
    detail; normal flows are only counted, so the table stays small."""

    def __init__(self, path: str | Path = ":memory:", max_alerts: int = 10_000):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        self._lock = threading.Lock()
        self.max_alerts = max_alerts
        self._normal_count = 0

    def add_batch(self, results: list[dict], flow_ids: list | None = None) -> int:
        """Store a batch of Detector.predict() results. Returns the number of alerts stored."""
        now, stored = time.time(), 0
        flow_ids = flow_ids or [None] * len(results)
        with self._lock:
            for r, fid in zip(results, flow_ids):
                if not r["is_attack"]:
                    self._normal_count += 1
                    continue
                detail = json.dumps({"ann_score": r.get("ann_score"), "rf_score": r.get("rf_score"),
                                     "novelty_score": r.get("novelty_score"), "top_deviations": r.get("top_deviations")})
                self._db.execute(
                    "INSERT INTO flows(ts, flow_id, is_attack, probability, severity, reason, policy, detail)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (now, None if fid is None else str(fid), 1, float(r["attack_probability"]), r["severity"],
                     r.get("flag_reason"), r["policy"], detail))
                stored += 1
            self._db.execute("DELETE FROM flows WHERE id <= (SELECT MAX(id) FROM flows) - ?", (self.max_alerts,))
            self._db.commit()
        return stored

    def recent(self, limit: int = 50, min_severity: str | None = None) -> list[dict]:
        limit = max(1, min(int(limit), 1000))
        sql = "SELECT id, ts, flow_id, probability, severity, reason, policy, detail FROM flows"
        args: list = []
        if min_severity == "high":
            sql += " WHERE severity = 'high'"
        elif min_severity not in (None, "medium"):
            raise ValueError("min_severity must be 'medium' or 'high'")
        sql += " ORDER BY id DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [{"alert_id": i, "ts": ts, "flow_id": fid, "attack_probability": p, "severity": sev, "reason": reason,
                 "policy": pol, **json.loads(detail or "{}")} for i, ts, fid, p, sev, reason, pol, detail in rows]

    def stats(self) -> dict:
        with self._lock:
            n_alert = self._db.execute("SELECT COUNT(*) FROM flows").fetchone()[0]
            high = self._db.execute("SELECT COUNT(*) FROM flows WHERE severity='high'").fetchone()[0]
        total = n_alert + self._normal_count
        return {"flows_seen": total, "alerts_stored": n_alert, "high_severity": high,
                "alert_rate": (n_alert / total) if total else 0.0}

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM flows")
            self._db.commit()
            self._normal_count = 0

    def close(self) -> None:
        with self._lock:
            self._db.close()
