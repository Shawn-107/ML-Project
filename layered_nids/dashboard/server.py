"""NIDS dashboard server (standard library only; localhost only).

   python -m layered_nids.dashboard --init-user NAME     # once: makes local/users.json and local/PASSWORD.txt (random password)
   python -m layered_nids.dashboard --data reports/pipeline   [--port 8338]

Pages:   /            host alerts and rule events from the pipeline, CSV upload
         /app/        the flow-analysis app of the owner's earlier project (React, built files), served by /api/*
API:     /nids/api/summary | stats | host | alerts | events | upload | download      (this dashboard; add source=live for live captures)
         /nids/api/capture/status | capture | capture/clear      (live capture, localhost only; see dashboard/live_capture.py)
         /api/info | evaluation | sample | score | explain | adapt | recalibrate   (flow model; same contract as the earlier app)

Security: binds to 127.0.0.1 only (refuses anything else); every page and API call needs a signed-in session (HttpOnly,
SameSite=Strict cookie); POSTs must come from the same origin; failed sign-ins are throttled; static paths cannot leave their folder;
bodies are size-capped; responses carry no-sniff, no-frame and content-security-policy headers.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
from http import cookies
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from layered_nids.common.paths import ROOT
from layered_nids.dashboard import csv_upload
from layered_nids.dashboard.auth import Sessions, create_user

HERE = Path(__file__).resolve().parent
STATIC = HERE / "static"
STATE = ROOT / "local"
COOKIE = "nids_session"
PUBLIC_ASSETS = ("/static/style.css", "/static/login.js")          # nothing else is served without a session
MAX_JSON = 5_000_000
LOOPBACK = ("127.0.0.1", "localhost", "::1")
SEVERITY_ORDER = {"critical": 4, "high": 3, "medium": 2, "low": 1}
CSP = "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"


class Data:
    """Alerts and events read from *.alerts.jsonl / *.events.jsonl in a folder (and the upload events file), cached by file time."""

    def __init__(self, directory: Path, state: Path):
        self.dir, self.state = Path(directory), Path(state)
        self._cache: dict = {}

    def _read(self, pattern: str, extra: list[Path]) -> list[dict]:
        files = sorted(self.dir.glob(pattern)) + [p for p in extra if p.exists()]
        return self._read_files(pattern, files)

    def _read_files(self, name: str, files: list[Path]) -> list[dict]:
        key = tuple((str(p), p.stat().st_mtime_ns) for p in files)
        if self._cache.get(name, (None,))[0] == key:
            return self._cache[name][1]
        rows = []
        for p in files:
            for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    r = json.loads(line)
                    if isinstance(r, dict):
                        rows.append(r)
                except json.JSONDecodeError:
                    pass
        self._cache[name] = (key, rows)
        return rows

    def alerts(self, source: str = "recorded") -> list[dict]:
        if source == "live":
            return self._read_files("live_alerts", [p for p in [self.state / "live_alerts.jsonl"] if p.exists()])
        return self._read("*.alerts.jsonl", [])

    def events(self, source: str = "recorded") -> list[dict]:
        if source == "live":
            return self._read_files("live_events", [p for p in [self.state / "live_events.jsonl"] if p.exists()])
        return self._read("*.events.jsonl", [self.state / "upload_events.jsonl"])


def _filter(rows, q: dict, host_key="host", text_keys=("host", "reason")):
    host, text = q.get("host", [""])[0], q.get("q", [""])[0].lower()
    out = [r for r in rows if (not host or r.get(host_key) == host) and (not text or any(text in str(r.get(k, "")).lower() for k in text_keys))]
    return out


def _stats(alerts: list, events: list, bins: int = 48) -> dict:
    """Numbers for the overview charts: alerts over time by severity, busiest hosts, evidence share per layer, events per rule."""
    ts = [float(r.get("ts", 0)) for r in alerts] + [float(r.get("ts", 0)) for r in events]
    t0, t1 = (min(ts), max(ts)) if ts else (0.0, 0.0)
    width = max((t1 - t0) / bins, 1.0)
    hist = [{"t0": t0 + i * width, "t1": t0 + (i + 1) * width, "critical": 0, "high": 0, "medium": 0, "low": 0, "events": 0, "signatures": 0} for i in range(bins)]
    for r in alerts:
        i = min(int((float(r.get("ts", 0)) - t0) // width), bins - 1)
        hist[i][r.get("severity", "low") if r.get("severity") in SEVERITY_ORDER else "low"] += 1
    for r in events:
        b = hist[min(int((float(r.get("ts", 0)) - t0) // width), bins - 1)]
        b["events"] += 1
        b["signatures"] += 1 if (r.get("detail") or {}).get("signature") else 0
    hosts: dict = {}
    for r in alerts:
        h = hosts.setdefault(r.get("host", "?"), {"host": r.get("host", "?"), "alerts": 0, "events": 0, "max_severity": "low", "last": 0.0})
        h["alerts"] += 1
        h["last"] = max(h["last"], float(r.get("ts", 0)))
        if SEVERITY_ORDER.get(r.get("severity", "low"), 0) > SEVERITY_ORDER.get(h["max_severity"], 0):
            h["max_severity"] = r.get("severity", "low")
    for r in events:
        h = hosts.setdefault(r.get("host", "?"), {"host": r.get("host", "?"), "alerts": 0, "events": 0, "max_severity": "none", "last": 0.0})
        h["events"] += 1
        h["last"] = max(h["last"], float(r.get("ts", 0)))
    top = sorted(hosts.values(), key=lambda h: (-h["alerts"], -h["events"], h["host"]))[:8]
    layers: dict = {}
    for r in alerts:
        for k, v in (r.get("layers") or {}).items():
            layers[k] = layers.get(k, 0.0) + float(v)
    total = sum(layers.values()) or 1.0
    rules: dict = {}
    for r in events:
        d = r.get("detail") or {}
        k = str(d.get("trail", r.get("layer", "?")))
        x = rules.setdefault(k, {"rule": k, "count": 0, "signature": bool(d.get("signature"))})
        x["count"] += 1
    return {"t0": t0, "t1": t1, "hist": hist, "top_hosts": top, "layers": {k: v / total for k, v in sorted(layers.items(), key=lambda kv: -kv[1])},
            "rules": sorted(rules.values(), key=lambda x: -x["count"])[:8],
            "severity": {k: sum(1 for r in alerts if r.get("severity") == k) for k in ("critical", "high", "medium", "low")}}


def make_handler(data: Data, sessions: Sessions, state: Path, get_flow_service, capture):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 60
        server_version = "NIDS"
        sys_version = ""

        def log_message(self, *a):
            pass

        # -------------------------------------------------------------------------------------------- helpers
        def _hdrs(self):
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Cache-Control", "no-store")

        def _send(self, code, body: bytes, ctype="application/json", extra=()):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            for k, v in extra:
                self.send_header(k, v)
            self._hdrs()
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj, extra=()):
            self._send(code, json.dumps(obj, default=float).encode(), extra=extra)

        def _token(self):
            c = cookies.SimpleCookie(self.headers.get("Cookie", ""))
            return c[COOKIE].value if COOKIE in c else None

        def _same_origin(self) -> bool:
            o = self.headers.get("Origin")
            if not o:
                return True
            return urlparse(o).netloc == self.headers.get("Host", "")

        def _body(self, limit: int) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0 or n > limit:
                raise ValueError(f"body must be 1..{limit} bytes")
            return self.rfile.read(n)

        def _static(self, root: Path, rel: str):
            p = (root / rel).resolve()
            if not p.is_relative_to(root.resolve()) or not p.is_file():
                return self._json(404, {"error": "not found"})
            self._send(200, p.read_bytes(), mimetypes.guess_type(p.name)[0] or "application/octet-stream")

        # -------------------------------------------------------------------------------------------- routing
        def _dispatch(self, method: str):
            u = urlparse(self.path)
            path, q = u.path, parse_qs(u.query)
            if method == "POST" and not self._same_origin():
                return self._json(403, {"error": "cross-origin request refused"})
            if path == "/login":
                if method == "GET":
                    return self._static(STATIC, "login.html")
                try:
                    b = json.loads(self._body(10_000))
                    tok = sessions.login(str(b.get("user", "")), str(b.get("password", "")), self.client_address[0])
                except (ValueError, json.JSONDecodeError):
                    return self._json(400, {"error": "bad request"})
                if not tok:
                    return self._json(401, {"error": "wrong user or password (or too many attempts: wait a minute)"})
                return self._json(200, {"ok": True}, extra=[("Set-Cookie", f"{COOKIE}={tok}; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800")])
            if method == "GET" and path in PUBLIC_ASSETS:                  # the sign-in page needs its own style and script before anyone is signed in
                return self._static(STATIC, path[len("/static/"):])
            user = sessions.user(self._token())
            if path == "/logout" and method == "POST":
                sessions.logout(self._token())
                return self._json(200, {"ok": True}, extra=[("Set-Cookie", f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0")])
            if not user:
                if path.startswith(("/api/", "/nids/api/")):
                    return self._json(401, {"error": "sign in first"})
                self.send_response(302)
                self.send_header("Location", "/login")
                self.send_header("Content-Length", "0")
                self._hdrs()
                self.end_headers()
                return
            if path == "/":
                return self._static(STATIC, "index.html")
            if path.startswith("/static/"):
                return self._static(STATIC, path[len("/static/"):])
            if path == "/app":
                self.send_response(302)
                self.send_header("Location", "/app/")
                self.send_header("Content-Length", "0")
                self._hdrs()
                self.end_headers()
                return
            if path.startswith("/app/"):
                return self._static(STATIC / "app", path[len("/app/"):] or "index.html")
            if path.startswith("/nids/api/"):
                return self._nids(method, path[len("/nids/api/"):], q)
            if path.startswith("/api/"):
                return self._flow_api(method, path, q)
            self._json(404, {"error": "not found"})

        # -------------------------------------------------------------------------------------------- dashboard API
        def _nids(self, method, name, q):
            src = "live" if q.get("source", [""])[0] == "live" else "recorded"
            if name == "capture/status" and method == "GET":
                return self._json(200, capture.status())
            if name in ("capture", "capture/clear") and method == "POST":
                if self.headers.get("X-NIDS-Capture") != "1":
                    return self._json(403, {"error": "missing capture header"})
                try:
                    if name == "capture/clear":
                        return self._json(200, capture.clear())
                    b = json.loads(self._body(2000))
                    if not isinstance(b, dict):
                        raise ValueError("body must be a JSON object")
                    return self._json(200, capture.start(b.get("seconds"), b.get("max_packets"), bool(b.get("acknowledged"))))
                except PermissionError as e:
                    return self._json(403, {"error": str(e), "needs_acknowledgement": True})
                except RuntimeError as e:
                    return self._json(409, {"error": str(e)})
                except (ValueError, json.JSONDecodeError) as e:
                    return self._json(400, {"error": str(e)})
            if name == "summary" and method == "GET":
                a, e = data.alerts(src), data.events(src)
                sev = {}
                for r in a:
                    sev[r.get("severity", "low")] = sev.get(r.get("severity", "low"), 0) + 1
                return self._json(200, {"alerts": len(a), "alert_hosts": len({r.get("host") for r in a}), "severity": sev, "events": len(e),
                                        "signature_events": sum(1 for r in e if (r.get("detail") or {}).get("signature")), "event_hosts": len({r.get("host") for r in e})})
            if name == "stats" and method == "GET":
                return self._json(200, _stats(data.alerts(src), data.events(src)))
            if name == "host" and method == "GET":
                h = q.get("host", [""])[0]
                a = sorted((r for r in data.alerts(src) if r.get("host") == h), key=lambda r: float(r.get("ts", 0)))
                e = sorted((r for r in data.events(src) if r.get("host") == h), key=lambda r: float(r.get("ts", 0)))
                return self._json(200, {"host": h, "alerts": a[-200:], "events": e[-200:], "alert_total": len(a), "event_total": len(e)})
            if name == "alerts" and method == "GET":
                rows = _filter(data.alerts(src), q)
                sev = q.get("severity", [""])[0]
                if sev:
                    rows = [r for r in rows if r.get("severity") == sev]
                rows.sort(key=lambda r: (-float(r.get("ts", 0)), -SEVERITY_ORDER.get(r.get("severity", "low"), 0)))
                return self._json(200, {"total": len(rows), "rows": rows[:int(q.get("limit", ["500"])[0])]})
            if name == "events" and method == "GET":
                rows = _filter(data.events(src), q)
                kind = q.get("kind", [""])[0]
                if kind in ("signature", "heuristic"):
                    rows = [r for r in rows if bool((r.get("detail") or {}).get("signature")) == (kind == "signature")]
                rows.sort(key=lambda r: -float(r.get("ts", 0)))
                return self._json(200, {"total": len(rows), "rows": rows[:int(q.get("limit", ["500"])[0])]})
            if name == "upload" and method == "POST":
                if self.headers.get("X-NIDS-Upload") != "1":
                    return self._json(403, {"error": "missing upload header"})
                try:
                    raw = self._body(csv_upload.MAX_BYTES)
                except ValueError:
                    return self._json(413, {"error": f"file is empty or larger than {csv_upload.MAX_BYTES // (1024 * 1024)} MB"})
                from urllib.parse import unquote
                res = csv_upload.handle_upload(raw, unquote(self.headers.get("X-Filename") or "upload.csv"), self.headers.get("X-Policy") or "fpr_3pct",
                                               state, state / "upload_events.jsonl")
                return self._json(200 if "error" not in res else 400, res)
            if name == "download" and method == "GET":
                found = csv_upload.result_file(state, q.get("id", [""])[0], q.get("kind", [""])[0])
                if not found:
                    return self._json(404, {"error": "not found"})
                p, fname = found
                return self._send(200, p.read_bytes(), "text/csv; charset=utf-8", extra=[("Content-Disposition", f'attachment; filename="{fname}"')])
            self._json(404, {"error": "unknown endpoint"})

        # -------------------------------------------------------------------------------------------- flow model API (earlier app)
        def _flow_api(self, method, path, q):
            from layered_nids.flowmodel.features import SchemaError
            from layered_nids.flowmodel.server import HttpError
            try:
                svc = get_flow_service()
                if method == "GET" and path == "/api/info":
                    return self._json(200, svc.info())
                if method == "GET" and path == "/api/evaluation":
                    return self._json(200, svc.evaluation())
                if method == "GET" and path == "/api/sample":
                    try:
                        n = int(q.get("n", ["1500"])[0])
                    except ValueError:
                        raise HttpError(400, "'n' must be an integer")
                    return self._json(200, {"flows": svc.sample(n)})
                if method == "POST":
                    b = json.loads(self._body(MAX_JSON))
                    if not isinstance(b, dict):
                        raise HttpError(400, "body must be a JSON object")
                    if path == "/api/score":
                        return self._json(200, {"results": svc.score(b.get("rows"))})
                    if path == "/api/explain":
                        return self._json(200, svc.explain(b.get("flow"), b.get("n_permutations", 32)))
                    if path == "/api/recalibrate":
                        return self._json(200, svc.recalibrate(b.get("rows"), b.get("label", "label"), b.get("fpr_cap", 0.08)))
                    if path == "/api/adapt":
                        return self._json(200, svc.adapt(b.get("rows")))
                raise HttpError(404, "unknown endpoint")
            except HttpError as e:
                self._json(e.code, {"error": e.message})
            except (SchemaError, ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": str(e)})
            except Exception:                                   # never show a stack trace to the browser
                self._json(500, {"error": "internal error"})

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

    return Handler


def make_server(host: str, port: int, data_dir: Path, state: Path = STATE, capture=None) -> ThreadingHTTPServer:
    if host not in LOOPBACK:
        raise SystemExit(f"refusing to listen on {host!r}: the dashboard is for this computer only (use 127.0.0.1)")
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    users = state / "users.json"
    if not users.exists():
        raise SystemExit("no users yet: run  python -m layered_nids.dashboard --init-user NAME  first")
    lock, holder = threading.Lock(), {}

    def flow_service():
        with lock:
            if "svc" not in holder:
                import pandas as pd
                from layered_nids.flowmodel.bundle import load_bundle
                from layered_nids.flowmodel.contract import FEATURES
                from layered_nids.flowmodel.detector import Detector
                from layered_nids.flowmodel.server import Service
                pool = None
                p = ROOT / "data" / "unsw" / "official_testing_82332.csv"
                if p.exists():
                    cols = pd.read_csv(p, nrows=1).columns
                    pool = pd.read_csv(p).drop_duplicates(subset=[c for c in FEATURES + ["label"] if c in cols]).reset_index(drop=True)
                holder["svc"] = Service(Detector(load_bundle()), pool, HERE / "data" / "final_evaluation.json")
            return holder["svc"]

    if capture is None:
        from layered_nids.dashboard.live_capture import CaptureManager
        capture = CaptureManager(state)
    handler = make_handler(Data(data_dir, state), Sessions(users), state, flow_service, capture)
    return ThreadingHTTPServer((host, port), handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m layered_nids.dashboard")
    ap.add_argument("--data", default=str(ROOT / "reports" / "pipeline"), help="folder with *.alerts.jsonl and *.events.jsonl")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8338)
    ap.add_argument("--state", default=str(STATE), help="folder for users, password file and uploads (never commit it)")
    ap.add_argument("--init-user", metavar="NAME", help="create or reset a user with a random password (written to PASSWORD.txt in the state folder)")
    a = ap.parse_args(argv)
    if a.init_user:
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,32}", a.init_user):
            print("user names: 1-32 letters, digits, . _ -")
            return 2
        state = Path(a.state)
        create_user(state / "users.json", a.init_user, state / "PASSWORD.txt")
        print(f"user '{a.init_user}' created; the password is in {state / 'PASSWORD.txt'}")
        return 0
    srv = make_server(a.host, a.port, Path(a.data), Path(a.state))
    print(f"NIDS dashboard on http://{a.host}:{a.port}   data: {a.data}   (Ctrl+C to stop)", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
