"""Small dependency-free JSON-over-HTTP service around a Detector (for the dashboard or other clients).

Endpoints
  GET  /health        status, active policy/threshold, model info
  GET  /drift         score-drift monitor status
  GET  /stats         counters of the alert store (flows seen, alerts, high severity)
  GET  /alerts        ?limit=50&min_severity=medium|high   most recent alerts
  POST /predict       {"flows": [ {...34 fields (+ optional "id")...}, ... ], "ids": [...optional...]} or one flow object
  POST /explain       {"flow": {...}, "n_permutations": 16, "n_background": 6, "top": 10}
  -- admin endpoints, only when started with allow_admin (CLI: --allow-admin) --
  POST /config        {"policy": "fpr_1pct", "use_novelty": true}
  POST /recalibrate   {"flows": [...with "label"...], "fpr_cap": 0.08}
  POST /reset         {"what": "recalibration" | "alerts" | "drift"}

Binds to 127.0.0.1 by default and has NO authentication: keep it local. Not a production server."""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .alerts import AlertStore
from .features import SchemaError
from .inference import Detector

log = logging.getLogger(__name__)
MAX_BODY = 5 * 1024 * 1024
MAX_FLOWS = 10_000
REQUEST_TIMEOUT_S = 30


def make_server(detector: Detector, host: str = "127.0.0.1", port: int = 8008, allow_admin: bool = False,
                store: AlertStore | None = None, allow_recalibrate: bool | None = None) -> ThreadingHTTPServer:
    """`allow_recalibrate` is a deprecated alias of `allow_admin`."""
    if allow_recalibrate is not None:
        allow_admin = allow_admin or allow_recalibrate
    store = store if store is not None else AlertStore()

    class Handler(BaseHTTPRequestHandler):
        timeout = REQUEST_TIMEOUT_S  # a stalled client cannot hold a thread forever

        def log_message(self, fmt, *args):  # route access logs through logging
            log.info("%s - %s", self.address_string(), fmt % args)

        def _send(self, code: int, obj: dict) -> None:
            body = json.dumps(obj, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self):
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise SchemaError("bad Content-Length")
            if n <= 0:
                raise SchemaError("empty request body")
            if n > MAX_BODY:
                raise SchemaError(f"request body too large (max {MAX_BODY} bytes)")
            try:
                return json.loads(self.rfile.read(n))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                raise SchemaError(f"invalid JSON: {e}") from e

        @staticmethod
        def _flows(body):
            flows = body.get("flows", body) if isinstance(body, dict) else body
            if isinstance(flows, list) and len(flows) > MAX_FLOWS:
                raise SchemaError(f"too many flows (max {MAX_FLOWS})")
            return flows

        def _drain_body(self) -> None:
            """Read and discard the request body so an early error reply is not turned into a TCP reset
            (seen on Windows when the server closes with unread data)."""
            try:
                n = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return
            if 0 < n <= MAX_BODY:
                self.rfile.read(n)

        def _admin(self) -> bool:
            if not allow_admin:
                self._drain_body()
                self._send(403, {"error": "admin endpoints are disabled (start with --allow-admin)"})
                return False
            return True

        def do_GET(self):
            try:
                url = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(url.query).items()}
                if url.path == "/health":
                    self._send(200, {"status": "ok", "admin_enabled": allow_admin, **detector.info()})
                elif url.path == "/drift":
                    self._send(200, detector.monitor.status())
                elif url.path == "/stats":
                    self._send(200, store.stats())
                elif url.path == "/alerts":
                    try:
                        limit = int(q.get("limit", 50))
                    except ValueError:
                        raise SchemaError("limit must be an integer")
                    self._send(200, {"alerts": store.recent(limit, q.get("min_severity"))})
                else:
                    self._send(404, {"error": "not found"})
            except (SchemaError, ValueError) as e:
                self._send(400, {"error": str(e)})
            except Exception:
                log.exception("GET failed")
                self._send(500, {"error": "internal error"})

        def do_POST(self):
            try:
                path = urlparse(self.path).path
                if path == "/predict":
                    body = self._read_json()
                    flows = self._flows(body)
                    results = detector.predict(flows)
                    ids = body.get("ids") if isinstance(body, dict) else None
                    if ids is None and isinstance(flows, list):
                        ids = [f.get("id") if isinstance(f, dict) else None for f in flows]
                    elif ids is None and isinstance(flows, dict):
                        ids = [flows.get("id")]
                    if ids is not None and len(ids) != len(results):
                        raise SchemaError("'ids' must have one entry per flow")
                    for r, i in zip(results, ids or [None] * len(results)):
                        if i is not None:
                            r["id"] = i
                    store.add_batch(results, ids)
                    self._send(200, {"results": results})
                elif path == "/explain":
                    body = self._read_json()
                    if not isinstance(body, dict):
                        raise SchemaError("body must be an object")
                    flow = body.get("flow", body.get("flows"))
                    if isinstance(flow, list) and len(flow) == 1:
                        flow = flow[0]
                    if not isinstance(flow, dict):
                        raise SchemaError("'flow' must be one flow object")
                    self._send(200, detector.explain(flow, n_permutations=int(body.get("n_permutations", 16)),
                                                     n_background=int(body.get("n_background", 6)),
                                                     seed=int(body.get("seed", 0)), top=int(body.get("top", 10))))
                elif path == "/config":
                    if not self._admin():
                        return
                    body = self._read_json()
                    if not isinstance(body, dict):
                        raise SchemaError("body must be an object")
                    if "policy" in body:
                        detector.set_policy(str(body["policy"]))
                    if "use_novelty" in body:
                        if not isinstance(body["use_novelty"], bool):
                            raise SchemaError("use_novelty must be true or false")
                        detector.use_novelty = body["use_novelty"]
                    self._send(200, detector.info())
                elif path == "/recalibrate":
                    if not self._admin():
                        return
                    body = self._read_json()
                    cap = float(body.get("fpr_cap", 0.08)) if isinstance(body, dict) else 0.08
                    self._send(200, detector.recalibrate(self._flows(body), fpr_cap=cap))
                elif path == "/reset":
                    if not self._admin():
                        return
                    body = self._read_json()
                    what = body.get("what") if isinstance(body, dict) else None
                    if what == "recalibration":
                        detector.reset_recalibration()
                    elif what == "alerts":
                        store.clear()
                    elif what == "drift":
                        detector.monitor.reset()
                    else:
                        raise SchemaError("'what' must be one of: recalibration, alerts, drift")
                    self._send(200, {"reset": what})
                else:
                    self._send(404, {"error": "not found"})
            except (SchemaError, ValueError, TypeError) as e:  # client errors: bad input, not enough labelled flows, ...
                self._send(400, {"error": str(e)})
            except Exception:
                log.exception("POST failed")
                self._send(500, {"error": "internal error"})

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    srv.alert_store = store
    return srv
