"""Local JSON HTTP service for the web frontend (standard library only). Endpoints are described in frontend/API.md.
Listens on 127.0.0.1 only; there is no authentication or TLS."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import pandas as pd

from .contract import CATEGORICAL, FEATURES
from .detector import Detector
from .features import SchemaError, coerce

MAX_BODY = 5_000_000
MAX_ROWS = 10_000
REPORT = Path(__file__).resolve().parent.parent / "reports" / "final_evaluation.json"
LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$")


class HttpError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code, self.message = code, message


def _num(x, nd=6):
    return None if x is None else round(float(x), nd)


class Service:
    """The model plus the replay pool. All model calls are serialised with a lock."""

    def __init__(self, detector: Detector, pool: pd.DataFrame | None = None, report_path: Path = REPORT):
        self.d, self.pool, self.report_path, self.lock = detector, pool, Path(report_path), threading.Lock()

    def info(self) -> dict:
        b, m = self.d.b, self.d.b.meta
        return dict(defaultPolicy=m["default_policy"], thresholds={k: _num(v) for k, v in b.thresholds.items()}, noveltyThreshold=_num(b.novelty_threshold, 4),
                    features=m["features"], trainedAtUtc=m["trained_at_utc"], nTrain=m["n_train"], nVal=m["n_val"],
                    model=m.get("model", "gradient boosting"), valAuc=_num(m["val_auc"]), trainAuc=_num(m["train_auc"]), kmeansK=m["kmeans_k"], nTrees=m.get("n_trees"))

    def evaluation(self) -> dict:
        if not self.report_path.exists():
            raise HttpError(404, "evaluation report not found; run final/evaluate_final.py")
        ev = json.loads(self.report_path.read_text())
        return dict(policies=ev["policies"], auc=ev["auc"], perAttackRecall=ev["per_attack_recall_official"], loao=ev["loao"], clusterCv=ev["cluster_cv_auc"],
                    shuffled=ev["shuffled_label_auc"], bootstrap=ev["bootstrap"], noise=ev["noise"], zeroed=ev["input_set_to_zero"], latencyMs=ev["latency_ms"],
                    checks=ev["checks"], nearCopyNote='rows with no near-copy of any training flow ("far-only") give the leak-free accuracy')

    def score(self, rows) -> list[dict]:
        if not isinstance(rows, list) or not rows:
            raise HttpError(400, "'rows' must be a non-empty list of flows")
        if len(rows) > MAX_ROWS:
            raise HttpError(413, f"too many rows ({len(rows)}); the limit is {MAX_ROWS} per request")
        if not all(isinstance(r, dict) for r in rows):
            raise HttpError(400, "every row must be an object with the model inputs")
        with self.lock:
            p = self.d.predict(coerce(rows))
        return [dict(probability=_num(r.probability), novelty_distance=_num(r.novelty_distance, 4), unusual=bool(r.unusual))
                for r in p.itertuples()]

    def sample(self, n: int) -> list[dict]:
        if self.pool is None or not len(self.pool):
            raise HttpError(404, "no replay data loaded (start the service with --pool path/to/flows.csv)")
        s = self.pool.sample(min(max(1, n), len(self.pool)), random_state=7).reset_index(drop=True)
        scored = self.score(s[FEATURES].to_dict("records"))
        clean = coerce(s[FEATURES])
        out = []
        for i, sc in enumerate(scored):
            f = {k: (str(clean.iloc[i][k]) if k in CATEGORICAL else _num(clean.iloc[i][k], 4)) for k in FEATURES}
            flow = dict(id=i, features=f, **sc)
            if "label" in s.columns:
                flow["label"] = int(s.label[i])
            if "attack_cat" in s.columns:
                flow["attack_cat"] = str(s.attack_cat[i])
            out.append(flow)
        return out

    def explain(self, flow, n_permutations) -> dict:
        if not isinstance(flow, dict):
            raise HttpError(400, "'flow' must be an object with the model inputs")
        if not isinstance(n_permutations, int) or not 1 <= n_permutations <= 64:
            raise HttpError(400, "'n_permutations' must be an integer between 1 and 64")
        with self.lock:
            e = self.d.explain(flow, n_permutations=n_permutations)
        return dict(probability=_num(e["probability"]), baseline=_num(e["baseline"]),
                    contributions=[dict(feature=c["feature"], value=(c["value"] if isinstance(c["value"], str) else _num(c["value"], 4)),
                                        contribution=_num(c["contribution"]), std_err=_num(c["std_err"])) for c in e["contributions"]])

    def adapt(self, rows) -> dict:
        if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
            raise HttpError(400, "'rows' must be a non-empty list of flows from your network")
        if len(rows) > MAX_ROWS:
            raise HttpError(413, f"too many rows ({len(rows)}); the limit is {MAX_ROWS} per request")
        with self.lock:
            rep = Detector(self.d.b).adapt(coerce(rows), apply=False)
        return dict(n_flows=rep["n_flows"], domain_auc=_num(rep["domain_auc"], 4), effective_sample=_num(rep["effective_sample"], 1),
                    old_thresholds={k: _num(v) for k, v in rep["old_thresholds"].items()}, new_thresholds={k: _num(v) for k, v in rep["new_thresholds"].items()}, notes=rep["notes"])

    def recalibrate(self, rows, label, fpr_cap) -> dict:
        if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
            raise HttpError(400, "'rows' must be a non-empty list of labelled flows")
        if len(rows) > MAX_ROWS:
            raise HttpError(413, f"too many rows ({len(rows)}); the limit is {MAX_ROWS} per request")
        if not isinstance(label, str) or not label:
            raise HttpError(400, "'label' must name the label column")
        if not isinstance(fpr_cap, (int, float)) or not 0 < fpr_cap <= 0.5:
            raise HttpError(400, "'fpr_cap' must be a number between 0 and 0.5")
        df = pd.DataFrame(rows)
        if label not in df.columns:
            raise HttpError(400, f"rows need a '{label}' column (0 = normal, 1 = attack)")
        df[label] = pd.to_numeric(df[label], errors="coerce")
        if df[label].isna().any():
            raise HttpError(400, f"'{label}' must be 0 or 1 in every row")
        df[label] = df[label].astype(int)
        with self.lock:
            rep = Detector(self.d.b, policy=self.d.policy if self.d.policy in self.d.b.thresholds else "fpr_3pct").recalibrate(df, label=label, fpr_cap=float(fpr_cap))
        return {k: (bool(v) if isinstance(v, (bool, np.bool_)) else _num(v) if isinstance(v, (float, np.floating)) else v) for k, v in rep.items()}


def make_handler(service: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        timeout = 30

        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code: int, obj) -> None:
            body = json.dumps(obj, default=float).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            origin = self.headers.get("Origin", "")
            if LOCAL_ORIGIN.match(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                raise HttpError(400, "empty request body")
            if n > MAX_BODY:
                # read (and discard) the body so the client can still receive the clean 413 message; give up on absurd sizes
                left = n if n <= 10 * MAX_BODY else 0
                while left > 0:
                    chunk = self.rfile.read(min(left, 1 << 20))
                    if not chunk:
                        break
                    left -= len(chunk)
                self.close_connection = True
                raise HttpError(413, f"request too large (limit {MAX_BODY // 1_000_000} MB)")
            try:
                data = json.loads(self._raw if self._raw is not None else self.rfile.read(n))
            except json.JSONDecodeError:
                raise HttpError(400, "body is not valid JSON") from None
            if not isinstance(data, dict):
                raise HttpError(400, "body must be a JSON object")
            return data

        def _route(self, method: str):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if method == "GET" and u.path == "/api/info":
                return service.info()
            if method == "GET" and u.path == "/api/evaluation":
                return service.evaluation()
            if method == "GET" and u.path == "/api/sample":
                try:
                    n = int(q.get("n", ["1500"])[0])
                except ValueError:
                    raise HttpError(400, "'n' must be an integer") from None
                return {"flows": service.sample(n)}
            if method == "POST" and u.path == "/api/score":
                return {"results": service.score(self._body().get("rows"))}
            if method == "POST" and u.path == "/api/explain":
                b = self._body()
                return service.explain(b.get("flow"), b.get("n_permutations", 32))
            if method == "POST" and u.path == "/api/recalibrate":
                b = self._body()
                return service.recalibrate(b.get("rows"), b.get("label", "label"), b.get("fpr_cap", 0.08))
            if method == "POST" and u.path == "/api/adapt":
                return service.adapt(self._body().get("rows"))
            if u.path.startswith("/api/"):
                known = {"/api/info", "/api/evaluation", "/api/sample", "/api/score", "/api/explain", "/api/recalibrate", "/api/adapt"}
                raise HttpError(405 if u.path in known else 404, "method not allowed" if u.path in known else "unknown endpoint")
            raise HttpError(404, "unknown endpoint")

        def _handle(self, method: str):
            # always read a normal-sized body first, even if the path turns out to be wrong; answering before the body is
            # read can make the client's connection reset on Windows
            self._raw = None
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if 0 < n <= MAX_BODY:
                self._raw = self.rfile.read(n)
            try:
                self._send(200, self._route(method))
            except HttpError as e:
                self._send(e.code, {"error": e.message})
            except (SchemaError, ValueError) as e:
                self._send(400, {"error": str(e)})
            except Exception:  # never leak a stack trace to the client
                self._send(500, {"error": "internal error"})

        def do_GET(self):
            self._handle("GET")

        def do_POST(self):
            self._handle("POST")

        def do_OPTIONS(self):
            self.send_response(204)
            origin = self.headers.get("Origin", "")
            if LOCAL_ORIGIN.match(origin):
                self.send_header("Access-Control-Allow-Origin", origin)
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def make_server(detector: Detector, host: str = "127.0.0.1", port: int = 8008, pool: pd.DataFrame | None = None, report_path: Path = REPORT) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("the service has no authentication and may only listen on localhost")
    srv = ThreadingHTTPServer((host, port), make_handler(Service(detector, pool, report_path)))
    srv.daemon_threads = True
    return srv
