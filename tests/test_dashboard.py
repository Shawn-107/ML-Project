"""NIDS dashboard: sign-in, access control, path safety, data APIs and upload checks (real server on an ephemeral localhost port)."""
import json
import threading
import urllib.error
import urllib.request
import http.cookiejar

import pytest

from layered_nids.dashboard import csv_upload
from layered_nids.dashboard.auth import create_user, hash_password, verify_password
from layered_nids.dashboard.server import make_server


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    state = tmp_path_factory.mktemp("state")
    data = tmp_path_factory.mktemp("data")
    (data / "t.alerts.jsonl").write_text("\n".join(json.dumps(a) for a in [
        {"host": "192.168.2.5", "ts": 1000.0, "probability": 0.99999, "severity": "critical", "layers": {"packets": 0.9, "rules": 0.1}, "reasons": ["scan-like"]},
        {"host": "192.168.2.6", "ts": 2000.0, "probability": 0.97, "severity": "medium", "layers": {"flows": 1.0}, "reasons": ["flagged flows"]}]) + "\n")
    (data / "t.events.jsonl").write_text("\n".join(json.dumps(e) for e in [
        {"layer": "rules", "host": "192.168.2.5", "ts": 1000.0, "score": 0.85, "reason": "directory traversal", "detail": {"signature": True, "trail": "directory traversal"}},
        {"layer": "rules", "host": "192.168.2.6", "ts": 2000.0, "score": 0.6, "reason": "arp flood", "detail": {"signature": False, "trail": "arp_flood"}}]) + "\n")
    create_user(state / "users.json", "tester", state / "PASSWORD.txt")
    password = [l for l in (state / "PASSWORD.txt").read_text().splitlines() if l.startswith("password: ")][0][10:]
    server = make_server("127.0.0.1", 0, data, state)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", password, state
    server.shutdown()


def call(base, path, data=None, headers=None, opener=None):
    opener = opener or urllib.request.build_opener(NoRedirect)
    try:
        r = opener.open(urllib.request.Request(base + path, data, headers or {}))
        return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def signed_in(srv):
    base, pw, _ = srv
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()), NoRedirect)
    s, _, h = call(base, "/login", json.dumps({"user": "tester", "password": pw}).encode(), {"Content-Type": "application/json"}, op)
    assert s == 200
    return op


def test_password_hashing_roundtrip():
    rec = hash_password("correct horse")
    assert verify_password("correct horse", rec) and not verify_password("wrong", rec)


def test_server_refuses_non_loopback_and_missing_users(tmp_path):
    with pytest.raises(SystemExit):
        make_server("0.0.0.0", 0, tmp_path, tmp_path)
    with pytest.raises(SystemExit):
        make_server("127.0.0.1", 0, tmp_path, tmp_path)         # no users.json yet


def test_everything_needs_a_session(srv):
    base = srv[0]
    assert call(base, "/")[0] == 302 and call(base, "/app/")[0] == 302
    for p in ("/nids/api/alerts", "/nids/api/summary", "/api/info", "/nids/api/download?id=" + "0" * 32 + "&kind=scored"):
        assert call(base, p)[0] == 401
    assert call(base, "/login")[0] == 200                         # the sign-in page itself is public


def test_wrong_password_unknown_user_and_cookie_flags(srv):
    base, pw, _ = srv
    J = {"Content-Type": "application/json"}
    assert call(base, "/login", json.dumps({"user": "tester", "password": "x"}).encode(), J)[0] == 401
    assert call(base, "/login", json.dumps({"user": "nobody", "password": pw}).encode(), J)[0] == 401
    s, _, h = call(base, "/login", json.dumps({"user": "tester", "password": pw}).encode(), J)
    assert s == 200 and "HttpOnly" in h["Set-Cookie"] and "SameSite=Strict" in h["Set-Cookie"]


def test_cross_origin_post_is_refused_and_security_headers_present(srv):
    base = srv[0]
    op = signed_in(srv)
    assert call(base, "/logout", b"{}", {"Origin": "http://evil.example"}, op)[0] == 403
    s, _, h = call(base, "/", opener=op)
    assert s == 200 and h["X-Frame-Options"] == "DENY" and "script-src 'self'" in h["Content-Security-Policy"] and h["X-Content-Type-Options"] == "nosniff"


@pytest.mark.parametrize("path", ["/static/../server.py", "/static/%2e%2e/auth.py", "/app/../../server.py", "/app/%2e%2e/%2e%2e/auth.py", "/static/..%5cserver.py"])
def test_static_paths_cannot_leave_their_folder(srv, path):
    assert call(srv[0], path, opener=signed_in(srv))[0] == 404


def test_alert_and_event_apis_filter_and_count(srv):
    base = srv[0]
    op = signed_in(srv)
    get = lambda p: json.loads(call(base, p, opener=op)[1])
    assert get("/nids/api/summary") == {"alerts": 2, "alert_hosts": 2, "severity": {"critical": 1, "medium": 1}, "events": 2, "signature_events": 1, "event_hosts": 2}
    assert [a["host"] for a in get("/nids/api/alerts?severity=critical")["rows"]] == ["192.168.2.5"]
    assert [a["host"] for a in get("/nids/api/alerts")["rows"]] == ["192.168.2.6", "192.168.2.5"]          # newest first
    assert get("/nids/api/events?kind=signature")["total"] == 1 and get("/nids/api/events?kind=heuristic&q=arp")["total"] == 1


def test_owner_app_and_flow_api_are_served(srv):
    base = srv[0]
    op = signed_in(srv)
    s, b, _ = call(base, "/app/", opener=op)
    assert s == 200 and b.startswith(b"<!doctype html>")
    s, b, _ = call(base, "/api/info", opener=op)
    info = json.loads(b)
    assert s == 200 and set(info["thresholds"]) == {"fpr_1pct", "fpr_3pct", "fpr_5pct"}
    s, b, _ = call(base, "/api/score", b"not json", {"Content-Type": "application/json"}, op)
    assert s == 400 and "error" in json.loads(b)


def test_upload_checks_and_download_safety(srv):
    base, _, state = srv
    op = signed_in(srv)
    U = {"Content-Type": "application/octet-stream", "X-NIDS-Upload": "1"}
    assert call(base, "/nids/api/upload", b"x", {"Content-Type": "application/octet-stream"}, op)[0] == 403       # header missing
    assert call(base, "/nids/api/upload", b"", U, op)[0] in (400, 413)
    s, b, _ = call(base, "/nids/api/upload", b"hello,world\n1,2\n", U, op)
    assert s == 400 and "unrecognised flow file" in json.loads(b)["error"]
    assert not list((state / "uploads").glob("*.input"))                                                          # the upload is not kept
    for q in ("id=../../x&kind=scored", "id=" + "0" * 32 + "&kind=scored", "id=" + "g" * 32 + "&kind=scored", "id=" + "0" * 32 + "&kind=../x"):
        assert call(base, "/nids/api/download?" + q, opener=op)[0] == 404


def test_csv_upload_result_file_validation():
    assert csv_upload.result_file(csv_upload.PROJECT, "../../x", "scored") is None
    assert csv_upload.result_file(csv_upload.PROJECT, "0" * 32, "../x") is None


def test_stats_and_host_endpoints(srv):
    base = srv[0]
    op = signed_in(srv)
    get = lambda p: json.loads(call(base, p, opener=op)[1])
    st = get("/nids/api/stats")
    assert st["severity"] == {"critical": 1, "high": 0, "medium": 1, "low": 0} and len(st["hist"]) == 48
    assert sum(b["critical"] + b["medium"] for b in st["hist"]) == 2 and sum(b["events"] for b in st["hist"]) == 2
    assert {h["host"] for h in st["top_hosts"]} == {"192.168.2.5", "192.168.2.6"} and abs(sum(st["layers"].values()) - 1.0) < 1e-9
    assert {r["rule"] for r in st["rules"]} == {"directory traversal", "arp_flood"}
    host = get("/nids/api/host?host=192.168.2.5")
    assert host["alert_total"] == 1 and host["event_total"] == 1 and host["alerts"][0]["severity"] == "critical"
    assert get("/nids/api/host?host=10.9.9.9")["alert_total"] == 0
    assert call(base, "/nids/api/stats")[0] == 401                          # needs a session


def test_lockout_after_repeated_failures(srv):
    base, pw, _ = srv
    J = {"Content-Type": "application/json"}
    for _ in range(6):
        call(base, "/login", json.dumps({"user": "tester", "password": "bad"}).encode(), J)
    assert call(base, "/login", json.dumps({"user": "tester", "password": pw}).encode(), J)[0] == 401              # locked for a minute
