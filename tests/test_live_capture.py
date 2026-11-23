"""Dashboard live capture: limits, acknowledgement on non-private networks, one job at a time, findings stored as 'live', pcap always deleted,
and the HTTP API (session, header, source filter). A fake capture and a fake analysis are injected: nothing here touches the network."""
import http.cookiejar
import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from layered_nids.common.schema import Evidence
from layered_nids.dashboard.auth import create_user
from layered_nids.dashboard.live_capture import CaptureManager, MAX_SECONDS
from layered_nids.dashboard.server import make_server

PRIVATE = [{"name": "home", "category": "Private"}]
PUBLIC = [{"name": "campus", "category": "Public"}]


def _evidence():
    return [Evidence("rules", "10.0.0.5", 1000.0, 0.9, "directory traversal request", {"signature": True, "trail": "directory traversal"}),
            Evidence("rules", "10.0.0.5", 1010.0, 0.9, "directory traversal request again", {"signature": True, "trail": "directory traversal"}),
            Evidence("rules", "10.0.0.7", 1020.0, 0.6, "many new destinations", {"signature": False, "trail": "scan"})]


def _manager(tmp_path, networks=PRIVATE, capture=None, analyse=None):
    return CaptureManager(tmp_path, capture=capture or (lambda p, s, n, t=False: {"packets": 1234, "seconds": 5.0}), analyse=analyse or (lambda p: _evidence()),
                          network=lambda: networks)


def _wait(m, states=("done", "failed"), timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if m.status()["job"].get("state") in states:
            return m.status()["job"]
        time.sleep(0.02)
    raise AssertionError("capture did not finish")


def test_capture_stores_live_findings_and_alerts_only_for_signature_hits(tmp_path):
    m = _manager(tmp_path)
    m.start(20)
    job = _wait(m)
    assert job["state"] == "done" and job["packets"] == 1234 and job["events"] == 3 and job["signature_hits"] == 2
    ev = [json.loads(l) for l in (tmp_path / "live_events.jsonl").read_text().splitlines()]
    assert len(ev) == 3 and all(e["detail"]["source"] == "live" and e["detail"]["capture"] == job["id"] for e in ev)
    al = [json.loads(l) for l in (tmp_path / "live_alerts.jsonl").read_text().splitlines()]
    assert len(al) == 1 and al[0]["host"] == "10.0.0.5" and al[0]["severity"] == "high" and al[0]["source"] == "live"     # two hits within the cooldown = one alert


def test_limits_and_acknowledgement(tmp_path):
    m = _manager(tmp_path)
    for bad in (0, -1, MAX_SECONDS + 1, "abc", None):
        with pytest.raises(ValueError):
            m.start(bad)
    with pytest.raises(ValueError):
        m.start(10, 10 ** 9)
    shared = _manager(tmp_path, networks=PUBLIC)
    with pytest.raises(PermissionError):
        shared.start(10)
    unknown = _manager(tmp_path, networks=[])
    with pytest.raises(PermissionError):
        unknown.start(10)
    shared.start(10, acknowledged=True)
    assert _wait(shared)["state"] == "done"


def test_one_capture_at_a_time_and_failure_is_reported_and_file_removed(tmp_path):
    gate = threading.Event()
    seen = {}

    def slow(path, s, n):
        seen["path"] = path
        path.write_bytes(b"x")
        gate.wait(5)
        raise RuntimeError("the capture failed: was the Windows approval declined, or is Npcap missing?")

    m = _manager(tmp_path, capture=slow)
    m.start(10)
    with pytest.raises(RuntimeError):
        m.start(10)                                                       # busy
    with pytest.raises(RuntimeError):
        m.clear()                                                         # cannot clear while running
    gate.set()
    job = _wait(m)
    assert job["state"] == "failed" and "approval" in job["message"]
    assert not seen["path"].exists()                                      # the pcap is deleted also on failure
    assert not (tmp_path / "live_events.jsonl").exists()


def test_clear_removes_live_results(tmp_path):
    m = _manager(tmp_path)
    m.start(10)
    _wait(m)
    assert m.status()["live_events"] == 3 and m.status()["live_alerts"] == 1
    assert m.clear()["removed_files"] == 3                                       # alerts, events and the capture history
    assert m.status()["live_events"] == 0 and m.status()["live_alerts"] == 0


# ---------------------------------------------------------------------------------------------------- HTTP API
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    state, data = tmp_path_factory.mktemp("state"), tmp_path_factory.mktemp("data")
    (data / "r.alerts.jsonl").write_text(json.dumps({"host": "192.168.2.9", "ts": 5.0, "probability": 0.99, "severity": "high", "layers": {"rules": 1.0}, "reasons": ["recorded"]}) + "\n")
    create_user(state / "users.json", "tester", state / "PASSWORD.txt")
    pw = [l for l in (state / "PASSWORD.txt").read_text().splitlines() if l.startswith("password: ")][0][10:]
    mgr = _manager(state, networks=PUBLIC)
    srv = make_server("127.0.0.1", 0, data, state, capture=mgr)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()), NoRedirect)
    req = urllib.request.Request(base + "/login", json.dumps({"user": "tester", "password": pw}).encode(), {"Content-Type": "application/json"})
    assert op.open(req).status == 200
    yield base, op, mgr
    srv.shutdown()


def _call(base, path, data=None, headers=None, opener=None):
    try:
        r = (opener or urllib.request.build_opener(NoRedirect)).open(urllib.request.Request(base + path, data, headers or {}))
        return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


H = {"Content-Type": "application/json", "X-NIDS-Capture": "1"}


def test_api_needs_session_and_header(web):
    base, op, _ = web
    assert _call(base, "/nids/api/capture/status")[0] == 401
    assert _call(base, "/nids/api/capture", b"{}", H)[0] == 401
    assert _call(base, "/nids/api/capture", json.dumps({"seconds": 10}).encode(), {"Content-Type": "application/json"}, op)[0] == 403     # header missing
    assert _call(base, "/nids/api/capture/clear", b"{}", {"Content-Type": "application/json"}, op)[0] == 403


def test_api_validation_acknowledgement_run_and_source_filter(web):
    base, op, mgr = web
    s, b = _call(base, "/nids/api/capture", json.dumps({"seconds": 9999}).encode(), H, op)
    assert s == 400 and "seconds" in b["error"]
    s, b = _call(base, "/nids/api/capture", b"not json", H, op)
    assert s == 400
    s, b = _call(base, "/nids/api/capture", json.dumps({"seconds": 10}).encode(), H, op)                  # public network, no acknowledgement
    assert s == 403 and b["needs_acknowledgement"] is True
    st = _call(base, "/nids/api/capture/status", opener=op)[1]
    assert st["needs_acknowledgement"] is True and st["networks"] == PUBLIC and st["job"]["state"] == "idle"
    s, b = _call(base, "/nids/api/capture", json.dumps({"seconds": 10, "acknowledged": True}).encode(), H, op)
    assert s == 200 and b["state"] == "capturing"
    _wait(mgr)
    st = _call(base, "/nids/api/capture/status", opener=op)[1]
    assert st["job"]["state"] == "done" and st["live_events"] == 3 and st["live_alerts"] == 1
    # the two data spaces stay apart
    assert _call(base, "/nids/api/summary", opener=op)[1]["alerts"] == 1                                    # recorded: the one file alert, no live rows
    live = _call(base, "/nids/api/summary?source=live", opener=op)[1]
    assert live["alerts"] == 1 and live["events"] == 3 and live["signature_events"] == 2
    assert _call(base, "/nids/api/alerts?source=live", opener=op)[1]["rows"][0]["host"] == "10.0.0.5"
    assert _call(base, "/nids/api/alerts", opener=op)[1]["rows"][0]["host"] == "192.168.2.9"
    assert _call(base, "/nids/api/stats?source=live", opener=op)[1]["severity"]["high"] == 1
    s, b = _call(base, "/nids/api/capture/clear", b"{}", H, op)
    assert s == 200 and _call(base, "/nids/api/summary?source=live", opener=op)[1]["events"] == 0


def test_history_records_every_capture_even_an_empty_one_and_the_test_request_flag(tmp_path):
    seen = []
    m = _manager(tmp_path, capture=lambda p, s, n, t=False: seen.append(t) or {"packets": 718, "seconds": 32.4}, analyse=lambda p: [])
    m.start(10)
    _wait(m)
    m.start(10, test_request=True)
    _wait(m)
    assert seen == [False, True]                                          # the option reaches the capture step; off by default
    h = m.status()["history"]
    assert [(x["packets"], x["events"], x["alerts"], x["test_request"]) for x in h] == [(718, 0, 0, False), (718, 0, 0, True)]
    assert m.status()["live_events"] == 0 and m.status()["live_alerts"] == 0   # nothing found, and the history still explains it
    m.clear()
    assert m.status()["history"] == []
