"""Dashboard pages: the Flow analysis tab and the NIDS bar on the owner's app, favicon, 404 pages, access rules for the new static files."""
import http.cookiejar
import json
import threading
import urllib.error
import urllib.request

import pytest

from layered_nids.dashboard.auth import create_user
from layered_nids.dashboard.server import make_server


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


@pytest.fixture(scope="module")
def web(tmp_path_factory):
    state, data = tmp_path_factory.mktemp("state"), tmp_path_factory.mktemp("data")
    create_user(state / "users.json", "tester", state / "PASSWORD.txt")
    pw = [l for l in (state / "PASSWORD.txt").read_text().splitlines() if l.startswith("password: ")][0][10:]
    srv = make_server("127.0.0.1", 0, data, state)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()), NoRedirect)
    assert op.open(urllib.request.Request(base + "/login", json.dumps({"user": "tester", "password": pw}).encode(), {"Content-Type": "application/json"})).status == 200
    yield base, op
    srv.shutdown()


def _get(base, path, opener=None):
    try:
        r = (opener or urllib.request.build_opener(NoRedirect)).open(base + path)
        return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def test_favicon_is_public_and_an_image(web):
    s, b, h = _get(web[0], "/favicon.ico")
    assert s == 200 and h["Content-Type"].startswith("image/svg+xml") and b.startswith(b"<svg")


def test_unknown_page_gets_a_html_404_and_unknown_api_a_json_404(web):
    base, op = web
    s, b, h = _get(base, "/no-such-page", op)
    assert s == 404 and h["Content-Type"].startswith("text/html") and b"Page not found" in b and b"Back to the dashboard" in b
    s, b, h = _get(base, "/nids/api/no-such-endpoint", op)
    assert s == 404 and json.loads(b)["error"]
    s, b, h = _get(base, "/static/no-such-file.js", op)
    assert s == 404 and b"Page not found" in b


def test_navigation_connects_dashboard_and_flow_analysis(web):
    base, op = web
    home = _get(base, "/", op)[1].decode()
    assert 'href="/app/">Flow analysis<' in home and 'id="live-t">Auto-refresh<' in home and "Show live results" not in home
    app = _get(base, "/app/", op)[1].decode()
    assert "<title>Flow analysis - NIDS</title>" in app and 'id="nids-bar"' in app
    for tab in ("/#/overview", "/#/alerts", "/#/events", "/#/upload", "/#/live"):
        assert f'href="{tab}"' in app                                          # the way back to every dashboard page
    assert _get(base, "/static/nids-bar.css", op)[0] == 200 and _get(base, "/static/nids-bar.js", op)[0] == 200
    bundle = _get(base, "/app/assets/index-nKKlqnia.js", op)[1].decode()
    assert "Replay monitor" in bundle and "Live monitor" not in bundle         # a replay is not called live


def test_new_static_files_need_a_session(web):
    base = web[0]
    for p in ("/static/nids-bar.css", "/static/nids-bar.js", "/app/"):
        assert _get(base, p)[0] in (302, 401)
