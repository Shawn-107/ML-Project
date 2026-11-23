"""Live self-test: does the capture -> detection chain work on REAL traffic?

Run only when the owner asks (it was run once on 2026-11-23: 218 packets captured, parsed, and the planted request flagged by the rules layer).

  python -m layered_nids.pipeline.live_selftest [--seconds 20] [--keep]

What it does, in order:
  1. captures your own traffic for a few seconds into data/live (Windows asks for approval: click Yes only if you started this);
  2. while capturing, sends ONE harmless HTTP request to example.com (IANA's public test domain) whose path looks like a directory
     traversal, with the made-up Host header trav.example. Our rules layer has a signature for exactly that pattern, so there is a known answer;
  3. checks that the capture really contains that request (searches the saved bytes; nothing else is printed from the capture);
  4. runs our rules layer on the saved file and looks for the traversal finding;
  5. deletes the capture unless --keep is given.
Only counts and the planted finding are printed. It proves the capture path, our packet parser and the rules layer work on real packets.
It does NOT prove detection accuracy and it does not exercise the packet detector (which needs tens of thousands of clean packets to learn).
"""
from __future__ import annotations

import argparse
import http.client
import socket
import threading
import time
from pathlib import Path

from layered_nids.packets.pcap import iter_packets
from layered_nids.pipeline.live import LIVE_DIR, capture_to_pcap
from layered_nids.pipeline.rules_layer import rules_evidence

PLANTED_PATH = "/download?file=../../../../etc/passwd"           # a classic directory-traversal request
HOST = "example.com"                                             # where the request is sent (IANA's public test domain)
HOST_HEADER = "trav.example"                                     # the Host header of the planted request


def send_planted_request() -> dict:
    """One harmless GET to example.com. Returns the status and our local address (not printed beyond the summary)."""
    ip = socket.gethostbyname(HOST)
    conn = http.client.HTTPConnection(ip, 80, timeout=10)
    try:
        conn.request("GET", PLANTED_PATH, headers={"Host": HOST_HEADER, "User-Agent": "curl/8.0", "Accept": "*/*"})
        r = conn.getresponse()
        r.read(256)
        return {"status": r.status, "dst_ip": ip}
    finally:
        conn.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--keep", action="store_true", help="keep the capture file (it contains your traffic)")
    a = ap.parse_args(argv)
    path = LIVE_DIR / time.strftime("selftest_%Y%m%d_%H%M%S.pcap")
    result: dict = {}
    started = threading.Event()
    t = threading.Thread(target=lambda: result.update(capture_to_pcap(path, a.seconds, 200_000, on_started=started.set)), daemon=True)
    print("capturing... (click Yes on the Windows approval prompt)", flush=True)
    t.start()
    if not started.wait(timeout=90):                             # wait until packets are really being captured (approval prompt included)
        print("the capture did not start within 90 s (approval refused?)", flush=True)
    time.sleep(1.5)
    try:
        sent = send_planted_request()
        print(f"planted request sent: HTTP {sent['status']} from {HOST}", flush=True)
    except Exception as e:
        sent = None
        print(f"planted request FAILED: {type(e).__name__}: {e}", flush=True)
    t.join(timeout=a.seconds + 60)
    if not result:
        print("RESULT: no capture was made (approval refused or capture failed).")
        return 1
    print(f"captured {result['packets']:,} packets in {result['seconds']} s ({result['bytes'] / 1e6:.2f} MB)", flush=True)

    ok_parse = ok_req = False
    n_parsed = 0
    try:
        n_parsed = sum(1 for _ in iter_packets(path))
        ok_parse = n_parsed == result["packets"]
        ok_req = PLANTED_PATH.encode() in Path(path).read_bytes()
        print(f"our parser reads {n_parsed:,} packets back (match: {ok_parse}); planted request present in the capture: {ok_req}", flush=True)
        if result["packets"] < 20:
            print("(very few packets: check that the right network interface was used)", flush=True)
        ev = rules_evidence(iter_packets(path))
        trav = [e for e in ev if "traversal" in (e.reason + str(e.detail.get("trail", ""))).lower()]
        print(f"rules layer on the saved capture: {len(ev)} evidence items total; {len(trav)} directory-traversal finding(s)", flush=True)
        for e in trav[:2]:
            print(f'   finding: "{e.reason}" - internal host attributed: {e.host}', flush=True)
        others = sorted({e.reason[:60] for e in ev if e not in trav})
        if others:
            print(f"   other findings on your real traffic: {others[:5]}", flush=True)
        verdict = bool(sent) and ok_parse and ok_req and len(trav) >= 1
    finally:
        if not a.keep:
            Path(path).unlink(missing_ok=True)
            print("capture file deleted", flush=True)
        else:
            print(f"capture kept (contains your traffic): {path}", flush=True)
    print("RESULT:", "PASS: real capture -> parser -> rules layer flagged the planted request" if verdict else
          "FAIL or INCOMPLETE: see the lines above", flush=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
