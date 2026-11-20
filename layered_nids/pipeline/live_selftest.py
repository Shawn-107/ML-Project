"""Live self-test: does the capture -> detection chain work on REAL traffic?

  python -m layered_nids.pipeline.live_selftest [--seconds 20] [--keep]

What it does, in order:
  1. captures your own traffic for a few seconds into data/live (Windows asks for approval: click Yes only if you started this);
  2. while capturing, sends ONE harmless HTTP request to example.com (IANA's public test domain) whose path looks like a directory
     traversal, with the made-up Host header trav.example (Maltrail whitelists example.com, so it would ignore that name; trav.example is
     what Maltrail's own test uses). Maltrail has a heuristic for exactly that pattern, so there is a known answer;
  3. checks that the capture really contains that request (searches the saved bytes; nothing else is printed from the capture);
  4. runs the real Maltrail sensor (WSL2) on the saved file and looks for the traversal finding;
  5. deletes the capture unless --keep is given.
Only counts and the planted finding are printed. It proves the capture path, our packet parser and the sensor work on real packets.
It does NOT prove detection accuracy and it does not exercise Kitsune (which needs tens of thousands of clean packets to learn).
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
from layered_nids.pipeline.maltrail_layer import events_to_evidence, parse_log, run_sensor_wsl

PLANTED_PATH = "/download?file=../../../../etc/passwd"           # Maltrail's own directory-traversal test pattern
HOST = "example.com"                                             # where the request is sent (IANA's public test domain)
HOST_HEADER = "trav.example"                                     # the Host header: Maltrail whitelists example.com itself, so it would ignore it


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
        events, bad = parse_log(run_sensor_wsl(path))
        trav = [e for e in events if "traversal" in e["info"].lower()]
        print(f"Maltrail sensor on the saved capture: {len(events)} events total ({bad} unparsed); "
              f"{len(trav)} directory-traversal finding(s)", flush=True)
        for e in trav[:2]:
            print(f'   finding: "{e["info"]}" ({e["ref"]}) - internal host attributed: {events_to_evidence([e])[0].host if events_to_evidence([e]) else "none"}', flush=True)
        others = sorted({e["info"] for e in events if "traversal" not in e["info"].lower()})
        if others:
            print(f"   other finding types on your real traffic: {others[:5]}", flush=True)
        verdict = bool(sent) and ok_parse and ok_req and len(trav) >= 1
    finally:
        if not a.keep:
            Path(path).unlink(missing_ok=True)
            print("capture file deleted", flush=True)
        else:
            print(f"capture kept (contains your traffic): {path}", flush=True)
    print("RESULT:", "PASS: real capture -> parser -> real Maltrail sensor flagged the planted request" if verdict else
          "FAIL or INCOMPLETE: see the lines above", flush=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
