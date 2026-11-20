"""Long live test of Kitsune on YOUR real traffic, with a planted anomaly that gives known answers.

  python -m layered_nids.pipeline.live_kitsune_test --minutes 60 --plant-at 50

  1. captures your own traffic for --minutes (one Windows approval prompt at the start; use the laptop normally);
  2. at minute --plant-at it sends a short burst of DNS lookups for random names under example.com (400 lookups, about 20 per second):
     harmless, but unlike normal browsing. Our parser reads DNS query names, so those query packets are KNOWN anomalous packets;
  3. afterwards Kitsune (reduced learning windows, see below) scores the capture; we report how many of the planted query packets
     score above the cut-off, versus how many other packets do (the false-alarm rate), plus what Maltrail's real sensor says;
  4. deletes the capture unless --keep (it contains your traffic).

Reduced learning: Kitsune's defaults need ~70,000 packets (55,000 to learn + 15,000 to calibrate); a quiet laptop makes 10-50 packets/s, so the
defaults may never finish in an hour. This test uses fm=2000, ad=20000, calibration=5000. Learning on fewer packets is LESS reliable; the
result is a feasibility check, not a benchmark. Only aggregate numbers are printed from the capture.
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import string
import threading
import time
from pathlib import Path

import numpy as np

from layered_nids.common.paths import REPORTS
from layered_nids.packets.pcap import iter_packets
from layered_nids.pipeline.kitsune_layer import calibrate_cutoff, score_packets
from layered_nids.pipeline.live import LIVE_DIR, MAX_SECONDS, capture_to_pcap
from layered_nids.pipeline.maltrail_layer import parse_log, run_sensor_wsl

PLANT_SUFFIX = ".example.com"
PLANT_PREFIX = "kt"


def planted_name() -> str:
    return PLANT_PREFIX + "".join(random.choices(string.ascii_lowercase, k=10)) + PLANT_SUFFIX


def is_planted_query(p) -> bool:
    return bool(p.qname) and p.qname.startswith(PLANT_PREFIX) and p.qname.endswith(PLANT_SUFFIX) and len(p.qname) == len(PLANT_PREFIX) + 10 + len(PLANT_SUFFIX)


def send_burst(n: int = 400, per_second: float = 20.0) -> tuple[float, float]:
    """Look up n random names (they do not exist). Returns (start, end) epoch seconds."""
    t0 = time.time()
    for _ in range(n):
        try:
            socket.getaddrinfo(planted_name(), 80)
        except OSError:
            pass                                                 # NXDOMAIN is the expected answer
        time.sleep(1.0 / per_second)
    return t0, time.time()


def analyse(packets: list, scores: np.ndarray, cutoff: float, first: int, window: tuple[float, float]) -> dict:
    """Packet-level result against the planted queries. `first` = index where scoring/calibration is over."""
    planted = np.array([is_planted_query(p) for p in packets])
    ts = np.array([p.ts for p in packets])
    idx = np.arange(len(packets))
    scored = idx >= first
    in_window = (ts >= window[0] - 1.0) & (ts <= window[1] + 2.0)
    target = planted & scored
    others = scored & ~planted & ~in_window                       # everything outside the planted burst: the false-alarm baseline
    flag = scores >= cutoff
    out = {"planted_query_packets_scored": int(target.sum()),
           "planted_flagged": int((flag & target).sum()),
           "planted_recall": float(flag[target].mean()) if target.any() else None,
           "other_packets_scored": int(others.sum()),
           "other_flagged_rate": float(flag[others].mean()) if others.any() else None,
           "packets_inside_window_not_planted": int((scored & ~planted & in_window).sum()),
           "flagged_rate_inside_window_not_planted": float(flag[scored & ~planted & in_window].mean()) if (scored & ~planted & in_window).any() else None}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=60.0)
    ap.add_argument("--plant-at", type=float, default=50.0, help="minute at which the planted burst is sent")
    ap.add_argument("--fm", type=int, default=2000)
    ap.add_argument("--ad", type=int, default=20000)
    ap.add_argument("--calibration", type=int, default=5000)
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--out", default=str(REPORTS / "pipeline" / "live_kitsune_test.json"))
    a = ap.parse_args(argv)
    seconds = a.minutes * 60.0
    if seconds > MAX_SECONDS or a.plant_at * 60.0 >= seconds:
        raise SystemExit(f"--minutes must be at most {MAX_SECONDS // 60} and --plant-at must be earlier than the end")
    path = LIVE_DIR / time.strftime("kitsune_live_%Y%m%d_%H%M%S.pcap")
    result: dict = {}
    started = threading.Event()
    t = threading.Thread(target=lambda: result.update(capture_to_pcap(path, seconds, 3_000_000, on_started=started.set)), daemon=True)
    print(f"[{time.strftime('%H:%M:%S')}] capturing for {a.minutes:g} min; click Yes on the Windows approval prompt", flush=True)
    t.start()
    if not started.wait(timeout=180):
        print("the capture did not start within 3 minutes (approval refused?). Stopping.", flush=True)
        return 1
    t_start = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] capture started; the planted burst will be sent at minute {a.plant_at:g}", flush=True)
    while time.time() - t_start < a.plant_at * 60.0:
        time.sleep(30)
        if int((time.time() - t_start)) % 300 < 30:
            size = Path(path).stat().st_size if Path(path).exists() else 0
            print(f"[{time.strftime('%H:%M:%S')}] minute {(time.time() - t_start) / 60:.0f}: capture file {size / 1e6:.1f} MB", flush=True)
    window = send_burst()
    print(f"[{time.strftime('%H:%M:%S')}] planted burst sent (400 lookups, {window[1] - window[0]:.0f} s)", flush=True)
    t.join(timeout=seconds + 120)
    if not result:
        print("no capture result: the capture failed", flush=True)
        return 1
    print(f"[{time.strftime('%H:%M:%S')}] captured {result['packets']:,} packets ({result['bytes'] / 1e6:.1f} MB)", flush=True)

    report = {"settings": vars(a), "capture": {k: result[k] for k in ("packets", "seconds", "bytes")}, "planted_window_epoch": window}
    needed = a.fm + a.ad + a.calibration + 2000
    try:
        if result["packets"] < needed:
            report["verdict"] = f"NOT ENOUGH PACKETS: {result['packets']:,} captured, {needed:,} needed with the reduced settings; Kitsune could not finish learning"
            print(report["verdict"], flush=True)
            a.keep = True
            print(f"the capture is kept so it can be extended or analysed later: {path}", flush=True)
        else:
            packets = list(iter_packets(path))
            data = score_packets(iter(packets), a.fm, a.ad, progress_every=20000)
            cutoff, first = calibrate_cutoff(data["scores"], a.fm, a.ad, a.calibration, a.fpr)
            res = analyse(packets, data["scores"], cutoff, first, window)
            report["kitsune"] = {"cutoff": cutoff, "scoring_from_packet": first, **res}
            print(f"Kitsune (reduced learning): cut-off {cutoff:.4f}; planted DNS query packets scored: {res['planted_query_packets_scored']}, "
                  f"flagged: {res['planted_flagged']} (recall {res['planted_recall']})", flush=True)
            print(f"   all other packets outside the burst: flagged {res['other_flagged_rate']:.4f} of {res['other_packets_scored']:,} (target {a.fpr})", flush=True)
            if res["flagged_rate_inside_window_not_planted"] is not None:
                print(f"   other packets inside the burst window: flagged {res['flagged_rate_inside_window_not_planted']:.3f} of {res['packets_inside_window_not_planted']:,}", flush=True)
            events, bad = parse_log(run_sensor_wsl(path))
            report["maltrail"] = {"events": len(events), "unparsed": bad, "types": sorted({e["info"] for e in events})[:8]}
            print(f"Maltrail sensor on the capture: {len(events)} events; types: {report['maltrail']['types']}", flush=True)
            report["verdict"] = ("measured on planted queries; reduced learning windows; one run" if res["planted_recall"] is not None else "no planted query packets were scored")
    finally:
        if not a.keep:
            Path(path).unlink(missing_ok=True)
            print("capture file deleted", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1, default=float))
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
