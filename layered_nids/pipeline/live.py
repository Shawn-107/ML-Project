"""Live capture adapter: record your OWN traffic for a limited time into a local pcap, then analyse it with the same code
path as a replayed file ("capture, then analyse"; the Kitsune wrapper is packet-by-packet, so true streaming can be added
later without changing the layers).

  python -m layered_nids.pipeline.live --seconds 60 --max-packets 200000          # needs Npcap; Windows asks for approval (UAC)

Safety rules built in:
  - hard limits: at most MAX_SECONDS and MAX_PACKETS per run, whatever you pass;
  - output only inside the project's data/live folder (nothing is uploaded; the file is yours to delete);
  - capture only starts when you run this command; it never runs in the background;
  - Npcap is installed admin-only: Windows shows an approval prompt; click Yes only for a capture you started.
Do not capture on shared or public networks: other people's traffic is not yours to record.

Limits: Kitsune needs tens of thousands of packets of clean traffic to learn before it can score (about 55,000 at its defaults),
so a short capture shows the capture path working but cannot produce Kitsune alerts. Maltrail runs on the saved file in WSL2.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from layered_nids.common.paths import DATA

LIVE_DIR = DATA / "live"
MAX_SECONDS = 3600
MAX_PACKETS = 2_000_000


def safe_output_path(path) -> Path:
    """The capture file must live inside data/live. Raises ValueError otherwise."""
    p = Path(path).resolve()
    root = LIVE_DIR.resolve()
    if root != p and root not in p.parents:
        raise ValueError(f"captures may only be written inside {root}")
    return p


def capture_to_pcap(path, seconds: float, max_packets: int, iface: str | None = None, sniffer=None, on_started=None) -> dict:
    """Capture up to `seconds` seconds or `max_packets` packets into a classic pcap at `path` (inside data/live).
    `sniffer(timeout, count, iface, on_packet, on_started)` is injectable for tests; the default is scapy's sniff (needs Npcap).
    `on_started()` is called once packets are actually being captured (after any approval prompt)."""
    if not (0 < seconds <= MAX_SECONDS):
        raise ValueError(f"seconds must be in (0, {MAX_SECONDS}]")
    if not (0 < max_packets <= MAX_PACKETS):
        raise ValueError(f"max_packets must be in (0, {MAX_PACKETS}]")
    out = safe_output_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    from scapy.utils import PcapWriter
    writer = PcapWriter(str(out), sync=False)
    seen = {"n": 0}

    def on_packet(pkt):
        writer.write(pkt)
        seen["n"] += 1

    if sniffer is None:
        def sniffer(timeout, count, iface, on_packet, on_started):
            from scapy.all import sniff
            sniff(timeout=timeout, count=count, iface=iface, store=False, prn=on_packet, started_callback=on_started)

    t0 = time.time()
    try:
        sniffer(seconds, max_packets, iface, on_packet, on_started or (lambda: None))
    finally:
        writer.close()
    return {"path": str(out), "packets": seen["n"], "seconds": round(time.time() - t0, 1), "bytes": out.stat().st_size}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Capture your own traffic for a limited time, then (optionally) analyse it.")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--max-packets", type=int, default=100_000)
    ap.add_argument("--iface", default=None, help="interface name (default: scapy's default)")
    ap.add_argument("--out", default=None, help="file name inside data/live (default: capture_<time>.pcap)")
    ap.add_argument("--analyse", action="store_true", help="run the pipeline on the capture afterwards (needs enough packets)")
    a = ap.parse_args(argv)
    out = a.out or time.strftime("capture_%Y%m%d_%H%M%S.pcap")
    path = LIVE_DIR / Path(out).name
    print(f"capturing up to {a.seconds:g} s / {a.max_packets:,} packets into {path}", flush=True)
    print("Windows will ask for approval: click Yes only if you started this command.", flush=True)
    info = capture_to_pcap(path, a.seconds, a.max_packets, a.iface)
    print(f"captured {info['packets']:,} packets in {info['seconds']} s ({info['bytes'] / 1e6:.2f} MB)", flush=True)
    print(f"the file stays on this PC: delete it when you are done: {info['path']}", flush=True)
    if a.analyse:
        need = 5000 + 50000 + 15000
        if info["packets"] < need:
            print(f"not enough packets to analyse with Kitsune's defaults: need about {need:,}, got {info['packets']:,}. "
                  f"Capture longer, or analyse a replayed file instead.", flush=True)
        else:
            from layered_nids.pipeline.run import main as run_main
            run_main(["--pcap", info["path"], "--maltrail", "--name", Path(out).stem])


if __name__ == "__main__":
    main()
