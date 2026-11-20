"""L1 baseline on the Kitsune Mirai capture.   python -m layered_nids.rules.run [--fpr 0.01]

Same clean period and time split as L3: thresholds are calibrated on the first 80% of the paper's 52-minute clean
period; the last 20% is held out to measure false alarms. Unit = (source host, 60 s window) that sent at least one packet.
No blocklist is used here (none is bundled). Detection numbers on minute >= 80 are PROXIES, not ground truth.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from layered_nids.common.paths import DATA, REPORTS
from layered_nids.packets.pcap import iter_packets
from layered_nids.rules.engine import calibrate_rules, run_rules

PCAP = DATA / "kitsune" / "mirai_example" / "mirai.pcap"
TRAIN_MIN, PROXY_MIN = 52.0, 80.0


def window_flags(evidence, window_s):
    return {(e.host, int(e.ts // window_s)): e for e in evidence}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fpr", type=float, default=0.01)
    ap.add_argument("--pcap", default=str(PCAP))
    ap.add_argument("--out", default=str(REPORTS / "rules" / "results.json"))
    a = ap.parse_args(argv)

    pk = list(iter_packets(a.pcap))
    ts = np.array([p.ts for p in pk])
    rel = (ts - ts.min()) / 60.0
    n52 = int(np.searchsorted(rel, TRAIN_MIN))
    cut = int(n52 * 0.8)
    cfg = calibrate_rules(pk[:cut], fpr=a.fpr)
    ev = run_rules(pk, cfg)
    flagged = window_flags(ev, cfg.window_s)
    t0 = ts.min()
    wid = lambda p: (p.src, int((p.ts) // cfg.window_s))
    hold = {wid(p) for p in pk[cut:n52]}
    hold_fa = len(hold & flagged.keys()) / len(hold)
    # Attack-period description (no labels): who sends the traffic, and which hosts did the rules flag?
    att = [p for p, r in zip(pk, rel) if r >= PROXY_MIN]
    senders = Counter(p.src for p in att)
    ev_att = [e for e in ev if e.ts / 60 >= PROXY_MIN - 1]
    flagged_hosts = Counter(e.host for e in ev_att)
    top_host, top_n = senders.most_common(1)[0]
    report = {
        "config": cfg.__dict__, "target_fpr": a.fpr,
        "units": "(source host, 60 s window) with at least one packet",
        "held_out_windows": len(hold), "held_out_false_alarm_rate": hold_fa,
        "windows_flagged_before_minute_72": sum(1 for e in ev if e.ts / 60 < 72),
        "attack_period_description_min80plus": {
            "packets": len(att), "distinct_senders": len(senders),
            "top_sender": top_host, "top_sender_packet_share": top_n / len(att),
            "flagged_hosts_and_windows": dict(flagged_hosts),
            "share_of_packets_sent_by_flagged_hosts": sum(n for h, n in senders.items() if h in flagged_hosts) / len(att),
            "top_sender_flagged": top_host in flagged_hosts,
            "firings_by_rule": dict(Counter(e.detail["rule"] for e in ev_att)),
        },
        "note": "Minute >= 80 is chosen from the traffic profile, NOT ground-truth labels, and the host-level reading "
                "(which host is the scanner) is an inference from volume. No labelled recall is measured here. "
                "No blocklist data is bundled, so the blocklist rule was not exercised on this capture.",
    }
    print(json.dumps(report, indent=1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
