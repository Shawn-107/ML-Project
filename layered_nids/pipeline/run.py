"""Combined pipeline on a pcap: our packet detector + our rules -> per-host evidence -> fused alerts.

  python -m layered_nids.pipeline.run --pcap data/kitsune/mirai.pcap --rules

Packet scores are cached. Output: reports/pipeline/<name>.json and <name>.alerts.jsonl. The capture is UNLABELLED, so the report
is DESCRIPTIVE: it shows what each layer and the fusion say and when, not detection rates.

Alert policy (per layer, because one threshold for every layer silences the weak ones):
  - rule SIGNATURE hits (patterns, blocklist matches) are precise, so each one alerts on its own;
  - packet anomalies and rule HEURISTICS accumulate per host, and alert when the accumulated evidence passes a threshold
    that is tuned on the first half of the assumed-normal stretch and checked on the second half;
  - evidence uses a burst-aware null fitted on normal windows (default) or the plain binomial null (--null binomial).
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from layered_nids.common.paths import REPORTS
from layered_nids.packets.pcap import iter_packets
from layered_nids.common.schema import Evidence
from layered_nids.fusion.accumulator import Alert, EvidenceAccumulator, FusionConfig
from layered_nids.pipeline.packet_layer import DEFAULT_TRAIN, calibrate_cutoff, fit_null, packet_evidence, score_pcap
from layered_nids.pipeline.rules_layer import is_signature, rules_evidence

ALERT_P_GRID = [1 - 10 ** -k for k in (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)]       # least to most strict


def find_jump(scores, cutoff: float, start: int, window: int = 2000):
    """First index after `start` where more than half of the next `window` packets are at or above the cut-off, else None.
    Descriptive only: it marks where the traffic first looks unlike what the detector learned, not a labelled attack start."""
    flags = (np.asarray(scores[start:]) >= cutoff).astype(float)
    if len(flags) < window:
        return None
    roll = np.convolve(flags, np.ones(window) / window, mode="valid")
    hit = np.flatnonzero(roll > 0.5)
    return int(hit[0]) + start if len(hit) else None


def fuse(evidence, alert_p: float, min_layers: int = 1, half_life_s: float = 300.0, cooldown_s: float = 300.0) -> list[Alert]:
    cfg = FusionConfig(alert_p=alert_p, min_layers=min_layers, half_life_s=half_life_s, cooldown_s=cooldown_s)
    return EvidenceAccumulator(cfg).run(evidence)


def fuse_policy(ev_packets, ev_rules, alert_p: float, cooldown_s: float = 300.0) -> list[Alert]:
    """Per-layer policy: signature hits alert directly (once per host and trail per cooldown); packet anomalies and rule
    heuristics accumulate together per host and alert at alert_p. Heuristics can corroborate an anomaly but cannot alert alone
    once the threshold is strict."""
    direct, last = [], {}
    for e in sorted((e for e in ev_rules if is_signature(e)), key=lambda e: e.ts):
        key = (e.host, e.detail.get("trail"))
        if e.ts - last.get(key, -1e300) >= cooldown_s:
            last[key] = e.ts
            direct.append(Alert(e.host, e.ts, e.score, "high", {"rules": 1.0}, (e.reason,)))
    soft = list(ev_packets) + [e for e in ev_rules if not is_signature(e)]
    return sorted(direct + fuse(soft, alert_p), key=lambda a: a.ts)


def alert_curve(evidence, bounds: dict, grid=ALERT_P_GRID, min_layers: int = 1) -> list[dict]:
    """For each alert threshold, the alerts (and distinct hosts) that fall in each named time segment.
    `bounds` maps a segment name to (start_ts, end_ts); a segment includes start and excludes end."""
    rows = []
    for p in grid:
        alerts = fuse(evidence, p, min_layers)
        row = {"alert_p": p}
        for name, (lo, hi) in bounds.items():
            seg = [x for x in alerts if lo <= x.ts < hi]
            row[name] = {"alerts": len(seg), "hosts": len({x.host for x in seg})}
        rows.append(row)
    return rows


def choose_alert_p(rows: list[dict], tune_segment: str, budget_alerts: int = 0) -> float:
    """The least strict threshold whose alerts in `tune_segment` do not exceed the budget; the strictest one if none does."""
    for r in rows:
        if r[tune_segment]["alerts"] <= budget_alerts:
            return r["alert_p"]
    return rows[-1]["alert_p"]


def segment_facts(data: dict, first: int, lo: float, hi: float) -> dict:
    """Duration and distinct internal hosts in a time segment (packets that have a host), for per-host-hour rates."""
    ts, hosts = data["ts"][first:], data["hosts"][first:]
    m = (ts >= lo) & (ts < hi) & (hosts != "")
    n_hosts = len(set(hosts[m]))
    minutes = (hi - lo) / 60.0
    return {"minutes": round(minutes, 1), "hosts_seen": n_hosts, "host_hours": round(n_hosts * minutes / 60.0, 2)}


def tune(data, cutoff, first, fpr, window_s, t_first, t_mid, jump_ts, method, swapped, budget):
    """One tuning run. method 'binomial' or 'burst'; swapped=True tunes on the SECOND normal half and checks on the first."""
    tune_seg, held_seg = ((t_mid, jump_ts), (t_first, t_mid)) if swapped else ((t_first, t_mid), (t_mid, jump_ts))
    if method == "burst":
        mu, rho, nwin = fit_null(data, cutoff, first, *tune_seg, window_s)
        ev = packet_evidence(data, cutoff, first, mu, window_s, rho=rho)
        null = {"mu": round(mu, 5), "rho": round(rho, 4), "windows_used": nwin}
    else:
        ev = packet_evidence(data, cutoff, first, fpr, window_s)
        null = {"mu": fpr, "rho": 0.0}
    bounds = {"tuning_half": tune_seg, "heldout_normal_half": held_seg, "after_jump": (jump_ts, float("inf"))}
    rows = alert_curve(ev, bounds)
    p = choose_alert_p(rows, "tuning_half", budget)
    row = next(r for r in rows if r["alert_p"] == p)
    return {"method": method, "swapped": swapped, "null": null, "chosen_alert_p": p, "at_chosen": row, "curve": rows, "evidence": ev}


def describe(alerts: list[Alert], t0: float, jump_ts: float | None) -> dict:
    by_host = Counter(a.host for a in alerts)
    first = {}
    for a in sorted(alerts, key=lambda a: a.ts):
        first.setdefault(a.host, round((a.ts - t0) / 60.0, 1))
    out = {"alerts": len(alerts), "hosts_alerted": len(by_host), "alerts_per_host_top5": dict(by_host.most_common(5)),
           "first_alert_minute_per_host": dict(sorted(first.items(), key=lambda kv: kv[1])[:8]),
           "severity": dict(Counter(a.severity for a in alerts)),
           "layers_in_alerts": dict(Counter(l for a in alerts for l in a.layers))}
    if jump_ts is not None:
        out["alerts_before_jump"] = sum(1 for a in alerts if a.ts < jump_ts)
        out["alerts_after_jump"] = sum(1 for a in alerts if a.ts >= jump_ts)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", required=True)
    ap.add_argument("--name", default=None, help="report name (default: pcap file name)")
    ap.add_argument("--rules", action="store_true", help="also run the rules layer (signatures and heuristics) on the packets")
    ap.add_argument("--cache", default=None, help="packet scores cache (.npz); default reports/pipeline/<name>_packet_scores.npz")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--train", type=int, default=DEFAULT_TRAIN, help="packets the detector learns from (must be clean traffic)")
    ap.add_argument("--calibration", type=int, default=15000, help="packets after training used to set the cut-off (assumed normal)")
    ap.add_argument("--fpr", type=float, default=0.01, help="target per-packet false-alarm rate for the cut-off (and the binomial null)")
    ap.add_argument("--window", type=float, default=60.0, help="evidence window in seconds")
    ap.add_argument("--null", choices=["burst", "binomial"], default="burst", help="evidence null for the final alerts")
    ap.add_argument("--alert-p", type=float, default=None, help="fix the alert threshold instead of tuning it")
    ap.add_argument("--budget", type=int, default=0, help="false alerts allowed on the tuning half (default 0)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    name = a.name or Path(a.pcap).stem
    cache = a.cache or str(REPORTS / "pipeline" / f"{name}_packet_scores.npz")
    out_path = Path(a.out or REPORTS / "pipeline" / f"{name}.json")

    data = score_pcap(a.pcap, cache, a.train, a.limit)
    n = len(data["scores"])
    cutoff, first = calibrate_cutoff(data["scores"], a.train, a.calibration, a.fpr)
    t0 = float(data["ts"][0])
    jump = find_jump(data["scores"], cutoff, first)
    jump_ts = float(data["ts"][jump]) if jump is not None else None
    print(f"packets {n:,} | trained on {a.train:,} | calibrated on next {first - a.train:,} -> cut-off {cutoff:.4f}"
          f" | scoring from packet {first:,}", flush=True)
    if jump is not None:
        print(f"packet scores jump at packet {jump:,} = minute {(jump_ts - t0) / 60:.1f} of the capture", flush=True)

    ev_m = []
    if a.rules:
        ev_m = rules_evidence(iter_packets(a.pcap, a.limit))
        print(f"rules: {len(ev_m)} evidence items for internal hosts "
              f"(signature hits {sum(is_signature(e) for e in ev_m)}, heuristics {sum(not is_signature(e) for e in ev_m)})", flush=True)

    tuning, alert_p, ev_k = None, a.alert_p, None
    if jump_ts is not None:
        t_first = float(data["ts"][first])
        t_mid = (t_first + jump_ts) / 2.0
        runs = {}
        for method in ("binomial", "burst"):
            for swapped in (False, True):
                runs[(method, swapped)] = tune(data, cutoff, first, a.fpr, a.window, t_first, t_mid, jump_ts, method, swapped, a.budget)
        main_run = runs[(a.null if a.null == "burst" else "binomial", False)]
        ev_k = main_run["evidence"]
        if alert_p is None:
            alert_p = main_run["chosen_alert_p"]
        tuning = {"assumption": "everything before the jump is normal (NOT a label)", "budget_alerts_on_tuning_half": a.budget,
                  "segments": {"first_half": segment_facts(data, first, t_first, t_mid), "second_half": segment_facts(data, first, t_mid, jump_ts),
                               "after_jump": segment_facts(data, first, jump_ts, float(data["ts"][-1]) + 1)},
                  "runs": {f"{m}{'_swapped' if s else ''}": {k: v for k, v in r.items() if k != "evidence"} for (m, s), r in runs.items()}}
        print("\nnull        order     fitted null (mu, rho)   chosen alert_p      held-out normal half / after-jump alerts (hosts)", flush=True)
        for (m, s), r in runs.items():
            at = r["at_chosen"]
            nl = r["null"]
            print(f"{m:10s}  {'swapped' if s else 'normal ':8s}  mu {nl['mu']:<8} rho {nl['rho']:<7}  {r['chosen_alert_p']:.12f}   "
                  f"{at['heldout_normal_half']['alerts']:3d} false / {at['after_jump']['alerts']:3d} ({at['after_jump']['hosts']} hosts)", flush=True)
    else:
        print("no jump found: no assumed-normal stretch to tune on; using alert_p 0.99 and the binomial null", flush=True)
        alert_p = alert_p or 0.99
        ev_k = packet_evidence(data, cutoff, first, a.fpr, a.window)
    print(f"\nusing alert_p {alert_p:.12f} with the {a.null if jump_ts is not None else 'binomial'} null", flush=True)

    setups = {"packets_only": fuse(ev_k, alert_p)}
    if a.rules:
        setups["rules_signature_hits_direct"] = fuse_policy([], [e for e in ev_m if is_signature(e)], alert_p)
        setups["rules_heuristics_alone"] = fuse([e for e in ev_m if not is_signature(e)], alert_p)
        setups["policy_all_layers"] = fuse_policy(ev_k, ev_m, alert_p)
        setups["need_packets_and_rules"] = fuse(ev_k + ev_m, alert_p, min_layers=2)
    report = {"pcap": str(a.pcap), "packets": n, "settings": {k: v for k, v in vars(a).items() if k != "pcap"},
              "alert_p_used": alert_p, "tuning": tuning,
              "packet_detector": {"cutoff": cutoff, "scoring_from_packet": first, "jump_packet": jump,
                          "jump_minute": None if jump is None else round((jump_ts - t0) / 60, 1), "evidence_windows": len(ev_k),
                          "cache_used": bool(data.get("cached"))},
              "rules": {"evidence_items": len(ev_m),
                           "signature_hits": sum(is_signature(e) for e in ev_m),
                           "first_event_minutes": sorted({round((e.ts - t0) / 60, 1) for e in ev_m})[:10]},
              "setups": {}, "note": "UNLABELLED capture: descriptive only. 'before/after jump' splits at the first point where the packet detector's "
                                    "scores stay above its cut-off; it is not a labelled attack start. No recall or false-alarm rate is measured."}
    print("", flush=True)
    chosen_setup = "policy_all_layers" if "policy_all_layers" in setups else "packets_only"
    alerts_path = out_path.with_suffix(".alerts.jsonl")
    alerts_path.parent.mkdir(parents=True, exist_ok=True)
    with alerts_path.open("w", encoding="utf-8") as f:                 # one alert per line; read by pipeline/to_dashboard.py
        for al in setups[chosen_setup]:
            f.write(json.dumps({"host": al.host, "ts": al.ts, "probability": al.probability, "severity": al.severity,
                                "layers": al.layers, "reasons": list(al.reasons), "setup": chosen_setup}, default=float) + "\n")
    report["alerts_file"] = str(alerts_path)
    for sname, alerts in setups.items():
        d = describe(alerts, t0, jump_ts)
        report["setups"][sname] = d
        print(f"{sname:32s} alerts {d['alerts']:4d} | hosts {d['hosts_alerted']:2d}"
              + (f" | before jump {d['alerts_before_jump']:3d} after {d['alerts_after_jump']:3d}" if "alerts_before_jump" in d else ""),
              flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=1, default=float))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
