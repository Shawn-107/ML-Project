"""Combined-pipeline tests on generated data (no capture, no WSL needed)."""
import json

import numpy as np
import pytest

from layered_nids.common.schema import Evidence
from layered_nids.packets.pcap import Packet
from layered_nids.pipeline.kitsune_layer import calibrate_cutoff, internal_host, kitsune_evidence, score_packets
from layered_nids.pipeline.maltrail_layer import HEURISTIC_SCORE, TRAIL_SCORE, events_to_evidence, parse_log
from layered_nids.pipeline.run import describe, find_jump, fuse

HEUR = '"2018-10-25 05:48:11.704154" mtlab 192.168.2.112 59290 61.188.37.216 8000 TCP IPORT 61.188.37.216:8000 "potential periodic beaconing (suspicious)" (heuristic)'
TRAIL = '"2018-10-25 05:50:00.000000" mtlab 10.0.0.66 40000 66.66.66.66 80 TCP IP 66.66.66.66 "malware (test)" (static)'


def pkt(src="10.0.0.1", dst="10.0.0.2", proto="tcp", sport=1234, dport=80, ts=1.0, size=100):
    return Packet(ts, size, "aa:aa:aa:aa:aa:01", "aa:aa:aa:aa:aa:02", src, dst, proto, sport, dport)


def test_internal_host_picks_the_real_internal_end_or_nothing():
    assert internal_host(pkt("192.168.2.5", "8.8.8.8")) == "192.168.2.5"
    assert internal_host(pkt("8.8.8.8", "192.168.2.5")) == "192.168.2.5"
    assert internal_host(pkt("10.0.0.1", "10.0.0.2")) == "10.0.0.1"
    assert internal_host(pkt("192.168.2.5", "224.0.0.251")) == "192.168.2.5"                 # multicast is not a host
    assert internal_host(pkt("0.0.0.0", "255.255.255.255")) == ""                            # DHCP discover: no host
    assert internal_host(pkt("0.0.0.0", "192.168.2.1")) == "192.168.2.1"                     # the other end is a real host
    assert internal_host(pkt("8.8.8.8", "1.1.1.1")) == ""                                    # purely external traffic
    assert internal_host(pkt("aa:bb:cc:dd:ee:01", "aa:bb:cc:dd:ee:02", proto="eth")) == ""   # link-layer only: no host
    assert internal_host(pkt("fe80::1", "ff02::1", proto="udp")) == "fe80::1"                # IPv6 link-local unicast is a host


def test_hosts_without_an_internal_end_take_no_part_in_evidence():
    from layered_nids.pipeline.kitsune_layer import kitsune_evidence
    n = 200
    data = {"scores": np.r_[np.zeros(50), np.ones(150) * 9.0], "ts": np.arange(n) * 0.1,
            "hosts": np.array(["192.168.2.5"] * 100 + [""] * 100)}
    ev = kitsune_evidence(data, cutoff=1.0, first_index=50, fpr=0.01, window_s=60.0)
    assert {e.host for e in ev} == {"192.168.2.5"} and ev[0].detail["events"] == 50           # the 100 host-less packets are not counted


def stream(n_normal=5000, n_flood=600, seed=0):
    """Stationary random normal traffic (random hosts, ports, sizes, exponential gaps), then a flood from one host.
    Regular arithmetic patterns are avoided on purpose: Kitsune's running counters drift on them and 'normal' scores creep up.
    The normal part lasts ~500 s because Kitsune's slowest windows are ~100 s long and need time to settle."""
    rng = np.random.default_rng(seed)
    out, t = [], 0.0
    for _ in range(n_normal):
        t += rng.exponential(0.1)
        out.append(pkt(f"10.0.0.{int(rng.integers(1, 6))}", "10.0.0.9", sport=int(rng.integers(40000, 40050)), dport=80,
                       ts=t, size=int(rng.integers(60, 120))))
    for _ in range(n_flood):
        t += rng.exponential(0.002)
        out.append(pkt("10.0.0.66", f"10.0.0.{int(rng.integers(1, 6))}", sport=5555, dport=int(rng.integers(1, 900)), ts=t, size=40))
    return out


def test_score_calibrate_and_evidence_flag_the_flood_host_only():
    data = score_packets(stream(), fm_grace=300, ad_grace=1500)
    s = data["scores"]
    assert (s[:1801] == 0).all() and (s[1801:] > 0).all()
    cutoff, first = calibrate_cutoff(s, 300, 1500, 1500, 0.01)
    assert first == 1801 + 1500
    assert (s[1801:first] >= cutoff).mean() <= 0.011                               # cut-off honours the target on its own window
    ev = kitsune_evidence(data, cutoff, first, 0.01, window_s=60.0, min_score=0.5)
    assert all(e.layer == "packets" for e in ev)
    ratio = lambda e: e.detail["flagged"] / e.detail["events"]
    flood = [e for e in ev if e.host == "10.0.0.66"]
    normal = [e for e in ev if e.host != "10.0.0.66"]
    assert flood and max(e.score for e in flood) >= 0.999 and min(ratio(e) for e in flood) >= 0.9   # every flood packet is flagged
    # Normal hosts exceed a 1% cut-off a bit more often than 1% (calibration window is small, exceedances come in bursts),
    # so they can show strong-looking windows, but only a minority of their packets are flagged:
    assert all(ratio(e) < 0.5 for e in normal)


def test_calibration_needs_enough_packets():
    with pytest.raises(ValueError):
        calibrate_cutoff(np.zeros(300), 100, 100, 50, 0.01)


def test_parse_log_reads_real_maltrail_lines_and_counts_bad_ones():
    events, bad = parse_log("\n".join([HEUR, "", "garbage line", TRAIL, '"bad time" a b c d e f g h "i" j']))
    assert len(events) == 2 and bad == 2
    assert events[0]["src"] == "192.168.2.112" and events[0]["ref"] == "(heuristic)" and events[0]["dport"] == "8000"
    assert events[0]["ts"] < events[1]["ts"]
    assert events[1]["info"] == "malware (test)" and events[1]["trail"] == "66.66.66.66"


def test_events_to_evidence_scores_hosts_and_reasons():
    ev = events_to_evidence(parse_log(HEUR + "\n" + TRAIL)[0])
    assert [e.score for e in ev] == [HEURISTIC_SCORE, TRAIL_SCORE] and all(e.layer == "rules" for e in ev)
    assert ev[0].host == "192.168.2.112" and "periodic beaconing" in ev[0].reason and ev[1].host == "10.0.0.66"
    inbound = parse_log('"2018-10-25 05:50:00.000000" mtlab 8.8.4.4 53 192.168.2.7 4444 UDP IP 8.8.4.4 "x (test)" (static)')[0]
    assert events_to_evidence(inbound)[0].host == "192.168.2.7"                     # the private end is the host
    external = parse_log('"2018-10-25 05:50:00.000000" mtlab 8.8.4.4 53 1.1.1.1 4444 UDP IP 8.8.4.4 "x (test)" (static)')[0]
    assert events_to_evidence(external) == []                                        # no internal end: skipped, like the Kitsune layer


def test_find_jump_marks_where_scores_stay_high():
    s = np.r_[np.zeros(5000) + 0.01, np.zeros(6000) + 5.0]
    assert find_jump(s, 0.5, 1000, window=500) in range(4500, 5200)
    assert find_jump(np.zeros(4000), 0.5, 100, window=500) is None


def test_alert_threshold_tuning_ignores_one_off_bursts_and_keeps_persistent_attacks():
    from layered_nids.pipeline.run import alert_curve, choose_alert_p
    ev = [Evidence("packets", "a", 50.0, 0.999, "one-off burst")]                            # a single strong window in the normal stretch
    ev += [Evidence("packets", "b", 200.0 + i, 0.999, "persistent") for i in range(0, 120, 10)]   # many windows in a row after the jump
    bounds = {"tune": (0.0, 100.0), "after": (150.0, 1e9)}
    rows = alert_curve(ev, bounds)
    tune = [r["tune"]["alerts"] for r in rows]
    after = [r["after"]["alerts"] for r in rows]
    assert tune == sorted(tune, reverse=True) and after == sorted(after, reverse=True)       # stricter threshold never adds alerts
    p = choose_alert_p(rows, "tune", budget_alerts=0)
    row = next(r for r in rows if r["alert_p"] == p)
    assert row["tune"]["alerts"] == 0 and row["after"]["alerts"] >= 1                       # burst silenced, persistent host still alerts
    assert p > rows[0]["alert_p"]                                                           # it had to be stricter than the loosest setting
    assert choose_alert_p(rows, "tune", budget_alerts=10) == rows[0]["alert_p"]            # a generous budget keeps the loosest setting


def test_fusion_needs_both_layers_when_asked_and_describe_splits_at_the_jump():
    k = [Evidence("packets", "h1", 100.0 + i, 0.97, "odd packets") for i in range(3)]
    m = [Evidence("rules", "h1", 105.0, 0.95, "Maltrail: x"), Evidence("rules", "h2", 105.0, 0.95, "Maltrail: x")]
    both = fuse(k + m, alert_p=0.99, min_layers=2)
    assert {a.host for a in both} == {"h1"} and set(both[0].layers) == {"packets", "rules"}
    assert not fuse(m, alert_p=0.99, min_layers=2)                                   # one layer alone cannot satisfy it
    d = describe(fuse(k + m, 0.99, 1), t0=100.0, jump_ts=102.0)
    assert d["alerts"] == d["alerts_before_jump"] + d["alerts_after_jump"] and d["hosts_alerted"] >= 1


def test_signature_hits_alert_alone_but_heuristics_only_corroborate():
    from layered_nids.pipeline.run import fuse_policy, is_signature
    ev_rules = events_to_evidence(parse_log(HEUR + "\n" + TRAIL)[0])
    assert [is_signature(e) for e in ev_rules] == [False, True]                      # heuristic vs blocklist hit
    strict = 1 - 1e-12
    alerts = fuse_policy([], ev_rules, strict)
    assert [a.host for a in alerts] == ["10.0.0.66"] and alerts[0].layers == {"rules": 1.0}   # the signature alerts alone
    assert not fuse_policy([], [e for e in ev_rules if not is_signature(e)], strict)         # a lone heuristic cannot reach a strict threshold
    k = [Evidence("packets", "192.168.2.112", 1540446500.0 + i, 0.999, "odd packets") for i in range(6)]
    joint = fuse_policy(k, [e for e in ev_rules if not is_signature(e)], 0.99)
    assert any(a.host == "192.168.2.112" for a in joint)                              # anomalies still alert; heuristics add to the same host


def test_cooldown_stops_the_same_signature_alerting_every_second():
    from layered_nids.pipeline.run import fuse_policy
    lines = "\n".join(f'"2018-10-25 05:50:{s:02d}.000000" mtlab 10.0.0.66 40000 66.66.66.66 80 TCP IP 66.66.66.66 "malware (test)" (static)' for s in range(10))
    ev = events_to_evidence(parse_log(lines)[0])
    assert len(fuse_policy([], ev, 0.99, cooldown_s=300.0)) == 1                      # ten identical hits in 10 s -> one alert
    assert len(fuse_policy([], ev, 0.99, cooldown_s=1.0)) == 10                       # hits are 1 s apart: a 1 s cooldown lets all through


def test_live_capture_adapter_limits_paths_and_writes_a_readable_pcap(tmp_path, monkeypatch):
    from scapy.all import IP, TCP, Ether
    from layered_nids.packets.pcap import iter_packets
    from layered_nids.pipeline import live
    monkeypatch.setattr(live, "LIVE_DIR", tmp_path / "live")
    with pytest.raises(ValueError):
        live.capture_to_pcap(tmp_path / "elsewhere.pcap", 5, 10)                    # outside data/live: refused
    with pytest.raises(ValueError):
        live.capture_to_pcap(tmp_path / "live" / "a.pcap", live.MAX_SECONDS + 1, 10)   # too long: refused
    with pytest.raises(ValueError):
        live.capture_to_pcap(tmp_path / "live" / "a.pcap", 5, live.MAX_PACKETS + 1)    # too many packets: refused

    def fake_sniffer(timeout, count, iface, on_packet, on_started):                    # no Npcap, no real traffic
        on_started()
        for i in range(25):
            on_packet(Ether(src="02:00:00:00:00:01", dst="02:00:00:00:00:02") / IP(src="192.168.2.5", dst="8.8.8.8") / TCP(sport=40000 + i, dport=443))

    started = []
    info = live.capture_to_pcap(tmp_path / "live" / "a.pcap", 5, 100, sniffer=fake_sniffer, on_started=lambda: started.append(1))
    assert started == [1]                                                              # the started signal fires
    assert info["packets"] == 25 and info["bytes"] > 0
    pk = list(iter_packets(info["path"]))
    assert len(pk) == 25 and pk[0].src == "192.168.2.5" and pk[0].dport == 443 and pk[0].proto == "tcp"


def test_live_kitsune_test_labels_only_the_planted_queries_and_measures_recall():
    from layered_nids.pipeline.live_kitsune_test import analyse, is_planted_query, planted_name
    name = planted_name()
    assert name.startswith("kt") and name.endswith(".example.com") and len(name) == 2 + 10 + len(".example.com")
    mk = lambda q, ts: Packet(ts, 80, "a", "b", "192.168.2.5", "8.8.8.8", "udp", 5353, 53, q)
    assert is_planted_query(mk(name, 1.0)) and not is_planted_query(mk("google.com", 1.0)) and not is_planted_query(mk("kt12.example.com", 1.0))
    packets = [mk("", float(i)) for i in range(100)] + [mk(planted_name(), 100.0 + i * 0.05) for i in range(20)] + [mk("", 200.0 + i) for i in range(50)]
    scores = np.r_[np.zeros(50), np.ones(50) * 0.02, np.ones(20) * 5.0, np.ones(50) * 0.02]      # only the planted packets score high
    res = analyse(packets, scores, cutoff=0.5, first=50, window=(100.0, 101.0))
    assert res["planted_query_packets_scored"] == 20 and res["planted_recall"] == 1.0
    assert res["other_packets_scored"] == 99 and res["other_flagged_rate"] == 0.0               # 100 minus the packet at t=99 inside the window margin; none flagged


def test_alerts_written_in_maltrail_format_read_back_through_our_own_maltrail_parser(tmp_path):
    from layered_nids.pipeline.to_maltrail import alert_to_line, write_logs
    a = {"host": "192.168.2.110", "ts": 1540450844.581, "probability": 0.9999999999, "severity": "high",
         "layers": {"packets": 0.9, "rules": 0.1}, "reasons": ['14 of 84 packets flagged in 60s "burst-aware"'], "setup": "x"}
    events, bad = parse_log(alert_to_line(a))
    assert bad == 0 and len(events) == 1
    e = events[0]
    assert e["src"] == "192.168.2.110" and e["ref"] == "(nids)" and e["sensor"] == "nids" and "high ML alert (packets+rules)" in e["info"]
    ev = events_to_evidence(events)[0]
    assert ev.host == "192.168.2.110" and ev.layer == "rules"                          # a reader of Maltrail logs sees it as a normal event
    src = tmp_path / "alerts.jsonl"
    src.write_text(json.dumps(a) + "\n" + json.dumps(dict(a, ts=a["ts"] + 86400 * 2)) + "\n")
    written = write_logs(src, tmp_path / "logs")
    assert sorted(written) == ["2018-10-25.log", "2018-10-27.log"] and all(v == 1 for v in written.values())
    assert (tmp_path / "logs" / "2018-10-25.log").read_text().count("\n") == 1
