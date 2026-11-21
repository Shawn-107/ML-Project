# Combined pipeline on the Mirai capture: real reference detector + real reference sensor + fusion

Run: `python -m layered_nids.pipeline.run --pcap data/mirai/mirai.pcap --reference_sensor --name mirai` (raw output: `mirai.json`).
Capture: the `mirai.pcap`, 764,137 packets, 118.9 minutes, UNLABELLED. Everything below is DESCRIPTIVE. No recall, precision or
false-alarm rate is measured, because nothing says which packets are attacks. "Before/after the jump" splits at the first point where
reference detector's scores stay above its cut-off (packet 121,714 = minute 74.4); it is not a labelled attack start.

## What ran
- reference detector: the vendored real code (identical to the original on the first 8,000 packets, `tests/test_reference_detector_vendored.py`), learned on the
  first 55,000 packets, cut-off from the next 15,000 (99th percentile = 0.0805, assumed normal). Scoring starts at packet 70,001 = minute 42.6.
  The whole capture was scored in 1,281 s (scores cached).
- reference sensor: the real Linux sensor 3.6 in WSL2, offline on the same pcap: 8 events, 0 unparsed lines.
- Host attribution (fixed, shared by both layers, `common/hosts.py`): a packet or event belongs to its source if that is a real private unicast
  address, else its destination if that is, else to no host. MAC-only traffic, 0.0.0.0, broadcast, multicast and purely external traffic take no part.
  Effect: reference detector evidence windows 257 -> 171.
- Evidence: per (host, 60 s window) binomial rule; fusion: noisy-OR accumulation with decay; one alert per host per 5 minutes unless it escalates.
- Alert threshold tuning (new): the stretch before the jump (minutes 42.6-74.4, 31.8 min) is ASSUMED normal and split in two halves. The threshold
  is chosen on the first half (budget: 0 alerts) and checked on the second half, which it never saw. After the jump is only reported.

## Results
Alerts from reference detector evidence as the alert threshold gets stricter (tuning half: 15.9 min, 20 hosts, 5.3 host-hours; held-out half: 15.9 min,
19 hosts, 5.0 host-hours; after the jump: 44.6 min, 21 hosts):
- alert_p 0.99 (the old default): 9 / 6 / 38 alerts (tuning / held-out / after), after-jump alerts on 14 hosts.
- 0.999: 6 / 6 / 31 (11 hosts). 0.9999: 4 / 3 / 20 (6 hosts). 0.99999: 3 / 2 / 18 (5 hosts). 0.999999: 2 / 2 / 18 (5 hosts).
- 1 - 1e-7: 1 / 0 / 16 (4 hosts). 1 - 1e-9: 1 / 0 / 16 (4 hosts). 1 - 1e-10: 1 / 0 / 13 (3 hosts).
- 1 - 1e-12 (chosen, budget 0 on the tuning half): 0 / 0 / 11 (2 hosts).

Reading it:
- False alerts on the held-out normal half fall from 6 (about 1.2 per host-hour) at the old 0.99 setting to 0 at 1 - 1e-7 and stricter. That is the gain from tuning,
  measured on traffic the tuning did not use, under the assumption that this stretch is normal.
- The price is coverage: after the jump, alerts fall from 38 on 14 hosts to 16 on 4 hosts (1 - 1e-7) or 11 on 2 hosts (chosen). Without labels we cannot say whether the
  hosts that stop alerting were under attack.
- At the chosen threshold the two alerting hosts are 192.168.2.110 (8 alerts, first at minute 79.6, 5.2 minutes after the jump) and 192.168.2.115 (3 alerts, first at minute 106.6).
- reference sensor-only: 0 alerts now. Its 8 heuristic events (score 0.6 each) cannot reach a threshold this strict. Both layers required to agree: 0 alerts (its events are in
  minutes 1.8-8.5, before reference detector can score at all).

## Update: per-layer policy, burst-aware evidence, and a robustness check (all tested; see tests/test_pipeline.py, tests/test_windows.py)
- Per-layer alert policy (`fuse_policy`): a reference sensor SIGNATURE hit (blocklist match) alerts on its own; reference detector anomalies and reference sensor HEURISTICS accumulate per
  host and alert at the tuned threshold. Verified on generated data (a signature alerts alone at the strictest threshold; a lone heuristic does not; heuristics
  still add to a host that reference detector flags). On Mirai it changes nothing: the capture has 0 signature hits (all 8 reference sensor events are heuristics).
- Burst-aware evidence: the null is a beta-binomial whose intra-window correlation rho is fitted on the tuning half. It does find bursts (rho 0.39 on the first
  half, 0.21 on the second) but changes nothing measurable here. Outcome after the jump: 10 alerts on 2 hosts (burst-aware) vs 11 on 2 hosts (plain binomial);
  held-out normal half: 0 false alerts for both. Reason: the tuned alert threshold already absorbs the burstiness. Reported as a null result.
- Robustness to the "assumed normal" caveat: tuning on the SECOND half and checking on the FIRST half picks a looser threshold (1 - 1e-7): 1 false alert on the
  held-out half and 16-17 alerts on 4 hosts after the jump (both nulls). So the answer moves between "2 hosts" and "4 hosts" depending on which half tunes it.
  The conclusion is therefore soft: some hosts are clearly flagged, others depend on how strict we are, and only labels can settle it.

| null | tuned on | fitted (mu, rho) | chosen alert_p | false alerts on the other half | after the jump |
| --- | --- | --- | --- | --- | --- |
| binomial | first half | (0.01, 0) | 1 - 1e-12 | 0 | 11 alerts, 2 hosts |
| burst-aware | first half | (0.0043, 0.39) | 1 - 1e-12 | 0 | 10 alerts, 2 hosts |
| binomial | second half | (0.01, 0) | 1 - 1e-7 | 1 | 16 alerts, 4 hosts |
| burst-aware | second half | (0.0062, 0.21) | 1 - 1e-7 | 1 | 17 alerts, 4 hosts |

## What this means (and does not)
- reference detector does the work on this capture. reference sensor raised nothing after minute 8.5.
- A single alert threshold for all layers is wrong: it suppresses the reference sensor's weaker evidence entirely once tuned for reference detector. The next design step is per-layer handling
  (signature/blocklist hits can alert on their own; heuristics and anomalies accumulate), then re-check.
- The chosen threshold is the strictest value in the grid, so it is not converged; 1 - 1e-7 gives the same held-out result (0 false alerts) with more coverage. I kept the
  rule I declared before looking (budget 0 on the tuning half) and show the whole curve instead of picking a point by its held-out result.

## Limits
- Unlabelled single capture; no detection rate. The calibration window and the "normal before the jump" stretch are ASSUMED normal.
- The evidence rule assumes flags are independent; real exceedances come in bursts (shown with synthetic traffic in `tests/test_pipeline.py`), which is why a strict threshold is needed.
- A cut-off set on one stretch of traffic only holds if normal traffic stays statistically stable. reference detector's slowest windows are about 100 s long, so learning must be much longer (here 42 min).
- reference detector runs at about 600 packets/s in pure Python: fine for replay, not for a busy link.
- reference sensor ran offline in WSL2; live capture is not tested. No learned combiner was trained (needs labels).

## Next
- Labelled packet data (UCI Mirai dataset) to measure recall and false alerts per host-hour and to train the learned combiner: the one thing that turns this
  descriptive report into a measured one.
- Live capture: the adapter exists (`pipeline/live.py`, tested with a fake sniffer, hard limits, local folder only). A real capture needs your approval prompt and
  is NOT yet run. reference detector needs about 55,000 clean packets before it can score, so a short capture shows the capture path only.
