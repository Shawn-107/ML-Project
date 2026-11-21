# Acceptance run on unseen data (2026-11-21)

Everything below was run after the rewrite, on data the models were not trained or tuned on. Raw numbers: `acceptance_result.json`. No network capture was made (owner decision); the "live" path is tested by delivering packets one at a time from files.

## CSV part
- UNSW-NB15 official test, cleaned (46,830 flows the model never saw): 3% policy accuracy 89.2%, recall 77.6%, false alarms 4.8% (1.4 s).
- NetFlow model, source hosts never seen in training (from its training record): AUC 0.99995, 3% policy recall 96.1% at 0.0% false alarms. This dataset is easy (see `reports/flows/NETFLOW_REPORT.md`), so it says little about other networks. 100,000 rows through the CSV tool: seconds, no warnings.
- Argus flow files from another project (5 files): scans found in 3 (port scan: vertical scan + ICMP sweep; two horizontal-scan files); none in the two files whose malware traffic is not scanning. The rules only cover scanning.
- A file from a dataset the tool has no model for: clear error naming the closest format and the missing columns; no crash.

## Packet part, one packet at a time on unseen traffic
- Detector trained on the first 52 minutes of the Mirai capture (calibrated at 1% false alarms), then 104,621 later packets scored one at a time: same scores as batch (largest difference 2e-7, flags agree 100%).
- Flagged share by period: minutes 52-60: 0.09%; 60-70: 1.0%; 70-74: 0.4%; 74-80: 88%; 80-90: 92% (the traffic changes at minute 74; the capture has no labels, so this is descriptive).
- Speed one packet at a time: about 830 packets per second (batch scoring is more than ten times faster).

## Unseen real captures through the rules (known attack captures from another project)
- ICMP ping sweep capture: 1 `icmp_sweep` finding (found by us; the external reference sensor found nothing). This capture exposed a missing rule, which was added.
- Malware capture (2,000 packets): 3 `scan` findings. ARP-sweep capture: 1 `arp_flood` finding. The reference sensor reported no events on these three files.
- Reading the unseen captures with the indicator file: no indicator hits.

## Capture path ("live" code path without the network)
- 6,000 packets delivered one by one through the capture adapter into the local capture file, read back: 6,000 packets, identical packet fields and identical rule evidence to reading the original file.

## Dashboard
- Sign-in works; with pipeline outputs plus rule events from the unseen captures loaded, the summary counts equal the lines in the files (86 alerts, 332 events). Search and filters work (e.g. "icmp" finds the sweep).
- CSV upload through the dashboard: an Argus file added 2 events (visible in the events table), a 3,000-row UNSW file was analysed (flagged 2,478: the first rows of that file are 95% attacks).
- The earlier project's app is served at `/app/` and its API works: sample flows scored (89.5% accuracy on 200 sampled flows at the 3% threshold), explanations returned.

## Not checked
- How the pages look: no browser was available to the assistant, so layout and clarity were not looked at. Open it and check.
- A real network capture (not done by decision) and signature hits on unseen data (none of the unseen captures contained web-attack requests or known-bad indicators).
