# NIDS: final report (status 2026-11-22)

Course: Advanced Machine Learning (MAI511-2), MSc AI/ML. Every number below was measured in this project unless marked "untested". Details and commands are in the linked reports.

## 1. What was built
A network intrusion detection system whose layers are all our own code:
- **Packet anomaly detector** (`layered_nids/anomaly`): 120 damped-statistics features per packet, an ensemble of small autoencoders trained on normal traffic only, a threshold set by a target false-alarm rate on held-out normal packets, evidence per (host, time window) from binomial or beta-binomial counts. Streaming and batch scoring give the same scores; models are saved with a SHA-256 check. A first, simpler version (`layered_nids/packets`) is kept as a baseline.
- **Signature and heuristic rules** (`layered_nids/signatures`): 10 web-attack patterns on HTTP request lines, matching of known-bad addresses, domains, URLs and TLS server names against a local indicator file (with an optional popular-domain whitelist), user blocklists, and scan, flood, ICMP-sweep, ARP, DNS-burst, DNS-tunnel, random-name and beacon rules. No machine learning. An optional adapter can add events from an external sensor program. A first rule set (`layered_nids/rules`) is kept as a baseline.
- **DGA classifier**, **flow classifiers** (UNSW-NB15 and NetFlow) and the **CSV entry point** (`layered_nids/csvmode`).
- **Fusion** (`layered_nids/fusion`, `layered_nids/pipeline`): per-host evidence accumulation (noisy-OR with decay), a per-layer alert policy, and a learned combiner for simulation studies.
- **Dashboard** (`layered_nids/dashboard`): own stdlib server, localhost only, sign-in with a salted password hash and a session cookie. Overview with charts (alerts over time, severity, host map, evidence share, rules), alert and event tables with filters, a per-host timeline panel, CSV upload, live refresh, a command palette, light and dark themes, an animated sign-in page; it also serves the earlier project's flow-analysis app.

## 2. Results
### Packet anomaly detector (`reports/anomaly/REPORT.md`)
Injection benchmark on real normal traffic (the first 52 minutes of the Mirai capture, split in time order) with synthetic attacks; 3 model seeds x 3 injection seeds; models compared at the SAME false-alarm rate (recall at 1% false alarms on the normal packets of uninvolved hosts). This measures whether a behaviour is noticed, not accuracy on real attacks. Our detector against our first version against the original reference detector:
- Horizontal scan: recall 0.99 against 0.14 and 0.15 (AUC 0.986 against 0.822 and 0.815).
- Port scan, SYN flood, UDP flood, bulk upload: about 1.00 for all three.
- DNS burst: **0.66 against 0.88 and 0.88 (worse)**. The optional sensitive profile (`views=2`) gives 0.92, at the cost of more false-alert windows (95.8 per hour against 60). The rule layer also finds this behaviour.
- Low-and-slow beacon: not found at packet level by any of the three; the rule layer finds regular new connections.
- False alarms on normal packets of uninvolved hosts: 1.88% against 2.11% and 3.12% (target 1%: all three are above it because traffic drifts).
- Speed: about 10,600 packets/s feature extraction; about 1,800 packets/s scoring one packet at a time (the original ran at 684 packets/s end to end on this machine).
- Several design variants were tried on separate validation attacks and rejected (listed in the report).

### Signature and heuristic rules (`reports/signatures/REPORT.md`)
- Injected behaviours (scan, port scan, SYN flood, UDP flood, DNS burst, web attacks, regular beacon connections) found 3 of 3 times each; bulk upload not found (the packet model finds it). Synthetic attacks written by the same author as the rules.
- Unseen real captures from another project: the ICMP ping-sweep capture exposed a missing rule (added; now found), the scan capture gave 3 scan findings and the ARP-sweep capture 1 finding. The external reference sensor reported nothing on these three files.
- Known-bad matching with a local 1.86-million-indicator file (loads in about 4 s; not part of the repository): side by side with the reference sensor on sampled indicators, ours found 98% of domains, 90% of addresses, 100% of address:port pairs, 98% of URL paths and 100% of host+path entries; the sensor found 100%, 14%, 100%, 73% and 57%. "Suspicious"-class indicators are treated as weak evidence. Both gave zero indicator hits on the whole Mirai capture. Ours finding more is not proof it is better: the sensor applies its own filtering.
- Limits: plain HTTP request lines and TLS server names only (no bodies, headers or TLS fingerprints); TLS server-name parsing was tested on crafted data, the Mirai capture contains none.

### Combined pipeline on the Mirai capture (`reports/pipeline/mirai.json`, unlabelled, descriptive)
The packet scores jump at minute 74.3. Alert threshold tuned on the first half of the stretch assumed normal and checked on the second half. Packet layer alone: 50 alerts on 7 hosts after the jump, none before. With the rules layer: 66 alerts, 7 of them before the jump (2 hosts, minutes 47.6 and 66.6): possible false alerts or real early activity, which cannot be told without labels. No recall or false-alarm rate is measured on this capture.

### DGA classifier (`reports/dga/REPORT.md`)
Unseen-family evaluation, 5 folds, recall at 1% false alarms: n-gram 72.1% +- 8.4, character CNN 74.4% +- 10.5, average of both 74.8% +- 9.2 (AUC 0.961, Brier 0.076). The average beats the n-gram model in all 5 folds (about +2.7 points) but the CNN in only 3 of 5 (+0.4, inside the fold spread). The benign side is a top-sites list, so false alarms on real DNS are untested.

### Flow classifiers (`reports/flows/`, `reports/csv/dataset_tests.json`)
- The earlier project's ANN + forest blend reproduces its shipped numbers within 0.3 points (87.9% accuracy, 79.0% recall, 7.5% false alarms at the 3% policy).
- Gradient boosting: 89.1% accuracy, 81.5% recall, about 7.1% false alarms; leave-one-attack-out 73.9% against 68.4%.
- The owner's earlier flow model, scored through CSV mode on the cleaned UNSW official test (46,830 flows), 3% policy: 89.2% accuracy, 77.5% recall, 4.8% false alarms. By the rule "equal-or-higher accuracy with fewer false alarms" it is the model used for UNSW CSV scoring. At the 1% policy: 86.3% accuracy, 60.8% recall, 0.6% false alarms.
- Negative results, reported as such: hyperparameter search (no gain), dropping shifted features (lower accuracy), reweighting to the test distribution (no gain), multi-class (no gain), derived NetFlow features (no gain).
- Fuzzers and Reconnaissance stay weak (about 16% and 36% detected). The official test is shifted from training (a classifier separates them with AUC 0.79).
- NetFlow: all 40 features give AUC 1.0000 on unseen hosts (the dataset is easy, not the model strong); with only the 24 features a packet-to-flow builder could compute, AUC 0.979 and recall 64% at 3% false alarms. No packet-to-flow builder was written.

### Fusion (`reports/fusion/`) — simulation
Built from real per-layer scores assuming independent layers: an upper bound, not a network result. Budget 0.5 false alerts per benign host-day; 20 test scenarios.
- When the evidence rule is told a layer's nominal false-alarm rate but the layer really gives more (flows: 7.2-7.6% against 3%), accumulation gave 9-27 false alerts per host-day; the learned combiner met the budget.
- When the rule is told each layer's locally measured false-alarm rate (fair baseline), accumulation also meets the budget: DGA + flows, 100% of attackers detected at 0.08 false alerts per host-day, against the learned combiner's 100% at 0.30. The learned combiner is therefore NOT better than a calibrated accumulator; what matters is calibration.

### CSV mode (`reports/csv/CSV_REPORT.md`)
Reads UNSW, NetFlow and Argus files; tolerant of bad input (every row kept, bad cells reported). NetFlow: 100,000 rows in seconds. Argus files get scan rules only; on five real sample files scans were found in 3 and nothing in the two whose malware traffic is not scanning. A file from a dataset with no model gives a clear error naming the closest format.

### Acceptance run on unseen data (`reports/acceptance/ACCEPTANCE.md`)
CSV (UNSW official test the models never saw, NetFlow, five flow files from another project), packets streamed one at a time on later traffic of the Mirai capture (flagged 0.1-1% before minute 74, 88-92% after), three unseen real captures through the rules, the capture path with an injected packet source (6,000 packets in and out, identical results), and the dashboard wired to the pipeline outputs (summary counts equal the lines in the files; an uploaded file adds events; the earlier project's app answers). No real network capture was made.

### Dashboard (`reports/acceptance/ACCEPTANCE.md`)
Checked over HTTP and in a real browser engine (headless Edge): every page renders its data, no JavaScript errors, phone-width layout measured without sideways page scroll, dark and light themes. A bug was found and fixed on the way: the sign-in page's stylesheet and script were blocked until sign-in. Not tested: typing a password into the sign-in form in a browser.

## 3. Engineering and safety choices
- Dataset tests only; no live capture was run (owner decision).
- The dashboard listens on 127.0.0.1 only, with a random password stored as a salted hash; uploads are size-capped, processed in a separate process and deleted after analysis.
- Model files for the flow models are pickles with a SHA-256 check (detects accidental change, not a malicious author); the packet detector loads tensors only.
- Windows Smart App Control was not bypassed: Python 3.13 with established package versions is used.

## 4. Limits (summary)
- UNSW-NB15 is 2015 lab traffic; accuracy about 89%, never 90%.
- No labelled packet data was used; the benchmarks use injected synthetic attacks. The packet detector is worse than the original on DNS bursts by default, and nothing finds slow beacons at packet level.
- The indicator data is a local, non-redistributable file; without it the rule layer cannot match known-bad indicators.
- Fusion numbers are from a simulation.
- Replay and CSV are not live detection.
- The explanation module is our own Shapley estimator, not the `shap` package.

## 5. Related work and credits
Third-party credits, papers and licence texts are listed in `credits/CREDITS.md`, which is kept with the project and not published. Verify every citation there before submitting.
