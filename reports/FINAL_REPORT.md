# Layered NIDS: final report (status 2026-11-20)

Course: Advanced Machine Learning (MAI511-2), MSc AI/ML. Every number below was measured in this project unless marked "untested". Details and the exact commands are in the linked reports.

## 1. What was built
A network intrusion detection system that combines evidence from several detectors per host:
- **Maltrail** (real sensor, MIT): blocklists and heuristics, run offline in WSL2 on pcap files. No ML.
- **Kitsune** (real code, vendored, MIT): unsupervised ensemble of small autoencoders on per-packet statistics. A packet-by-packet wrapper was added; a parity test shows identical features and scores to the original.
- **Our ML layers:** DGA domain classifier, supervised flow classifiers (UNSW-NB15 34 features, NetFlow 40 features), and a learned alert combiner.
- **Fusion:** per-host evidence over time windows (binomial and beta-binomial p-values, noisy-OR with decay), with a per-layer policy (signature hits alert directly; Kitsune and heuristics accumulate).
- **Inputs:** pcap files (replay), CSV flow files (`layered_nids/csvmode`), and a limited live capture (Npcap, admin-only, one UAC prompt per capture).
- **Output:** Maltrail's own web dashboard on localhost (`tools/dashboard`), loaded with the real Maltrail events plus our alerts.

Slips was studied (`reports/repo_evaluation.md`) but is not used: GPL, Linux/Docker only. No Slips code is in this project.

## 2. Results
### Real tools on the Mirai capture (764,137 packets, unlabelled)
- Maltrail: 8 events, all heuristic "periodic beaconing" in minutes 2-8; none where Kitsune's scores change.
- Kitsune: 0.7% of packets above the cut-off until about packet 121,700 (minute 74.4), then 88% flagged. Speed about 600-700 packets/s in pure Python.
- Combined pipeline (alert threshold tuned on the first half of the assumed-normal stretch, checked on the second half): false alerts on the held-out half fall from 6 (about 1.2 per host-hour) at p = 0.99 to 0 at 1-1e-7 or stricter; alerts after the jump fall from 38 on 14 hosts to 10-11 on 2 hosts at the chosen threshold.
- **Limit:** no labels. Recall and precision are NOT measured; "normal before the jump" is an assumption (swapping the halves gave 2 versus 4 alerting hosts).

### DGA classifier (`reports/dga/REPORT.md`)
Unseen-family evaluation, 5 folds, recall at 1% false alarms: n-gram 72.1% +- 8.4, character CNN 74.4% +- 10.5, average of both 74.8% +- 9.2 (AUC 0.961, Brier 0.076). The average beats n-gram in all 5 folds (about +2.7 points) but the CNN in only 3 of 5 (+0.4 points, inside the fold spread). The benign side is a top-sites list, so false alarms on real DNS are untested.

### Flow classifier, UNSW-NB15 (`reports/flows/`)
- The earlier project's ANN+forest blend reproduces its shipped numbers within 0.3 points (87.9% accuracy, 79.0% recall, 7.5% false alarms at the 3% policy).
- Gradient boosting is better: 89.1% accuracy, 81.5% recall, about 7.1% false alarms (3 seeds). Leave-one-attack-out, 9 families, one seed: 73.9% detected versus 68.4%.
- Negative results, reported as such: hyperparameter search (no gain, validation AUC about 0.982 for every setting), dropping shifted features (lower accuracy), reweighting to the test distribution (no gain), multi-class (no gain), derived NetFlow features (no gain).
- Fuzzers and Reconnaissance stay weak (about 16% and 36% detected). The official test set is shifted from training (a classifier separates them with AUC 0.79), so about 7% false alarms at a 3% target is expected.

### NetFlow flow classifier (`reports/flows/NETFLOW_REPORT.md`)
All 40 features give AUC 1.0000 on unseen hosts: the dataset is easy, not the model strong. With only the 24 features a live packet-to-flow builder could compute: AUC 0.979, recall 64% at 3% false alarms, scanning 19%. No packet-to-flow builder was written.

### Fusion (`reports/fusion/`) — simulation
Built from real per-layer scores assuming independent layers, so an upper bound, not a network result. Budget: 0.5 false alerts per benign host-day; 20 test scenarios, mean +- std over scenarios.
- **Unfair first run** (the evidence rule is told the nominal 3% false-alarm rate for flows, but the layer really gives 7.2-7.6%): cut-off and noisy-OR accumulation gave 9-27 false alerts per host-day and never met the budget; the learned combiner met it (98% of attackers detected at 0.23 false alerts per host-day with the improved flow model). This made the learned combiner look necessary.
- **Fair baseline** (`*_told_fpr.json`: same scores and thresholds, but the rule is told each layer's locally measured false-alarm rate): noisy-OR accumulation also meets the budget. DGA + flows, stacked flow model: accumulation 100% detected at 0.08 false alerts/host-day (median 156 s to detect); learned combiner 100% at 0.30 (258 s); single cut-off 100% at 0.93 (over budget). Flows only: accumulation 64% at 0.08, learned combiner 64% at 0.21. The legacy flow model gives the same picture.
- **Conclusion:** the learned combiner is NOT better than a correctly calibrated noisy-OR accumulator here. The earlier advantage came from the accumulator being mis-calibrated. What matters is telling each layer's evidence its real false-alarm rate (measured on local normal traffic). Accumulation reaches the grid edge (alert threshold 1 - 1e-10) in the two-layer case. Nested cross-validation not done.

### CSV mode (`reports/csv/CSV_REPORT.md`)
Reads UNSW, NetFlow and Argus files; tolerant of bad input (every row kept, bad cells reported). UNSW cleaned test through the tool: 89.1% accuracy, 81.4% recall, 7.0% false alarms. NetFlow: 200,000 rows in 8 s. Argus: scan rules only; found scans in 3 of 5 real sample files and nothing in the two whose malware traffic is not scanning (rules do not cover that).

### Live path
Self-test passed: a real 267-packet capture on this laptop's Wi-Fi, parsed completely, and the real Maltrail sensor flagged the planted harmless directory-traversal request. **Untested:** Kitsune on live traffic (needs about 55,000 packets to learn), long captures, true streaming, detection accuracy.

## 3. Engineering and safety choices
- Smart App Control was not bypassed: the blocked Windows build was replaced by the Linux build in a hardened WSL2 (no Windows drives, no interop); Python 3.13 with established package versions avoids the reputation blocking.
- Npcap runs in administrator-only mode; every capture needs a UAC approval, stays local, and is deleted afterwards.
- The dashboard listens on 127.0.0.1 only, with its own random password stored as a PBKDF2 hash.
- Model files are pickles with a SHA-256 check (detects accidental change, not a malicious author); the repository does not publish them.
- 104 tests pass in the new package; the legacy backend has 54 passed, 1 skipped.

## 4. Limits (summary)
- UNSW-NB15 is 2015 lab traffic; accuracy about 89%, never 90%.
- No labelled packet data was used; the UCI Kitsune dataset could not be fetched (TLS check failed; not bypassed).
- Fusion numbers are from a simulation.
- Replay and CSV are not live detection.
- The explanation module is our own Shapley estimator, not the `shap` package.

## 5. Related work and credits
- Maltrail: M. Stampar, https://github.com/stamparm/maltrail (MIT; static trail data under separate terms).
- Kitsune: Y. Mirsky, T. Doitshman, Y. Elovici, A. Shabtai, "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS 2018; code https://github.com/ymirsky/Kitsune-py (MIT).
- Slips: Stratosphere Laboratory, https://github.com/stratosphereips/StratosphereLinuxIPS (GPL-2.0; studied only).
- UNSW-NB15: N. Moustafa, J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems", MilCIS 2015.
- NF-ToN-IoT and the NetFlow v2 feature set: University of Queensland NetFlow datasets (check the exact citation before submission).
- DGA data: Mendeley "Legitimate domains and DGA categorized morphologically and by families" (CC BY 4.0).
- Citations were written from memory of the sources and the repositories' own pages; verify titles, venues and years before submitting.
