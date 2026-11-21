confi# Layered NIDS — project brief (hand-off to Claude Code)

Working title (short and changeable): **Layered NIDS**.
Owner: Shawn, MSc AI/ML, Christ University. Course: Advanced Machine Learning (MAI511-2).
Preferences: direct and brief, plain language, list formats over tables, no filler, answer exactly what was asked, fact-check and correct without deflecting.

## 0. Direction update (2026-11-19, from the owner — overrides anything below that conflicts)
- The earlier approach (re-implementing the ideas of three tools in our own code) was too basic for PG level. The owner found three completed GitHub projects that also do live detection — Maltrail, Kitsune-py, Slips — and wants to build on the REAL tools.
- Current task: test each repo separately (use case, inputs, CSV vs live, what runs on this machine), then report whether they can be combined into one project and exactly what can be combined. Write it to `reports/repo_evaluation.md`. Do not claim anything untested.
- Licence rules:
  - Maltrail engine (MIT) and Kitsune-py (MIT): real code may be reused, with their licence notice kept. Maltrail's static trail data has separate terms (free for internal defence, research, teaching; commercial needs a licence).
  - Slips is GPL-2.0: run it as a separate program and read its output files. Do NOT copy or import its code into this project (that would make the whole project GPL).
  - Cite all three as related work.
- 2026-11-20, owner decision: **Slips is NOT used** in the project (heavy, Linux/Docker only, GPL). The project is Kitsune-py + Maltrail + our own ML layers. Slips stays in the repo evaluation report only as a studied alternative. Live detection therefore comes from Maltrail's sensor (Linux/VM, or Npcap on Windows) and from our own capture feeding the Kitsune wrapper, our rules port and our ML layers.
- Our own earlier layers (`dga/`, `packets/`, `rules/`, `flows/`, `fusion/`) are NOT deleted. They remain as the ML contribution and as baselines to compare against the real tools.

## 1. Goal
A network intrusion detection system where **the machine learning is the core**, built on three real open-source tools, combined at the alert level, plus our own ML layers:
- Maltrail: blocklist + heuristic detection on live traffic or pcap (no ML in Maltrail).
- Kitsune-py: unsupervised ensemble of small autoencoders on per-packet statistics (AfterImage + KitNET).
- Slips: behavioural IDS that accumulates evidence per host (has its own ML modules and threat-intel feeds; needs Docker/Linux, runs as a separate process).
- Our own: DGA classifier, supervised flow classifier (UNSW-NB15), learned alert fusion.

Must be PG-level, honest, and simple enough to explain. Never claim a number that was not measured.

## 2. What already exists (separate project, finished; now `legacy/nids_backend`, importable as `nids`)
UNSW-NB15 backend, package `nids`. Key facts:
- Input: 34 flow features (UNSW-NB15 mirror copy). 3 categorical (proto, service, state), 1 binary, 30 continuous.
- Cleaning: drop feature vectors with conflicting labels, collapse repeats, remove test rows also in training. Preprocessor fitted on train only; thresholds from a validation split; official test used only by `evaluate`.
- Models: 5-seed MLP ensemble (128,64; L2 alpha 1e-4; early stopping) + random forest (150 trees), blended 0.5/0.5. Autoencoder (MLPRegressor) trained on Normal flows for novelty.
- Thresholds are policies named by false-alarm target (fpr_1pct / 3pct / 5pct / f1). Default fpr_3pct.
- Extras: Monte-Carlo Shapley explanations (own implementation), local recalibration, score-drift monitor (misses the dataset shift), SQLite alert store, replay generator, stdlib HTTP service (localhost, no auth), CLI, report generator.
- Re-run on this machine (2026-11-19): tests 54 passed, 1 skipped (needs optional parquet files). Official test (46,830 flows), retrained: fpr_1pct 88.0% acc / 2.4% false alarms / 69.3% recall; fpr_3pct 87.9% / 7.5% / 79.0%; with novelty 84.6% / 14.6% / 83.0%. Matches the shipped numbers within 0.1 point. Leave-one-attack-out and stability from the old report are NOT re-verified (shipped: LOAO mean 69.0% / 76.4% with novelty; Fuzzers 31.5%, Reconnaissance 36.9%).
- Known limit: official test is distribution-shifted vs train (a classifier tells them apart with cross-validated AUC 0.79). Never reached 90%.
- No live capture. The 34 features come from Argus/Zeek-style tooling and cannot be reproduced exactly from raw packets, so L4 is replay/CSV only.

## 3. The three reference repos (cloned to `reference/`; read the READMEs and code; tested where noted)
- **Maltrail** (stamparm/maltrail, release 3.6): engine MIT. Rust sensor (libpcap) + Python server/web UI. Matches traffic against "trails" (domains, URLs, IPs, IP:port, User-Agent, JA3/JA4) plus heuristics (scanning, DNS exhaustion, DGA-like lookups, suspicious downloads, proxy probes). No ML. Input: live interface or pcap (`--offline -r file.pcap`); no CSV input. Output: event-log lines `"<time>" <sensor> <src_ip> <src_port> <dst_ip> <dst_port> <proto> <type> <trail> <info> <ref>`. Windows build exists but needs Npcap. Default login admin/changeme!, listens on 0.0.0.0 — change both before running.
  - Tested here: the Windows `.exe` is unsigned and **Smart App Control blocked it** (not bypassed), so the official Linux build (release 3.6, SHA-256 verified, with the verified 1.86M-row trail set) was run offline inside hardened WSL2 (`~/work/maltrail` in Ubuntu; config localhost-only, trail updates off). Its own detection self-test passed 27/27. On the whole Mirai capture (764,137 packets, 0.06 s) it logged 8 events, all heuristic "periodic beaconing" in minutes 2-8, none in the phase where Kitsune's scores jump (minute 74 on). One unlabelled capture only. The sensor reads real captures made on this laptop (live self-test passed, see section 10 item 7); it does not capture live itself here. Its Python side also ran on Windows: 36 of 43 test files pass, 6 fail for Windows-only reasons.
- **Kitsune-py** (ymirsky/Kitsune-py): MIT. Unsupervised, online. AfterImage = damped incremental statistics per packet channel (100 features measured; its example file says 115); KitNET = ensemble of small autoencoders feeding an output autoencoder. Input: pcap, pcapng or its own TSV (pre-parsed packets); no CSV flows; no live capture in its code (statistics are incremental, so a live wrapper is possible). Uses tshark if installed, otherwise scapy (much slower; loads the whole pcap into memory). Repo ships an unlabelled `mirai.pcap` (62.9 MB).
  - Tested here (full Mirai pcap, 764,137 packets): needs `Cython` installed and a NumPy-2 shim in the caller. 684 packets/s in pure Python. Scores stay near benign (0.7% above the 99th-percentile cut-off) until about packet 121,700, then 88% of packets are flagged. No labels, so recall/precision are NOT measured. Details in `reports/repo_evaluation.md`.
- **Slips** (stratosphereips/StratosphereLinuxIPS, v1.1.24): GPL-2.0. Behavioural IDS/IPS with threat-intel feeds, heuristics and ML modules: `ml_linear_model` (sklearn SGDClassifier + PCA), `ml_online_model`, RNN C&C detection (TensorFlow/Keras on "Stratosphere letters"), HTTPS anomaly detection (River ADWIN). Input: live interface, pcap (via Zeek), Argus flows (CSV or tab), Zeek logs, Suricata JSON, nfdump, stdin. Platform: Docker or Linux; needs Zeek and Redis; docs say Windows is supported through Docker only, and that the Docker image does not capture the host's own interface on every host type.
  - Tested here: not runnable — no Docker, WSL, Zeek or Redis on this machine. Untested.
- GitHub search found no well-established (high-star) project combining flow classification, unseen-attack testing, cleaning and recalibration. Only two small repos were opened, so this is not proof none exist.

## 4. Architecture (each layer its own module, merged only at the alert level via `common/schema.py` Evidence)
Real tools (target design; combine by reading their outputs, not by merging code where the licence forbids it):
- Maltrail → event-log lines → Evidence (layer `rules`).
- Kitsune-py → per-packet RMSE scores → Evidence (layer `packets`).
- Slips → its alerts/evidence files (separate process) → Evidence.
Our own layers (kept; also the baselines for the real tools):
- L1 `rules/` — own blocklist + heuristics baseline.
- L2 `dga/` — domain classifier, real vs generated (n-gram LR, lexical MLP, char-CNN; family-held-out evaluation).
- L3 `packets/` — own AfterImage-style statistics + autoencoder ensemble (baseline for Kitsune-py).
- L4 `flows/` — the legacy UNSW-NB15 `nids` package + improvement experiments (gradient boosting, stacker, adversarial validation, novelty options).
- L5 `fusion/` — per-host evidence accumulation (noisy-OR with decay) + learned logistic-regression combiner; evaluated in a simulation because no dataset has the same hosts seen by all layers.
- Dashboard (Streamlit) is a later, separate step.

## 5. ML experiments on the flow model (measured; see `reports/flows/`)
- Gradient boosting beat the legacy 0.5/0.5 ANN+forest blend on validation and test: at the 3% policy, 89.1% vs 88.0% accuracy, recall 81.5% vs 78.6%, false alarms about 7% vs 7.2% (new models over 3 seeds; ANN held fixed). The ANN is the weakest component; the learned stacker gives it about zero weight. Multi-class gradient boosting adds nothing over binary.
- Adversarial validation: dropping the most shifted features (ackdat, dbytes, service, dmean…) lowers test accuracy (88.9% → 86.8% at 10 features dropped); reweighting training rows to look like test gives no gain. Negative result, reported as such.
- Leave-one-attack-out, 9 families, one seed (spread is across families, not repeated runs): legacy blend 68.4% detected / 7.7% false alarms (matches the old report's 69.0%); gradient boosting 73.9% / 7.1%. Novelty detectors alone: autoencoder 47.8%, isolation forest 26.4%, K-Means 18.8%. Autoencoder + gradient boosting, "flag if either": 77.2% / 14.2% false alarms; gated (novelty counts only when the classifier is not clearly sure it is normal): 76.4% / 9.7%. Fuzzers (~16%) and Reconnaissance (~36%) stay weak whatever we tried.
- Fusion simulation (`reports/fusion/ablation_*.json`; budget 0.5 false alerts per benign host-day): the flow layer's measured false-alarm rate (7.2-7.6%) is above the 3% the evidence rule assumes, so cut-off and noisy-OR accumulation on the flow layer give 9-27 false alerts per host-day and never meet the budget (their detection numbers are not valid). The learned combiner is the only method that met it: 98% of attackers detected at 0.23 false alerts per host-day with the improved flow model. CORRECTION (2026-11-20, fair baseline run, `reports/fusion/ablation_*_told_fpr.json`): when the accumulator is told the locally measured false-alarm rate it ALSO meets the budget (DGA+flows, stacked flows: 100% detected at 0.08 false alerts/host-day vs the learned combiner's 100% at 0.30; flows only 64% at 0.08 vs 64% at 0.21). So the learned combiner is not better than a calibrated accumulator; its earlier advantage came from the mis-calibrated baseline. Nested CV not done.
- Gradient-boosting hyperparameter search (12 random settings + default, chosen by validation AUC only, `reports/flows/tuning.json`): NO gain. Every setting has validation AUC about 0.982 (differences in the 4th decimal), the default is selected, test unchanged (89.1% accuracy, 81.6% recall, 7.1% false alarms). The limit is the distribution shift, not the settings. Null result.
- Legacy model retrained under Python 3.13 / scikit-learn 1.6.1 reproduces the shipped numbers within 0.3 points (accuracy 87.89% vs 87.95%, recall 78.8% vs 79.0%, false alarms 7.45% vs 7.49%); old model files backed up in `legacy/models_backup_sklearn19`.
- NetFlow flow model (`reports/flows/NETFLOW_REPORT.md`, NF-ToN-IoT-v2, split by source host): all 40 features AUC 1.0000 (a sign the dataset is easy, not that the model is strong); only the 24 features a live packet-to-flow builder can compute: AUC 0.979, recall 64% at 3% false alarms (vs 96%), scanning 19% (vs 94%); derived features (TCP flag bits, ratios) did NOT help (negative result); unseen attack families 52% on average. A live flow classifier on exactly reproducible features is much weaker; no packet-to-flow builder was written.

## 6. Evaluation rules (non-negotiable)
- Leakage control as in section 2; split by DGA family / attack family where relevant; test data touched once; never choose a model by its test score.
- Always report mean +- std over repeated runs; say what the spread is over (seeds, folds, families).
- Baselines for every model (rule-based, logistic regression). Ablations: each layer on/off in the fusion, at the same false-alarm budget.
- Report false-alarm rate next to recall, never accuracy alone. Report calibration (Brier, ECE).
- Leave-one-family-out for unseen-threat claims.
- Anything not run or not measured is labelled "untested". No invented numbers. A simulation is labelled a simulation.

## 7. Syllabus mapping (MAI511-2)
- Unit 1: regularization (L2), logistic regression baselines, bias/variance, trees.
- Unit 2: feature selection / PCA / correlation filter on flow and DGA features.
- Unit 3: K-Means (optimum k by silhouette) as an unsupervised novelty detector.
- Unit 5: MLP, activation functions; Lab 9 hand-computed neuron with binary and bipolar sigmoid.
- Self-learning (beyond syllabus): leakage control, threshold policies, autoencoder ensembles, Shapley explanations, recalibration, drift, fusion.

## 8. Data (see `data/SOURCES.md`)
- DGA: Mendeley "Legitimate domains and DGA categorized morphologically and by families" (CC BY 4.0), SHA-256 verified, in `data/dga/`. Benign side is a top-sites list (easier than real DNS).
- UNSW-NB15 official CSVs in `data/unsw/` (copied from the earlier project).
- Packets: Kitsune's `mirai.pcap` is unlabelled. The labelled UCI Kitsune dataset (17.7 GB zip, CC BY 4.0) is NOT downloaded (TLS verification failed for archive.ics.uci.edu; not bypassed). Other datasets from the earlier project exist on disk (`C:\ML project\data`: CIC-IDS2018, NF-ToN-IoT, ToN-IoT, NSL-KDD).

## 9. Environment notes (this machine: Windows 11, owner's laptop)
- USE `.venv313` (Python 3.13.17 from `.runtime/py313`, pinned long-established versions in `requirements-313.txt`: numpy 2.2.6, scipy 1.15.3, pandas 2.2.3, scikit-learn 1.6.1, torch 2.7.1+cpu). Run things with `.venv313\Scripts\python.exe`. Tests there: 74 passed (layered_nids), 54 passed + 1 skipped (legacy). 12 logical CPUs.
- The old `.venv` (Python 3.15.2, numpy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1) is partly BLOCKED by Smart App Control since the 2026-11-20 restart (numpy.fft, scipy.stats, pandas, sklearn). Likely cause (a theory the test supports): brand-new unsigned files have no Microsoft reputation. Established versions load fine, so no security setting needed changing. The legacy model bundle in `legacy/nids_backend/models` was saved with sklearn 1.9.1; if loading it fails or warns under 1.6.1, retrain it (about 4 minutes) after copying the old one.
- Windows Application Control (Smart App Control ON) blocks pyarrow and unsigned executables. Do not bypass it or turn it off; the owner may choose to. First runs in a fresh environment can be very slow (Windows scanning new files; one legacy test took 14 min once, 1 s later).
- WSL2 with Ubuntu 26.04 LTS is installed (2026-11-20, owner-approved UAC install). Hardened in `/etc/wsl.conf`: no Windows drives mounted, no Windows interop (tested). Owner's Linux user is `shawn`; never ask for or handle the password. Move files in/out via `\\wsl.localhost\Ubuntu\home\shawn\`. WSL2 cannot see the Windows network card, so it is for offline pcap work only.
- Npcap 1.90 (free edition, from npcap.com; installer signature valid, Nmap Software LLC) is installed (2026-11-20, owner-run) with "Restrict access to Administrators only" ON and WinPcap-compatible mode OFF. A capture by a normal program triggers a UAC approval prompt each time; the owner approved two test captures (1 packet each, nothing stored). Scapy in `.venv313` finds Npcap (`conf.use_pcap` True). Only start captures when the owner expects the prompt; never capture on shared/public networks; keep pcaps local and delete them after use.
- Not installed: Docker, Rust, Zeek, Redis, tshark (not needed).
- Other Claude sessions may work in this folder at the same time; message before editing shared files (`pyproject.toml`, package `__init__.py`).

## 10. Build plan (status)
1. Skeleton + shared schema; L4 wired: done.
2. L2 DGA: done (`reports/dga/REPORT.md`). Ensemble check (`reports/dga/ensemble.json`, same 5 unseen-family folds, paired): recall at 1% false alarms n-gram 72.1% +- 8.4, CNN 74.4% +- 10.5, average of both 74.8% +- 9.2 (AUC 0.961, Brier 0.076 vs 0.960/0.078 and 0.955/0.083). The average beats the n-gram model in all 5 folds (+2.7 points on average) but beats the CNN in only 3 of 5 (+0.4 points, far inside the fold spread): a small gain over n-gram, no meaningful gain over the CNN.
3. L3 own packet detector, L1 own rules: done by a second session (own tests; no labels, so no recall measured).
4. L5 fusion: accumulator, simulator, learned combiner, ablation: done (simulation only; results in section 5); fair accumulator baseline now run (see the correction in section 5).
5. Repo evaluation of the three real tools (section 0): done as far as this machine allows (`reports/repo_evaluation.md`); Maltrail's sensor and Slips still need a Linux environment (WSL2/VM/Docker) to run.
6. Decide the combined design from the repo evaluation, then rebuild around the real tools; keep our layers as baselines.
   - Done: real Kitsune vendored into `layered_nids/realtools/kitsune/` (MIT, NOTICE.md lists every change) with a packet-by-packet wrapper (`stream.py`); parity test proves identical features and scores to the original on Mirai (`tests/test_kitsune_vendored.py`).
   - Done (2026-11-20): combined pipeline on pcap, `layered_nids/pipeline/` (real Kitsune scores -> per-host evidence; real Maltrail sensor run in WSL2 -> parsed events -> evidence; noisy-OR fusion). 83 tests pass. Host attribution fixed (shared rule `common/hosts.py`: only real private unicast hosts; MACs, 0.0.0.0, broadcast, multicast, external-only traffic take no part). Alert threshold tuned on the first half of the assumed-normal stretch before Kitsune's jump (minute 74.4) and checked on the second half. On the unlabelled Mirai capture (`reports/pipeline/REPORT.md`): false alerts on the held-out normal half fall from 6 (about 1.2 per host-hour) at alert_p 0.99 to 0 at 1-1e-7 or stricter; after the jump alerts fall from 38 on 14 hosts to 16 on 4 hosts (1-1e-7) or 11 on 2 hosts (chosen 1-1e-12, grid edge). Maltrail-only now 0 alerts (weak heuristic events cannot reach a strict single threshold); both-must-agree 0. Descriptive only, no labels; the "normal before the jump" stretch is assumed.
   - Next: per-layer alert handling (signature hits direct, heuristics/anomalies accumulate); packet-to-flow builder + NetFlow-feature flow model; CSV and live input adapters (Npcap is installed); learned fusion once labels exist; cross-dataset test.
7. Live path: Npcap installed (admin-only). `layered_nids/pipeline/live.py` (limited capture into `data/live`, same code path as replay) and `live_selftest.py` exist. LIVE SELF-TEST PASSED (2026-11-20, owner clicked the approval prompt): a real 267-packet capture on the Wi-Fi card -> our parser (all packets read back) -> the planted harmless HTTP request (directory-traversal pattern, Host trav.example) was in the capture -> the real Maltrail sensor in WSL2 flagged "potential directory traversal (suspicious)" attributed to the laptop's address; no other events on the real traffic. NOT tested: Kitsune on live traffic (needs ~55,000 clean packets to learn), a long capture, true streaming (capture-then-analyse only), the Maltrail sensor capturing live itself (WSL2 cannot see the Windows network card), detection accuracy. Note: Maltrail whitelists example.com, so the test uses the Host header trav.example.
8. CSV mode (2026-11-20): `layered_nids/csvmode/` (`python -m layered_nids.csvmode FILE.csv`), report `reports/csv/CSV_REPORT.md`, 14 tests. One robust entry point for flow files: tolerant reader (delimiter, BOM, padding, malformed lines, empty files), format detection (UNSW 34 features / NetFlow v2 40 features / Argus), the matching model (gradient boosting on the old project's preprocessor; NetFlow gradient boosting; models in `models/csv/`, SHA-256 checked) or, for Argus, scan rules (vertical, horizontal on uncommon ports, ICMP sweep; sliding 60 s windows). Unreadable numbers become NaN and are reported per row; every row is kept; out-of-range inputs raise a warning. MODEL DECISION (2026-11-21, owner's rule: equal accuracy, fewer false alarms wins): the UNSW model is now the owner's earlier-project model (`layered_nids/flowmodel`, code copied from `C:/ML project/final/ids`, retrained here, reproduces its shipped numbers). Same 46,830 cleaned test rows, 3% policy: earlier model 89.2% accuracy / 77.6% recall / 4.8% false alarms (AUC 0.954); our 34-feature model 89.1% / 81.4% / 7.0% (AUC 0.957); ours with the earlier project's cluster-grouped thresholds 89.1% / 79.0% / 5.7%. The earlier model's thresholds are better calibrated; ours ranks 0.003 AUC better. Details `reports/csv/CSV_REPORT.md`, `reports/flows/compare_earlier_project.json`. Measured through the CSV tool now: cleaned test 89.2% / 77.5% / 4.8% (1.5 s); raw official file (contains training copies, optimistic) 90.9% / 90.5% / 8.6%. Older 34-feature measurement: NetFlow 200,000 rows in 8 s; the five real Slips Argus samples processed with warnings, scans found in 3 of 5, none in the two files whose malware traffic is not scanning (rules cover scanning only). Limits: UNSW/NetFlow CSVs have no timeline (no timestamps, UNSW no addresses); NetFlow model trained on an easy dataset only; rule thresholds not tuned on labelled data.
9. Dashboard (2026-11-20): Maltrail's own web UI (MIT, unmodified copy in `tools/dashboard/`) runs on 127.0.0.1:8338 with our own password hash (`PASSWORD.txt`, local only). It shows the 8 real Maltrail events plus our 10 fused alerts from the Mirai run, converted by `pipeline/to_dashboard.py` (ports 0 / placeholder protocol because ours are host findings). Tested over HTTP: wrong password 401, right password 200, `/events` 18 lines, unauthenticated `/events` 401. Not tested: how it looks in a browser (owner to check). Viewer only: no capture, no updates. CSV upload added (2026-11-20): the "Upload CSV" button posts a flow file to `/nids/csv/upload` (signed-in only, custom header against forged requests, 100 MB cap, CSV mode run in a separate process, results downloadable, flagged hosts added to the event table). Tested over HTTP on a real Argus sample and 5,000 real UNSW rows (the raw first 5,000 rows are 95% attacks: 93% recall, 14% false alarms there, not a benchmark). Branding: only the name NIDS is shown in the UI; credits and licence texts live in the local, unpublished `credits/` folder.
10. README and report with honest limits.
Each step ends with passing tests and a short measured result.

## 11. Open decisions
- Course project or portfolio project (rubric not shared).
- Which of the real tools actually run on this machine (Docker/WSL2 install is the owner's call).
- Whether to keep the Streamlit dashboard in scope (deferred).

## 12. Honest limits to keep in every report
- UNSW-NB15 is 2015 lab traffic; official test is shifted from train; accuracy ~88-89%, not 90%.
- Replay is simulation, not live capture. Flow features cannot be reproduced exactly from packets.
- Fusion results are from a simulation built from real per-layer scores with independent layers; they are an upper bound, not a network result.
- The DGA benign data is a top-sites list; false alarms on real DNS are untested.
- The explanation module is our own Shapley estimator, not the `shap` package.
- HTTP service has no authentication; keep it on localhost. Model files are pickles; load only your own.
