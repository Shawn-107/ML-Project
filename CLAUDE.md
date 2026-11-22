# Layered NIDS — project brief (hand-off to Claude Code)

Working title (short and changeable): **Layered NIDS**.
Owner: Shawn, MSc AI/ML, Christ University. Course: Advanced Machine Learning (MAI511-2).
Preferences: direct and brief, plain language, list formats over tables, no filler, answer exactly what was asked, fact-check and correct without deflecting.

## 0. Direction (owner decisions; the newest overrides anything below that conflicts)
- 2026-11-21: this is the owner's own project. The code base must be our own work and only the name **NIDS** is shown (UI, docs, code comments). No third-party code is in the repository.
  - Third-party tools were studied as references only. Their names, papers, licence texts and the earlier evaluation live in the local, git-ignored `credits/` folder (`credits/CREDITS.md`). Keep it with the project and include the credits wherever the work is submitted or shown; never present others' ideas as the owner's.
  - Local copies of the reference code for comparison are in `reference/` (git-ignored). Never copy their code into the repository; write from the published method and compare outputs.
  - No live capture tests: test on pcap files and CSV datasets only (the capture adapter `pipeline/live.py` exists but is not used).
  - When two models compete, use the one with equal or higher accuracy and fewer false alarms, compared on the SAME rows; record the head-to-head in `reports/`.
- Our earlier layers (`dga/`, `packets/`, `rules/`, `flows/`, `fusion/`) are kept as the ML contribution and as baselines for the newer ones.
- The owner's earlier project (`C:\ML project`, read only) is theirs and may be reused: its flow model is `layered_nids/flowmodel/`.

## 1. Goal
A network intrusion detection system where **the machine learning is the core**, combined at the alert level from independent layers:
- packet anomaly detector (unsupervised autoencoder ensemble on per-packet statistics);
- signature and heuristic rules (patterns, blocklists, scan/flood/beacon rules);
- DGA domain classifier; supervised flow classifiers (UNSW-NB15, NetFlow); learned and noisy-OR alert fusion;
- a dashboard with CSV upload.
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

## 3. Reference tools (studied, not shipped)
Three open-source network-security tools were evaluated earlier (what they do, inputs, CSV versus live, what runs here) and used to validate our own code. Their evaluation and the measured comparisons are in the local `credits/` folder.
- Measured on the same Mirai capture (764,137 packets, unlabelled): our first packet detector and the original reference detector flagged the same packets 99.84% of the time (score correlation 0.945); ours took 3.7 minutes against about 19.
- Newer detector (`anomaly/`) versus the original on injected synthetic attacks: `reports/anomaly/REPORT.md`.
- Licence rule: never copy reference code into the repository. (The earlier public history contained MIT-licensed copies with their notices; a history rewrite is pending the owner's yes.)

## 4. Architecture (each layer its own module, merged only at the alert level via `common/schema.py` Evidence)
- `packets/` first packet detector (baseline) and `anomaly/` newer packet detector (120 features incl. scan novelty and DNS entropy; streaming; SHA-256 save/load) -> Evidence layer `packets`.
- `rules/` first rules (baseline) and `signatures/` newer signature + heuristic layer -> Evidence layer `rules` (`detail["signature"]` True for pattern/blocklist hits).
- `dga/` domain classifier (n-gram LR, lexical MLP, char-CNN; family-held-out evaluation).
- `flows/` + `flowmodel/` flow classifiers (UNSW-NB15 and NetFlow; the owner's earlier model is used for UNSW CSV scoring) and `csvmode/` the robust CSV entry point.
- `fusion/` per-host evidence accumulation (noisy-OR with decay) + learned combiner; `pipeline/` glue (packet layer + rules layer -> fused alerts -> `*.alerts.jsonl` and `*.events.jsonl` for the dashboard).
- `dashboard/` own stdlib dashboard (localhost, sign-in, alerts/events tables, CSV upload).

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
- Packets: the `mirai.pcap` is unlabelled. The labelled UCI Mirai dataset (17.7 GB zip, CC BY 4.0) is NOT downloaded (TLS verification failed for archive.ics.uci.edu; not bypassed). Other datasets from the earlier project exist on disk (`C:\ML project\data`: CIC-IDS2018, NF-ToN-IoT, ToN-IoT, NSL-KDD).

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
5. Reference-tool evaluation (done earlier; local `credits/repo_evaluation.md`).
6. Combined pipeline on a pcap (`layered_nids/pipeline/`): packet detector -> per-host evidence, rules layer -> evidence, noisy-OR fusion with a per-layer policy (signature hits alert directly; anomalies and heuristics accumulate). Alert threshold tuned on the first half of the assumed-normal stretch before the jump (minute 74.3) and checked on the second half. Unlabelled Mirai capture, descriptive only. Latest run (`reports/pipeline/mirai.json`): packet layer alone 50 alerts on 7 hosts after the jump and 0 before; with the rules layer 67 alerts, 8 of them before the jump (3 hosts, minutes 47.6, 66.6 and 69.6; re-run 2026-11-22 with the newer detector default and rules): possible false alerts or real early activity, cannot be told without labels.
7. Live capture: not used (owner decision 2026-11-21). The adapter exists; do not run it.
8. CSV mode (`layered_nids/csvmode/`, `reports/csv/CSV_REPORT.md`, `reports/csv/dataset_tests.json`): tolerant reader, format detection (UNSW 34 / NetFlow 40 / Argus), models, scan rules for Argus files. UNSW cleaned test with the owner's earlier flow model: 89.2% accuracy, 77.5% recall, 4.8% false alarms (the gradient-boosting model had 81.4% recall at 7.0% false alarms; the owner's rule picks the one with fewer false alarms at equal-or-higher accuracy).
9. Dashboard: own code (`layered_nids/dashboard/`), see item 10.
10. OWN-CODE REBUILD (2026-11-21, owner decision: only NIDS shown, code should be the owner's own; no live test; datasets for CSV tests):
   - `layered_nids/anomaly/` = own packet detector v2 (120 features incl. scan novelty and DNS entropy; streaming; save/load with SHA-256) + injection benchmark with labelled synthetic attacks (`reports/anomaly/REPORT.md`). Test seeds, 3 model seeds x 3 injections, recall at equal 1% false alarms vs our first version vs the reference detector: horizontal scan 0.98 / 0.14 / 0.15; floods, port scan, bulk upload about 1.0 for all; DNS burst 0.97 / 0.88 / 0.88 (fixed 2026-11-22 by mixed scaling, was 0.66); slow beacon unseen by all (episodes 0% / 22% / 33%). False alarms on uninvolved hosts 1.65% / 2.11% / 3.12% (target 1%, all above it); false-alert windows per hour 65.1 / 60.7 / 95.8. Synthetic attacks only. Several variants tried and rejected, listed in the report. Optional two-view profile `views=2`; streaming about 1,800 packets/s.
   - `layered_nids/signatures/` = own signature + heuristic layer (10 web-attack patterns on the request line, blocklists, scan/flood/ARP/DNS/beacon rules); 7 of 9 injected attack kinds found 3/3, bulk upload not found (`reports/signatures/REPORT.md`).
   - `layered_nids/dashboard/` = own stdlib dashboard (sign-in with PBKDF2 + session cookie, localhost only, alerts/events tables, CSV upload) that also serves the owner's earlier-project React app at `/app/` (their built files) over `/api/*` backed by `layered_nids/flowmodel`. Run `python -m layered_nids.dashboard --init-user NAME` then `python -m layered_nids.dashboard`. 16 tests; checked over HTTP and in a real browser engine (headless Edge; see `reports/acceptance/ACCEPTANCE.md`). Redesigned 2026-11-22: overview charts, host map, host timeline panel, live refresh, command palette (Ctrl+K), animated sign-in page, light and dark themes. Not tested: typing a password into the form in a browser.
   - `layered_nids/flowmodel/` = the owner's earlier flow model (see item 8). The third-party copies were removed from the working tree on 2026-11-21 (kept locally under `reference/`).
11. README and final report (`reports/FINAL_REPORT.md`): written and brought up to date on 2026-11-22; re-check every number against the linked reports before submitting.
Each step ends with passing tests and a short measured result.

## 11. Open decisions
- Course project or portfolio project (rubric not shared).
- Whether to replace the public GitHub history (it still contains earlier MIT-licensed third-party copies with their notices). Needs the owner's explicit yes.
- Choice of a licence for the owner's own code.
- Whether to keep the Streamlit dashboard in scope (deferred).

## 12. Honest limits to keep in every report
- UNSW-NB15 is 2015 lab traffic; official test is shifted from train; accuracy ~88-89%, not 90%.
- Replay is simulation, not live capture. Flow features cannot be reproduced exactly from packets.
- Fusion results are from a simulation built from real per-layer scores with independent layers; they are an upper bound, not a network result.
- The DGA benign data is a top-sites list; false alarms on real DNS are untested.
- The explanation module is our own Shapley estimator, not the `shap` package.
- HTTP service has no authentication; keep it on localhost. Model files are pickles; load only your own.
