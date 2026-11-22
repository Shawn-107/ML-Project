# NIDS

A network intrusion detection system where machine learning is the core, combined at the alert level from independent layers:
- a packet anomaly detector (unsupervised autoencoder ensemble on per-packet statistics);
- signature and heuristic rules (web-attack patterns, blocklists, scan / flood / beacon / DNS rules);
- a DGA domain classifier and supervised flow classifiers (UNSW-NB15, NetFlow);
- per-host evidence fusion (noisy-OR with decay, and a learned combiner);
- a dashboard with CSV upload.

MSc AI/ML coursework (Advanced Machine Learning, MAI511-2). This repository holds the code and the trained flow models only; the project brief, measured reports and data are kept locally by the author.

## Set up from a fresh clone (Windows, Python 3.13)
1. Environment: follow the build lines at the top of `requirements-313.txt` (a Python 3.13 venv, `pip install -r requirements-313.txt`, then `pip install --no-deps -e . -e legacy/nids_backend`).
2. Data: put the files listed in `data/SOURCES.md` where it says (UNSW-NB15 files in `data/unsw/` are required; the Mirai capture, DGA list, NetFlow file and an indicator file are optional).
3. Check what is missing: `python -m layered_nids.setup_check` prints each missing piece and the command that fixes it.
4. Models: trained flow models are included in `models/csv/` (each has a SHA-256 in its `.meta.json`; they are pickles, so only load files from a source you trust). To retrain them: `python -m layered_nids.csvmode.models train` (add `--which unsw34` for only the UNSW flow model).
5. Tests: `python -m pytest tests` (tests that need data you did not supply are skipped or fail with a clear message from step 3).

## Run
- Pipeline on a pcap: `python -m layered_nids.pipeline.run --pcap data/mirai/mirai.pcap --rules`  (writes `reports/pipeline/<name>.json`, `.alerts.jsonl`, `.events.jsonl`; an indicator file in `data/trails/` is used automatically)
- CSV flow files: `python -m layered_nids.csvmode FILE.csv`
- Dashboard: `python -m layered_nids.dashboard --init-user NAME` (prints where the random password is stored), then
  `python -m layered_nids.dashboard --data reports/pipeline --port 8338` and open http://127.0.0.1:8338 (localhost only). Press Ctrl+K in the page for the command palette.
- Packet detector: the default scaling is chosen automatically (mixed scaling with at least 20,000 training packets, robust log scaling below that). `PacketAnomalyDetector(views=2)` combines two scalings (more sensitive, more false-alert windows); see `reports/anomaly/REPORT.md`.

## Not in this repository (by design)
- Reports and the project brief (kept locally), datasets (see `data/SOURCES.md` for where to get them), score caches, the credits and reference material, virtual environments, passwords and local configuration.
- The packet detector is trained per run from a capture and is not shipped as a file.

## Honest limits
Results on the unlabelled Mirai capture are descriptive (no labels, so no recall is measured there). Packet-detector and rule benchmarks use synthetic injected attacks, which measure whether a behaviour is noticed, not accuracy on real attacks. Fusion results come from a simulation. UNSW-NB15 is 2015 lab traffic and its official test is shifted from training. No live capture was tested (dataset tests only). See `CLAUDE.md` section 12.
