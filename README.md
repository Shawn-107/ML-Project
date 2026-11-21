# NIDS

A network intrusion detection system where machine learning is the core, combined at the alert level from independent layers:
- a packet anomaly detector (unsupervised autoencoder ensemble on per-packet statistics);
- signature and heuristic rules (web-attack patterns, blocklists, scan / flood / beacon / DNS rules);
- a DGA domain classifier and supervised flow classifiers (UNSW-NB15, NetFlow);
- per-host evidence fusion (noisy-OR with decay, and a learned combiner);
- a dashboard with CSV upload.

MSc AI/ML coursework (Advanced Machine Learning, MAI511-2). Full brief, measured results and honest limits: `CLAUDE.md`. Final report: `reports/FINAL_REPORT.md`.

## Run (Windows, Python 3.13 venv with `requirements-313.txt`)
- Tests: `python -m pytest tests`
- Pipeline on a pcap: `python -m layered_nids.pipeline.run --pcap data/mirai/mirai.pcap --rules`  (writes `reports/pipeline/<name>.json`, `.alerts.jsonl`, `.events.jsonl`)
- CSV flow files: `python -m layered_nids.csvmode FILE.csv`  (train the models first: `python -m layered_nids.csvmode.models train`)
- Dashboard: `python -m layered_nids.dashboard --init-user NAME` (prints where the random password is stored), then
  `python -m layered_nids.dashboard --data reports/pipeline --port 8338` and open http://127.0.0.1:8338 (localhost only).

## Not in this repository (by design)
- Datasets (UNSW-NB15, DGA lists, the Mirai capture): see `data/SOURCES.md`.
- Trained model files (pickles; rebuild with the training commands), score caches, virtual environments, passwords and local configuration.

## Honest limits
Results on the unlabelled Mirai capture are descriptive (no labels, so no recall is measured there). Packet-detector and rule benchmarks use synthetic injected attacks, which measure whether a behaviour is noticed, not accuracy on real attacks. Fusion results come from a simulation. UNSW-NB15 is 2015 lab traffic and its official test is shifted from training. No live capture was tested (dataset tests only). See `CLAUDE.md` section 12.
