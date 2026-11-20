# Layered NIDS

A network intrusion detection system built around machine learning, combined at the alert level from three kinds of evidence:
the real Maltrail sensor (blocklists + heuristics), the real Kitsune autoencoder ensemble (vendored, MIT), and our own ML layers
(DGA domain classifier, UNSW-NB15 / NetFlow flow classifiers, learned fusion). MSc AI/ML coursework (Advanced Machine Learning, MAI511-2).

Final report: `reports/FINAL_REPORT.md`. Project brief and details: `CLAUDE.md` (full project brief, measured results and honest limits); reports in `reports/` (`repo_evaluation.md`, `pipeline/REPORT.md`, `dga/`, `flows/`, `csv/`, `fusion/`).

## Run (Windows, Python 3.13 venv with `requirements-313.txt`)
- Tests: `python -m pytest tests`
- Pipeline on a pcap: `python -m layered_nids.pipeline.run --pcap data/kitsune/mirai.pcap --maltrail`  (the Maltrail sensor runs in WSL2)
- CSV flow files: `python -m layered_nids.csvmode FILE.csv`  (train the models first: `python -m layered_nids.csvmode.models train`)
- Dashboard: `tools/maltrail_ui` (Maltrail's own web UI, localhost only; create your own `server.conf` from `server.conf.example`)

## Not in this repository (by design)
- Datasets (UNSW-NB15, DGA lists, the Kitsune Mirai capture): see `data/SOURCES.md` for where to get them.
- Trained model files (pickles; rebuild with the training commands), score caches, the three cloned reference repos, virtual environments.
- Any password or local configuration.

## Honest limits
Results on the unlabelled Mirai capture are descriptive (no labels, so no recall measured). Fusion results come from a simulation. UNSW-NB15 is 2015 lab traffic and the official test is shifted from training (accuracy about 89%, not 90%). Live capture: only a self-test and a capture-then-analyse path were tested; Kitsune on live traffic is untested. See `CLAUDE.md` section 12.

## Third-party code and credits
- Maltrail (Miroslav Stampar), MIT: sensor run as a separate program; web UI copied unmodified into `tools/maltrail_ui/` with its LICENSE. Its trail data has separate terms (free for research and teaching).
- Kitsune-py (Yisroel Mirsky), MIT: vendored in `layered_nids/realtools/kitsune/` with its LICENSE; every change is listed in its `NOTICE.md`. Paper: Mirsky et al., "Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection", NDSS 2018.
- Slips (Stratosphere Laboratory), GPL-2.0: studied only; no code copied or imported.
- UNSW-NB15 (Moustafa & Slay), NF-ToN-IoT (UQ), Mendeley DGA dataset (CC BY 4.0): used for training and evaluation; not redistributed here.
- No licence has been chosen for our own code yet, so by default all rights are reserved.
