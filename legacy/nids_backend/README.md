# NIDS backend (UNSW-NB15 flow features)

Machine-learning network intrusion detection: an ANN ensemble blended with a random forest, alert thresholds
set by a target false-alarm rate, a normal-traffic autoencoder for unseen attacks, Shapley-value explanations,
local recalibration, an alert log, a replay stream for the "real-time" demo, repeatable experiments, and a small
local HTTP service. The dashboard (Streamlit) is a separate, later step and talks to this backend through
`Detector` or the HTTP endpoints.

## Layout
```
nids/
  config.py       feature contract (34 inputs, in order) + validated Config (all hyper-parameters)
  data.py         load / validate / clean / split (no model code)
  features.py     the ONE preprocessing used for training and inference; input validation
  models.py       AnnEnsemble, ForestModel, NoveltyAutoencoder
  thresholds.py   alert policies (false-alarm target, F1, labelled recalibration)
  pipeline.py     train() and bundle save/load (atomic write + sha256 sidecar, schema version check)
  inference.py    Detector: predict, explain, policies, novelty, recalibrate, drift
  explain.py      Monte-Carlo Shapley values for one flow (own implementation, see below)
  evaluate.py     metrics, calibration (Brier/ECE), per-class rates, latency benchmark
  experiments.py  leave-one-attack-out, repeated-run stability, recalibration sample-size curve
  alerts.py       SQLite alert log (thread-safe) for the dashboard
  replay.py       stream stored flows in batches (simulated live traffic)
  drift.py        score-drift monitor (heuristic, see its docstring)
  report.py       builds reports/REPORT.md from the JSON reports
  server.py       stdlib JSON HTTP service (local only)
  cli.py          python -m nids ...
tests/            pytest suite
data/             official_training_175341.csv, official_testing_82332.csv
models/           nids_bundle.joblib + nids_bundle.meta.json
reports/          REPORT.md, evaluation / unseen-attack / stability / recalibration JSON, logs
```

## Use
```
pip install numpy pandas scikit-learn joblib            # pyarrow only if you read .parquet files
python -m nids train                                    # ~4 min on 2 CPUs
python -m nids evaluate --policy fpr_3pct [--novelty]   # official test set -> reports/evaluation_*.json
python -m nids predict   --input flows.csv --output out.csv
python -m nids explain   --input flows.csv --row 0      # which features pushed this flow's score
python -m nids recalibrate --input labelled_local.csv --fpr-cap 0.08
python -m nids unseen      [--attacks Generic,Worms]    # leave-one-attack-out (slow: retrains per attack)
python -m nids stability   --repeats 3                  # repeat training with other seeds/splits
python -m nids recal-curve                              # accuracy vs number of labelled local flows
python -m nids report                                   # -> reports/REPORT.md
python -m nids serve --port 8008 [--allow-admin]
python -m pytest tests                                  # add --fast to CLI commands for quick smoke runs
```
```python
from nids import Detector
from nids.replay import iter_replay
d = Detector.load()                                     # default policy: fpr_3pct
d.predict([{...34 fields...}])                          # -> list of dicts
d.explain({...34 fields...})                            # -> baseline, probability, per-feature contributions
d.set_policy("fpr_1pct"); d.use_novelty = True
d.recalibrate(labelled_df, fpr_cap=0.08)                # small labelled sample of YOUR traffic
for batch in iter_replay(test_df, batch_size=20, flows_per_second=50): d.predict(batch.to_dict("records"))
```

## HTTP API (127.0.0.1, no authentication)
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, policy, threshold, validation metrics, `admin_enabled` |
| GET | `/drift`, `/stats` | score-drift status; alert counters |
| GET | `/alerts?limit=50&min_severity=high` | latest alerts |
| POST | `/predict` | `{"flows":[...], "ids":[...]}` or one flow; a flow may carry its own `"id"` |
| POST | `/explain` | `{"flow":{...}, "n_permutations":16, "n_background":6, "top":10}` |
| POST | `/config`, `/recalibrate`, `/reset` | admin only (start with `--allow-admin`), otherwise 403 |

Limits: 5 MB per request, 10,000 flows per request, 30 s socket timeout. Errors are JSON `{"error": ...}` with 400/403/404/500.

## Design decisions
- **Leakage control.** Rows are judged on the 34 features: ambiguous feature vectors (same features, both
  Normal and Attack labels) are dropped, repeats collapsed, and test rows that also occur in training removed.
  The preprocessor is fitted on the training split only; thresholds come from a validation split; the official
  test set is used only by `evaluate` and the experiments.
- **Training = serving.** The fitted preprocessor travels inside the bundle; a test checks that `Detector`
  reproduces the training-time scores exactly.
- **Thresholds are policies**, named by the false-alarm rate they target on validation Normal traffic
  (`fpr_1pct`, `fpr_3pct`, `fpr_5pct`, `f1`). Default `fpr_3pct`.
- **Recalibration** re-tunes the threshold on labelled local flows (best accuracy under a false-alarm cap); with
  400+ flows 30% are held out and the change is only applied if it does not lose accuracy there.
- **Explanations.** The `shap` package was not available when this was built, so `explain.py` implements the same
  quantity directly: Shapley values estimated by switching features one at a time, in random order, from typical
  Normal flows to the flow being explained. Contributions always sum exactly to `probability - baseline` (tested);
  `std_err` shows the sampling noise of each value. More `n_permutations` = less noise, slower.
- **Failure behaviour.** Bad input raises `SchemaError` (HTTP 400, no stack traces leaked); unseen categories are
  ignored; negative counters are clipped; a corrupted or outdated bundle is refused.

## Known limits (read before quoting numbers)
- The data is the 34-feature mirror copy (9 original columns missing) of a 2015 dataset.
- Replay of stored flows is **not** live capture; these features are not computable from raw packets by a simple tool.
- The official test set is distributed differently from the training data; accuracy there is much lower than on a
  random split. The drift monitor did not detect this shift in our experiments.
- Bundles are pickles: load only files you trained. The checksum detects corruption, not a malicious author.
- The HTTP service has no authentication or TLS; keep it on 127.0.0.1.
