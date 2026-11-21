# CSV mode: one robust tool for flow files

Run: `python -m layered_nids.csvmode FILE.csv [--out OUT.csv] [--policy fpr_1pct|fpr_3pct] [--schema unsw34|netflow_v2|argus]`
Code: `layered_nids/csvmode/` (reader, schemas, models, flowrules, run). Tests: `tests/test_csvmode.py` (14).
Models are trained with `python -m layered_nids.csvmode.models train` and saved in `models/csv/` with a SHA-256 checked on load.

## What it combines
- The flow model of the owner's earlier project (`layered_nids/flowmodel`): gradient boosting on 11 UNSW-NB15 inputs, thresholds set on a validation split of whole K-Means clusters. It replaced our 34-feature gradient boosting after a head-to-head on the same rows (see below).
- The NetFlow-v2 model (40 features, NF-ToN-IoT-v2, split by source host).
- Host/time rules for Argus-format flow files, the format the Slips project reads (scan rules; no ML, since nothing labelled exists for it here).

## Robustness (each point has a test)
- Reads comma, tab, semicolon and pipe separated files; strips an invisible byte-order mark and padding spaces; skips malformed lines and COUNTS them.
- Clear errors (no crash) for empty files, header-only files, missing files, unknown formats (names the closest format and the missing columns), unknown policies,
  untrained or tampered model files.
- Unreadable, empty or infinite numbers become NaN (the models handle them) and are reported per row; a row with half or more of its features missing is marked `unusable`.
  Every input row appears in the output with a `row_status` (ok / warn / unusable) and notes. Nothing is dropped silently.
- A file that looks unlike the training data (more than 10% of rows with 3+ values far outside the training range) produces a warning that scores may be unreliable.
- Big files are read in chunks; chunk size does not change the answers (tested). Rule mode refuses files over 3 million rows and says to split by time.
- Real-data regressions found and fixed while building it: header-only files slipped through; a model pickled from `__main__` could not be loaded elsewhere;
  fixed 60-second windows missed a slow scan that straddled a minute boundary (now sliding windows); a real file had an ICMP ping sweep that the rules did not cover.

## Results on real data (measured here)
- UNSW cleaned official test set (46,830 flows, rows also in training removed) through the CSV tool, 1.5 s: 3% policy accuracy 89.2%, recall 77.5%, false alarms 4.8%; 1% policy accuracy 86.3%, recall 60.8%, false alarms 0.6% (`dataset_tests.json`).
- UNSW raw official test file as downloaded (82,332 rows, includes copies of training rows, so optimistic): 3% policy accuracy 90.9%, recall 90.5%, false alarms 8.6%; 1% policy 88.1% / 78.9% / 0.6%. Do not quote this as the model's accuracy.
- Model choice (same 46,830 rows, `reports/flows/compare_earlier_project.json`, `hybrid_grouped_thresholds.json`):
  - earlier project's model: AUC 0.954; 3% policy 89.2% accuracy / 77.6% recall / 4.8% false alarms; 1% policy 86.3% / 60.8% / 0.6%.
  - our 34-feature gradient boosting: AUC 0.957; 3% policy 89.1% / 81.4% / 7.0%; 1% policy 89.0% / 72.0% / 2.4%.
  - our model with the earlier project's cluster-grouped thresholds (3 seeds): AUC 0.954; 3% policy 89.1% / 79.0% / 5.7%; 1% policy 88.4% / 68.6% / 1.4%.
  - At the same false-alarm rate (thresholds chosen on the test itself, equal for both) ours ranks slightly better (AUC +0.003, interval 0.0025-0.0043 from row resampling, optimistic) but the earlier model's own thresholds are better calibrated (4.8% vs 7.0% false alarms at a 3% target). Rule used: equal accuracy, fewer false alarms wins, so the earlier model is used. Trade-off: about 4 points less recall at the 3% policy.
- NetFlow file, first 200,000 rows: 8 seconds, 59 rows flagged with bad values, 0.03% of rows far outside the training range. Accuracy NOT measured on this slice (hosts overlap training).
- The five real Slips sample files (Argus format): all processed with warnings, none crashed. Scan findings: port-scan file 1 vertical scan + 1 ICMP sweep; "mixed" file 1 horizontal scan; "malicious" file 4: 2 horizontal scans; tab-separated mixed file none; the 6,751-row "malicious" file none.

## Limits (read before relying on it)
- The Argus rules only detect scanning. The two files with malware traffic that is not scanning produce NO findings; there is no ML for Argus format. Recall of the rules on those files is zero.
  The rule thresholds (15 ports, 15 hosts, 60 s) are defaults, not measured on labelled data.
- UNSW model: trained on 2015 lab traffic; the official test is shifted from training, so false alarms land above the target (4.8% at a 3% target on the cleaned test).
- NetFlow model: trained on NF-ToN-IoT only. That dataset is easy (AUC 1.0 even on unseen hosts), so its scores say little about other networks; no cross-dataset test exists.
- The UNSW and NetFlow CSVs carry no timestamps (UNSW has no addresses at all), so there is no per-host timeline for them; only the Argus format has host and time.
- No domain-name (DGA) column support yet; the DGA model works on lists of names separately.
- One seed; model files are pickles (load only your own, the SHA-256 check detects accidental change, not a malicious author).
