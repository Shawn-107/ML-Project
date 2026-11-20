# CSV mode: one robust tool for flow files

Run: `python -m layered_nids.csvmode FILE.csv [--out OUT.csv] [--policy fpr_1pct|fpr_3pct] [--schema unsw34|netflow_v2|argus]`
Code: `layered_nids/csvmode/` (reader, schemas, models, flowrules, run). Tests: `tests/test_csvmode.py` (14).
Models are trained with `python -m layered_nids.csvmode.models train` and saved in `models/csv/` with a SHA-256 checked on load.

## What it combines
- The earlier UNSW-NB15 project's 34-feature preprocessor, with a gradient-boosting model on top (validated improvement over the old ANN+forest blend).
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
- UNSW cleaned official test set (46,830 flows) through the CSV tool: at the 3% policy accuracy 89.1%, recall 81.4%, false alarms 7.0%; at the 1% policy accuracy 89.0%,
  recall 72.0%, false alarms 2.4%. This matches the earlier gradient-boosting measurement (89.1% / 81.6% / 7.1%). 2 seconds.
- NetFlow file, first 200,000 rows: 8 seconds, 59 rows flagged with bad values, 0.03% of rows far outside the training range. Accuracy NOT measured on this slice (hosts overlap training).
- The five real Slips sample files (Argus format, one tab-separated, one missing a column, one with a 1970 clock): all processed with warnings, none crashed. Scan findings:
  port-scan file: 1 vertical scan + 1 ICMP sweep (from 10.8.0.69); "mixed" file: 1 horizontal scan; "malicious" file 4: 2 horizontal scans; tab-separated mixed file: none;
  the 6,751-row "malicious" file: none.

## Limits (read before relying on it)
- The Argus rules only detect scanning. The two files with malware traffic that is not scanning produce NO findings; there is no ML for Argus format. Recall of the rules on those files is zero.
  The rule thresholds (15 ports, 15 hosts, 60 s) are defaults, not measured on labelled data.
- UNSW model: trained on 2015 lab traffic; the official test is shifted from training, so expect about 7% false alarms at the 3% policy (a 3% target).
- NetFlow model: trained on NF-ToN-IoT only. That dataset is easy (AUC 1.0 even on unseen hosts), so its scores say little about other networks; no cross-dataset test exists.
- The UNSW and NetFlow CSVs carry no timestamps (UNSW has no addresses at all), so there is no per-host timeline for them; only the Argus format has host and time.
- No domain-name (DGA) column support yet; the DGA model works on lists of names separately.
- One seed; model files are pickles (load only your own, the SHA-256 check detects accidental change, not a malicious author).
