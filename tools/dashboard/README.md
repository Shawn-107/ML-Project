# Dashboard: the NIDS dashboard (Maltrail's web UI, rebranded), on localhost

`server.py`, `core/`, `html/` and `LICENSE` come from Maltrail 3.6 (MIT, Miroslav Stampar); the MIT licence and copyright notices are kept. Our changes: visible names changed to NIDS (page title, header logo text, footer, sign-in text, export file names, notification title; the footer still credits Maltrail), and `server.conf`.

Start (from this folder):  `..\..\.venv313\Scripts\python.exe server.py -c server.conf`  then open http://127.0.0.1:8338  (user `shawn`, password in `PASSWORD.txt`).
Stop: Ctrl-C in that window, or end the python process.

Safety settings in `server.conf`: listens on 127.0.0.1 only, own random password (stored as a PBKDF2 hash; the shipped `changeme!` is not used),
no trail updates, no internet access from the server, no packet capture (this is the viewer only).

What it shows: every `*.log` file in `logs/`, in Maltrail's event format.
- 8 events: the real Maltrail sensor on the Mirai capture (sensor name `nids-sensor`, reference `(heuristic)`).
- Our alerts: written by `python -m layered_nids.pipeline.to_dashboard --alerts reports/pipeline/mirai.alerts.jsonl --out tools/dashboard/logs`
  (sensor `nids`, reference `(nids)`). They are ML findings about a HOST, so ports are 0, destination 0.0.0.0 and the protocol column is a placeholder.
- Both are from the unlabelled Mirai capture, dated 2018-10-25 (the capture's own date). Not live data.

## CSV upload
The "Upload CSV" button (top right) sends a UNSW-NB15, NetFlow v2 or Argus flow file to the CSV analyser (`nids_csv.py`, up to 100 MB, signed-in users only). It shows rows, flagged share, warnings and lets you download the scored file; flagged hosts are added to the event table (sensor `nids-csv`).
Uploaded files are deleted after analysis; results stay in `uploads/` (ignored by git, pruned after 7 days).
