# Dashboard: the NIDS dashboard (Maltrail's web UI, rebranded), on localhost

`server.py`, `core/`, `html/` and `LICENSE` come from Maltrail 3.6 (MIT, Miroslav Stampar); the MIT licence and copyright notices are kept. Our changes: visible names changed to NIDS (page title, header logo text, footer, sign-in text, export file names, notification title; the footer still credits Maltrail), and `server.conf`.

Start (from this folder):  `..\..\.venv313\Scripts\python.exe server.py -c server.conf`  then open http://127.0.0.1:8338  (user `shawn`, password in `PASSWORD.txt`).
Stop: Ctrl-C in that window, or end the python process.

Safety settings in `server.conf`: listens on 127.0.0.1 only, own random password (stored as a PBKDF2 hash; the shipped `changeme!` is not used),
no trail updates, no internet access from the server, no packet capture (this is the viewer only).

What it shows: every `*.log` file in `logs/`, in Maltrail's event format.
- 8 events: the real Maltrail sensor on the Mirai capture (sensor name `nids-maltrail-sensor`, reference `(heuristic)`).
- Our alerts: written by `python -m layered_nids.pipeline.to_maltrail --alerts reports/pipeline/mirai.alerts.jsonl --out tools/maltrail_ui/logs`
  (sensor `nids`, reference `(nids)`). They are ML findings about a HOST, so ports are 0, destination 0.0.0.0 and the protocol column is a placeholder.
- Both are from the unlabelled Mirai capture, dated 2018-10-25 (the capture's own date). Not live data.
