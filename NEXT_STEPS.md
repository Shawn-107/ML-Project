# Next steps (written 2026-11-19)

## Where things stand
- Tests: 71 passed in the new project, 54 passed + 1 skipped in the legacy backend. No background jobs running from this work.
- Real-tools report: `reports/repo_evaluation.md` (Kitsune-py was run; Maltrail was downloaded but blocked by Smart App Control; Slips not run).
- Results: `reports/dga/REPORT.md`, `reports/flows/*.json` (improvements, adversarial, loao), `reports/fusion/ablation_*.json` (simulation).
- Project rules and direction: `CLAUDE.md`.

## Your side (all optional until you want that feature)
1. Safe housekeeping, say "yes" and I do it:
   - delete the unsigned Maltrail download (a `.exe` and a `.tar.gz` in the session scratch folder),
   - remove the unused `pypdf` package from `.venv`.
2. Stop the old project's service if you are not using it. It listens on `127.0.0.1:8008` with no login:
   `Get-NetTCPConnection -LocalPort 8008 | Select OwningProcess` then `Stop-Process -Id <that id>`.
3. Decisions (reply in one line each):
   - May the project be released under GPL-2.0? (Only matters if Slips code is copied in. "No" is fine: we keep Slips as a separate program.)
   - Build the flow model on NetFlow-style features so it works on CSV and live? (My recommendation: yes.)
   - For running real Maltrail and Slips: WSL2 or a Linux VM?
4. Only when you want those features (these need admin rights, you run them):
   - Real Maltrail/Slips: install WSL2 (`wsl --install` in an admin PowerShell, then reboot) or set up a Linux VM.
   - Live capture on Windows: install Npcap from npcap.com. Best done inside an isolated VM.
   - Labelled packet data: download the Mirai part of the UCI Kitsune dataset (DOI 10.24432/C5D90Q) in a browser into `data/kitsune/`.

## My side, when you say "go" (all offline, no admin, no drivers)
1. Housekeeping above.
2. One program with three input modes (CSV, pcap, live-ready): Kitsune's real code (patched for NumPy 2, unsafe shell call replaced),
   Maltrail-style rules and lists, per-host evidence, our DGA and flow models, learned alert combiner.
3. Packet-to-flow builder and a flow model on NetFlow-style features (data already on disk: `C:\ML project\data\crossds\nfton`).
4. Cross-dataset test (train on one dataset, test on another on disk).
5. Fair fusion baseline (accumulator given the locally measured false-alarm rate), tests, one report with honest limits.

## Safety rules we keep
- Do not turn off Smart App Control. Do not bypass Application Control.
- Change Maltrail's default login (admin/changeme!) and keep its web page on localhost before ever running it.
- Keep services on 127.0.0.1. Load model pickles only if you trained them.
- Do not run any downloaded program before reading what it does.

## WSL2 status (2026-11-20)
- Official `wsl --install` was run with the owner's UAC approval and exited 0. WSL package 3.0.5.0, kernel 6.18.58.1 installed.
- No Linux distribution yet. Windows has a REBOOT PENDING; `wsl --status` said WSL2 cannot start until the
  "Virtual Machine Platform" component is enabled (expected to finish after the reboot).
- Owner: restart the laptop. Afterwards an Ubuntu window should open to finish setup (choose a Linux username and a strong,
  different password). If no window opens, run `wsl --install -d Ubuntu` in an admin PowerShell.
- If after the restart `wsl --status` still says virtualization is not enabled, turn on virtualization (Intel VT-x) in the
  laptop's BIOS/firmware settings. That is the only thing that could still block WSL2.
- Then (Claude): `wsl --status`; read Microsoft's wsl.conf page and turn off the Windows-drive mount and Windows interop;
  download the Linux Maltrail sensor release, verify its SHA-256, run it on a COPY of the Mirai pcap; keep its web page on localhost.

## WSL2 update (2026-11-20, later): DONE and hardened
- Ubuntu 26.04.1 LTS installed in WSL2 (kernel 6.18.58.1). Owner created the Linux user `shawn` (sudo group, default login).
- `/etc/wsl.conf` now has `[automount] enabled=false`, `[interop] enabled=false`, `appendWindowsPath=false` (settings and restart
  method confirmed on Microsoft's wsl-config page). Backup of the original: `/etc/wsl.conf.bak` inside Ubuntu.
- Verified after `wsl --shutdown`: `C:` is not mounted (`/mnt/c` empty), no Windows folders on PATH, `cmd.exe` not reachable, and a
  test that tried to make a copied Windows `cmd.exe` write a file into the Windows temp folder FAILED (no file written).
  The WSLInterop handler is still listed by the kernel, but launches fail, so this is cosmetic.
- Moving files in and out: from Windows use the share `\wsl.localhost\Ubuntu\home\shawn\` (works with the drive mount off).
- Next (Claude, when told to continue): inside Ubuntu check curl/sha256sum/tar; download the Linux Maltrail sensor release, verify its
  SHA-256 against the published file, read its README, run it OFFLINE on a copy of the Mirai pcap with a localhost-only config.

## Maltrail in WSL2 (2026-11-20): set up and run offline
- Location inside Ubuntu: `~/work/maltrail` (sensor binary in `bin/`, config `maltrail.conf` localhost-only, trails in `state/`, events in `logs/`,
  pcaps in `pcaps/`, matching 3.6 source in `src/`). Release and trail-set SHA-256 verified against the published files.
- Re-run self-test: `wsl -d Ubuntu -- sh /home/shawn/work/run_detect_test.sh` (expected: 27/27 detection(s) fired).
- Re-run on a pcap: put it in `\wsl.localhost\Ubuntu\home\shawn\work\maltrail\pcaps\`, then run `./bin/maltrail-sensor -c maltrail.conf --offline -r pcaps/<file>`
  from `~/work/maltrail` with `MALTRAIL_ROOT` set to that folder (see `run_mirai.sh`). Events: `logs/<date>.log`.
- Result on Mirai: 8 events (all "periodic beaconing", minutes 2-8, none during Kitsune's jump at minute 74). See reports/repo_evaluation.md.
- Do NOT run the Maltrail web server with its default login; it is not needed for offline work.
- Next: map both tools' outputs to Evidence per host and time window; compare on the same pcaps; then the NetFlow flow model.

## Npcap (2026-11-20): installed and restricted
- npcap-1.90.exe from npcap.com, signature valid (Nmap Software LLC). Installed with admin-only ON, 802.11 OFF, WinPcap mode OFF.
- Checked: driver 1.90 signed by Microsoft's hardware publisher and running; AdminOnly=1; no wpcap.dll in System32; no restart pending.
- Test: a normal (non-admin) scapy capture attempt shows a UAC approval prompt; the owner approved twice (1 packet each, not stored).
  A refusal when clicking No was not tested (the owner clicked Yes both times); declining a UAC prompt simply fails the capture.
- Rules: only click Yes on a prompt you triggered; never capture on shared/public Wi-Fi; keep captures local and delete after use.
- To remove: Settings > Apps > Installed apps > Npcap > Uninstall.
- Next: build the live capture adapter (limited duration/size, local folder), feed Kitsune wrapper + Maltrail-style rules + DGA; first on pcap replay.

## Combined pipeline built (2026-11-20)
- Code: `layered_nids/pipeline/` (kitsune_layer.py, maltrail_layer.py, run.py). Tests: tests/test_pipeline.py (7). Whole suite: 81 passed in `.venv313`.
- Run: `.venv313\Scripts\python.exe -m layered_nids.pipeline.run --pcap data\kitsune\mirai.pcap --maltrail --name mirai`
  (Kitsune scores are cached in `reports/pipeline/mirai_kitsune_scores.npz`; a fresh scoring run takes ~21 minutes).
- Result and limits: `reports/pipeline/REPORT.md` (descriptive: no labels).
- Next: host attribution fix, alert tuning on a normal stretch, then the live capture adapter (Npcap installed, admin-only) and the NetFlow flow model.

## Pipeline fixes (2026-11-20, later)
- Host attribution fixed (`layered_nids/common/hosts.py`, used by both layers). Cached Kitsune scores are reused; only the host column is rebuilt.
- Alert threshold is now tuned on the first half of the assumed-normal stretch before the jump and checked on the second half (`--alert-p` fixes it by hand, `--budget` sets the allowed false alerts).
- Results and the full curve: `reports/pipeline/REPORT.md`. Whole suite: 83 passed.
- Next: per-layer alert handling (a single threshold suppresses Maltrail's weaker evidence), then the live adapter and the NetFlow flow model.

## Live self-test PASSED (2026-11-20)
- Run: `.venv313\Scripts\python.exe -m layered_nids.pipeline.live_selftest --seconds 20` (a Windows approval prompt appears: click Yes only if you started it).
- Real Wi-Fi capture (267 packets) -> our parser read all back -> planted harmless request present -> real Maltrail sensor (WSL2) flagged "potential directory traversal (suspicious)".
- Bugs found and fixed by running it: (1) the request was sent before the capture had started (approval prompt delay): now waits for scapy's started signal; (2) Maltrail
  whitelists example.com, so the test uses the Host header trav.example.
- The capture file is deleted after the test (it contains your traffic); use --keep only if you want it.
- Not yet tested live: Kitsune (needs ~55,000 clean packets), a long capture, true streaming.

## CSV mode built (2026-11-20)
- `python -m layered_nids.csvmode FILE.csv` (details and limits: `reports/csv/CSV_REPORT.md`). Models: `python -m layered_nids.csvmode.models train` (about 25 s).
- Not yet built: domain-name (DGA) columns, a CSV timeline for the UNSW/NetFlow files (they have no timestamps), per-host fusion from CSV evidence.
- The one-hour live Kitsune test (`python -m layered_nids.pipeline.live_kitsune_test --minutes 60 --plant-at 50`) is written and unit-tested but NOT started: it needs one
  Windows approval prompt, the laptop awake and used normally for the hour, and it deletes the capture afterwards.
