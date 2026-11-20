# Maltrail, Kitsune-py, Slips: what was tested, what each does, what can be combined

Date: 2026-11-20. Machine: Windows 11 Home laptop, Python 3.15 venv, Smart App Control ON; WSL2 with Ubuntu 26.04 installed later the same day (hardened); no Docker/Npcap/Rust/Zeek/Redis.
TESTED = run here. DOCS = read from the repo's docs/source, not run. Repos are in `reference/` (unmodified); tests ran in throwaway copies.

## 1. Testing status
- Kitsune-py: TESTED. Ran on the whole Mirai capture (764,137 packets).
- Maltrail: TESTED (offline). On Windows the sensor `.exe` is unsigned and Smart App Control blocked it (not bypassed), so it was run as the official Linux build inside WSL2 (Ubuntu 26.04, hardened: no Windows drives, no Windows programs). Checksums of the release and the trail set matched. Its own detection self-test passed 27/27, and it replayed the whole Mirai capture. LIVE capture was NOT tested (WSL2 cannot see the Windows network card; needs Npcap on Windows or a Linux machine). Its Python side also ran on Windows (own test suite, 36 of 43 files pass; the 6 failures are Windows-only differences).
- Slips: NOT RUN. Needs Docker or Linux with Zeek and Redis. Read its docs and its shipped sample data only.

## 2. Maltrail (stamparm/maltrail 3.6, MIT engine)
Use case: cheap, fast detection of KNOWN-bad traffic: blocklisted domains/IPs/URLs/User-Agents/JA3, plus heuristics (scanning, DNS floods, DGA-like names, web attacks). No machine learning.
- Parts: Rust sensor (libpcap) + Python server and web UI (standard library only).
- Input: live interface, or pcap (`--offline -r file.pcap`). No CSV input.
- Output: event-log lines `"<time>" <sensor> <src_ip> <src_port> <dst_ip> <dst_port> <proto> <type> <trail> <info> <ref>`; the parser/format is exercised by its own tests.
- TESTED here:
  - Download of the Windows sensor release, SHA-256 matched the published value (proves the file is intact, not that the code is safe).
  - Python suite, one process per file, in a copy: 36 pass, 6 fail, 1 skipped (`test_detect_test` needs the sensor). The failures are platform-only: they call Linux-only functions (`os.geteuid`, `time.tzset`), hit Windows file locking, expect LF line endings (`\r\n` found), need git history (my copy had none), or are timing-sensitive. None showed a detection-logic fault.
  - Linux sensor in WSL2 (release 3.6 x86-64 glibc build; release and trail-set SHA-256 matched the published values; `-T` loaded 1,860,474 trails with 0 malformed; the only failed check was live-capture privileges, which offline replay does not need):
    - Own detection self-test (`server.py --detect-test`, crafted malicious traffic, controlled trails): 27 of 27 expected detections fired.
    - Whole Mirai capture (764,137 packets, 59.9 MB), offline: finished in 0.06 s of sensor time (about 2.5 million packets/s; Kitsune took 1,117 s on the same file). 8 events, all heuristic "potential periodic beaconing (suspicious)", from IoT hosts 192.168.2.112 (6 events), .109 and .110. No trail (blocklist) hits.
    - All 8 events are between 05:48 and 05:54, minutes 2-8 of a 119-minute capture (it starts 05:46:22). Kitsune's scores jump at packet 121,713 = 07:00:44, minute 74. So during the phase where Kitsune flags 88% of packets, Maltrail raised nothing. Whether the 8 early events are false alarms (regular polling by IoT devices is plausible) cannot be said: the capture has no labels.
    - Single capture: this shows what happened on this file, not a general detection rate.
  - NOT tested: live capture; detection quality on labelled traffic.
- Caveats: signature-based, so it cannot flag attacks it has no trail or heuristic for. Static trail data has its own licence terms. Default web login `admin/changeme!` on all interfaces: change both. Windows needs Npcap and runs the sensor as SYSTEM via a scheduled task (install.ps1).

## 3. Kitsune-py (ymirsky/Kitsune-py, MIT)
Use case: catching UNKNOWN anomalies in packet traffic, with no labels. Learns what normal traffic looks like, then scores every packet.
- Method: AfterImage keeps damped running statistics per packet channel (100 features measured in this version; a comment in its example file says 115); KitNET is an ensemble of small autoencoders (14 for the first 8,000 packets of Mirai); the score is reconstruction error (RMSE).
- Input: pcap, pcapng or its own TSV. No CSV flow input. No live capture in the shipped code.
- Output: one number per packet.
- TESTED here (whole `mirai.pcap`, 764,137 packets; its example settings: 5,000 packets to learn feature mapping, 50,000 to train):
  - Needed `Cython` installed and a NumPy 2 shim (`np.Inf`); I did not edit their code.
  - Speed: 684 packets/s (1,117 s) in pure Python with scapy; 3.3 GB of memory because the scapy path loads the whole file; the scapy path also ignores any packet limit.
  - Cut-off = 99th percentile of its own benign window (packets 55,001-70,000) = 0.0794.
  - Before the attack: 0.7% of packets above the cut-off (target 1%), mean score 0.014.
  - From about packet 121,700 on, scores jump: more than half of the next 2,000 packets exceed the cut-off, then 88% of all later packets are flagged (mean score 20.8 vs 0.016 benign).
  - The capture has no labels, so recall and precision are NOT measured. The 12% not flagged may be benign background traffic mixed into the attack, or attack packets it misses; unknown. Labelled data (UCI Kitsune dataset) would settle this.
- Caveats: needs a clean learning period at the start; scores packets, not hosts (a packet-to-host mapping must be added); pure Python is slow for a busy link; code dates from 2018 (NumPy 2 break; one `subprocess.call(..., shell=True)` that builds a command from the pcap path when tshark is installed, which is unsafe with untrusted file names); a live mode would need a wrapper that feeds packets to its statistics (untested).

## 4. Slips (stratosphereips/StratosphereLinuxIPS 1.1.24, GPL-2.0)
Use case: behavioural detection for a whole network: builds a profile per host, runs many detectors, collects evidence per host and alerts when enough accumulates. Also uses 40+ threat-intel feeds.
- ML inside (DOCS + code imports): flow classifier `ml_linear_model` (scikit-learn SGDClassifier + PCA; models come from a separate Stratosphere training repo), `ml_online_model`, an RNN for command-and-control (TensorFlow/Keras), HTTPS anomaly detection (River ADWIN).
- Input: live interface (via Zeek), pcap (via Zeek), Argus flow CSV (comma or tab), Zeek log folders, Suricata JSON, nfdump, stdin.
- Output: `alerts.json` and `alerts.txt` in the output folder, plus a local web page.
- Shipped sample inputs (`dataset/`): Argus CSV files (columns `StartTime,Dur,Proto,SrcAddr,Sport,Dir,DstAddr,Dport,State,sTos,dTos,TotPkts,TotBytes,SrcBytes,SrcPkts,Label`), small pcaps (for example a 2 KB ICMP port-scan and two "malicious" pcaps), Suricata and Zeek samples. These can serve as shared test inputs for all tools.
- NOT run. Platform: Docker or Linux, 4 GB RAM and about 5 GB disk; Windows only through Docker; the docs say the Docker image does not capture the host's own interface in every setup.
- Caveats: GPL, so copying its code makes the project GPL (running it as a separate program does not). Many moving parts (Zeek, Redis, many processes). Its ML models are pickles: do not load third-party model files.

## 5. CSV and live, side by side
- Maltrail: live yes (Linux; Windows needs Npcap), pcap yes, CSV no.
- Kitsune-py: live no as shipped (TESTED pcap), CSV no (TSV of parsed packets only).
- Slips: live yes (Linux/Docker), pcap yes, CSV yes (Argus), plus Zeek/Suricata/nfdump (DOCS).
- So no single tool covers everything; CSV comes from Slips' Argus input and from our own NetFlow-style flow model. Live comes from Maltrail and Slips on Linux, and from a Kitsune wrapper plus our layers.

## 6. Can they be combined?
Verdict: yes, at the alert level (each stays its own program or module, outputs are turned into one score per host and time window). Not as one merged codebase. DESIGN judgement; the combination itself is untested.
- All three: possible. Best coverage: known-bad (Maltrail), unknown packet anomalies (Kitsune), per-host behaviour and flow ML (Slips), plus our DGA model, NetFlow flow model and learned fusion. Needs Linux (WSL2/Docker/VM) for Maltrail and Slips.
- Kitsune + Maltrail: the best pair. Both read the same pcap. Complementary (signatures vs learned normal). Simple outputs (a score per packet; event-log lines). Both MIT. Both now ran on the same capture: Maltrail raised 8 events, all in minutes 2-8 and none in the phase where Kitsune's scores jump (minute 74 on); Kitsune stayed at benign level until then. That is one capture, but it is the pattern that makes the pair worth combining: each saw something the other did not.
- Kitsune + Slips: workable. Overlap in the "anomaly" role; Slips needs Zeek-parsed input and Linux/Docker. Combine by reading `alerts.json`.
- Maltrail + Slips: least added value. Slips already has threat-intel and heuristics; Maltrail's trail lists could be given to Slips as local threat-intel files, but that adds little over Slips alone.
- Common record: our `Evidence(layer, host, ts, score in [0,1], reason)`. Kitsune scores are per packet, so they are summarised per (host, window) with a binomial rule; Maltrail events and Slips alerts map straight in.
- Does not combine directly: CSV schemas (Slips reads Argus CSV; our flow model needs its own feature set); Slips code cannot be linked in (GPL).

## 7. Recommendation
Core = Kitsune-py + Maltrail (real code or real sensor) + our ML layers + learned fusion, built offline on pcap and CSV first. Slips as an external comparison and optional third source once a Linux environment exists. Live capture last, in an isolated VM.

## 8. What is still unverified
- Maltrail's detection quality on LABELLED traffic, and live capture (it ran offline on one unlabelled capture plus its own 27-case self-test).
- Everything about Slips beyond its docs and sample files.
- Kitsune's precision/recall (no labels), and any live use.
- Behaviour when the tools run together.
