# NIDS signature and heuristic layer (own code, no machine learning)

Code: `layered_nids/signatures/` (patterns, heuristics, scan). Tests: `tests/test_signatures.py` (24). Benchmark: `python -m layered_nids.signatures.benchmark` -> `benchmark.json`.

## What it does
- Signatures (evidence flagged `signature: True`): 10 web-attack patterns on the HTTP request line (directory traversal incl. double-encoded, sensitive files, SQL injection, XSS, command injection, remote file inclusion, shellshock, web shell and admin probes, log4j lookups, router/IoT exploit paths) and user-supplied blocklists of addresses and domains.
- Heuristics (`signature: False`): scan (many distinct targets), SYN flood and UDP flood to one target, ARP flood, DNS burst, DNS random-looking names (name entropy), beaconing (very regular new connections to one external destination).
- Count thresholds are defaults raised to the 99th percentile of clean traffic (`calibrate`, clean traffic only).

## Measured (injection benchmark, 3 injection seeds; thresholds calibrated on the first 80% of the clean 52 minutes of the Mirai capture)
- horizontal scan, port scan, SYN flood, UDP flood, DNS burst, web attacks (6 request lines), regular beacon connections: detected in 3 of 3 injections each.
- bulk upload (1,400-byte packets, 80/s): detected 0 of 3. The rules count targets and packets, not volume; the packet anomaly model finds it (recall 1.00).
- old-style beacon made of packets inside one connection (no new connections): 1 of 3, by luck (scan rule). Regular new connections is the pattern the rule looks for.
- Clean real traffic (last 10.4 minutes of the clean period, no injections): 2 scan evidence items. Whether they are false alarms is not known (no labels); the same hosts are active in the attack period.
- Whole Mirai capture: 79 evidence items (45 ARP flood, 17 scan, 14 random DNS names, 3 beaconing); no web attacks in that capture, so no signature hit.

## Limits
- Plain HTTP request lines only: no encrypted traffic, no request bodies, no headers. A hit means "looks like", not "succeeded".
- Injected attacks are synthetic and were written by the same author as the rules; real attackers vary. Untested on real attack traffic.
- No blocklist data is shipped; blocklists are supplied by the user.
