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

## Indicator matching (known-bad addresses, domains, URLs) - added 2026-11-21
Code: `layered_nids/signatures/trails.py` (`Trails.load`, `scan(..., trails=...)`). The indicator data is NOT in this repository: it is a local file in `data/trails/` (git-ignored; community data that is free for defending your own systems and for research and education, not for redistribution or building into a service for others). With a 1.86-million-indicator file it loads in about 4 seconds.
- Indicator kinds handled: address, address:port, domain (and every sub-domain), URL path, host+path, name patterns (36 regular expressions).
- Policy: indicators described as "suspicious" (public resolvers, mass scanners) become weak evidence (score 0.3, not a signature hit); malware and malicious ones are signature hits (score 0.85). Without this policy the Mirai capture produced 238 "hits", all one public DNS server.
- Side by side with the reference sensor on the SAME pcaps (sampled indicators of every kind turned into DNS lookups, connections and HTTP requests; 1,110 packets; counted as signature hits):
  - domains: ours 98%, reference sensor 100%; address:port: ours 100%, sensor 100%; host+path: ours 100%, sensor 57%; URL paths: ours 98%, sensor 73%; plain addresses: ours 90%, sensor 14%.
  - Ours misses only indicators described as "suspicious", which we report as weak evidence instead. The sensor reports far fewer plain addresses and URLs; I did not investigate why (it applies its own filtering rules), so "ours finds more" must not be read as "ours is better": a sensor that stays quiet on weak indicators can be the better choice.
  - Normal traffic (the whole Mirai capture, 764,137 packets): ours 0 signature hits, sensor 0 indicator events.
- Limits: the test traffic is synthetic and was built from the same indicator file the matcher uses; one capture of normal traffic; no whitelist of popular domains (the reference sensor has one); matching is on DNS names, addresses, and the HTTP request line only.

## Additions on 2026-11-21 (after the first report)
- ICMP sweep rule (one host sends ICMP to 15 or more different destinations in a window): found the ping-sweep capture that the other rules missed; 20 pings to ONE host do not fire it.
- TLS server names: the packet reader now reads the server name from a TLS ClientHello (`Packet.sni`). Indicator domains are matched against it, so known-bad domains are caught in HTTPS too, and random-looking server names count in the random-name rule. Tested on crafted ClientHello bytes; the Mirai capture contains no TLS server names, so nothing was measured on real HTTPS traffic here.
- DNS tunnelling rule (10 or more DNS names of 60+ characters in a window).
- Popular-domain whitelist: `Trails.load_whitelist(file)` (or `data/trails/whitelist.txt`, used automatically by the pipeline) ignores indicator matches on those domains and their sub-domains.
- The optional external sensor can be added to the pipeline with `--rules --external-sensor` when it is configured (see `signatures/external.py`); the indicator file in `data/trails/` is used automatically when present.
- Still not covered: TLS fingerprints, user-agent rules, request bodies and headers, a ready-made whitelist (supply your own list).
