# Data sources and verification status

Everything here was checked on 2026-11-19. "Verified" means checked by us; "not verified" means taken from a web page only.

## unsw/  (L4 flows)
- `official_training_175341.csv`, `official_testing_82332.csv` - copied from `C:\ML project\unsw_work\data\` (the earlier finished project).
- SHA-256 training: BEC7DD5EC88DC2A0CCC7A07879D338395ED7421750F675FD0339E07DFE0648FA
- SHA-256 testing:  734FE6642EDF758F7C94D7D9149426B49D202FE8E7BF0BEF47392489C3C0A559
- Row counts match the file names (175,341 / 82,332) and the legacy backend loads them (54 tests pass).
- Origin of the CSVs (mirror copy, 34 features) is as stated in the legacy README; the upstream source was not re-checked here.

## dga/  (L2 domains)
- `dga_legit.csv` - "Legitimate domains and DGA categorized morphologically and by families", Gregorio & Cansian,
  Mendeley Data v1, DOI 10.17632/nhvyvytn2h.1, published 2025-11-11, licence CC BY 4.0.
- SHA-256 matches the value published by the Mendeley API: ad783f5dbef10bf2836d37f107db202c1a5b76f053c45ec394b81bdb42ccca1f (verified).
- Columns: `domain, family, type`. 4,090,661 rows, no duplicate domains, no domain in two families (measured).
- 161 family values = 160 DGA families + `legit` (1,000,000 rows, Majestic Million per the dataset page).
- Page says 998,313 legitimate domains; the file has 1,000,000 rows with family `legit`. Explained (measured): 1,687 of the
  legit rows have type `subdomain` (e.g. `play.google.com`); the other 998,313 have type `domain`.
- Leakage facts (measured): only 3,502,069 of 4,090,661 rows have a unique name once the TLD is dropped; 946,445 rows share a
  name with another row (the same generated string under several TLDs); 15,513 names appear as both legit and DGA.
  Any split must group or deduplicate on the name without its TLD, and drop the conflicting names.
- Family sizes are very uneven (largest DGA family 519,426 rows, smallest 2). Evaluation must cap or weight by family.
- Caveat: benign side is a top-sites list, so "real vs generated" is easier here than on real DNS traffic.

## mirai/  (L3 packets)
- The public Mirai network-attack capture (UCI dataset #516, DOI 10.24432/C5D90Q, CC BY 4.0, Mirsky et al.) as `data/mirai/mirai.pcap` (62,859,800 bytes, SHA-256 DA69165A7B47353DE854E82B3E8C88741383BF94099DB2AEAAC35D27C7C921FE when measured here). The file has NO labels. Put it there yourself (the dataset page and the original authors' repository both provide it).
- It is the FULL Mirai capture, not just the demo subset (measured + read from the paper in the repo):
  764,137 packets, 7,137 s (~119 min). The dataset paper (Table III) lists Mirai as
  764,137 packets, train 52.0 min, execute 66.9 min. The first 52.0 min = 85,380 packets is the paper's clean training phase.
  The repo demo's "first 70,000 clean" is a rougher, more conservative statement.
- Traffic profile (measured, not labels): minutes 0-72 steady (~13k packets per 8 min, ARP 4-7%, ~50 destinations);
  from ~minute 72 ~118k packets per 8 min, ARP 78-87%, ~290 destinations. One host, 192.168.2.110, sends 88.6% of the
  packets after minute 80 (498,585 ARP packets): the scanner. This is an inference from volume, not a label.
- 173 timestamp steps go backwards (tiny); the parser keeps file order, which is what a label file would follow.
- The labelled dataset (UCI #516, DOI 10.24432/C5D90Q, CC BY 4.0, Mirsky et al.; nine attack captures with per-packet 0/1 labels) was downloaded on 2026-11-23 into
  `data/labelled_iot/` (local only, not committed). The download needed the owner's approval to skip the certificate REVOCATION check
  (`curl --ssl-no-revoke`; the certificate itself was still verified). The zip arrived 18.0 GB of about 18.7 GB: the end of the video-injection files
  and the zip index were missing, so that capture's pcap is absent (its labels are there). Eight captures were extracted and used: Active Wiretap,
  ARP MitM, Fuzzing, Mirai, OS Scan, SSDP Flood, SSL Renegotiation, SYN DoS. Checked: packet count equals label count in all eight; the Mirai capture
  has 764,137 packets like `data/mirai/mirai.pcap`, and its labels start the attack at packet 121,621 = minute 74.3. No checksum is published, so
  the files are "extracted without error, counts consistent", not hash-verified.
- UCI states packets are truncated to 200 bytes and labels are 0/1 per packet.

## netflow/  (optional: NetFlow flow model)
- `NF-ToN-IoT-v2-test.csv` (University of Queensland NetFlow v2 datasets). Put it in `data/netflow/` or set `NIDS_NETFLOW_CSV` to its path. Only needed to retrain the NetFlow model.

## trails/  (optional: known-bad indicator data)
- `trails.csv` and an optional `whitelist.txt` of popular domains (one per line, or `rank,domain`). Local files, never committed. The community indicator data is free for defending
  your own systems and for research and education; it must not be redistributed or built into a service for others. Without it the rule layer still works (patterns and behaviour rules), it just
  cannot match known-bad indicators.
