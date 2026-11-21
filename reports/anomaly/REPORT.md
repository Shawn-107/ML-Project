# NIDS packet anomaly detector v2 (own code) and the injection benchmark

Code: `layered_nids/anomaly/` (features, model, detector, inject, benchmark). Tests: `tests/test_anomaly.py` (10).
Raw numbers: `benchmark_validate.json`-style files in this folder (`validate_v2.json`, `validate_baselines.json`, `benchmark_test.json`) and the logs next to them.

## What it is
- Written from the maths of damped incremental statistics and the idea of an ensemble of small autoencoders (Kitsune paper); no code copied from the original.
- 120 features per packet (the original and our first version have 100): host, channel, jitter and socket size statistics at 4 decay rates, plus two new groups:
  - novelty: per source host, never-before-seen targets and destinations (a scan is a burst of new targets even if every packet looks ordinary);
  - mix: per source host, TCP/UDP/other shares, SYN-without-ACK share, RST share, DNS query count and DNS name entropy.
- Model: correlation-grouped autoencoders on log-and-quantile-scaled features, output autoencoder over the group errors (a second scoring mode, `tail_sum`, was tested and rejected, see below).
- Threshold policy: score that at most 1% of held-out normal packets exceed; evidence per (host, 60 s window) from the count of flagged packets against the false-alarm rate (beta-binomial when the calibration windows show bursty false alarms).
- Streaming (`process`, one packet at a time) gives the same scores as batch (tested). Save/load with SHA-256 check; weights loaded with `weights_only` (no pickled code).

## How it was tested (injection benchmark)
No labelled packet capture is on this machine. So: real normal traffic (the first 52 minutes of the Kitsune Mirai capture, split in time order 60/20/20 into train / calibration / hold) with synthetic attacks merged into the hold part, each from an existing internal host:
horizontal scan (300 targets), port scan (1,000 ports), SYN flood, UDP flood, DNS burst of random names, bulk upload, and a low-and-slow beacon (28 packets, one every 15 s).
**This measures whether a detector notices these behaviours. It does not measure accuracy on real attacks.**
Every model: same training data, same 1% calibration policy, same attacked stream. Spread = 3 model seeds x 3 injection seeds (9 runs); choice between our own variants used different injection seeds (validation) from the final numbers (test).

## Results (test stage, 9 runs, mean; the spread is in `benchmark_test.json`)
Packet recall at the 1% policy, packet AUC, and whether the attacker host got an improbable window (p <= 1e-6) during the attack:
- horizontal scan: v2 recall 1.00 / AUC 0.986 / episode found 100%; our first version 0.29 / 0.824 / 78%; original Kitsune 0.65 / 0.820 / 100%.
- port scan: v2 0.99 / 0.994 / 100%; first version 0.99 / 0.990 / 100%; original 0.99 / 0.938 / 100%.
- SYN flood and UDP flood: all three about 1.00 recall; AUC v2 0.999 and 1.000, first version 0.998 and 0.999, original 0.966 and 0.967.
- DNS burst: v2 0.86 / 0.942 / 100%; first version 0.93 / 0.959 / 100%; original 0.95 / 0.904 / 100%.
- bulk upload: all three recall 1.00; AUC v2 1.000, first 0.999, original 0.993.
- low-and-slow beacon: nobody finds it at packet level (v2 recall 0.04, AUC 0.706, episode 0%; first version 22%; original 33% with AUC 0.382). The rule layer (`signatures`) does catch it, because it counts regular new connections.
- False alarms on normal packets of hosts NOT involved in any attack (target 1%): v2 1.88% +- 0.62, first version 2.11% +- 0.72, original 3.12% +- 1.03. All three are above the 1% target because traffic in the last 10 minutes drifts away from the calibration part. False-alert windows per hour (any host, outside attack episodes): v2 60.7, first 62.6, original 99.6 (windows of 60 s with p <= 1e-6; the unit is small, the hold stretch is only 10 minutes).
- Speed (feature extraction only, 100,000 packets): 10,600 packets/s for v2. The original Kitsune ran at 684 packets/s end to end on this machine (earlier measurement), so the two numbers are not the same thing.

## Rejected variant (validation stage, 1 model seed x 2 injections)
`tail_sum` scoring (sum of the rarest group-error tails instead of an output autoencoder): recall 1.0 on every attack but false alarms on uninvolved hosts 2.5-3.0% against 0.9% for the output autoencoder and lower AUC on floods (0.978 vs 1.000). Not used.

## Limits
- Synthetic attacks, one capture, one 10-minute hold stretch, IoT-heavy network. Real attacks, other networks and encrypted traffic are untested.
- v2 is worse than the first version and the original on DNS-burst recall (0.86 vs 0.93 and 0.95): the rule layer covers this behaviour (3 of 3 in `reports/signatures/benchmark.json`).
- False alarms exceed the 1% target on drifting traffic; the evidence layer (windowed counts, fusion) is what keeps alerts per host-hour low, and that is only measured on the Mirai pipeline, not on labelled data.
- No live traffic was captured or tested (owner decision).
