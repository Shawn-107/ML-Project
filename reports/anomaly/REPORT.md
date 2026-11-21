# NIDS packet anomaly detector v2 (own code) and the injection benchmark

Code: `layered_nids/anomaly/` (features, model, detector, inject, benchmark). Tests: `tests/test_anomaly.py` (10).
Final numbers: `benchmark_test_v2.json` (v2, default), `benchmark_test_v2_minmax.json` (v2, sensitive option), `benchmark_test_final.json` (our first version and the original reference detector, same injections). Validation runs used for choosing: `validate_*.json`.

## What it is
- Written from the published method (damped incremental statistics + an ensemble of small autoencoders) with its own choices; no code copied.
- 120 features per packet (the reference and our first version have 100): host, channel, jitter and socket size statistics at 4 decay rates, plus two new groups:
  - novelty: per source host, never-before-seen targets and destinations (a scan is a burst of new targets even if every packet looks ordinary);
  - mix: per source host, TCP/UDP/other shares, SYN-without-ACK share, RST share, DNS query count and DNS name entropy.
- Model: correlation-grouped autoencoders, features log-transformed and scaled by their 0.5%-99.5% training quantiles, output autoencoder over the group errors.
- Threshold policy: score that at most 1% of held-out normal packets exceed. Evidence per (host, 60 s window) from the count of flagged packets against the false-alarm rate (beta-binomial when calibration windows show bursty false alarms).
- Streaming (`process`, one packet at a time) gives the same scores as batch (tested). Save/load with SHA-256 check; weights loaded with `weights_only`.

## How it was tested (injection benchmark)
No labelled packet capture is on this machine. So: real normal traffic (first 52 minutes of the Mirai capture, split in time order 60/20/20 into train / calibration / hold) with synthetic attacks merged into the hold part, each from an existing internal host: horizontal scan (300 targets), port scan (1,000 ports), SYN flood, UDP flood, DNS burst of random names, bulk upload, and a low-and-slow beacon (28 packets, one every 15 s).
**This measures whether a detector notices these behaviours. It does not measure accuracy on real attacks.**
All models: same training data, same 1% calibration policy, same attacked streams. Spread: 3 model seeds x 3 injection seeds (9 runs). Variants were chosen on different injections (seeds 100-105), never on the test seeds (0-2).
"Recall at 1% FA" uses, for every model alike, the threshold that 1% of the normal packets of uninvolved hosts exceed in that stream, so it compares models at the SAME false-alarm rate (the policy threshold does not: traffic drifts away from the calibration part, so models flag more than 1% on the hold part).

## Results (test seeds; mean over 9 runs)
Recall at 1% FA / packet AUC / attacker host got an improbable window (p <= 1e-6) during the attack:
- horizontal scan: v2 0.99 / 0.986 / 100%; first version 0.14 / 0.822 / 78%; reference 0.15 / 0.815 / 100%.
- port scan: v2 0.99 / 0.994 / 100%; first 0.97 / 0.990 / 100%; reference 0.98 / 0.938 / 100%.
- SYN flood: v2 0.99 / 0.999 / 100%; first 0.99 / 0.998; reference 0.99 / 0.966.
- UDP flood: v2 1.00 / 1.000; first 1.00 / 0.999; reference 1.00 / 0.967.
- bulk upload: v2 1.00 / 1.000; first 1.00 / 0.999; reference 1.00 / 0.993.
- DNS burst: v2 **0.66** / 0.942 / 100%; first **0.88** / 0.959 / 100%; reference **0.88** / 0.905 / 100%.
- low-and-slow beacon: nobody finds it at packet level (recall 0.00-0.01 at 1% FA). AUC v2 0.677, first 0.656, reference 0.370. Episode detection: v2 0%, first 22%, reference 33%.
- False alarms on normal packets of hosts NOT involved in any attack (target 1%): v2 1.88%, first 2.11%, reference 3.12%. False-alert windows per hour (any host, outside attack episodes): v2 60.0, first 60.7, reference 95.8.
- Normal packets of the attacking hosts and targets (their statistics are disturbed by the attack): v2 8.9% flagged, first 10.5%, reference 14.4%.
- Speed: feature extraction 10,600 packets/s (100,000 packets); the reference ran at 684 packets/s end to end on this machine (earlier measurement), so the two are not the same thing.

## Where v2 is NOT better
- DNS-burst packets at equal false alarms: 0.66 against 0.88 for both others. The episode is still found (100%) and the rule layer finds it too (3 of 3, `reports/signatures/REPORT.md`).
- Slow beacon: still unseen at packet level; the first version and the reference catch a minority of episodes by luck of the window count; the rule layer finds regular new connections (3 of 3).
- All three are above the 1% false-alarm target on uninvolved hosts (traffic drifts).

## Sensitive profile (`PacketAnomalyDetector(views=2)`), added 2026-11-21
Two ensembles see the same features through different scalings (log+quantile and plain min-max); each score is put on a rank scale against normal reference scores and the larger one counts. Chosen on the validation injections (6 injections x 2 model seeds), final numbers on the test injections (`benchmark_test_two_view.json`):
- DNS burst: recall at equal false alarms 0.92 (default profile 0.66; first version and reference 0.88); AUC 0.926 (default 0.942, first version 0.959, reference 0.905).
- Other attacks: horizontal scan 0.99 / AUC 0.967, port scan 0.99 / 0.985, floods and bulk upload about 1.00; slow beacon still unseen (episode 11%).
- Costs: false-alert windows per hour 95.8 (default 60.0, first version 60.7, reference 95.8); normal packets of attacked hosts flagged 19.8% (default 8.9%). False alarms on uninvolved hosts 1.87% (same as the default).
- So it removes the DNS-burst weakness against both older detectors but is noisier per window. It is an option, not the default; saving and loading work for both profiles.
- Streaming speed after adding a numpy path for one-packet scoring: about 1,800 packets per second (was 830); batch scoring is much faster.

## Variants tried and rejected (validation seeds)
- `tail_sum` scoring (sum of the rarest group-error tails, correlation or family-grouped groups, with and without a tail extension): recall 1.0 on floods and DNS at the policy threshold in some runs, but unstable (on the test seeds the family-grouped version found 0-24% of DNS-burst packets at equal false alarms) and prone to ties; a rounding bug in it was found and fixed (identical group errors compared with a tolerance), after which the numbers stayed about the same. Not used.
- `hybrid` (max of both views on a common rank scale): more false-alert windows (94/hour) and worse DNS at equal false alarms. Not used.
- Family-based grouping of features (one network per feature family and decay rate): same as correlation grouping for the output score. Not used.
- Regularity features for beacons (gap mean, spread, ratio per destination): no gain on beacon AUC (0.72 against 0.75), removed.
- Plain min-max scaling (`scaling="minmax"`, kept as an option): DNS burst 0.94 at equal false alarms (better than both others) and port-scan AUC about the same, but more false-alert windows (86 per hour), more flagged normal packets of attacked hosts (20%), lower scan AUC (0.966) and DNS AUC (0.926), and it flagged most normal windows on a tiny synthetic training set in `tests/test_pipeline.py` (needs about 50,000 normal packets, like the reference). The default stays the robust log-and-quantile scaling.

## Limits
- Synthetic attacks, one capture, one 10-minute hold stretch, an IoT-heavy network. Real attacks, other networks and encrypted traffic are untested.
- The reference detector in the comparison column is a local, unshipped copy (`layered_nids/anomaly/_reference_original.py`, git-ignored); only its saved numbers are in the repository.
- No live traffic was captured or tested (owner decision).
