# Network Intrusion Detection - results

## Data

UNSW-NB15 official split. Training 175,341 rows -> 91,839 after removing rows whose feature vector repeats or carries conflicting labels (5,585 conflicting). Test 82,332 -> 46,830 after the same cleaning and removing rows that also occur in training.
Model fitting used 73,471 rows, threshold selection 18,368 validation rows, final evaluation 46,830 test rows (touched only by `evaluate`).

## Validation (same distribution as training)

Policy `fpr_3pct`: accuracy 88.8%, recall 79.7%, false-alarm rate 3.0%, F1 87.0%.

## Official test set

| Policy | Novelty | Accuracy | Balanced acc. | Recall | False alarms | F1 | Brier | ECE |
|---|---|---|---|---|---|---|---|---|
| fpr_1pct | no | 88.1% | 83.5% | 69.4% | 2.4% | 79.7% | 0.123 | 0.127 |
| fpr_3pct | no | 87.9% | 85.8% | 79.0% | 7.5% | 81.6% | 0.123 | 0.127 |
| fpr_3pct | yes | 84.6% | 84.2% | 83.0% | 14.6% | 78.4% | 0.123 | 0.127 |

The official test set is harder than the training distribution (known distribution shift), so test accuracy is lower than validation accuracy. Report both.

Latency: 28.3 ms per flow one at a time, 0.27 ms per flow in batches.

## Attacks the model never saw (leave-one-attack-out)

Each row retrains without that attack family, then measures how many of its test flows are caught (policy `fpr_3pct`, novelty detector at 5.0% validation false alarms).

| Held-out attack | Flows | Classifier | Novelty | Combined | Detected when trained on it |
|---|---|---|---|---|---|
| Analysis | 224 | 99.6% | 76.8% | 100.0% | 99.6% |
| Backdoor | 68 | 76.5% | 39.7% | 79.4% | 91.2% |
| DoS | 1,162 | 82.1% | 50.7% | 88.5% | 87.3% |
| Exploits | 7,048 | 79.4% | 61.7% | 89.2% | 91.8% |
| Fuzzers | 3,909 | 16.2% | 20.0% | 31.5% | 39.3% |
| Generic | 1,220 | 66.2% | 78.7% | 95.0% | 93.8% |
| Reconnaissance | 1,874 | 34.7% | 15.9% | 36.9% | 96.4% |
| Shellcode | 292 | 74.0% | 9.2% | 75.0% | 74.7% |
| Worms | 39 | 92.3% | 69.2% | 92.3% | 92.3% |

Mean over families: classifier 69.0%, combined 76.4%. False alarms on Normal test flows: 7.8% (classifier) vs 15.1% (with novelty).

## Stability over repeated training runs

3 runs with different seeds and validation splits.

| Metric | Validation mean ± std | Test mean ± std |
|---|---|---|
| accuracy | 89.2% ± 0.3% | 87.7% ± 0.2% |
| recall | 80.5% ± 0.7% | 79.8% ± 0.6% |
| false_alarm_rate | 3.0% ± 0.0% | 8.2% ± 0.6% |
| f1 | 87.6% ± 0.4% | 81.5% ± 0.2% |

## Local recalibration

Without any local labels: accuracy 87.9%, false alarms 7.5%. Accuracy after tuning the threshold on n labelled local flows:

| Labelled flows | Accuracy | Recall | False alarms |
|---|---|---|---|
| 50 | 87.7% ± 0.7% | 77.4% | 7.0% |
| 100 | 87.9% ± 0.5% | 76.2% | 6.1% |
| 200 | 88.1% ± 0.3% | 74.2% | 4.9% |
| 500 | 88.1% ± 0.2% | 75.9% | 5.6% |
| 1000 | 88.2% ± 0.2% | 74.3% | 4.7% |
| 3000 | 88.2% ± 0.1% | 74.2% | 4.6% |

## Limits

- Trained on UNSW-NB15 (2015, synthetic lab traffic). Real networks differ; recalibrate on local labelled flows.
- Works on flow records, not raw packets. The 'real-time' demo replays stored flows; there is no live capture.
- The drift monitor did not detect the shift between UNSW-NB15 training and test data; it is only a sanity check.
- Explanations are Shapley-value estimates from a Monte-Carlo sampler written for this project (the `shap` package was not available).
- The HTTP service has no authentication and must stay on localhost.
