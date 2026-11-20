# L2 - DGA domain classifier: measured results

Everything below comes from `reports/dga/results.json` (5-fold family-held-out run, 1,469 s wall time) and
`reports/dga/compare.json` (random split and leave-one-family-out). Commands: `python -m layered_nids.dga.run`,
`python -m layered_nids.dga.compare`. Single seed (0). The +- values are the spread across the 5 folds, which hold
out different families, so they measure how much results depend on which families are unseen. They are not
seed-to-seed variation; repeating over several seeds has not been done (untested).

## Data and protocol
- Source: Mendeley "Legitimate domains and DGA categorized morphologically and by families" (CC BY 4.0), hash-verified (see `data/SOURCES.md`).
- Sample used: 253,375 names = 100,000 benign + 153,375 generated from 144 families (capped at 2,000 names per family; families with fewer than 20 names dropped).
- TLD removed; names de-duplicated; names appearing as both benign and generated dropped.
- Each fold holds out about 29 whole families (never seen in training or validation) plus a share of benign names.
- The alert threshold is chosen on validation benign names (target false-alarm rate 1% or 5%) and applied once to the test fold.

## Results: unseen families, 5 folds, mean +- std
Threshold at 1% target false alarms (measured false alarms on the test folds ranged 0.8-1.2% across models and folds):

- entropy rule (no learning): AUC 0.671 +- 0.056, recall 12.4% +- 4.9%
- lexical features + logistic regression: AUC 0.864 +- 0.065, recall 54.0% +- 10.4%
- lexical features + MLP (PyTorch): AUC 0.873 +- 0.054, recall 59.8% +- 10.7%
- char n-gram TF-IDF + logistic regression: AUC 0.960 +- 0.019, recall 71.9% +- 8.1%
- char CNN (PyTorch): AUC 0.955 +- 0.023, recall 74.3% +- 10.6%

At 5% target false alarms: n-gram LR recall 85.7% +- 6.2%, char CNN 85.3% +- 7.2%, lexical LR 65.4%, lexical MLP 69.9%.

Calibration on the test folds (Brier / ECE, lower is better; priors differ from real traffic, so treat as relative):
- n-gram LR 0.078 / 0.050, char CNN 0.083 / 0.056, lexical MLP 0.143 / 0.084, lexical LR 0.149 / 0.088.

Families almost never caught (recall under 5% at the 1% policy), out of 144 tested:
- entropy rule 99, lexical LR 27, lexical MLP 27, n-gram LR 12, char CNN 11.
- Examples missed by both best models: tsifiri, fosniw, kfos.

## What the comparison supports
- Learning from characters beats hand-built lexical features by a wide margin (AUC 0.96 vs 0.86-0.87).
- The n-gram LR and the char CNN are not distinguishable here: the CNN is ahead in 3 of 5 folds and behind in 2, and the gap (2.4 points) is smaller than the fold spread (8-11 points).
- The n-gram LR trains in 5-12 s per fold; the CNN takes about 180-210 s on CPU. For this task the simple model is the better engineering choice.
- The MLP on lexical features adds about 6 points of recall over logistic regression on the same features, so some non-linearity helps, but it does not close the gap to character models.

## Random split vs held-out families (n-gram LR, single split)
- Random split (families shared between train and test): recall 79.6% at 1.0% false alarms, AUC 0.979.
- Family-held-out (5-fold mean): recall 71.9%, AUC 0.960.
- A family-blind evaluation overstates unseen-family recall by about 8 points here. Different test sets, so indicative only.

## Leave-one-family-out, 10 largest families (1% policy)
- n-gram LR: mean recall 91.4%, lowest 72.3% (pykspa2s); false alarms about 1.0%.
- lexical LR: mean recall 64.8%, lowest 16.4% (pykspa2s).
- The 10 largest families are more machine-looking than the average family, so 91.4% must not be read as the typical unseen-family recall. The 5-fold average over all 144 families (68.6% per-family mean for n-gram LR) is the fairer number.

## Limits
- Benign names are a top-sites list. Real DNS traffic includes CDN, tracking and mistyped names that look random, so the false-alarm rate on real traffic is untested and probably higher.
- Many tested families are word-list or hash-style generators of a kind that are hard to distinguish from names by characters alone; these are exactly the ones missed.
- One seed; no repeated-seed variance. Hyper-parameters were set by hand and not tuned (untested whether tuning helps).
- Models are not yet connected to live DNS logs; L2 is evaluated offline on names only.

## Ensemble check (added 2026-11-20; `layered_nids/dga/ensemble.py`, raw output `ensemble.json`)
Same protocol and the same 5 folds of unseen families; in each fold the n-gram model and the CNN are fitted once and scored three ways (paired).
- n-gram LR: AUC 0.960 +- 0.019, recall 72.1% +- 8.4% at 1% false alarms (measured 0.93%), 85.6% at 5%, Brier 0.078.
- char CNN: AUC 0.955 +- 0.023, recall 74.4% +- 10.5%, 85.2% at 5%, Brier 0.083.
- average of both: AUC 0.961 +- 0.021, recall 74.8% +- 9.2%, 86.5% at 5%, Brier 0.076.
- Per fold at 1%: average minus n-gram is +2.0, +2.6, +5.2, +0.3, +3.3 points (better in all 5); average minus CNN is +2.1, +0.7, -1.8, +2.9, -2.0 (better in 3 of 5).
- Reading: a small, consistent gain over the n-gram model; no meaningful gain over the CNN (0.4 points against a fold spread of about 10). The ensemble costs the CNN's
  training time. Use it where the CNN is available; the n-gram model alone is the cheap choice. One seed, so the seed-to-seed spread is untested.
