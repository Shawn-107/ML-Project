"""Two extra evaluations for L2, using the same models and metrics as run.py:

  random split   : test rows are random, so test families ALSO appear in training. Shows how much a family-blind
                   evaluation flatters the model (the number most DGA papers report).
  leave-one-family-out (LOFO): for each of the N largest families, train on every other family, test on that one.

python -m layered_nids.dga.compare [--top 10] [--models ngram_lr,lexical_lr] -> reports/dga/compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from layered_nids.common.paths import REPORTS
from layered_nids.dga.data import FoldSplit, load_domains
from layered_nids.dga.evaluate import evaluate_fold
from layered_nids.dga.models import REGISTRY


def _train_val(rest: pd.DataFrame, rng, val_frac: float = 0.15):
    is_val = rng.random(len(rest)) < val_frac
    return rest[~is_val].reset_index(drop=True), rest[is_val].reset_index(drop=True)


def random_split(df: pd.DataFrame, test_frac: float = 0.2, seed: int = 0) -> FoldSplit:
    rng = np.random.default_rng(seed)
    is_test = rng.random(len(df)) < test_frac
    train, val = _train_val(df[~is_test], rng)
    test = df[is_test].reset_index(drop=True)
    return FoldSplit(-1, train, val, test, sorted(set(test.loc[test["y"] == 1, "family"])))


def lofo_split(df: pd.DataFrame, family: str, benign_test_frac: float = 0.2, seed: int = 0) -> FoldSplit:
    """Test = every row of `family` plus a random share of benign rows. Train/val = everything else."""
    if family not in set(df["family"]):
        raise KeyError(family)
    rng = np.random.default_rng(seed)
    is_fam = (df["family"] == family).to_numpy()
    is_ben_test = (df["y"].to_numpy() == 0) & (rng.random(len(df)) < benign_test_frac)
    test = df[is_fam | is_ben_test].reset_index(drop=True)
    train, val = _train_val(df[~(is_fam | is_ben_test)], rng)
    return FoldSplit(-1, train, val, test, [family])


def _build(name: str, seed: int, epochs: int):
    kw = {} if name == "entropy_rule" else {"seed": seed}
    if name in ("lexical_mlp", "char_cnn"):
        kw["epochs"] = epochs
    return REGISTRY[name](**kw)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="lexical_lr,ngram_lr")
    ap.add_argument("--top", type=int, default=10, help="LOFO for the N largest families in the sample")
    ap.add_argument("--max-per-family", type=int, default=2000)
    ap.add_argument("--n-benign", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS / "dga" / "compare.json"))
    a = ap.parse_args(argv)

    df = load_domains(max_per_family=a.max_per_family, n_benign=a.n_benign, seed=a.seed)
    names = a.models.split(",")
    out = {"config": vars(a), "random_split": {}, "lofo": {}}

    s = random_split(df, seed=a.seed)
    for m in names:
        r = evaluate_fold(_build(m, a.seed, a.epochs), s)
        out["random_split"][m] = {k: r[k] for k in ("auc", "policies", "calibration") if k in r}
        p = r["policies"]["fpr_1pct"]
        print(f"random split  {m:12s} auc {r['auc']:.4f} recall {p['recall']:.3f} false-alarm {p['false_alarm_rate']:.4f}", flush=True)

    top = df.loc[df.y == 1, "family"].value_counts().head(a.top).index.tolist()
    for fam in top:
        s = lofo_split(df, fam, seed=a.seed)
        out["lofo"][fam] = {"n_family_rows": int((df["family"] == fam).sum())}
        for m in names:
            r = evaluate_fold(_build(m, a.seed, a.epochs), s)
            p = r["policies"]["fpr_1pct"]
            out["lofo"][fam][m] = {"recall": p["recall"], "false_alarm_rate": p["false_alarm_rate"], "auc": r["auc"]}
            print(f"LOFO {fam:16s} {m:12s} recall {p['recall']:.3f} false-alarm {p['false_alarm_rate']:.4f}", flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
