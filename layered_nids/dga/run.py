"""python -m layered_nids.dga.run [--fast] [--models a,b] [--folds 5]  ->  reports/dga/results.json"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from layered_nids.common.paths import REPORTS
from layered_nids.dga.data import check_split, family_folds, load_domains
from layered_nids.dga.evaluate import evaluate_fold, summarise
from layered_nids.dga.models import REGISTRY


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="entropy_rule,lexical_lr,ngram_lr,lexical_mlp,char_cnn")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", default="", help="comma list, e.g. 0,1 (default all)")
    ap.add_argument("--max-per-family", type=int, default=4000)
    ap.add_argument("--n-benign", type=int, default=150_000)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="small sample + 2 epochs, for a smoke run")
    ap.add_argument("--out", default=str(REPORTS / "dga" / "results.json"))
    a = ap.parse_args(argv)
    if a.fast:
        a.max_per_family, a.n_benign, a.epochs = 300, 20_000, 2

    t0 = time.time()
    df = load_domains(max_per_family=a.max_per_family, n_benign=a.n_benign, seed=a.seed)
    print(f"data: {len(df):,} names, {df['y'].sum():,} generated across "
          f"{df.loc[df.y == 1, 'family'].nunique()} families", flush=True)
    only = {int(x) for x in a.only_folds.split(",") if x != ""}
    results = {m: [] for m in a.models.split(",")}
    for split in family_folds(df, a.folds, seed=a.seed):
        if only and split.fold not in only:
            continue
        check_split(split)
        print(f"fold {split.fold}: train {len(split.train):,} val {len(split.val):,} test {len(split.test):,} "
              f"({len(split.test_families)} unseen families)", flush=True)
        for m in results:
            kw = {} if m == "entropy_rule" else {"seed": a.seed}
            if m in ("lexical_mlp", "char_cnn"):
                kw["epochs"] = a.epochs
            r = evaluate_fold(REGISTRY[m](**kw), split)
            results[m].append(r)
            p = r["policies"]["fpr_1pct"]
            print(f"  {m:13s} auc {r['auc']:.4f}  @1%FPR: recall {p['recall']:.3f} "
                  f"false-alarm {p['false_alarm_rate']:.4f}  ({r['fit_seconds']}s)", flush=True)
    out = {"config": vars(a), "data": {"names": int(len(df)), "generated": int(df["y"].sum())},
           "summary": {m: summarise(rs) for m, rs in results.items() if rs}, "folds": results,
           "note": "test folds contain only DGA families unseen in training; threshold set on validation benign names; "
                   "TLD removed; benign names are a top-sites list (see data/SOURCES.md)",
           "wall_seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
