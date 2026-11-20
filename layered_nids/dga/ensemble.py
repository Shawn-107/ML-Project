"""Does averaging the n-gram model and the character CNN beat either alone?   python -m layered_nids.dga.ensemble

Same protocol as run.py (5 folds of unseen DGA families, threshold from validation benign names, test used once). In each fold
BOTH models are fitted once and scored three ways: n-gram alone, CNN alone, and the average of their probabilities, so the
comparison is paired. -> reports/dga/ensemble.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from layered_nids.common.paths import REPORTS
from layered_nids.dga.data import check_split, family_folds, load_domains
from layered_nids.dga.evaluate import evaluate_fold, summarise
from layered_nids.dga.models import CharCNN, DGAModel, NgramLR


class _Scores(DGAModel):
    """Wraps an already-fitted scoring function so evaluate_fold can reuse it without refitting."""

    def __init__(self, name, fn):
        self.name, self._fn = name, fn

    def fit(self, names, y, val=None):
        return self

    def predict_proba(self, names):
        return self._fn(names)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--out", default=str(REPORTS / "dga" / "ensemble.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    df = load_domains(max_per_family=2000, n_benign=100_000, seed=0)
    res = {"ngram_lr": [], "char_cnn": [], "average": []}
    for split in family_folds(df, 5, seed=0):
        check_split(split)
        tr, va = split.train, split.val
        ng = NgramLR(seed=0).fit(tr["name"].tolist(), tr["y"].to_numpy())
        cnn = CharCNN(seed=0, epochs=a.epochs).fit(tr["name"].tolist(), tr["y"].to_numpy(), val=(va["name"].tolist(), va["y"].to_numpy()))
        fns = {"ngram_lr": ng.predict_proba, "char_cnn": cnn.predict_proba,
               "average": lambda x: (ng.predict_proba(x) + cnn.predict_proba(x)) / 2.0}
        for name, fn in fns.items():
            r = evaluate_fold(_Scores(name, fn), split)
            res[name].append(r)
        p = {k: res[k][-1]["policies"]["fpr_1pct"] for k in res}
        print(f"fold {split.fold}: recall@1% ngram {p['ngram_lr']['recall']:.3f} cnn {p['char_cnn']['recall']:.3f} average {p['average']['recall']:.3f}"
              f" | false alarms {p['ngram_lr']['false_alarm_rate']:.4f} {p['char_cnn']['false_alarm_rate']:.4f} {p['average']['false_alarm_rate']:.4f}"
              f" ({time.time() - t0:.0f}s)", flush=True)
    out = {"summary": {k: summarise(v) for k, v in res.items()}, "folds": res, "wall_seconds": round(time.time() - t0, 1),
           "note": "paired: the same fitted models scored three ways on the same unseen-family test folds"}
    Path(a.out).write_text(json.dumps(out, indent=1))
    for k, s in out["summary"].items():
        print(f"{k:10s} AUC {s['auc']['mean']:.4f}+-{s['auc']['std']:.3f} | @1%: recall {s['fpr_1pct.recall']['mean']:.3f}+-{s['fpr_1pct.recall']['std']:.3f}"
              f" FA {s['fpr_1pct.false_alarm_rate']['mean']:.4f} | @5%: recall {s['fpr_5pct.recall']['mean']:.3f} | brier {s['brier']['mean']:.3f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
