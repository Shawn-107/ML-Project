"""Leave-one-attack-out (LOAO) for the flow classifier and novelty options, one protocol for everything.

   python -m layered_nids.flows.loao [--attacks Fuzzers,Reconnaissance] -> reports/flows/loao.json

For each attack family A: refit the preprocessor and every model WITHOUT any A flow (train and validation), set
thresholds on validation Normal flows without A, then measure on the official test set how many A flows are caught and
how many Normal flows are falsely flagged. Single seed (each fold already retrains ~8 models): the spread across
families is NOT a repeat-run spread, and seed variance of LOAO is untested.

Classifiers: legacy blend (ANN 2 seeds + forest, 0.5/0.5), gradient boosting, mean(forest, gradient boosting).
Novelty detectors (fitted on Normal flows only, threshold at 5% validation false alarms): legacy autoencoder,
isolation forest, K-Means distance (k by silhouette).
Combinations with the gradient-boosting classifier: OR (flag if either flags) and GATED (flag if the classifier flags,
or if novelty flags AND the classifier score is above its 15%-false-alarm level, i.e. not clearly normal).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from nids.config import ATTACK_CAT, FEATURES, LABEL, NORMAL, Config
from nids.features import build_preprocessor, dense
from nids.models import AnnEnsemble, ForestModel, NoveltyAutoencoder

from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.paths import REPORTS
from layered_nids.flows.data import load_flow_data
from layered_nids.flows.models import HGB, IsoNovelty, KMeansNovelty

CLF_FPR, NOV_FPR, GATE_FPR = 0.03, 0.05, 0.15


def fold(d, A: str, cfg: Config, seed: int) -> dict:
    tr_raw, va_raw, te_raw = d.raw("tr"), d.raw("v"), d.raw("te")
    tr, va = tr_raw[tr_raw[ATTACK_CAT] != A], va_raw[va_raw[ATTACK_CAT] != A]
    te = te_raw[(te_raw[ATTACK_CAT] == A) | (te_raw[ATTACK_CAT] == NORMAL)]
    pre = build_preprocessor()
    Xtr = dense(pre.fit_transform(tr[FEATURES])).astype(np.float32)
    Xv, Xt = (dense(pre.transform(x[FEATURES])).astype(np.float32) for x in (va, te))
    ytr, yv = tr[LABEL].to_numpy(), va[LABEL].to_numpy()
    is_a = (te[ATTACK_CAT] == A).to_numpy()
    is_n = (te[LABEL] == 0).to_numpy()

    ann = AnnEnsemble(cfg.ann_hidden, cfg.ann_alpha, cfg.loao_seeds, cfg.ann_max_iter, seed + 100).fit(Xtr, ytr)
    rf = ForestModel(cfg.rf_trees, cfg.rf_min_leaf, seed).fit(Xtr, ytr)
    gb = HGB(seed).fit(Xtr, ytr)
    s = {"legacy_blend": (0.5 * ann.predict_proba(Xv) + 0.5 * rf.predict_proba(Xv), 0.5 * ann.predict_proba(Xt) + 0.5 * rf.predict_proba(Xt)),
         "gb": (gb.score(Xv), gb.score(Xt)),
         "rf_gb_mean": ((rf.predict_proba(Xv) + gb.score(Xv)) / 2, (rf.predict_proba(Xt) + gb.score(Xt)) / 2)}
    xn = Xtr[ytr == 0]
    nov = {"autoencoder": NoveltyAutoencoder(cfg.ae_hidden, cfg.ae_max_iter, seed).fit(xn),
           "isoforest": IsoNovelty(seed).fit(xn), "kmeans": KMeansNovelty((4, 8, 16), seed).fit(xn)}
    out = {"n_attack_test": int(is_a.sum()), "n_normal_test": int(is_n.sum()), "classifiers": {}, "novelty_alone": {},
           "combined_with_gb": {}, "kmeans_k": int(nov["kmeans"].k_)}

    def rate(flag):
        return {"detect": float(flag[is_a].mean()), "false_alarm": float(flag[is_n].mean())}

    thr = {k: threshold_at_fpr(v[0][yv == 0], CLF_FPR) for k, v in s.items()}
    for k, (_, st) in s.items():
        out["classifiers"][k] = rate(st >= thr[k])
    gb_flag = s["gb"][1] >= thr["gb"]
    gb_lo = threshold_at_fpr(s["gb"][0][yv == 0], GATE_FPR)
    for k, m in nov.items():
        nv, nt = (m.score(dense(Xv)) if k == "autoencoder" else m.score(Xv)), (m.score(dense(Xt)) if k == "autoencoder" else m.score(Xt))
        nthr = threshold_at_fpr(nv[yv == 0], NOV_FPR)
        nflag = nt >= nthr
        out["novelty_alone"][k] = rate(nflag)
        out["combined_with_gb"][f"{k}_or"] = rate(gb_flag | nflag)
        out["combined_with_gb"][f"{k}_gated"] = rate(gb_flag | (nflag & (s["gb"][1] >= gb_lo)))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--attacks", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPORTS / "flows" / "loao.json"))
    a = ap.parse_args(argv)
    t0 = time.time()
    d = load_flow_data()
    cfg = Config()
    present = sorted(set(d.cte) - {NORMAL})
    attacks = [x for x in a.attacks.split(",") if x] or present
    folds = {}
    for A in attacks:
        folds[A] = fold(d, A, cfg, a.seed)
        c = folds[A]["classifiers"]
        print(f"{A:15s} detect: legacy {c['legacy_blend']['detect']:.3f} gb {c['gb']['detect']:.3f} rf+gb {c['rf_gb_mean']['detect']:.3f} | "
              f"false alarm: legacy {c['legacy_blend']['false_alarm']:.3f} gb {c['gb']['false_alarm']:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    def mean_of(path):
        v = [path(f) for f in folds.values()]
        return {"mean_detect": float(np.mean([x["detect"] for x in v])), "mean_false_alarm": float(np.mean([x["false_alarm"] for x in v]))}

    summary = {"classifiers": {k: mean_of(lambda f, k=k: f["classifiers"][k]) for k in ("legacy_blend", "gb", "rf_gb_mean")},
               "novelty_alone": {k: mean_of(lambda f, k=k: f["novelty_alone"][k]) for k in ("autoencoder", "isoforest", "kmeans")},
               "combined_with_gb": {k: mean_of(lambda f, k=k: f["combined_with_gb"][k]) for k in next(iter(folds.values()))["combined_with_gb"]}}
    out = {"protocol": __doc__, "seed": a.seed, "ann_seeds": cfg.loao_seeds, "folds": folds, "summary_mean_over_attacks": summary,
           "wall_seconds": round(time.time() - t0, 1)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
