"""EXPERIMENT (run:  python -m layered_nids.flows.openset ; about 20 minutes; writes reports/flows/openset_exp1.json and openset_exp2.json).
Not used by the product: "learn normal, call the rest abnormal" versus "learn normal plus known attacks" versus both together,
on UNSEEN attack families (UNSW-NB15, leave-one-attack-out), compared at the SAME false-alarm target and by AUC; plus whether teaching MORE KINDS of
normal helps (hold out one normal service type, see how often it is wrongly flagged when unseen versus taught).

Designs (all thresholds from validation NORMAL flows at a 3% false-alarm target; family A and its flows are absent from training/validation):
  normal_only_*      novelty detector trained on Normal flows only: autoencoder, isolation forest, K-Means distance
  supervised_gb      gradient boosting on Normal + the known attack families
  both_*             either view alarmed, on a common rank scale (min p-value): supervised_gb with each novelty detector
"""
import json, sys, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
from sklearn.metrics import roc_auc_score
from nids.config import ATTACK_CAT, FEATURES, LABEL, NORMAL, Config
from nids.features import build_preprocessor, dense
from nids.models import NoveltyAutoencoder
from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.flows.data import load_flow_data
from layered_nids.flows.models import HGB, IsoNovelty, KMeansNovelty

from layered_nids.common.paths import REPORTS
SCR = str(REPORTS / "flows")
FA = 0.03
cfg = Config()
d = load_flow_data()
tr_raw, va_raw, te_raw = d.raw("tr"), d.raw("v"), d.raw("te")
families = sorted(c for c in set(te_raw[ATTACK_CAT]) if c != NORMAL)

def surprise(v_normal_scores, x):
    """-log of the share of validation-normal scores at least as high (rank scale), continued beyond the largest by a log ratio."""
    ref = np.sort(v_normal_scores); n = len(ref)
    at_least = n - np.searchsorted(ref, x * (1 - 1e-6) - 1e-12, side="left")
    return -np.log((at_least + 1.0) / (n + 1.0)) + 2.0 * np.log(np.maximum(x / (max(ref[-1], 1e-12) * (1 + 1e-6)), 1.0))

def scores_for(tr, va, te, seed, normal_mask_tr=None):
    pre = build_preprocessor()
    Xtr = dense(pre.fit_transform(tr[FEATURES])).astype(np.float32)
    Xv, Xt = (dense(pre.transform(x[FEATURES])).astype(np.float32) for x in (va, te))
    ytr = tr[LABEL].to_numpy()
    nm = (ytr == 0) if normal_mask_tr is None else normal_mask_tr
    xn = Xtr[nm]
    gb = HGB(seed).fit(Xtr, ytr)
    out = {"supervised_gb": (gb.score(Xv), gb.score(Xt))}
    ae = NoveltyAutoencoder(cfg.ae_hidden, cfg.ae_max_iter, seed).fit(xn)
    out["normal_only_autoencoder"] = (ae.score(dense(Xv)), ae.score(dense(Xt)))
    iso = IsoNovelty(seed).fit(xn); out["normal_only_isoforest"] = (iso.score(Xv), iso.score(Xt))
    km = KMeansNovelty((4, 8, 16), seed).fit(xn); out["normal_only_kmeans"] = (km.score(Xv), km.score(Xt))
    return out

def evaluate(S, yv, is_target, is_normal_test):
    """S: name -> (val scores, test scores). Adds the 'both' combinations. Returns name -> detect / false alarm / AUC."""
    S = dict(S)
    sg = S["supervised_gb"]
    for k in ("autoencoder", "isoforest", "kmeans"):
        sn = S[f"normal_only_{k}"]
        S[f"both_gb+{k}"] = tuple(np.maximum(surprise(sg[0][yv == 0], sg[i]), surprise(sn[0][yv == 0], sn[i])) for i in (0, 1))
    res = {}
    for name, (sv, st) in S.items():
        thr = threshold_at_fpr(sv[yv == 0], FA)
        flag = st >= thr
        m = is_target | is_normal_test
        res[name] = {"detect": float(flag[is_target].mean()), "false_alarm": float(flag[is_normal_test].mean()),
                     "auc": float(roc_auc_score(is_target[m], st[m]))}
    return res

# ---------------------------------------------------------------------------------------------------- Experiment 1: unseen attack families
t0 = time.time(); exp1 = {}
for A in families:
    tr, va = tr_raw[tr_raw[ATTACK_CAT] != A], va_raw[va_raw[ATTACK_CAT] != A]
    te = te_raw[(te_raw[ATTACK_CAT] == A) | (te_raw[ATTACK_CAT] == NORMAL)]
    S = scores_for(tr, va, te, 0)
    exp1[A] = evaluate(S, va[LABEL].to_numpy(), (te[ATTACK_CAT] == A).to_numpy(), (te[LABEL] == 0).to_numpy())
    print(f"[{time.time()-t0:5.0f}s] {A:15s} " + " | ".join(f"{k.replace('normal_only_','n_')} {v['detect']:.2f}/{v['false_alarm']:.2f}" for k, v in exp1[A].items()), flush=True)
names = list(next(iter(exp1.values())))
summary1 = {n: {m: [float(np.mean([exp1[A][n][m] for A in families])), float(np.std([exp1[A][n][m] for A in families]))] for m in ("detect", "false_alarm", "auc")} for n in names}
print("\nEXPERIMENT 1: unseen attack families, 9 folds, false-alarm target 3%; mean +- std over families")
for n, v in summary1.items():
    print(f"  {n:26s} detects {100*v['detect'][0]:5.1f}% +- {100*v['detect'][1]:4.1f} | false alarms {100*v['false_alarm'][0]:5.1f}% | AUC {v['auc'][0]:.3f}")
json.dump({"per_family": exp1, "summary": summary1}, open(SCR + "/openset_exp1.json", "w"), indent=1)

# ---------------------------------------------------------------------------------------------------- Experiment 2: teaching more kinds of normal
svc = tr_raw[tr_raw[LABEL] == 0]["service"].value_counts()
types = [s for s in svc.index if s != "-"][:5]
exp2 = {}
for T in types:
    taught = scores_for(tr_raw, va_raw, te_raw, 0)
    in_t_tr = (tr_raw["service"] == T).to_numpy() & (tr_raw[LABEL] == 0).to_numpy()
    unseen = scores_for(tr_raw, va_raw, te_raw, 0, normal_mask_tr=(tr_raw[LABEL] == 0).to_numpy() & ~in_t_tr)
    ytest = te_raw[LABEL].to_numpy(); in_t_te = (te_raw["service"] == T).to_numpy() & (ytest == 0); other_n = (ytest == 0) & ~in_t_te
    yv = va_raw[LABEL].to_numpy(); in_t_va = (va_raw["service"] == T).to_numpy() & (yv == 0)
    row = {"normal_flows_of_this_type_in_test": int(in_t_te.sum()), "taught_in_training": int(in_t_tr.sum())}
    for nm in ("normal_only_autoencoder", "normal_only_isoforest", "normal_only_kmeans"):
        for lab, S in (("taught", taught), ("unseen", unseen)):
            sv, st = S[nm]
            keep = ~in_t_va                                                           # threshold from validation normals of the OTHER types, both times
            thr = threshold_at_fpr(sv[(yv == 0) & keep], FA)
            row[f"{nm}_{lab}"] = {"false_alarm_on_this_type": float((st[in_t_te] >= thr).mean()), "false_alarm_on_other_normals": float((st[other_n] >= thr).mean())}
    exp2[T] = row
    print(f"[{time.time()-t0:5.0f}s] type '{T}': " + " | ".join(f"{k.replace('normal_only_','')}: {v['false_alarm_on_this_type']:.2f}" for k, v in row.items() if isinstance(v, dict)), flush=True)
print("\nEXPERIMENT 2: share of normal flows of one TYPE (service) wrongly flagged, when that type was taught vs left out of training (3% target on other types)")
for nm in ("normal_only_autoencoder", "normal_only_isoforest", "normal_only_kmeans"):
    t = np.mean([exp2[T][f"{nm}_taught"]["false_alarm_on_this_type"] for T in types]); u = np.mean([exp2[T][f"{nm}_unseen"]["false_alarm_on_this_type"] for T in types])
    print(f"  {nm:26s} taught {100*t:5.1f}% -> unseen {100*u:5.1f}%")
json.dump(exp2, open(SCR + "/openset_exp2.json", "w"), indent=1)
print("\ndone", round(time.time() - t0), "s")
