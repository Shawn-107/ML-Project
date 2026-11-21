"""Serving API: predict, explain, policies."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .bundle import load_bundle, DEFAULT_PATH
from .adapt import MAX_TARGET, MIN_FLOWS, shift_weights, weighted_threshold
from .contract import FEATURES, DEFAULT_POLICY, POLICIES
from .features import coerce, SchemaError
from .models import p1
from .thresholds import labelled_threshold


class Detector:
    def __init__(self, bundle, policy=DEFAULT_POLICY):
        self.b = bundle
        self.thresholds = dict(bundle.thresholds)      # per-instance copy: adapt() can change these without touching the saved model
        self.set_policy(policy)

    @classmethod
    def load(cls, path=DEFAULT_PATH, policy=DEFAULT_POLICY, verify=True):
        return cls(load_bundle(path, verify), policy)

    def set_policy(self, policy):
        if policy not in self.thresholds:
            raise ValueError(f"unknown policy {policy!r}; choose from {sorted(self.thresholds)}")
        self.policy, self.threshold = policy, self.thresholds[policy]

    def recalibrate(self, labelled: pd.DataFrame, label: str = "label", fpr_cap: float = 0.08, holdout: float = 0.3, seed: int = 0) -> dict:
        """Re-tune the alert threshold on a small LABELLED sample of YOUR traffic (best accuracy under a false-alarm cap).
        With 400+ flows, 30% are held out and the new threshold is only applied if it does not lose accuracy there."""
        if label not in labelled.columns:
            raise SchemaError(f"labelled table needs a '{label}' column (0 = normal, 1 = attack)")
        y = labelled[label].to_numpy()
        if not set(np.unique(y)) <= {0, 1}:
            raise SchemaError(f"'{label}' must contain only 0/1")
        s = self.score_frame(coerce(labelled))
        old = self.threshold
        rep = dict(old_threshold=old, n_flows=int(len(y)), fpr_cap=fpr_cap, validated=False)
        if len(y) >= 400:
            idx = np.random.RandomState(seed).permutation(len(y)); cut = int(len(y) * (1 - holdout))
            tr_i, ho_i = idx[:cut], idx[cut:]
            t_try = labelled_threshold(s[tr_i], y[tr_i], fpr_cap)
            acc = lambda t: float(((s[ho_i] >= t) == (y[ho_i] == 1)).mean())
            rep.update(validated=True, holdout_accuracy_before=acc(old), holdout_accuracy_after=acc(t_try))
            rep["applied"] = rep["holdout_accuracy_after"] >= rep["holdout_accuracy_before"]
            new = labelled_threshold(s, y, fpr_cap) if rep["applied"] else old
        else:
            new = labelled_threshold(s, y, fpr_cap); rep["applied"] = True
        rep["new_threshold"] = new
        rep["accuracy_all_before"] = float(((s >= old) == (y == 1)).mean()); rep["accuracy_all_after"] = float(((s >= new) == (y == 1)).mean())
        if rep["applied"]:
            self.policy, self.threshold = "custom", float(new)
        return rep

    def adapt(self, unlabelled, apply: bool = False, seed: int = 0) -> dict:
        """OPTIONAL label-free threshold adaptation to YOUR traffic (see ids/adapt.py for what was and was not shown to work).
        unlabelled: flows from your network, at least 500. Returns a report; changes the thresholds of this Detector only if apply=True."""
        ref = getattr(self.b, "adapt_ref", None)
        if ref is None:
            raise ValueError("this model file has no adaptation reference; retrain with the current code (python final/train_final.py)")
        df = coerce(unlabelled)
        if len(df) < MIN_FLOWS:
            raise ValueError(f"need at least {MIN_FLOWS} flows from your network to adapt (got {len(df)})")
        if len(df) > MAX_TARGET:
            df = df.sample(MAX_TARGET, random_state=seed)
        Xt = self.b.prep.transform(df)
        w, dauc = shift_weights(ref["X"], Xt, seed)
        normal = ref["normal"]
        ess = float(w[normal].sum() ** 2 / (w[normal] ** 2).sum())
        new = {p: weighted_threshold(ref["scores"][normal], w[normal], t) for p, t in POLICIES.items()}
        notes = []
        if dauc < 0.6:
            notes.append("your traffic looks very similar to the validation data, so thresholds barely change")
        if ess < 1000:
            notes.append("few validation flows resemble your traffic (effective sample under 1,000): treat the new thresholds as unreliable")
        if dauc > 0.95:
            notes.append("your traffic is very different from the training data: expect weak results, retraining on labelled local flows is the real fix")
        rep = dict(n_flows=int(len(df)), domain_auc=dauc, effective_sample=ess, old_thresholds=dict(self.thresholds), new_thresholds=new, applied=bool(apply), notes=notes)
        if apply:
            self.thresholds = dict(new)
            self.set_policy(self.policy if self.policy in new else DEFAULT_POLICY)
        return rep

    def save_override(self, path):
        o = {"policy": self.policy, "threshold": self.threshold}
        if self.thresholds != self.b.thresholds:
            o["thresholds"] = self.thresholds
        Path(path).write_text(json.dumps(o, indent=2))

    def load_override(self, path):
        o = json.loads(Path(path).read_text())
        if "thresholds" in o:
            self.thresholds = {k: float(v) for k, v in o["thresholds"].items()}
        self.policy, self.threshold = "custom", float(o["threshold"])

    def score_frame(self, df) -> np.ndarray:
        """attack probability for an already-validated frame (used by explain and the evaluation scripts)"""
        return p1(self.b.model, self.b.prep.transform(df))

    def predict(self, flows, policy=None) -> pd.DataFrame:
        df = coerce(flows)
        X = self.b.prep.transform(df)
        p = p1(self.b.model, X)
        thr = self.thresholds[policy] if policy else self.threshold
        margin = np.clip((p - thr) / max(1 - thr, 1e-9), 0, 1)
        sev = np.where(p < thr, "none", np.where(margin >= 0.66, "high", np.where(margin >= 0.33, "medium", "low")))
        dist = self.b.kmeans.distance(X)
        return pd.DataFrame(dict(probability=p, threshold=thr, alert=p >= thr, severity=sev,
                                 novelty_distance=dist, unusual=dist >= self.b.novelty_threshold))

    def explain(self, flow, n_permutations=16, seed=0):
        """Shapley-value estimate of how each input moved the attack probability away from a typical Normal flow.
        Contributions always sum to probability - baseline."""
        x = coerce(flow)
        if len(x) != 1:
            raise SchemaError("explain takes exactly one flow")
        rng = np.random.RandomState(seed)
        bg = self.b.background
        rows, orders = [], []
        for _ in range(n_permutations):
            cur = bg.iloc[rng.randint(len(bg))].copy()
            order = rng.permutation(len(FEATURES))
            orders.append(order)
            rows.append(cur.copy())
            for i in order:
                cur[FEATURES[i]] = x.iloc[0][FEATURES[i]]
                rows.append(cur.copy())
        frame = pd.DataFrame(rows)[FEATURES].reset_index(drop=True)
        for c in FEATURES[3:]:
            frame[c] = frame[c].astype(float)
        s = self.score_frame(frame).reshape(n_permutations, len(FEATURES) + 1)
        contrib = np.zeros((n_permutations, len(FEATURES)))
        for j, order in enumerate(orders):
            contrib[j, order] = np.diff(s[j])
        prob, base = float(self.score_frame(x)[0]), float(s[:, 0].mean())
        m, se = contrib.mean(0), contrib.std(0) / np.sqrt(n_permutations)
        out = [dict(feature=f, value=x.iloc[0][f], contribution=float(m[i]), std_err=float(se[i])) for i, f in enumerate(FEATURES)]
        return dict(probability=prob, baseline=base, contributions=sorted(out, key=lambda d: -abs(d["contribution"])))
