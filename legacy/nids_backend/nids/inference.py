"""Detector: the one object the dashboard / API / CLI use to score flows."""
from __future__ import annotations

import datetime as dt
import json
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONTINUOUS, LABEL
from .drift import DriftMonitor
from .evaluate import binary_metrics
from .features import SchemaError, coerce_flows, dense, log1p_clip
from .pipeline import DEFAULT_BUNDLE, Bundle, blend, load_bundle
from .thresholds import labelled_threshold


class Detector:
    def __init__(self, bundle: Bundle, policy: str | None = None, use_novelty: bool = False):
        self.bundle = bundle
        self.policy = policy or bundle.meta["default_policy"]
        self.use_novelty = use_novelty
        self._override: dict | None = None
        self._lock = threading.Lock()
        self.monitor = DriftMonitor(bundle.reference_scores)
        bundle.rf.set_single_row_mode()
        if self.policy not in bundle.thresholds:
            raise ValueError(f"unknown policy {self.policy!r}; choose from {sorted(bundle.thresholds)}")

    @classmethod
    def load(cls, path: str | Path = DEFAULT_BUNDLE, **kw) -> "Detector":
        return cls(load_bundle(path), **kw)

    # ---- thresholds / policy ---------------------------------------------------------------------
    @property
    def policies(self) -> list[str]:
        return sorted(self.bundle.thresholds)

    @property
    def threshold(self) -> float:
        return self._override["threshold"] if self._override else self.bundle.thresholds[self.policy]

    def set_policy(self, name: str) -> None:
        if name not in self.bundle.thresholds:
            raise ValueError(f"unknown policy {name!r}; choose from {self.policies}")
        with self._lock:
            self.policy, self._override = name, None

    # ---- scoring -----------------------------------------------------------------------------------
    def score_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        b = self.bundle
        X = b.preprocessor.transform(df)
        ann, rf = b.ann.predict_proba(X), b.rf.predict_proba(X)
        return pd.DataFrame({"ann_score": ann, "rf_score": rf, "blend_score": blend(ann, rf, b.blend_weight),
                             "novelty_score": b.autoencoder.score(dense(X))}, index=df.index)

    def predict_frame(self, flows, update_monitor: bool = True) -> pd.DataFrame:
        df = coerce_flows(flows)
        s = self.score_frame(df)
        t = self.threshold
        by_classifier = s["blend_score"] >= t
        novel = s["novelty_score"] >= self.bundle.novelty_threshold
        flagged = (by_classifier | novel) if self.use_novelty else by_classifier
        reason = np.select([by_classifier & novel & self.use_novelty, by_classifier, novel & self.use_novelty],
                           ["classifier+novelty", "classifier", "novelty"], default="")
        severity = np.where(~flagged, "none", np.where(s["blend_score"] >= max(t, 0.9), "high", "medium"))
        out = pd.DataFrame({
            "is_attack": flagged.to_numpy(), "attack_probability": s["blend_score"].to_numpy(),
            "severity": severity, "flag_reason": reason, "ann_score": s["ann_score"].to_numpy(),
            "rf_score": s["rf_score"].to_numpy(), "novelty_score": s["novelty_score"].to_numpy(),
            "novel": novel.to_numpy(), "threshold": t, "policy": self.policy,
        }, index=df.index)
        if update_monitor:
            self.monitor.update(s["blend_score"].to_numpy(), flagged.to_numpy())
        return out

    def top_deviations(self, flows, k: int = 3) -> list[list[dict]]:
        """Features that deviate most from typical Normal traffic (z-scores of log1p values). A cheap,
        model-agnostic hint for the analyst, not a causal explanation (SHAP comes later)."""
        df = coerce_flows(flows)
        prof = self.bundle.profile
        z = (log1p_clip(df[CONTINUOUS].to_numpy()) - prof["mean"]) / prof["std"]
        res = []
        for row in z:
            idx = np.argsort(-np.abs(row))[:k]
            res.append([{"feature": prof["features"][i], "z": round(float(row[i]), 2)} for i in idx])
        return res

    def predict(self, flows, explain: bool = True) -> list[dict]:
        out = self.predict_frame(flows)
        dev = self.top_deviations(flows) if explain else [None] * len(out)
        recs = []
        for (_, r), d in zip(out.iterrows(), dev):
            rec = {"is_attack": bool(r.is_attack), "attack_probability": float(r.attack_probability),
                   "severity": str(r.severity), "flag_reason": str(r.flag_reason) or None,
                   "ann_score": float(r.ann_score), "rf_score": float(r.rf_score),
                   "novelty_score": float(r.novelty_score), "threshold": float(r.threshold), "policy": str(r.policy)}
            if d is not None:
                rec["top_deviations"] = d
            recs.append(rec)
        return recs

    # ---- explanations ------------------------------------------------------------------------------
    def explain(self, flow: dict, n_permutations: int = 16, n_background: int = 6, seed: int = 0, top: int = 10) -> dict:
        """Shapley-value explanation of the attack probability of ONE flow (see nids.explain)."""
        from .explain import shapley_contributions
        df = coerce_flows(flow if isinstance(flow, dict) else dict(flow))
        if len(df) != 1:
            raise SchemaError("explain() takes exactly one flow")
        score_fn = lambda d: self.score_frame(d)["blend_score"].to_numpy()
        res = shapley_contributions(score_fn, df.iloc[0], self.bundle.background, n_permutations, n_background, seed)
        res["contributions"] = res["contributions"][:max(1, top)]
        res["threshold"], res["is_attack"] = self.threshold, bool(res["probability"] >= self.threshold)
        return res

    # ---- recalibration with a small labelled sample of local traffic --------------------------------
    def recalibrate(self, labelled, fpr_cap: float = 0.08, seed: int = 0, apply: bool = True) -> dict:
        """Re-tune the alert threshold on labelled flows from the target network.
        With >= 400 flows, 30% are held out and the new threshold is applied only if it does not lose accuracy
        on that hold-out compared with the current one."""
        df = coerce_flows(labelled, require=[LABEL])
        y = df[LABEL].to_numpy().astype(int)
        s = self.score_frame(df)["blend_score"].to_numpy()
        rng = np.random.RandomState(seed)
        hold = np.zeros(len(y), bool)
        if len(y) >= 400:
            for c in (0, 1):
                idx = np.where(y == c)[0]
                hold[rng.choice(idx, int(0.3 * len(idx)), replace=False)] = True
        fit = ~hold
        new_t = labelled_threshold(s[fit], y[fit], fpr_cap)
        report = {"n_flows": int(len(y)), "n_fit": int(fit.sum()), "n_holdout": int(hold.sum()),
                  "old_threshold": self.threshold, "new_threshold": new_t, "fpr_cap": fpr_cap}
        if hold.any():
            old_m, new_m = binary_metrics(y[hold], s[hold] >= self.threshold), binary_metrics(y[hold], s[hold] >= new_t)
            report.update(holdout_old=old_m, holdout_new=new_m)
            ok = new_m["accuracy"] >= old_m["accuracy"] - 0.01
        else:
            report["note"] = "fewer than 400 flows: no hold-out check was possible"
            ok = True
        report["applied"] = bool(apply and ok)
        if report["applied"]:
            with self._lock:
                self._override = {"threshold": new_t, "fpr_cap": fpr_cap, "n_flows": int(len(y)),
                                  "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
        elif apply:
            report["reason"] = "new threshold was worse than the current one on the hold-out; kept the current one"
        return report

    def reset_recalibration(self) -> None:
        with self._lock:
            self._override = None

    def save_override(self, path: str | Path) -> None:
        if not self._override:
            raise RuntimeError("no recalibration to save")
        Path(path).write_text(json.dumps(self._override, indent=2))

    def load_override(self, path: str | Path) -> None:
        o = json.loads(Path(path).read_text())
        if not isinstance(o.get("threshold"), (int, float)) or not 0.0 <= o["threshold"] <= 1.0 + 1e-9:
            raise ValueError("invalid override file")
        with self._lock:
            self._override = o

    def info(self) -> dict:
        m = self.bundle.meta
        return {"policy": self.policy, "threshold": self.threshold, "use_novelty": self.use_novelty,
                "recalibrated": self._override is not None, "policies": self.policies,
                "trained_at_utc": m["trained_at_utc"], "validation_metrics": m["validation_metrics"],
                "schema_version": m["schema_version"]}
