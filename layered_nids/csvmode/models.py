"""The two ML models behind CSV mode, trained once, saved with a checksum, and scored in a way that tolerates bad input.

  unsw34      gradient boosting on the earlier project's 34 UNSW-NB15 features and its fitted preprocessor (cleaned official train split)
  netflow_v2  gradient boosting on the 40 NetFlow v2 features (NF-ToN-IoT-v2, split by source host)

Both use thresholds named by false-alarm target on VALIDATION benign flows (1% and 3%), like the rest of the project.
Missing or unreadable numbers become NaN (gradient boosting handles them); they are counted and reported, never silently zeroed.
Inputs far outside the training range are counted too, so a file that looks unlike the training data produces a warning.

SECURITY: model files are pickles. Only load files you trained yourself; load_model checks a SHA-256 written at training time.

   python -m layered_nids.csvmode.models train [--which unsw34,netflow_v2]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from layered_nids.common.metrics import threshold_at_fpr
from layered_nids.common.paths import MODELS
from layered_nids.csvmode.reader import CsvError
from layered_nids.csvmode.schemas import NETFLOW_FEATURES

MODEL_DIR = MODELS / "csv"
POLICIES = {"fpr_1pct": 0.01, "fpr_3pct": 0.03}


# ---------------------------------------------------------------------------------------------- input range check
class RangeCheck:
    """Per column, the 0.1%-99.9% range of log1p(value) seen in training. A value is 'out of range' if it lies more than one
    natural-log unit (a factor of e) beyond that range. Cheap, and enough to notice a file from a different kind of network."""

    def __init__(self, lo: dict, hi: dict):
        self.lo, self.hi = lo, hi

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "RangeCheck":
        z = np.log1p(df.clip(lower=0))
        return cls({c: float(z[c].quantile(0.001)) for c in df.columns}, {c: float(z[c].quantile(0.999)) for c in df.columns})

    def count(self, df: pd.DataFrame) -> np.ndarray:
        z = np.log1p(df.clip(lower=0))
        n = np.zeros(len(df), dtype=np.int32)
        for c in self.lo:
            if c in z:
                v = z[c].to_numpy()
                n += ((v < self.lo[c] - 1.0) | (v > self.hi[c] + 1.0)).astype(np.int32)
        return n


def to_numeric_frame(df: pd.DataFrame, cols) -> tuple[pd.DataFrame, np.ndarray]:
    """Numeric columns with NaN for anything unreadable or infinite. Also returns, per row, how many NON-EMPTY cells were not numbers."""
    out = pd.DataFrame(index=df.index)
    bad = np.zeros(len(df), dtype=np.int32)
    for c in cols:
        raw = df[c]
        v = pd.to_numeric(raw, errors="coerce").replace([np.inf, -np.inf], np.nan)
        bad += (v.isna() & raw.ne("")).to_numpy().astype(np.int32)
        out[c] = v
    return out, bad


# ---------------------------------------------------------------------------------------------- scoring
def score_unsw(df: pd.DataFrame, m: dict) -> tuple[np.ndarray, dict]:
    num, bad = to_numeric_frame(df, m["continuous"] + m["binary"])
    frame = pd.DataFrame(index=df.index)
    for c in m["categorical"]:
        frame[c] = df[c].str.lower().replace("", "-")
    for c in m["continuous"] + m["binary"]:
        frame[c] = num[c]
    X = np.asarray(m["pre"].transform(frame[m["features"]]), dtype=np.float32)
    p = m["gb"].predict_proba(X)[:, 1]
    return p, {"bad_cells": bad, "missing": num.isna().sum(axis=1).to_numpy(), "out_of_range": m["ranges"].count(num[m["continuous"]])}


def score_netflow(df: pd.DataFrame, m: dict) -> tuple[np.ndarray, dict]:
    num, bad = to_numeric_frame(df, m["features"])
    p = m["gb"].predict_proba(num[m["features"]].to_numpy(dtype=np.float32))[:, 1]
    return p, {"bad_cells": bad, "missing": num.isna().sum(axis=1).to_numpy(), "out_of_range": m["ranges"].count(num)}


SCORERS = {"unsw34": score_unsw, "netflow_v2": score_netflow}


# ---------------------------------------------------------------------------------------------- save / load
def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(kind: str, model: dict, meta: dict, directory: Path = MODEL_DIR) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}.joblib"
    tmp = path.with_suffix(".tmp")
    joblib.dump(model, tmp, compress=3)
    tmp.replace(path)
    meta = dict(meta, kind=kind, sha256=_sha(path), sklearn=sklearn.__version__, numpy=np.__version__, pandas=pd.__version__,
                trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (directory / f"{kind}.meta.json").write_text(json.dumps(meta, indent=2, default=float))
    return path


def load_model(kind: str, directory: Path = MODEL_DIR) -> dict:
    path, meta_path = directory / f"{kind}.joblib", directory / f"{kind}.meta.json"
    if not path.exists() or not meta_path.exists():
        raise CsvError(f"model '{kind}' is not trained yet: run  python -m layered_nids.csvmode.models train --which {kind}")
    if json.loads(meta_path.read_text()).get("sha256") != _sha(path):
        raise CsvError(f"model file {path.name} failed its integrity check (changed since training); retrain it")
    return joblib.load(path)


# ---------------------------------------------------------------------------------------------- training
def _thresholds(val_benign_scores) -> dict:
    return {name: threshold_at_fpr(val_benign_scores, fpr) for name, fpr in POLICIES.items()}


def train_unsw(seed: int = 0) -> tuple[dict, dict]:
    from nids.config import BINARY, CATEGORICAL, CONTINUOUS, FEATURES
    from layered_nids.flows.data import load_flow_data
    from layered_nids.flows.evaluate import evaluate_scores
    from layered_nids.flows.models import HGB
    d = load_flow_data()
    gb = HGB(seed, max_iter=400).fit(d.Xtr, d.ytr)
    sv, st = gb.score(d.Xv), gb.score(d.Xte)
    thr = _thresholds(sv[d.yv == 0])
    rc = RangeCheck.fit(d.raw("tr")[CONTINUOUS].astype(float))
    metrics = evaluate_scores(sv, d.yv, st, d.yte, d.cte)           # thresholds from the full validation set; test scored once
    model = {"kind": "unsw34", "pre": d.bundle.preprocessor, "gb": gb.m, "thresholds": thr, "ranges": rc, "features": list(FEATURES),
             "categorical": list(CATEGORICAL), "continuous": list(CONTINUOUS), "binary": list(BINARY)}
    meta = {"trained_on": {"rows": int(len(d.ytr)), "source": "UNSW-NB15 official training split, cleaned (legacy nids.data)"},
            "thresholds": thr, "official_test_cleaned": {k: metrics[k] for k in ("auc_test", "fpr_1pct", "fpr_3pct") if k in metrics}}
    for k in ("fpr_1pct", "fpr_3pct"):
        meta["official_test_cleaned"][k] = {x: y for x, y in metrics[k].items() if x != "flagged_by_family"}
    return model, meta


def train_netflow(frac: float = 0.2, seed: int = 0) -> tuple[dict, dict]:
    from sklearn.model_selection import GroupShuffleSplit
    from layered_nids.flows.evaluate import evaluate_scores
    from layered_nids.flows.models import HGB
    from layered_nids.flows.netflow import ID_COLS, LABELS, load_sample
    df = load_sample(frac, seed)
    all_cols = [c for c in df.columns if c not in ID_COLS + LABELS]
    df = df.drop_duplicates(subset=all_cols + ["IPV4_SRC_ADDR"]).reset_index(drop=True)
    groups = df["IPV4_SRC_ADDR"].to_numpy()
    tr, rest = next(GroupShuffleSplit(1, train_size=0.7, random_state=seed).split(df, groups=groups))
    va_rel, te_rel = next(GroupShuffleSplit(1, train_size=0.5, random_state=seed).split(rest, groups=groups[rest]))
    va, te = rest[va_rel], rest[te_rel]
    y, cat = df["Label"].to_numpy(), df["Attack"].to_numpy()
    low = df[all_cols].rename(columns=str.lower)
    assert list(low.columns) == NETFLOW_FEATURES, "NetFlow column order/names changed"
    X = low.to_numpy(dtype=np.float32)
    gb = HGB(seed, max_iter=300).fit(X[tr], y[tr])
    sv, st = gb.score(X[va]), gb.score(X[te])
    thr = _thresholds(sv[y[va] == 0])
    metrics = evaluate_scores(sv, y[va], st, y[te], cat[te])
    model = {"kind": "netflow_v2", "gb": gb.m, "thresholds": thr, "ranges": RangeCheck.fit(low.iloc[tr].astype(float)),
             "features": list(NETFLOW_FEATURES)}
    meta = {"trained_on": {"rows": int(len(tr)), "source": "NF-ToN-IoT-v2 sample, exact repeats removed, split by source host"},
            "thresholds": thr, "test_hosts_unseen": {"auc": metrics["auc_test"], **{k: {x: y2 for x, y2 in metrics[k].items() if x != "flagged_by_family"}
                                                                               for k in ("fpr_1pct", "fpr_3pct")}},
            "caution": "this dataset is easy (AUC 1.0 on unseen hosts with all features); do not read that as real-world accuracy"}
    return model, meta


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["train"])
    ap.add_argument("--which", default="unsw34,netflow_v2")
    a = ap.parse_args(argv)
    # When run with `python -m`, this file is the module `__main__`: classes pickled from it could not be loaded anywhere else.
    # Train through the importable copy so saved models record `layered_nids.csvmode.models.RangeCheck`.
    import layered_nids.csvmode.models as mod
    for kind in a.which.split(","):
        t0 = time.time()
        model, meta = {"unsw34": mod.train_unsw, "netflow_v2": mod.train_netflow}[kind]()
        path = mod.save_model(kind, model, meta)
        print(f"{kind}: trained and saved to {path} in {time.time() - t0:.0f}s; thresholds {meta['thresholds']}", flush=True)


if __name__ == "__main__":
    main()
