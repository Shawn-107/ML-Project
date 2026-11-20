"""L4 data access: the legacy `nids` splits and the legacy fitted preprocessor, as plain numpy arrays.

We REUSE the legacy package (cleaning, split, preprocessor, trained ANN + forest). Nothing here re-implements them.
The legacy bundle must have been trained on the same splits; this is checked, not assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from nids.config import ATTACK_CAT, FEATURES, LABEL, NORMAL, Config
from nids.data import load_splits
from nids.features import dense
from nids.pipeline import Bundle, load_bundle


@dataclass
class FlowData:
    bundle: Bundle
    splits: object
    feature_names: list          # transformed column names (one-hot expanded)
    Xtr: np.ndarray
    Xv: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yv: np.ndarray
    yte: np.ndarray
    ctr: np.ndarray              # attack_cat strings
    cv: np.ndarray
    cte: np.ndarray

    def raw(self, part: str) -> pd.DataFrame:
        return getattr(self.splits, {"tr": "train", "v": "val", "te": "test"}[part])


def load_flow_data(cfg: Config | None = None) -> FlowData:
    cfg = cfg or Config()
    bundle = load_bundle()
    sp = load_splits(cfg)
    info = bundle.meta.get("data_info", {})
    if info.get("n_train") != len(sp.train) or info.get("n_val") != len(sp.val):
        raise RuntimeError(f"bundle was trained on different splits ({info.get('n_train')}/{info.get('n_val')} rows) "
                           f"than load_splits gives ({len(sp.train)}/{len(sp.val)}); retrain with `python -m nids train`")
    pre = bundle.preprocessor

    def X(df):
        return dense(pre.transform(df[FEATURES])).astype(np.float32)

    Xtr = X(sp.train)
    names = transformed_names(pre)
    if len(names) != Xtr.shape[1]:
        raise RuntimeError(f"feature-name count {len(names)} != matrix width {Xtr.shape[1]}")
    return FlowData(bundle, sp, names, Xtr, X(sp.val), X(sp.test),
                    sp.train[LABEL].to_numpy(), sp.val[LABEL].to_numpy(), sp.test[LABEL].to_numpy(),
                    sp.train[ATTACK_CAT].to_numpy(), sp.val[ATTACK_CAT].to_numpy(), sp.test[ATTACK_CAT].to_numpy())


def transformed_names(pre) -> list:
    """Column names of the legacy preprocessor output (its log transform has no get_feature_names_out).
    Order is the ColumnTransformer order: one-hot categoricals, scaled continuous, binary."""
    from nids.config import BINARY, CATEGORICAL, CONTINUOUS
    cat = pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
    return [f"cat__{n}" for n in cat] + [f"num__{c}" for c in CONTINUOUS] + [f"bin__{c}" for c in BINARY]


def source_feature(transformed_name: str) -> str:
    """'cat__proto_tcp' -> 'proto'; 'num__sbytes' -> 'sbytes'; 'bin__is_sm_ips_ports' -> 'is_sm_ips_ports'."""
    base = transformed_name.split("__", 1)[-1]
    for f in ("proto", "service", "state"):
        if transformed_name.startswith("cat__") and base.startswith(f + "_"):
            return f
    return base
