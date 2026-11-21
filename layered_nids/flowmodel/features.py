"""Input validation and the single preprocessing used for training AND serving."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .contract import CATEGORICAL, NUMERIC, FEATURES


class SchemaError(ValueError):
    """Incoming flows do not match the data contract."""


def coerce(flows) -> pd.DataFrame:
    if isinstance(flows, pd.DataFrame):
        df = flows.copy()
    elif isinstance(flows, dict):
        df = pd.DataFrame([flows])
    elif isinstance(flows, (list, tuple)):
        if not flows or not all(isinstance(f, dict) for f in flows):
            raise SchemaError("flows must be a non-empty list of objects")
        df = pd.DataFrame(list(flows))
    else:
        raise SchemaError(f"unsupported input type: {type(flows).__name__}")
    if len(df) == 0:
        raise SchemaError("no flows given")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required fields: {missing}")
    df = df[FEATURES].copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().str.lower()
    try:
        df[NUMERIC] = df[NUMERIC].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as e:
        raise SchemaError(f"non-numeric value in a numeric field: {e}") from e
    v = df[NUMERIC].to_numpy(float)
    if not np.isfinite(v).all():
        raise SchemaError("numeric fields must be finite (no NaN/inf/empty values)")
    df[NUMERIC] = np.clip(v, 0, None)            # flow counters are never negative
    return df.reset_index(drop=True)


class Preprocessor:
    """one-hot categoricals (unseen categories ignored) + log1p + standard scaling, fitted on training rows only"""

    def fit(self, df):
        self.oh = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(df[CATEGORICAL])
        self.sc = StandardScaler().fit(self._n(df))
        return self

    @staticmethod
    def _n(df):
        return np.log1p(np.clip(df[NUMERIC].to_numpy(float), 0, None))

    def transform(self, df):
        return np.hstack([self.oh.transform(df[CATEGORICAL]), self.sc.transform(self._n(df))]).astype(np.float32)
