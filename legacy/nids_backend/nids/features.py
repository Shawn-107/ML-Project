"""The single preprocessing step used for training AND inference (so they can never diverge)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .config import BINARY, CATEGORICAL, CONTINUOUS, FEATURES, SCHEMA_VERSION


class SchemaError(ValueError):
    """Raised when incoming flows do not match the feature contract."""


def log1p_clip(x):
    """log(1+x) with negatives clipped to 0 (flow counters are never negative)."""
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0, None))


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ("num", Pipeline([("log", FunctionTransformer(log1p_clip)), ("scale", StandardScaler())]), CONTINUOUS),
            ("bin", "passthrough", BINARY),
        ]
    )


def dense(x) -> np.ndarray:
    return np.asarray(x.todense()) if sparse.issparse(x) else np.asarray(x)


def normal_profile(normal_rows: pd.DataFrame) -> dict:
    """Mean / std of log1p(feature) over Normal training flows, used for 'top deviating features'."""
    z = log1p_clip(normal_rows[CONTINUOUS].to_numpy())
    return {"features": list(CONTINUOUS), "mean": z.mean(0), "std": np.maximum(z.std(0), 1e-6)}


def normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Strip + lower-case the categorical columns. Used for training data and incoming flows alike, so
    'TCP', 'tcp ' and 'tcp' are the same category everywhere."""
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().str.lower()
    return df


def coerce_flows(flows, require: list[str] | None = None) -> pd.DataFrame:
    """Accept a dict, a list of dicts or a DataFrame and return a clean DataFrame with exactly FEATURES
    (plus any extra columns listed in `require`). Raises SchemaError with a precise message."""
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
    missing = [c for c in FEATURES + (require or []) if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required fields: {missing}")
    df = normalise_categoricals(df)
    nums = CONTINUOUS + BINARY
    try:
        df[nums] = df[nums].apply(pd.to_numeric, errors="raise")
    except (ValueError, TypeError) as e:
        raise SchemaError(f"non-numeric value in a numeric field: {e}") from e
    vals = df[nums].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        raise SchemaError("numeric fields must be finite (no NaN/inf/empty values)")
    return df
