"""Loading, validating, cleaning and splitting the data. No model code here."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import ATTACK_CAT, FEATURES, LABEL, NORMAL, Config
from .features import normalise_categoricals

log = logging.getLogger(__name__)


class DataError(ValueError):
    """Raised when a table does not satisfy the data contract."""


def read_table(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"data file not found: {p}")
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(p)
        except ImportError:  # no pyarrow/fastparquet: use the small built-in reader
            from ._mini_parquet import read_parquet
            log.warning("pyarrow not installed; using built-in minimal parquet reader")
            return read_parquet(str(p))
    raise DataError(f"unsupported file type: {p.suffix} (use .csv or .parquet)")


def validate_labelled(df: pd.DataFrame, name: str = "table") -> None:
    need = FEATURES + [LABEL, ATTACK_CAT]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise DataError(f"{name}: missing columns {missing}")
    if df[FEATURES].isna().any().any():
        raise DataError(f"{name}: NaN values in feature columns")
    if not set(df[LABEL].unique()) <= {0, 1}:
        raise DataError(f"{name}: '{LABEL}' must contain only 0/1")
    if not ((df[LABEL] == 0) == (df[ATTACK_CAT] == NORMAL)).all():
        raise DataError(f"{name}: '{LABEL}' disagrees with '{ATTACK_CAT}' ('{NORMAL}' must be exactly label 0)")


def _row_hash(df: pd.DataFrame) -> pd.Series:
    return pd.util.hash_pandas_object(df[FEATURES], index=False)


def clean_table(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove ambiguous and repeated flows, judged on the FEATURES (what the model sees):
    1. feature vectors that appear with conflicting attack/normal labels are dropped entirely (label noise);
    2. remaining repeated feature vectors are collapsed to one row.
    Full-row dedupe alone is not enough: rows identical in features but different in attack_cat survive it
    and then leak between train and validation."""
    n0 = len(df)
    conflict = df.groupby(FEATURES, dropna=False)[LABEL].transform("nunique") > 1
    df = df[~conflict]
    n_conflict = int(conflict.sum())
    df = df.drop_duplicates(subset=FEATURES).reset_index(drop=True)
    return df, {"rows": n0, "dropped_conflicting_label": n_conflict, "after_dedupe": len(df)}


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    info: dict = field(default_factory=dict)


def load_splits(cfg: Config) -> Splits:
    """Official train/test sets -> deduplicated; test rows that also appear in training are removed;
    a stratified validation set is carved out of the training pool. The test set is never used for fitting
    or for choosing thresholds."""
    train_raw, test_raw = read_table(cfg.train_path), read_table(cfg.test_path)
    validate_labelled(train_raw, "train"), validate_labelled(test_raw, "test")
    train_raw, test_raw = normalise_categoricals(train_raw), normalise_categoricals(test_raw)
    train, train_clean = clean_table(train_raw)
    test, test_clean = clean_table(test_raw)
    n_test_dedup = len(test)
    test = test[~_row_hash(test).isin(set(_row_hash(train)))].reset_index(drop=True)
    if cfg.max_train_rows and len(train) > cfg.max_train_rows:
        train, _ = train_test_split(train, train_size=cfg.max_train_rows, stratify=train[ATTACK_CAT],
                                    random_state=cfg.seed)
        train = train.reset_index(drop=True)
    tr, val = train_test_split(train, test_size=cfg.val_size, stratify=train[ATTACK_CAT], random_state=cfg.seed)
    info = dict(train_raw=len(train_raw), test_raw=len(test_raw), train_dedup=len(train),
                test_dedup=n_test_dedup, test_after_overlap_removal=len(test),
                train_cleaning=train_clean, test_cleaning=test_clean,
                n_train=len(tr), n_val=len(val), n_test=len(test))
    log.info("splits: %s", info)
    return Splits(tr.reset_index(drop=True), val.reset_index(drop=True), test, info)
