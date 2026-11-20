"""Replay stored flows as a stream, for the real-time demo (clearly a SIMULATION of live traffic)."""
from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import pandas as pd

from .config import ATTACK_CAT, FEATURES, LABEL


def iter_replay(df: pd.DataFrame, batch_size: int = 20, shuffle: bool = True, seed: int = 0,
                max_batches: int | None = None, flows_per_second: float | None = None) -> Iterator[pd.DataFrame]:
    """Yield consecutive batches of flows. If `flows_per_second` is given, sleeps to mimic that arrival rate.
    Ground-truth columns (label, attack_cat), when present, are kept so a demo can show hits and misses; they
    are never passed to the model (Detector only reads FEATURES)."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"replay table is missing feature columns: {missing}")
    if flows_per_second is not None and flows_per_second <= 0:
        raise ValueError("flows_per_second must be > 0")
    order = np.random.RandomState(seed).permutation(len(df)) if shuffle else np.arange(len(df))
    for b, start in enumerate(range(0, len(df), batch_size)):
        if max_batches is not None and b >= max_batches:
            return
        chunk = df.iloc[order[start:start + batch_size]]
        if flows_per_second:
            time.sleep(len(chunk) / flows_per_second)
        yield chunk
