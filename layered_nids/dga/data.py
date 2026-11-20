"""DGA data: load, clean, and split by FAMILY so test families are never seen in training.

Source file (data/dga/dga_legit.csv, see data/SOURCES.md): columns domain, family, type; family 'legit' = benign.

Choices that matter for honesty:
- The TLD is removed before modelling. In this file DGA domains often use .tk/.xyz/.top while benign ones are
  mostly .com, so a model could score well from the TLD alone. We want it to learn the name.
- Names are de-duplicated after the TLD is removed (abcbot has itekgvzlw.tk AND itekgvzlw.com).
- DGA families are capped (max_per_family) because the largest has 519k rows and the smallest has 2.
  Without a cap, big families dominate both training and the average.
- Test folds contain whole families the model never trained on. Benign rows are split by row (they have no family).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from layered_nids.common.paths import DATA

DEFAULT_PATH = DATA / "dga" / "dga_legit.csv"
BENIGN = "legit"


def strip_tld(domain: str) -> str:
    """'abc.example.co' -> 'abc.example'. Only the last label is dropped (no public-suffix list on purpose:
    it is not installed and a heuristic is enough for this purpose; multi-part suffixes keep their 2nd-level part)."""
    d = domain.strip().lower().rstrip(".")
    return d.rsplit(".", 1)[0] if "." in d else d


def load_domains(path: str | Path = DEFAULT_PATH, max_per_family: int = 4000, n_benign: int = 150_000,
                 min_family_size: int = 20, seed: int = 0) -> pd.DataFrame:
    """Return a frame with columns name, family, y (1 = generated), sampled and de-duplicated."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["domain", "family"])
    raw["name"] = raw["domain"].map(strip_tld)
    raw = raw[(raw["name"].str.len() >= 3) & (raw["name"].str.len() <= 120)]
    raw = raw.drop_duplicates("name", keep="first")
    # A name that appears in both the benign and a DGA group is ambiguous: drop it everywhere.
    ambiguous = raw.groupby("name")["family"].transform(lambda s: (s == BENIGN).any() and (s != BENIGN).any())
    raw = raw[~ambiguous]

    rng = np.random.default_rng(seed)
    parts = []
    benign = raw[raw["family"] == BENIGN]
    parts.append(benign.sample(n=min(n_benign, len(benign)), random_state=int(rng.integers(1 << 31))))
    dga = raw[raw["family"] != BENIGN]
    sizes = dga["family"].value_counts()
    keep = sizes[sizes >= min_family_size].index
    for fam, g in dga[dga["family"].isin(keep)].groupby("family"):
        parts.append(g.sample(n=min(max_per_family, len(g)), random_state=int(rng.integers(1 << 31))))
    out = pd.concat(parts, ignore_index=True)
    out["y"] = (out["family"] != BENIGN).astype(np.int8)
    return out[["name", "family", "y"]].sample(frac=1.0, random_state=seed).reset_index(drop=True)


@dataclass
class FoldSplit:
    fold: int
    train: pd.DataFrame
    val: pd.DataFrame      # same families as train, different rows: used only to set the threshold
    test: pd.DataFrame     # families never seen in train/val; benign rows disjoint from train/val
    test_families: list[str]


def family_folds(df: pd.DataFrame, n_folds: int = 5, val_frac: float = 0.15, seed: int = 0):
    """Yield FoldSplit objects. Families are shuffled once and dealt into n_folds groups; each group is the test
    set once. Benign rows are dealt the same way by row. Every row is tested exactly once across folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    rng = np.random.default_rng(seed)
    families = np.array(sorted(df.loc[df["y"] == 1, "family"].unique()))
    rng.shuffle(families)
    fam_fold = {f: i % n_folds for i, f in enumerate(families)}
    ben_idx = np.flatnonzero(df["y"].to_numpy() == 0)
    ben_fold = np.empty(len(df), dtype=np.int16)
    ben_fold[ben_idx] = rng.permutation(len(ben_idx)) % n_folds
    row_fold = np.where(df["y"].to_numpy() == 1, df["family"].map(fam_fold).fillna(-1).to_numpy(), ben_fold)

    for k in range(n_folds):
        test = df[row_fold == k].reset_index(drop=True)
        rest = df[row_fold != k]
        is_val = rng.random(len(rest)) < val_frac
        train, val = rest[~is_val].reset_index(drop=True), rest[is_val].reset_index(drop=True)
        yield FoldSplit(k, train, val, test, sorted(set(test.loc[test["y"] == 1, "family"])))


def check_split(s: FoldSplit) -> None:
    """Raise if the split leaks. Cheap enough to run on every fold."""
    seen = set(s.train.loc[s.train["y"] == 1, "family"]) | set(s.val.loc[s.val["y"] == 1, "family"])
    overlap = seen & set(s.test_families)
    if overlap:
        raise AssertionError(f"test families also in train/val: {sorted(overlap)[:5]}")
    names_tr = set(s.train["name"]) | set(s.val["name"])
    dup = names_tr & set(s.test["name"])
    if dup:
        raise AssertionError(f"{len(dup)} test names also in train/val, e.g. {sorted(dup)[:3]}")
