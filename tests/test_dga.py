"""L2 tests on GENERATED data only (the real dataset is not needed). They check plumbing and leakage rules,
not detection quality."""
import numpy as np
import pandas as pd
import pytest

from layered_nids.dga.data import BENIGN, check_split, family_folds, strip_tld
from layered_nids.dga.evaluate import evaluate_fold, summarise
from layered_nids.dga.features import FEATURE_NAMES, entropy, lexical_matrix
from layered_nids.dga.models import CharCNN, EntropyRule, LexicalLR, LexicalMLP, NgramLR

WORDS = ["mail", "shop", "news", "bank", "cloud", "game", "photo", "music", "travel", "sport", "login", "store"]


def make_frame(n_benign=600, n_families=10, per_family=60, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_benign):
        rows.append((rng.choice(WORDS) + rng.choice(WORDS) + str(rng.integers(0, 50)), BENIGN, 0))
    alphabet = np.array(list("abcdefghijklmnopqrstuvwxyz0123456789"))
    for f in range(n_families):
        for _ in range(per_family):
            rows.append(("".join(rng.choice(alphabet, size=int(rng.integers(10, 22)))), f"fam{f}", 1))
    df = pd.DataFrame(rows, columns=["name", "family", "y"]).drop_duplicates("name")
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def test_strip_tld():
    assert strip_tld("Google.com") == "google"
    assert strip_tld("a.b.example.co") == "a.b.example"
    assert strip_tld("nodot") == "nodot"


def test_family_folds_never_leak_and_cover_every_row_once():
    df = make_frame()
    seen_test_rows = 0
    all_test_fams = []
    for s in family_folds(df, n_folds=5, seed=1):
        check_split(s)
        seen_test_rows += len(s.test)
        all_test_fams += s.test_families
    assert seen_test_rows == len(df)                                      # each row is tested exactly once
    assert sorted(all_test_fams) == sorted(df.loc[df.y == 1, "family"].unique())   # each family tested exactly once


def test_check_split_catches_a_planted_leak():
    df = make_frame()
    s = next(iter(family_folds(df, n_folds=5, seed=1)))
    s.train = pd.concat([s.train, s.test[s.test.y == 1].head(3)])          # plant test rows in train
    with pytest.raises(AssertionError):
        check_split(s)


def test_features_shape_and_entropy():
    m = lexical_matrix(["google", "xkqzvjw8a9"])
    assert m.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(m).all()
    assert entropy("aaaa") == 0.0 and entropy("abcd") == pytest.approx(2.0)
    assert m[1, FEATURE_NAMES.index("entropy")] > m[0, FEATURE_NAMES.index("entropy")]


@pytest.mark.parametrize("cls,kw", [(LexicalLR, {}), (NgramLR, {"max_features": 5000}),
                                    (LexicalMLP, {"epochs": 3}), (CharCNN, {"epochs": 3})])
def test_models_learn_an_easy_synthetic_task(cls, kw):
    df = make_frame(seed=3)
    s = next(iter(family_folds(df, n_folds=5, seed=3)))
    r = evaluate_fold(cls(**kw), s)
    assert r["auc"] > 0.9                                                   # generated names are easy here
    assert 0.0 <= r["policies"]["fpr_1pct"]["false_alarm_rate"] <= 0.05
    assert "calibration" in r


def test_entropy_rule_runs_and_is_marked_uncalibrated():
    df = make_frame()
    s = next(iter(family_folds(df, n_folds=5, seed=1)))
    r = evaluate_fold(EntropyRule(), s)
    assert "calibration" not in r and r["auc"] > 0.5


def test_lofo_split_holds_out_the_whole_family():
    from layered_nids.dga.compare import lofo_split
    df = make_frame(seed=7)
    s = lofo_split(df, "fam3", seed=7)
    check_split(s)                                                       # no family or name overlap
    assert (s.test.loc[s.test.y == 1, "family"] == "fam3").all()
    assert "fam3" not in set(s.train.family) | set(s.val.family)
    assert (s.test.y == 0).any()                                         # benign rows exist so false alarms can be measured
    with pytest.raises(KeyError):
        lofo_split(df, "nope")


def test_random_split_shares_families_by_design():
    from layered_nids.dga.compare import random_split
    df = make_frame(seed=8)
    s = random_split(df, seed=8)
    assert set(s.test.family) & set(s.train.family)                      # that is the point: it is the flattering protocol
    assert not (set(s.test.name) & set(s.train.name))                    # but rows are still disjoint


def test_summarise_mean_std():
    df = make_frame(seed=5)
    folds = [evaluate_fold(LexicalLR(), s) for s in family_folds(df, n_folds=3, seed=5)]
    out = summarise(folds)
    assert out["auc"]["n"] == 3 and out["auc"]["std"] >= 0
    assert "fpr_1pct.recall" in out
