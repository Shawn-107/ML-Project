"""Hand-built lexical features for a domain name (TLD already removed). Used by the lexical baseline and the MLP."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

VOWELS = set("aeiou")
HEX = set("0123456789abcdef")
FEATURE_NAMES = (
    "length", "digit_ratio", "vowel_ratio", "consonant_ratio", "hyphen_count", "dot_count",
    "max_consonant_run", "max_vowel_run", "max_digit_run", "entropy", "unique_char_ratio",
    "hex_ratio", "longest_label", "n_labels", "digit_to_letter_switches",
)


def _max_run(s: str, pred) -> int:
    best = cur = 0
    for ch in s:
        if pred(ch):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in Counter(s).values())


def lexical_vector(name: str) -> list[float]:
    n = max(len(name), 1)
    letters = [c for c in name if c.isalpha()]
    digits = sum(c.isdigit() for c in name)
    vowels = sum(c in VOWELS for c in name)
    labels = name.split(".")
    switches = sum(1 for a, b in zip(name, name[1:]) if (a.isdigit() and b.isalpha()) or (a.isalpha() and b.isdigit()))
    return [
        float(len(name)), digits / n, vowels / n, (len(letters) - vowels) / n, float(name.count("-")), float(name.count(".")),
        float(_max_run(name, lambda c: c.isalpha() and c not in VOWELS)), float(_max_run(name, lambda c: c in VOWELS)),
        float(_max_run(name, str.isdigit)), entropy(name), len(set(name)) / n,
        sum(c in HEX for c in name) / n, float(max(len(l) for l in labels)), float(len(labels)), float(switches),
    ]


def lexical_matrix(names) -> np.ndarray:
    return np.asarray([lexical_vector(n) for n in names], dtype=np.float32)
