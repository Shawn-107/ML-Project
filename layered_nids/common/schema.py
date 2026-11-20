"""The one record every layer emits and the fusion layer consumes.

A layer never raises an alert on its own: it returns Evidence (a score in [0, 1] about one host at one
time, plus a human-readable reason). Fusion decides when evidence adds up to an alert.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

LAYERS = ("rules", "dga", "packets", "flows")


@dataclass(frozen=True)
class Evidence:
    layer: str            # one of LAYERS
    host: str             # the internal host the evidence is about (IP string or any stable id)
    ts: float             # seconds; event time, not wall-clock
    score: float          # 0 = looks normal, 1 = very suspicious; already comparable across layers by calibration
    reason: str           # short sentence a human can read
    detail: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.layer not in LAYERS:
            raise ValueError(f"unknown layer {self.layer!r}; expected one of {LAYERS}")
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0, 1], got {self.score}")

    def to_dict(self) -> dict:
        return asdict(self)
