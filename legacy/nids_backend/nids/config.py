"""Central configuration and the data contract (feature schema) shared by every module."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SCHEMA_VERSION = 2  # bump whenever the bundle layout or feature contract changes

# ---- data contract: the exact 34 input features, in order -------------------------------------
CATEGORICAL = ["proto", "service", "state"]
BINARY = ["is_sm_ips_ports"]
CONTINUOUS = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload", "sloss", "dloss",
    "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt", "synack",
    "ackdat", "smean", "dmean", "trans_depth", "response_body_len", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd",
]
FEATURES = [
    "dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sload", "dload",
    "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb", "dtcpb", "dwin", "tcprtt",
    "synack", "ackdat", "smean", "dmean", "trans_depth", "response_body_len", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd", "is_sm_ips_ports",
]
assert set(FEATURES) == set(CATEGORICAL) | set(BINARY) | set(CONTINUOUS) and len(FEATURES) == 34

LABEL = "label"           # 0 = Normal, 1 = Attack
ATTACK_CAT = "attack_cat"  # 'Normal' or one of 9 attack families
NORMAL = "Normal"
ATTACK_FAMILIES = ["Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance",
                   "Shellcode", "Worms"]


def policy_name(target_fpr: float) -> str:
    return f"fpr_{int(round(target_fpr * 100))}pct"


@dataclass
class Config:
    # data. NOTE: the Hugging Face mirror names these files the other way round; these are the OFFICIAL sets.
    train_path: Path = DATA_DIR / "official_training_175341.csv"
    test_path: Path = DATA_DIR / "official_testing_82332.csv"
    val_size: float = 0.2
    seed: int = 0
    max_train_rows: int | None = None  # for fast tests only

    # models
    ann_hidden: tuple = (128, 64)
    ann_alpha: float = 1e-4
    ann_seeds: int = 5
    ann_max_iter: int = 40
    rf_trees: int = 150
    rf_min_leaf: int = 2
    blend_weight: float = 0.5  # weight of the ANN in the ANN/RF blend
    ae_hidden: tuple = (64, 16, 64)
    ae_max_iter: int = 30

    # alert policies: name -> target false-alarm rate on validation Normal traffic
    fpr_targets: tuple = (0.01, 0.03, 0.05)
    default_policy: str = "fpr_3pct"
    novelty_fpr: float = 0.05

    # experiments / explanations
    loao_seeds: int = 2          # ANN seeds per leave-one-attack-out fold
    background_size: int = 200   # Normal training rows kept for explanations

    def __post_init__(self):
        self.train_path, self.test_path = Path(self.train_path), Path(self.test_path)
        if not 0.05 <= self.val_size <= 0.5:
            raise ValueError("val_size must be between 0.05 and 0.5")
        if self.ann_seeds < 1 or self.loao_seeds < 1:
            raise ValueError("ann_seeds and loao_seeds must be >= 1")
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValueError("blend_weight must be in [0, 1]")
        if not self.fpr_targets or any(not (0 < t < 1) or abs(t * 100 - round(t * 100)) > 1e-9 or round(t * 100) < 1
                                       for t in self.fpr_targets):
            raise ValueError("fpr_targets must be whole percentages between 1% and 99%, e.g. (0.01, 0.03)")
        if len({policy_name(t) for t in self.fpr_targets}) != len(self.fpr_targets):
            raise ValueError("fpr_targets contains duplicates")
        if self.default_policy not in {policy_name(t) for t in self.fpr_targets} | {"f1"}:
            raise ValueError(f"default_policy {self.default_policy!r} is not one of the configured policies")
        if not 0 < self.novelty_fpr < 1:
            raise ValueError("novelty_fpr must be in (0, 1)")
        if self.background_size < 10:
            raise ValueError("background_size must be >= 10")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["train_path"], d["test_path"] = str(self.train_path), str(self.test_path)
        return d

    @classmethod
    def fast(cls, rows: int = 6000, **kw) -> "Config":
        """Small, quick configuration used by the unit tests."""
        base = dict(max_train_rows=rows, ann_seeds=1, ann_max_iter=8, rf_trees=20, ae_max_iter=5,
                    loao_seeds=1, background_size=40)
        base.update(kw)
        return cls(**base)
