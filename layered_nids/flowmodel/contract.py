"""Data contract: the 11 raw inputs the final model needs (3 categorical + 8 numeric), chosen by forward selection."""
CATEGORICAL = ["proto", "service", "state"]
NUMERIC = ["dbytes", "smean", "dload", "ackdat", "ct_src_dport_ltm", "sbytes", "dmean", "spkts"]
FEATURES = CATEGORICAL + NUMERIC
SCHEMA_VERSION = 2   # 2 = single gradient-boosting model (1 was the ANN + forest + boosting blend)
POLICIES = {"fpr_1pct": 0.01, "fpr_3pct": 0.03, "fpr_5pct": 0.05}   # target false-alarm rate on validation Normal flows
DEFAULT_POLICY = "fpr_3pct"
NOVELTY_TARGET = 0.01
